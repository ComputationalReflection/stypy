
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' min-max othello player in 100 lines; copyleft Mark Dufour (GPL3 or later) '''
2: 
3: empty, black, white = 0, 1, -1
4: board = [[empty for x in range(8)] for y in range(8)]
5: board[3][3] = board[4][4] = white
6: board[3][4] = board[4][3] = black
7: player = {white: 'human', black: 'lalaoth'}
8: depth = 6
9: directions = [(1, 1), (-1, 1), (0, 1), (1, -1), (-1, -1), (0, -1), (1, 0), (-1, 0)]
10: corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
11: 
12: 
13: def possible_move(board, x, y, color):
14:     if board[x][y] != empty:
15:         return False
16:     for direction in directions:
17:         if flip_in_direction(board, x, y, direction, color):
18:             return True
19:     return False
20: 
21: 
22: def flip_in_direction(board, x, y, direction, color):
23:     other_color = False
24:     while True:
25:         x, y = x + direction[0], y + direction[1]
26:         if x not in range(8) or y not in range(8):
27:             return False
28:         square = board[x][y]
29:         if square == empty: return False
30:         if square != color:
31:             other_color = True
32:         else:
33:             return other_color
34: 
35: 
36: def flip_stones(board, move, color):
37:     for direction in directions:
38:         if flip_in_direction(board, move[0], move[1], direction, color):
39:             x, y = move[0] + direction[0], move[1] + direction[1]
40:             while board[x][y] != color:
41:                 board[x][y] = color
42:                 x, y = x + direction[0], y + direction[1]
43:     board[move[0]][move[1]] = color
44: 
45: 
46: def print_board(board, turn):
47:     ##    print '  '+' '.join('abcdefgh')
48:     for nr, line in enumerate(board):
49:         pass  # print nr+1, ' '.join([{white: 'O', black: 'X', empty: '.'}[square] for square in line])
50: 
51: 
52: ##    print 'turn:', player[turn]
53: ##    print 'black:', stone_count(board, black), 'white:', stone_count(board, white)
54: 
55: def possible_moves(board, color):
56:     return [(x, y) for x in range(8) for y in range(8) if possible_move(board, x, y, color)]
57: 
58: 
59: def coordinates(move):
60:     return (int(move[1]) - 1, 'abcdefgh'.index(move[0]))
61: 
62: 
63: def human_move(move):
64:     return 'abcdefgh'[move[1]] + str(move[0] + 1)
65: 
66: 
67: def stone_count(board, color):
68:     return sum([len([square for square in line if square == color]) for line in board])
69: 
70: 
71: def best_move(board, color, first, step=1):
72:     max_move, max_mobility, max_score = None, 0, 0
73:     for move in possible_moves(board, color):
74:         if move in corners:
75:             mobility, score = 64, 64
76:             if color != first:
77:                 mobility = 64 - mobility
78:         else:
79:             testboard = [[square for square in line] for line in board]
80:             flip_stones(testboard, move, color)
81:             if step < depth:
82:                 next_move, mobility = best_move(testboard, -color, first, step + 1)
83:             else:
84:                 mobility = len(possible_moves(testboard, first))
85:             score = mobility
86:             if color != first:
87:                 score = 64 - score
88:         if score >= max_score:
89:             max_move, max_mobility, max_score = move, mobility, score
90:     return max_move, max_mobility
91: 
92: 
93: def run():
94:     turn = black
95:     while possible_moves(board, black) or possible_moves(board, white):
96:         if possible_moves(board, turn):
97:             print_board(board, turn)
98:             if turn == black:
99:                 move, mobility = best_move(board, turn, turn)
100:             ##                print 'move:', human_move(move)
101:             else:
102:                 try:
103:                     move = coordinates(raw_input('move? '))
104:                 except ValueError:
105:                     ##                    print 'syntax error'
106:                     continue
107:             if not possible_move(board, move[0], move[1], turn):
108:                 ##                print 'impossible!'
109:                 continue
110:             else:
111:                 flip_stones(board, move, turn)
112:                 break  # XXX shedskin; remove to play against computer
113:         turn = -turn
114:     print_board(board, turn)
115:     if stone_count(board, black) == stone_count(board, white):
116:         pass
117:     ##        print 'draw!'
118:     else:
119:         if stone_count(board, black) > stone_count(board, white):
120:             pass  # print player[black], 'wins!'
121:         else:
122:             pass  # print player[white], 'wins!'
123:     return True
124: 
125: 
126: run()
127: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', ' min-max othello player in 100 lines; copyleft Mark Dufour (GPL3 or later) ')

# Assigning a Tuple to a Tuple (line 3):

# Assigning a Num to a Name (line 3):
int_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 22), 'int')
# Assigning a type to the variable 'tuple_assignment_1' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'tuple_assignment_1', int_25)

# Assigning a Num to a Name (line 3):
int_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 25), 'int')
# Assigning a type to the variable 'tuple_assignment_2' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'tuple_assignment_2', int_26)

# Assigning a Num to a Name (line 3):
int_27 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 28), 'int')
# Assigning a type to the variable 'tuple_assignment_3' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'tuple_assignment_3', int_27)

# Assigning a Name to a Name (line 3):
# Getting the type of 'tuple_assignment_1' (line 3)
tuple_assignment_1_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'tuple_assignment_1')
# Assigning a type to the variable 'empty' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'empty', tuple_assignment_1_28)

# Assigning a Name to a Name (line 3):
# Getting the type of 'tuple_assignment_2' (line 3)
tuple_assignment_2_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'tuple_assignment_2')
# Assigning a type to the variable 'black' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 7), 'black', tuple_assignment_2_29)

# Assigning a Name to a Name (line 3):
# Getting the type of 'tuple_assignment_3' (line 3)
tuple_assignment_3_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'tuple_assignment_3')
# Assigning a type to the variable 'white' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 14), 'white', tuple_assignment_3_30)

# Assigning a ListComp to a Name (line 4):

# Assigning a ListComp to a Name (line 4):
# Calculating list comprehension
# Calculating comprehension expression

# Call to range(...): (line 4)
# Processing the call arguments (line 4)
int_39 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 50), 'int')
# Processing the call keyword arguments (line 4)
kwargs_40 = {}
# Getting the type of 'range' (line 4)
range_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 44), 'range', False)
# Calling range(args, kwargs) (line 4)
range_call_result_41 = invoke(stypy.reporting.localization.Localization(__file__, 4, 44), range_38, *[int_39], **kwargs_40)

comprehension_42 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 9), range_call_result_41)
# Assigning a type to the variable 'y' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 9), 'y', comprehension_42)
# Calculating list comprehension
# Calculating comprehension expression

# Call to range(...): (line 4)
# Processing the call arguments (line 4)
int_33 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 31), 'int')
# Processing the call keyword arguments (line 4)
kwargs_34 = {}
# Getting the type of 'range' (line 4)
range_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 25), 'range', False)
# Calling range(args, kwargs) (line 4)
range_call_result_35 = invoke(stypy.reporting.localization.Localization(__file__, 4, 25), range_32, *[int_33], **kwargs_34)

comprehension_36 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 10), range_call_result_35)
# Assigning a type to the variable 'x' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 10), 'x', comprehension_36)
# Getting the type of 'empty' (line 4)
empty_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 10), 'empty')
list_37 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 10), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 10), list_37, empty_31)
list_43 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 9), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 9), list_43, list_37)
# Assigning a type to the variable 'board' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'board', list_43)

# Multiple assignment of 2 elements.

# Assigning a Name to a Subscript (line 5):
# Getting the type of 'white' (line 5)
white_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 28), 'white')

# Obtaining the type of the subscript
int_45 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 20), 'int')
# Getting the type of 'board' (line 5)
board_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 14), 'board')
# Obtaining the member '__getitem__' of a type (line 5)
getitem___47 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 14), board_46, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 5)
subscript_call_result_48 = invoke(stypy.reporting.localization.Localization(__file__, 5, 14), getitem___47, int_45)

int_49 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 23), 'int')
# Storing an element on a container (line 5)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), subscript_call_result_48, (int_49, white_44))

# Assigning a Subscript to a Subscript (line 5):

# Obtaining the type of the subscript
int_50 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 23), 'int')

# Obtaining the type of the subscript
int_51 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 20), 'int')
# Getting the type of 'board' (line 5)
board_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 14), 'board')
# Obtaining the member '__getitem__' of a type (line 5)
getitem___53 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 14), board_52, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 5)
subscript_call_result_54 = invoke(stypy.reporting.localization.Localization(__file__, 5, 14), getitem___53, int_51)

# Obtaining the member '__getitem__' of a type (line 5)
getitem___55 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 14), subscript_call_result_54, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 5)
subscript_call_result_56 = invoke(stypy.reporting.localization.Localization(__file__, 5, 14), getitem___55, int_50)


# Obtaining the type of the subscript
int_57 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 6), 'int')
# Getting the type of 'board' (line 5)
board_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'board')
# Obtaining the member '__getitem__' of a type (line 5)
getitem___59 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 0), board_58, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 5)
subscript_call_result_60 = invoke(stypy.reporting.localization.Localization(__file__, 5, 0), getitem___59, int_57)

int_61 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 9), 'int')
# Storing an element on a container (line 5)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 0), subscript_call_result_60, (int_61, subscript_call_result_56))

# Multiple assignment of 2 elements.

# Assigning a Name to a Subscript (line 6):
# Getting the type of 'black' (line 6)
black_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 28), 'black')

# Obtaining the type of the subscript
int_63 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 20), 'int')
# Getting the type of 'board' (line 6)
board_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 14), 'board')
# Obtaining the member '__getitem__' of a type (line 6)
getitem___65 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 14), board_64, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 6)
subscript_call_result_66 = invoke(stypy.reporting.localization.Localization(__file__, 6, 14), getitem___65, int_63)

int_67 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 23), 'int')
# Storing an element on a container (line 6)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 14), subscript_call_result_66, (int_67, black_62))

# Assigning a Subscript to a Subscript (line 6):

# Obtaining the type of the subscript
int_68 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 23), 'int')

# Obtaining the type of the subscript
int_69 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 20), 'int')
# Getting the type of 'board' (line 6)
board_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 14), 'board')
# Obtaining the member '__getitem__' of a type (line 6)
getitem___71 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 14), board_70, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 6)
subscript_call_result_72 = invoke(stypy.reporting.localization.Localization(__file__, 6, 14), getitem___71, int_69)

# Obtaining the member '__getitem__' of a type (line 6)
getitem___73 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 14), subscript_call_result_72, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 6)
subscript_call_result_74 = invoke(stypy.reporting.localization.Localization(__file__, 6, 14), getitem___73, int_68)


# Obtaining the type of the subscript
int_75 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 6), 'int')
# Getting the type of 'board' (line 6)
board_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'board')
# Obtaining the member '__getitem__' of a type (line 6)
getitem___77 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 0), board_76, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 6)
subscript_call_result_78 = invoke(stypy.reporting.localization.Localization(__file__, 6, 0), getitem___77, int_75)

int_79 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 9), 'int')
# Storing an element on a container (line 6)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 0), subscript_call_result_78, (int_79, subscript_call_result_74))

# Assigning a Dict to a Name (line 7):

# Assigning a Dict to a Name (line 7):

# Obtaining an instance of the builtin type 'dict' (line 7)
dict_80 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 9), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 7)
# Adding element type (key, value) (line 7)
# Getting the type of 'white' (line 7)
white_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 10), 'white')
str_82 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 17), 'str', 'human')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 9), dict_80, (white_81, str_82))
# Adding element type (key, value) (line 7)
# Getting the type of 'black' (line 7)
black_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 26), 'black')
str_84 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 33), 'str', 'lalaoth')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 9), dict_80, (black_83, str_84))

# Assigning a type to the variable 'player' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'player', dict_80)

# Assigning a Num to a Name (line 8):

# Assigning a Num to a Name (line 8):
int_85 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 8), 'int')
# Assigning a type to the variable 'depth' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'depth', int_85)

# Assigning a List to a Name (line 9):

# Assigning a List to a Name (line 9):

# Obtaining an instance of the builtin type 'list' (line 9)
list_86 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)

# Obtaining an instance of the builtin type 'tuple' (line 9)
tuple_87 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 9)
# Adding element type (line 9)
int_88 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 15), tuple_87, int_88)
# Adding element type (line 9)
int_89 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 15), tuple_87, int_89)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 13), list_86, tuple_87)
# Adding element type (line 9)

# Obtaining an instance of the builtin type 'tuple' (line 9)
tuple_90 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 9)
# Adding element type (line 9)
int_91 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 23), tuple_90, int_91)
# Adding element type (line 9)
int_92 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 23), tuple_90, int_92)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 13), list_86, tuple_90)
# Adding element type (line 9)

# Obtaining an instance of the builtin type 'tuple' (line 9)
tuple_93 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 32), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 9)
# Adding element type (line 9)
int_94 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 32), tuple_93, int_94)
# Adding element type (line 9)
int_95 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 32), tuple_93, int_95)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 13), list_86, tuple_93)
# Adding element type (line 9)

# Obtaining an instance of the builtin type 'tuple' (line 9)
tuple_96 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 40), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 9)
# Adding element type (line 9)
int_97 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 40), tuple_96, int_97)
# Adding element type (line 9)
int_98 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 40), tuple_96, int_98)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 13), list_86, tuple_96)
# Adding element type (line 9)

# Obtaining an instance of the builtin type 'tuple' (line 9)
tuple_99 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 49), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 9)
# Adding element type (line 9)
int_100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 49), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 49), tuple_99, int_100)
# Adding element type (line 9)
int_101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 53), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 49), tuple_99, int_101)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 13), list_86, tuple_99)
# Adding element type (line 9)

# Obtaining an instance of the builtin type 'tuple' (line 9)
tuple_102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 59), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 9)
# Adding element type (line 9)
int_103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 59), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 59), tuple_102, int_103)
# Adding element type (line 9)
int_104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 62), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 59), tuple_102, int_104)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 13), list_86, tuple_102)
# Adding element type (line 9)

# Obtaining an instance of the builtin type 'tuple' (line 9)
tuple_105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 68), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 9)
# Adding element type (line 9)
int_106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 68), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 68), tuple_105, int_106)
# Adding element type (line 9)
int_107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 71), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 68), tuple_105, int_107)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 13), list_86, tuple_105)
# Adding element type (line 9)

# Obtaining an instance of the builtin type 'tuple' (line 9)
tuple_108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 76), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 9)
# Adding element type (line 9)
int_109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 76), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 76), tuple_108, int_109)
# Adding element type (line 9)
int_110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 80), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 76), tuple_108, int_110)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 13), list_86, tuple_108)

# Assigning a type to the variable 'directions' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'directions', list_86)

# Assigning a List to a Name (line 10):

# Assigning a List to a Name (line 10):

# Obtaining an instance of the builtin type 'list' (line 10)
list_111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)

# Obtaining an instance of the builtin type 'tuple' (line 10)
tuple_112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 12), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 10)
# Adding element type (line 10)
int_113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 12), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 12), tuple_112, int_113)
# Adding element type (line 10)
int_114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 12), tuple_112, int_114)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_111, tuple_112)
# Adding element type (line 10)

# Obtaining an instance of the builtin type 'tuple' (line 10)
tuple_115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 10)
# Adding element type (line 10)
int_116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 20), tuple_115, int_116)
# Adding element type (line 10)
int_117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 20), tuple_115, int_117)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_111, tuple_115)
# Adding element type (line 10)

# Obtaining an instance of the builtin type 'tuple' (line 10)
tuple_118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 10)
# Adding element type (line 10)
int_119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 28), tuple_118, int_119)
# Adding element type (line 10)
int_120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 28), tuple_118, int_120)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_111, tuple_118)
# Adding element type (line 10)

# Obtaining an instance of the builtin type 'tuple' (line 10)
tuple_121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 36), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 10)
# Adding element type (line 10)
int_122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 36), tuple_121, int_122)
# Adding element type (line 10)
int_123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 39), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 36), tuple_121, int_123)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_111, tuple_121)

# Assigning a type to the variable 'corners' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'corners', list_111)

@norecursion
def possible_move(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'possible_move'
    module_type_store = module_type_store.open_function_context('possible_move', 13, 0, False)
    
    # Passed parameters checking function
    possible_move.stypy_localization = localization
    possible_move.stypy_type_of_self = None
    possible_move.stypy_type_store = module_type_store
    possible_move.stypy_function_name = 'possible_move'
    possible_move.stypy_param_names_list = ['board', 'x', 'y', 'color']
    possible_move.stypy_varargs_param_name = None
    possible_move.stypy_kwargs_param_name = None
    possible_move.stypy_call_defaults = defaults
    possible_move.stypy_call_varargs = varargs
    possible_move.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'possible_move', ['board', 'x', 'y', 'color'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'possible_move', localization, ['board', 'x', 'y', 'color'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'possible_move(...)' code ##################

    
    
    # Obtaining the type of the subscript
    # Getting the type of 'y' (line 14)
    y_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 16), 'y')
    
    # Obtaining the type of the subscript
    # Getting the type of 'x' (line 14)
    x_125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 13), 'x')
    # Getting the type of 'board' (line 14)
    board_126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 7), 'board')
    # Obtaining the member '__getitem__' of a type (line 14)
    getitem___127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 7), board_126, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 14)
    subscript_call_result_128 = invoke(stypy.reporting.localization.Localization(__file__, 14, 7), getitem___127, x_125)
    
    # Obtaining the member '__getitem__' of a type (line 14)
    getitem___129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 7), subscript_call_result_128, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 14)
    subscript_call_result_130 = invoke(stypy.reporting.localization.Localization(__file__, 14, 7), getitem___129, y_124)
    
    # Getting the type of 'empty' (line 14)
    empty_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 22), 'empty')
    # Applying the binary operator '!=' (line 14)
    result_ne_132 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 7), '!=', subscript_call_result_130, empty_131)
    
    # Testing if the type of an if condition is none (line 14)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 4), result_ne_132):
        pass
    else:
        
        # Testing the type of an if condition (line 14)
        if_condition_133 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 4), result_ne_132)
        # Assigning a type to the variable 'if_condition_133' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'if_condition_133', if_condition_133)
        # SSA begins for if statement (line 14)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 15)
        False_134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', False_134)
        # SSA join for if statement (line 14)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'directions' (line 16)
    directions_135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 21), 'directions')
    # Testing if the for loop is going to be iterated (line 16)
    # Testing the type of a for loop iterable (line 16)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 16, 4), directions_135)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 16, 4), directions_135):
        # Getting the type of the for loop variable (line 16)
        for_loop_var_136 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 16, 4), directions_135)
        # Assigning a type to the variable 'direction' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'direction', for_loop_var_136)
        # SSA begins for a for statement (line 16)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to flip_in_direction(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of 'board' (line 17)
        board_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 29), 'board', False)
        # Getting the type of 'x' (line 17)
        x_139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 36), 'x', False)
        # Getting the type of 'y' (line 17)
        y_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 39), 'y', False)
        # Getting the type of 'direction' (line 17)
        direction_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 42), 'direction', False)
        # Getting the type of 'color' (line 17)
        color_142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 53), 'color', False)
        # Processing the call keyword arguments (line 17)
        kwargs_143 = {}
        # Getting the type of 'flip_in_direction' (line 17)
        flip_in_direction_137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 11), 'flip_in_direction', False)
        # Calling flip_in_direction(args, kwargs) (line 17)
        flip_in_direction_call_result_144 = invoke(stypy.reporting.localization.Localization(__file__, 17, 11), flip_in_direction_137, *[board_138, x_139, y_140, direction_141, color_142], **kwargs_143)
        
        # Testing if the type of an if condition is none (line 17)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 17, 8), flip_in_direction_call_result_144):
            pass
        else:
            
            # Testing the type of an if condition (line 17)
            if_condition_145 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 17, 8), flip_in_direction_call_result_144)
            # Assigning a type to the variable 'if_condition_145' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'if_condition_145', if_condition_145)
            # SSA begins for if statement (line 17)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'True' (line 18)
            True_146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 19), 'True')
            # Assigning a type to the variable 'stypy_return_type' (line 18)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 12), 'stypy_return_type', True_146)
            # SSA join for if statement (line 17)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'False' (line 19)
    False_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'stypy_return_type', False_147)
    
    # ################# End of 'possible_move(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'possible_move' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_148)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'possible_move'
    return stypy_return_type_148

# Assigning a type to the variable 'possible_move' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'possible_move', possible_move)

@norecursion
def flip_in_direction(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'flip_in_direction'
    module_type_store = module_type_store.open_function_context('flip_in_direction', 22, 0, False)
    
    # Passed parameters checking function
    flip_in_direction.stypy_localization = localization
    flip_in_direction.stypy_type_of_self = None
    flip_in_direction.stypy_type_store = module_type_store
    flip_in_direction.stypy_function_name = 'flip_in_direction'
    flip_in_direction.stypy_param_names_list = ['board', 'x', 'y', 'direction', 'color']
    flip_in_direction.stypy_varargs_param_name = None
    flip_in_direction.stypy_kwargs_param_name = None
    flip_in_direction.stypy_call_defaults = defaults
    flip_in_direction.stypy_call_varargs = varargs
    flip_in_direction.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'flip_in_direction', ['board', 'x', 'y', 'direction', 'color'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'flip_in_direction', localization, ['board', 'x', 'y', 'direction', 'color'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'flip_in_direction(...)' code ##################

    
    # Assigning a Name to a Name (line 23):
    
    # Assigning a Name to a Name (line 23):
    # Getting the type of 'False' (line 23)
    False_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 18), 'False')
    # Assigning a type to the variable 'other_color' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'other_color', False_149)
    
    # Getting the type of 'True' (line 24)
    True_150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 10), 'True')
    # Testing if the while is going to be iterated (line 24)
    # Testing the type of an if condition (line 24)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 24, 4), True_150)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 24, 4), True_150):
        # SSA begins for while statement (line 24)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Tuple to a Tuple (line 25):
        
        # Assigning a BinOp to a Name (line 25):
        # Getting the type of 'x' (line 25)
        x_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'x')
        
        # Obtaining the type of the subscript
        int_152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 29), 'int')
        # Getting the type of 'direction' (line 25)
        direction_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 19), 'direction')
        # Obtaining the member '__getitem__' of a type (line 25)
        getitem___154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 19), direction_153, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 25)
        subscript_call_result_155 = invoke(stypy.reporting.localization.Localization(__file__, 25, 19), getitem___154, int_152)
        
        # Applying the binary operator '+' (line 25)
        result_add_156 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 15), '+', x_151, subscript_call_result_155)
        
        # Assigning a type to the variable 'tuple_assignment_4' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'tuple_assignment_4', result_add_156)
        
        # Assigning a BinOp to a Name (line 25):
        # Getting the type of 'y' (line 25)
        y_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 33), 'y')
        
        # Obtaining the type of the subscript
        int_158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 47), 'int')
        # Getting the type of 'direction' (line 25)
        direction_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 37), 'direction')
        # Obtaining the member '__getitem__' of a type (line 25)
        getitem___160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 37), direction_159, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 25)
        subscript_call_result_161 = invoke(stypy.reporting.localization.Localization(__file__, 25, 37), getitem___160, int_158)
        
        # Applying the binary operator '+' (line 25)
        result_add_162 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 33), '+', y_157, subscript_call_result_161)
        
        # Assigning a type to the variable 'tuple_assignment_5' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'tuple_assignment_5', result_add_162)
        
        # Assigning a Name to a Name (line 25):
        # Getting the type of 'tuple_assignment_4' (line 25)
        tuple_assignment_4_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'tuple_assignment_4')
        # Assigning a type to the variable 'x' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'x', tuple_assignment_4_163)
        
        # Assigning a Name to a Name (line 25):
        # Getting the type of 'tuple_assignment_5' (line 25)
        tuple_assignment_5_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'tuple_assignment_5')
        # Assigning a type to the variable 'y' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 11), 'y', tuple_assignment_5_164)
        
        # Evaluating a boolean operation
        
        # Getting the type of 'x' (line 26)
        x_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'x')
        
        # Call to range(...): (line 26)
        # Processing the call arguments (line 26)
        int_167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 26), 'int')
        # Processing the call keyword arguments (line 26)
        kwargs_168 = {}
        # Getting the type of 'range' (line 26)
        range_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 20), 'range', False)
        # Calling range(args, kwargs) (line 26)
        range_call_result_169 = invoke(stypy.reporting.localization.Localization(__file__, 26, 20), range_166, *[int_167], **kwargs_168)
        
        # Applying the binary operator 'notin' (line 26)
        result_contains_170 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 11), 'notin', x_165, range_call_result_169)
        
        
        # Getting the type of 'y' (line 26)
        y_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 32), 'y')
        
        # Call to range(...): (line 26)
        # Processing the call arguments (line 26)
        int_173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 47), 'int')
        # Processing the call keyword arguments (line 26)
        kwargs_174 = {}
        # Getting the type of 'range' (line 26)
        range_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 41), 'range', False)
        # Calling range(args, kwargs) (line 26)
        range_call_result_175 = invoke(stypy.reporting.localization.Localization(__file__, 26, 41), range_172, *[int_173], **kwargs_174)
        
        # Applying the binary operator 'notin' (line 26)
        result_contains_176 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 32), 'notin', y_171, range_call_result_175)
        
        # Applying the binary operator 'or' (line 26)
        result_or_keyword_177 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 11), 'or', result_contains_170, result_contains_176)
        
        # Testing if the type of an if condition is none (line 26)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 26, 8), result_or_keyword_177):
            pass
        else:
            
            # Testing the type of an if condition (line 26)
            if_condition_178 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 26, 8), result_or_keyword_177)
            # Assigning a type to the variable 'if_condition_178' (line 26)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'if_condition_178', if_condition_178)
            # SSA begins for if statement (line 26)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'False' (line 27)
            False_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 27)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'stypy_return_type', False_179)
            # SSA join for if statement (line 26)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Subscript to a Name (line 28):
        
        # Assigning a Subscript to a Name (line 28):
        
        # Obtaining the type of the subscript
        # Getting the type of 'y' (line 28)
        y_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 26), 'y')
        
        # Obtaining the type of the subscript
        # Getting the type of 'x' (line 28)
        x_181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 23), 'x')
        # Getting the type of 'board' (line 28)
        board_182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 17), 'board')
        # Obtaining the member '__getitem__' of a type (line 28)
        getitem___183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 17), board_182, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 28)
        subscript_call_result_184 = invoke(stypy.reporting.localization.Localization(__file__, 28, 17), getitem___183, x_181)
        
        # Obtaining the member '__getitem__' of a type (line 28)
        getitem___185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 17), subscript_call_result_184, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 28)
        subscript_call_result_186 = invoke(stypy.reporting.localization.Localization(__file__, 28, 17), getitem___185, y_180)
        
        # Assigning a type to the variable 'square' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'square', subscript_call_result_186)
        
        # Getting the type of 'square' (line 29)
        square_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 11), 'square')
        # Getting the type of 'empty' (line 29)
        empty_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 21), 'empty')
        # Applying the binary operator '==' (line 29)
        result_eq_189 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 11), '==', square_187, empty_188)
        
        # Testing if the type of an if condition is none (line 29)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 29, 8), result_eq_189):
            pass
        else:
            
            # Testing the type of an if condition (line 29)
            if_condition_190 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 8), result_eq_189)
            # Assigning a type to the variable 'if_condition_190' (line 29)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'if_condition_190', if_condition_190)
            # SSA begins for if statement (line 29)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'False' (line 29)
            False_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 35), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 29)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 28), 'stypy_return_type', False_191)
            # SSA join for if statement (line 29)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'square' (line 30)
        square_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 11), 'square')
        # Getting the type of 'color' (line 30)
        color_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 21), 'color')
        # Applying the binary operator '!=' (line 30)
        result_ne_194 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 11), '!=', square_192, color_193)
        
        # Testing if the type of an if condition is none (line 30)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 30, 8), result_ne_194):
            # Getting the type of 'other_color' (line 33)
            other_color_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 19), 'other_color')
            # Assigning a type to the variable 'stypy_return_type' (line 33)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'stypy_return_type', other_color_197)
        else:
            
            # Testing the type of an if condition (line 30)
            if_condition_195 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 30, 8), result_ne_194)
            # Assigning a type to the variable 'if_condition_195' (line 30)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'if_condition_195', if_condition_195)
            # SSA begins for if statement (line 30)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 31):
            
            # Assigning a Name to a Name (line 31):
            # Getting the type of 'True' (line 31)
            True_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 26), 'True')
            # Assigning a type to the variable 'other_color' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'other_color', True_196)
            # SSA branch for the else part of an if statement (line 30)
            module_type_store.open_ssa_branch('else')
            # Getting the type of 'other_color' (line 33)
            other_color_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 19), 'other_color')
            # Assigning a type to the variable 'stypy_return_type' (line 33)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'stypy_return_type', other_color_197)
            # SSA join for if statement (line 30)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for while statement (line 24)
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'flip_in_direction(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'flip_in_direction' in the type store
    # Getting the type of 'stypy_return_type' (line 22)
    stypy_return_type_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_198)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'flip_in_direction'
    return stypy_return_type_198

# Assigning a type to the variable 'flip_in_direction' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'flip_in_direction', flip_in_direction)

@norecursion
def flip_stones(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'flip_stones'
    module_type_store = module_type_store.open_function_context('flip_stones', 36, 0, False)
    
    # Passed parameters checking function
    flip_stones.stypy_localization = localization
    flip_stones.stypy_type_of_self = None
    flip_stones.stypy_type_store = module_type_store
    flip_stones.stypy_function_name = 'flip_stones'
    flip_stones.stypy_param_names_list = ['board', 'move', 'color']
    flip_stones.stypy_varargs_param_name = None
    flip_stones.stypy_kwargs_param_name = None
    flip_stones.stypy_call_defaults = defaults
    flip_stones.stypy_call_varargs = varargs
    flip_stones.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'flip_stones', ['board', 'move', 'color'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'flip_stones', localization, ['board', 'move', 'color'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'flip_stones(...)' code ##################

    
    # Getting the type of 'directions' (line 37)
    directions_199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 21), 'directions')
    # Testing if the for loop is going to be iterated (line 37)
    # Testing the type of a for loop iterable (line 37)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 37, 4), directions_199)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 37, 4), directions_199):
        # Getting the type of the for loop variable (line 37)
        for_loop_var_200 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 37, 4), directions_199)
        # Assigning a type to the variable 'direction' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'direction', for_loop_var_200)
        # SSA begins for a for statement (line 37)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to flip_in_direction(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'board' (line 38)
        board_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 29), 'board', False)
        
        # Obtaining the type of the subscript
        int_203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 41), 'int')
        # Getting the type of 'move' (line 38)
        move_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 36), 'move', False)
        # Obtaining the member '__getitem__' of a type (line 38)
        getitem___205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 36), move_204, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 38)
        subscript_call_result_206 = invoke(stypy.reporting.localization.Localization(__file__, 38, 36), getitem___205, int_203)
        
        
        # Obtaining the type of the subscript
        int_207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 50), 'int')
        # Getting the type of 'move' (line 38)
        move_208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 45), 'move', False)
        # Obtaining the member '__getitem__' of a type (line 38)
        getitem___209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 45), move_208, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 38)
        subscript_call_result_210 = invoke(stypy.reporting.localization.Localization(__file__, 38, 45), getitem___209, int_207)
        
        # Getting the type of 'direction' (line 38)
        direction_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 54), 'direction', False)
        # Getting the type of 'color' (line 38)
        color_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 65), 'color', False)
        # Processing the call keyword arguments (line 38)
        kwargs_213 = {}
        # Getting the type of 'flip_in_direction' (line 38)
        flip_in_direction_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 11), 'flip_in_direction', False)
        # Calling flip_in_direction(args, kwargs) (line 38)
        flip_in_direction_call_result_214 = invoke(stypy.reporting.localization.Localization(__file__, 38, 11), flip_in_direction_201, *[board_202, subscript_call_result_206, subscript_call_result_210, direction_211, color_212], **kwargs_213)
        
        # Testing if the type of an if condition is none (line 38)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 38, 8), flip_in_direction_call_result_214):
            pass
        else:
            
            # Testing the type of an if condition (line 38)
            if_condition_215 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 8), flip_in_direction_call_result_214)
            # Assigning a type to the variable 'if_condition_215' (line 38)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'if_condition_215', if_condition_215)
            # SSA begins for if statement (line 38)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Tuple to a Tuple (line 39):
            
            # Assigning a BinOp to a Name (line 39):
            
            # Obtaining the type of the subscript
            int_216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 24), 'int')
            # Getting the type of 'move' (line 39)
            move_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 19), 'move')
            # Obtaining the member '__getitem__' of a type (line 39)
            getitem___218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 19), move_217, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 39)
            subscript_call_result_219 = invoke(stypy.reporting.localization.Localization(__file__, 39, 19), getitem___218, int_216)
            
            
            # Obtaining the type of the subscript
            int_220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 39), 'int')
            # Getting the type of 'direction' (line 39)
            direction_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 29), 'direction')
            # Obtaining the member '__getitem__' of a type (line 39)
            getitem___222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 29), direction_221, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 39)
            subscript_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 39, 29), getitem___222, int_220)
            
            # Applying the binary operator '+' (line 39)
            result_add_224 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 19), '+', subscript_call_result_219, subscript_call_result_223)
            
            # Assigning a type to the variable 'tuple_assignment_6' (line 39)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'tuple_assignment_6', result_add_224)
            
            # Assigning a BinOp to a Name (line 39):
            
            # Obtaining the type of the subscript
            int_225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 48), 'int')
            # Getting the type of 'move' (line 39)
            move_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 43), 'move')
            # Obtaining the member '__getitem__' of a type (line 39)
            getitem___227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 43), move_226, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 39)
            subscript_call_result_228 = invoke(stypy.reporting.localization.Localization(__file__, 39, 43), getitem___227, int_225)
            
            
            # Obtaining the type of the subscript
            int_229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 63), 'int')
            # Getting the type of 'direction' (line 39)
            direction_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 53), 'direction')
            # Obtaining the member '__getitem__' of a type (line 39)
            getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 53), direction_230, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 39)
            subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 39, 53), getitem___231, int_229)
            
            # Applying the binary operator '+' (line 39)
            result_add_233 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 43), '+', subscript_call_result_228, subscript_call_result_232)
            
            # Assigning a type to the variable 'tuple_assignment_7' (line 39)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'tuple_assignment_7', result_add_233)
            
            # Assigning a Name to a Name (line 39):
            # Getting the type of 'tuple_assignment_6' (line 39)
            tuple_assignment_6_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'tuple_assignment_6')
            # Assigning a type to the variable 'x' (line 39)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'x', tuple_assignment_6_234)
            
            # Assigning a Name to a Name (line 39):
            # Getting the type of 'tuple_assignment_7' (line 39)
            tuple_assignment_7_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'tuple_assignment_7')
            # Assigning a type to the variable 'y' (line 39)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'y', tuple_assignment_7_235)
            
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'y' (line 40)
            y_236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 27), 'y')
            
            # Obtaining the type of the subscript
            # Getting the type of 'x' (line 40)
            x_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 24), 'x')
            # Getting the type of 'board' (line 40)
            board_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 18), 'board')
            # Obtaining the member '__getitem__' of a type (line 40)
            getitem___239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 18), board_238, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 40)
            subscript_call_result_240 = invoke(stypy.reporting.localization.Localization(__file__, 40, 18), getitem___239, x_237)
            
            # Obtaining the member '__getitem__' of a type (line 40)
            getitem___241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 18), subscript_call_result_240, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 40)
            subscript_call_result_242 = invoke(stypy.reporting.localization.Localization(__file__, 40, 18), getitem___241, y_236)
            
            # Getting the type of 'color' (line 40)
            color_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 33), 'color')
            # Applying the binary operator '!=' (line 40)
            result_ne_244 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 18), '!=', subscript_call_result_242, color_243)
            
            # Testing if the while is going to be iterated (line 40)
            # Testing the type of an if condition (line 40)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 12), result_ne_244)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 40, 12), result_ne_244):
                # SSA begins for while statement (line 40)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
                
                # Assigning a Name to a Subscript (line 41):
                
                # Assigning a Name to a Subscript (line 41):
                # Getting the type of 'color' (line 41)
                color_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 30), 'color')
                
                # Obtaining the type of the subscript
                # Getting the type of 'x' (line 41)
                x_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 22), 'x')
                # Getting the type of 'board' (line 41)
                board_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'board')
                # Obtaining the member '__getitem__' of a type (line 41)
                getitem___248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 16), board_247, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 41)
                subscript_call_result_249 = invoke(stypy.reporting.localization.Localization(__file__, 41, 16), getitem___248, x_246)
                
                # Getting the type of 'y' (line 41)
                y_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 25), 'y')
                # Storing an element on a container (line 41)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 16), subscript_call_result_249, (y_250, color_245))
                
                # Assigning a Tuple to a Tuple (line 42):
                
                # Assigning a BinOp to a Name (line 42):
                # Getting the type of 'x' (line 42)
                x_251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 23), 'x')
                
                # Obtaining the type of the subscript
                int_252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 37), 'int')
                # Getting the type of 'direction' (line 42)
                direction_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 27), 'direction')
                # Obtaining the member '__getitem__' of a type (line 42)
                getitem___254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 27), direction_253, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 42)
                subscript_call_result_255 = invoke(stypy.reporting.localization.Localization(__file__, 42, 27), getitem___254, int_252)
                
                # Applying the binary operator '+' (line 42)
                result_add_256 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 23), '+', x_251, subscript_call_result_255)
                
                # Assigning a type to the variable 'tuple_assignment_8' (line 42)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 16), 'tuple_assignment_8', result_add_256)
                
                # Assigning a BinOp to a Name (line 42):
                # Getting the type of 'y' (line 42)
                y_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 41), 'y')
                
                # Obtaining the type of the subscript
                int_258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 55), 'int')
                # Getting the type of 'direction' (line 42)
                direction_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 45), 'direction')
                # Obtaining the member '__getitem__' of a type (line 42)
                getitem___260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 45), direction_259, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 42)
                subscript_call_result_261 = invoke(stypy.reporting.localization.Localization(__file__, 42, 45), getitem___260, int_258)
                
                # Applying the binary operator '+' (line 42)
                result_add_262 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 41), '+', y_257, subscript_call_result_261)
                
                # Assigning a type to the variable 'tuple_assignment_9' (line 42)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 16), 'tuple_assignment_9', result_add_262)
                
                # Assigning a Name to a Name (line 42):
                # Getting the type of 'tuple_assignment_8' (line 42)
                tuple_assignment_8_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 16), 'tuple_assignment_8')
                # Assigning a type to the variable 'x' (line 42)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 16), 'x', tuple_assignment_8_263)
                
                # Assigning a Name to a Name (line 42):
                # Getting the type of 'tuple_assignment_9' (line 42)
                tuple_assignment_9_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 16), 'tuple_assignment_9')
                # Assigning a type to the variable 'y' (line 42)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 19), 'y', tuple_assignment_9_264)
                # SSA join for while statement (line 40)
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 38)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a Name to a Subscript (line 43):
    
    # Assigning a Name to a Subscript (line 43):
    # Getting the type of 'color' (line 43)
    color_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 30), 'color')
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    int_266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 15), 'int')
    # Getting the type of 'move' (line 43)
    move_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 10), 'move')
    # Obtaining the member '__getitem__' of a type (line 43)
    getitem___268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 10), move_267, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 43)
    subscript_call_result_269 = invoke(stypy.reporting.localization.Localization(__file__, 43, 10), getitem___268, int_266)
    
    # Getting the type of 'board' (line 43)
    board_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'board')
    # Obtaining the member '__getitem__' of a type (line 43)
    getitem___271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 4), board_270, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 43)
    subscript_call_result_272 = invoke(stypy.reporting.localization.Localization(__file__, 43, 4), getitem___271, subscript_call_result_269)
    
    
    # Obtaining the type of the subscript
    int_273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 24), 'int')
    # Getting the type of 'move' (line 43)
    move_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 19), 'move')
    # Obtaining the member '__getitem__' of a type (line 43)
    getitem___275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 19), move_274, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 43)
    subscript_call_result_276 = invoke(stypy.reporting.localization.Localization(__file__, 43, 19), getitem___275, int_273)
    
    # Storing an element on a container (line 43)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 4), subscript_call_result_272, (subscript_call_result_276, color_265))
    
    # ################# End of 'flip_stones(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'flip_stones' in the type store
    # Getting the type of 'stypy_return_type' (line 36)
    stypy_return_type_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_277)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'flip_stones'
    return stypy_return_type_277

# Assigning a type to the variable 'flip_stones' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'flip_stones', flip_stones)

@norecursion
def print_board(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'print_board'
    module_type_store = module_type_store.open_function_context('print_board', 46, 0, False)
    
    # Passed parameters checking function
    print_board.stypy_localization = localization
    print_board.stypy_type_of_self = None
    print_board.stypy_type_store = module_type_store
    print_board.stypy_function_name = 'print_board'
    print_board.stypy_param_names_list = ['board', 'turn']
    print_board.stypy_varargs_param_name = None
    print_board.stypy_kwargs_param_name = None
    print_board.stypy_call_defaults = defaults
    print_board.stypy_call_varargs = varargs
    print_board.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'print_board', ['board', 'turn'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'print_board', localization, ['board', 'turn'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'print_board(...)' code ##################

    
    
    # Call to enumerate(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'board' (line 48)
    board_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 30), 'board', False)
    # Processing the call keyword arguments (line 48)
    kwargs_280 = {}
    # Getting the type of 'enumerate' (line 48)
    enumerate_278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 20), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 48)
    enumerate_call_result_281 = invoke(stypy.reporting.localization.Localization(__file__, 48, 20), enumerate_278, *[board_279], **kwargs_280)
    
    # Testing if the for loop is going to be iterated (line 48)
    # Testing the type of a for loop iterable (line 48)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 48, 4), enumerate_call_result_281)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 48, 4), enumerate_call_result_281):
        # Getting the type of the for loop variable (line 48)
        for_loop_var_282 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 48, 4), enumerate_call_result_281)
        # Assigning a type to the variable 'nr' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'nr', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 4), for_loop_var_282))
        # Assigning a type to the variable 'line' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'line', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 4), for_loop_var_282))
        # SSA begins for a for statement (line 48)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        pass
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'print_board(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'print_board' in the type store
    # Getting the type of 'stypy_return_type' (line 46)
    stypy_return_type_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_283)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'print_board'
    return stypy_return_type_283

# Assigning a type to the variable 'print_board' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'print_board', print_board)

@norecursion
def possible_moves(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'possible_moves'
    module_type_store = module_type_store.open_function_context('possible_moves', 55, 0, False)
    
    # Passed parameters checking function
    possible_moves.stypy_localization = localization
    possible_moves.stypy_type_of_self = None
    possible_moves.stypy_type_store = module_type_store
    possible_moves.stypy_function_name = 'possible_moves'
    possible_moves.stypy_param_names_list = ['board', 'color']
    possible_moves.stypy_varargs_param_name = None
    possible_moves.stypy_kwargs_param_name = None
    possible_moves.stypy_call_defaults = defaults
    possible_moves.stypy_call_varargs = varargs
    possible_moves.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'possible_moves', ['board', 'color'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'possible_moves', localization, ['board', 'color'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'possible_moves(...)' code ##################

    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 56)
    # Processing the call arguments (line 56)
    int_288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 34), 'int')
    # Processing the call keyword arguments (line 56)
    kwargs_289 = {}
    # Getting the type of 'range' (line 56)
    range_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 28), 'range', False)
    # Calling range(args, kwargs) (line 56)
    range_call_result_290 = invoke(stypy.reporting.localization.Localization(__file__, 56, 28), range_287, *[int_288], **kwargs_289)
    
    comprehension_291 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 12), range_call_result_290)
    # Assigning a type to the variable 'x' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'x', comprehension_291)
    # Calculating comprehension expression
    
    # Call to range(...): (line 56)
    # Processing the call arguments (line 56)
    int_300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 52), 'int')
    # Processing the call keyword arguments (line 56)
    kwargs_301 = {}
    # Getting the type of 'range' (line 56)
    range_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 46), 'range', False)
    # Calling range(args, kwargs) (line 56)
    range_call_result_302 = invoke(stypy.reporting.localization.Localization(__file__, 56, 46), range_299, *[int_300], **kwargs_301)
    
    comprehension_303 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 12), range_call_result_302)
    # Assigning a type to the variable 'y' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'y', comprehension_303)
    
    # Call to possible_move(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'board' (line 56)
    board_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 72), 'board', False)
    # Getting the type of 'x' (line 56)
    x_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 79), 'x', False)
    # Getting the type of 'y' (line 56)
    y_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 82), 'y', False)
    # Getting the type of 'color' (line 56)
    color_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 85), 'color', False)
    # Processing the call keyword arguments (line 56)
    kwargs_297 = {}
    # Getting the type of 'possible_move' (line 56)
    possible_move_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 58), 'possible_move', False)
    # Calling possible_move(args, kwargs) (line 56)
    possible_move_call_result_298 = invoke(stypy.reporting.localization.Localization(__file__, 56, 58), possible_move_292, *[board_293, x_294, y_295, color_296], **kwargs_297)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 56)
    tuple_284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 56)
    # Adding element type (line 56)
    # Getting the type of 'x' (line 56)
    x_285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 13), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 13), tuple_284, x_285)
    # Adding element type (line 56)
    # Getting the type of 'y' (line 56)
    y_286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 16), 'y')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 13), tuple_284, y_286)
    
    list_304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 12), list_304, tuple_284)
    # Assigning a type to the variable 'stypy_return_type' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type', list_304)
    
    # ################# End of 'possible_moves(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'possible_moves' in the type store
    # Getting the type of 'stypy_return_type' (line 55)
    stypy_return_type_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_305)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'possible_moves'
    return stypy_return_type_305

# Assigning a type to the variable 'possible_moves' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'possible_moves', possible_moves)

@norecursion
def coordinates(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'coordinates'
    module_type_store = module_type_store.open_function_context('coordinates', 59, 0, False)
    
    # Passed parameters checking function
    coordinates.stypy_localization = localization
    coordinates.stypy_type_of_self = None
    coordinates.stypy_type_store = module_type_store
    coordinates.stypy_function_name = 'coordinates'
    coordinates.stypy_param_names_list = ['move']
    coordinates.stypy_varargs_param_name = None
    coordinates.stypy_kwargs_param_name = None
    coordinates.stypy_call_defaults = defaults
    coordinates.stypy_call_varargs = varargs
    coordinates.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'coordinates', ['move'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'coordinates', localization, ['move'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'coordinates(...)' code ##################

    
    # Obtaining an instance of the builtin type 'tuple' (line 60)
    tuple_306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 60)
    # Adding element type (line 60)
    
    # Call to int(...): (line 60)
    # Processing the call arguments (line 60)
    
    # Obtaining the type of the subscript
    int_308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 21), 'int')
    # Getting the type of 'move' (line 60)
    move_309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'move', False)
    # Obtaining the member '__getitem__' of a type (line 60)
    getitem___310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 16), move_309, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 60)
    subscript_call_result_311 = invoke(stypy.reporting.localization.Localization(__file__, 60, 16), getitem___310, int_308)
    
    # Processing the call keyword arguments (line 60)
    kwargs_312 = {}
    # Getting the type of 'int' (line 60)
    int_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'int', False)
    # Calling int(args, kwargs) (line 60)
    int_call_result_313 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), int_307, *[subscript_call_result_311], **kwargs_312)
    
    int_314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 27), 'int')
    # Applying the binary operator '-' (line 60)
    result_sub_315 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 12), '-', int_call_result_313, int_314)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 12), tuple_306, result_sub_315)
    # Adding element type (line 60)
    
    # Call to index(...): (line 60)
    # Processing the call arguments (line 60)
    
    # Obtaining the type of the subscript
    int_318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 52), 'int')
    # Getting the type of 'move' (line 60)
    move_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 47), 'move', False)
    # Obtaining the member '__getitem__' of a type (line 60)
    getitem___320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 47), move_319, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 60)
    subscript_call_result_321 = invoke(stypy.reporting.localization.Localization(__file__, 60, 47), getitem___320, int_318)
    
    # Processing the call keyword arguments (line 60)
    kwargs_322 = {}
    str_316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 30), 'str', 'abcdefgh')
    # Obtaining the member 'index' of a type (line 60)
    index_317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 30), str_316, 'index')
    # Calling index(args, kwargs) (line 60)
    index_call_result_323 = invoke(stypy.reporting.localization.Localization(__file__, 60, 30), index_317, *[subscript_call_result_321], **kwargs_322)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 12), tuple_306, index_call_result_323)
    
    # Assigning a type to the variable 'stypy_return_type' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type', tuple_306)
    
    # ################# End of 'coordinates(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'coordinates' in the type store
    # Getting the type of 'stypy_return_type' (line 59)
    stypy_return_type_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_324)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'coordinates'
    return stypy_return_type_324

# Assigning a type to the variable 'coordinates' (line 59)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'coordinates', coordinates)

@norecursion
def human_move(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'human_move'
    module_type_store = module_type_store.open_function_context('human_move', 63, 0, False)
    
    # Passed parameters checking function
    human_move.stypy_localization = localization
    human_move.stypy_type_of_self = None
    human_move.stypy_type_store = module_type_store
    human_move.stypy_function_name = 'human_move'
    human_move.stypy_param_names_list = ['move']
    human_move.stypy_varargs_param_name = None
    human_move.stypy_kwargs_param_name = None
    human_move.stypy_call_defaults = defaults
    human_move.stypy_call_varargs = varargs
    human_move.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'human_move', ['move'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'human_move', localization, ['move'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'human_move(...)' code ##################

    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    int_325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 27), 'int')
    # Getting the type of 'move' (line 64)
    move_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 22), 'move')
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 22), move_326, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_328 = invoke(stypy.reporting.localization.Localization(__file__, 64, 22), getitem___327, int_325)
    
    str_329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 11), 'str', 'abcdefgh')
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 11), str_329, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_331 = invoke(stypy.reporting.localization.Localization(__file__, 64, 11), getitem___330, subscript_call_result_328)
    
    
    # Call to str(...): (line 64)
    # Processing the call arguments (line 64)
    
    # Obtaining the type of the subscript
    int_333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 42), 'int')
    # Getting the type of 'move' (line 64)
    move_334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 37), 'move', False)
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 37), move_334, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_336 = invoke(stypy.reporting.localization.Localization(__file__, 64, 37), getitem___335, int_333)
    
    int_337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 47), 'int')
    # Applying the binary operator '+' (line 64)
    result_add_338 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 37), '+', subscript_call_result_336, int_337)
    
    # Processing the call keyword arguments (line 64)
    kwargs_339 = {}
    # Getting the type of 'str' (line 64)
    str_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 33), 'str', False)
    # Calling str(args, kwargs) (line 64)
    str_call_result_340 = invoke(stypy.reporting.localization.Localization(__file__, 64, 33), str_332, *[result_add_338], **kwargs_339)
    
    # Applying the binary operator '+' (line 64)
    result_add_341 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 11), '+', subscript_call_result_331, str_call_result_340)
    
    # Assigning a type to the variable 'stypy_return_type' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'stypy_return_type', result_add_341)
    
    # ################# End of 'human_move(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'human_move' in the type store
    # Getting the type of 'stypy_return_type' (line 63)
    stypy_return_type_342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_342)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'human_move'
    return stypy_return_type_342

# Assigning a type to the variable 'human_move' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'human_move', human_move)

@norecursion
def stone_count(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'stone_count'
    module_type_store = module_type_store.open_function_context('stone_count', 67, 0, False)
    
    # Passed parameters checking function
    stone_count.stypy_localization = localization
    stone_count.stypy_type_of_self = None
    stone_count.stypy_type_store = module_type_store
    stone_count.stypy_function_name = 'stone_count'
    stone_count.stypy_param_names_list = ['board', 'color']
    stone_count.stypy_varargs_param_name = None
    stone_count.stypy_kwargs_param_name = None
    stone_count.stypy_call_defaults = defaults
    stone_count.stypy_call_varargs = varargs
    stone_count.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'stone_count', ['board', 'color'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'stone_count', localization, ['board', 'color'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'stone_count(...)' code ##################

    
    # Call to sum(...): (line 68)
    # Processing the call arguments (line 68)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'board' (line 68)
    board_354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 80), 'board', False)
    comprehension_355 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 16), board_354)
    # Assigning a type to the variable 'line' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'line', comprehension_355)
    
    # Call to len(...): (line 68)
    # Processing the call arguments (line 68)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'line' (line 68)
    line_349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 42), 'line', False)
    comprehension_350 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 21), line_349)
    # Assigning a type to the variable 'square' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 21), 'square', comprehension_350)
    
    # Getting the type of 'square' (line 68)
    square_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 50), 'square', False)
    # Getting the type of 'color' (line 68)
    color_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 60), 'color', False)
    # Applying the binary operator '==' (line 68)
    result_eq_348 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 50), '==', square_346, color_347)
    
    # Getting the type of 'square' (line 68)
    square_345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 21), 'square', False)
    list_351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 21), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 21), list_351, square_345)
    # Processing the call keyword arguments (line 68)
    kwargs_352 = {}
    # Getting the type of 'len' (line 68)
    len_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'len', False)
    # Calling len(args, kwargs) (line 68)
    len_call_result_353 = invoke(stypy.reporting.localization.Localization(__file__, 68, 16), len_344, *[list_351], **kwargs_352)
    
    list_356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 16), list_356, len_call_result_353)
    # Processing the call keyword arguments (line 68)
    kwargs_357 = {}
    # Getting the type of 'sum' (line 68)
    sum_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'sum', False)
    # Calling sum(args, kwargs) (line 68)
    sum_call_result_358 = invoke(stypy.reporting.localization.Localization(__file__, 68, 11), sum_343, *[list_356], **kwargs_357)
    
    # Assigning a type to the variable 'stypy_return_type' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type', sum_call_result_358)
    
    # ################# End of 'stone_count(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'stone_count' in the type store
    # Getting the type of 'stypy_return_type' (line 67)
    stypy_return_type_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_359)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'stone_count'
    return stypy_return_type_359

# Assigning a type to the variable 'stone_count' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'stone_count', stone_count)

@norecursion
def best_move(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 40), 'int')
    defaults = [int_360]
    # Create a new context for function 'best_move'
    module_type_store = module_type_store.open_function_context('best_move', 71, 0, False)
    
    # Passed parameters checking function
    best_move.stypy_localization = localization
    best_move.stypy_type_of_self = None
    best_move.stypy_type_store = module_type_store
    best_move.stypy_function_name = 'best_move'
    best_move.stypy_param_names_list = ['board', 'color', 'first', 'step']
    best_move.stypy_varargs_param_name = None
    best_move.stypy_kwargs_param_name = None
    best_move.stypy_call_defaults = defaults
    best_move.stypy_call_varargs = varargs
    best_move.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'best_move', ['board', 'color', 'first', 'step'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'best_move', localization, ['board', 'color', 'first', 'step'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'best_move(...)' code ##################

    
    # Assigning a Tuple to a Tuple (line 72):
    
    # Assigning a Name to a Name (line 72):
    # Getting the type of 'None' (line 72)
    None_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 40), 'None')
    # Assigning a type to the variable 'tuple_assignment_10' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_assignment_10', None_361)
    
    # Assigning a Num to a Name (line 72):
    int_362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 46), 'int')
    # Assigning a type to the variable 'tuple_assignment_11' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_assignment_11', int_362)
    
    # Assigning a Num to a Name (line 72):
    int_363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 49), 'int')
    # Assigning a type to the variable 'tuple_assignment_12' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_assignment_12', int_363)
    
    # Assigning a Name to a Name (line 72):
    # Getting the type of 'tuple_assignment_10' (line 72)
    tuple_assignment_10_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_assignment_10')
    # Assigning a type to the variable 'max_move' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'max_move', tuple_assignment_10_364)
    
    # Assigning a Name to a Name (line 72):
    # Getting the type of 'tuple_assignment_11' (line 72)
    tuple_assignment_11_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_assignment_11')
    # Assigning a type to the variable 'max_mobility' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 14), 'max_mobility', tuple_assignment_11_365)
    
    # Assigning a Name to a Name (line 72):
    # Getting the type of 'tuple_assignment_12' (line 72)
    tuple_assignment_12_366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_assignment_12')
    # Assigning a type to the variable 'max_score' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 28), 'max_score', tuple_assignment_12_366)
    
    
    # Call to possible_moves(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'board' (line 73)
    board_368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 31), 'board', False)
    # Getting the type of 'color' (line 73)
    color_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 38), 'color', False)
    # Processing the call keyword arguments (line 73)
    kwargs_370 = {}
    # Getting the type of 'possible_moves' (line 73)
    possible_moves_367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'possible_moves', False)
    # Calling possible_moves(args, kwargs) (line 73)
    possible_moves_call_result_371 = invoke(stypy.reporting.localization.Localization(__file__, 73, 16), possible_moves_367, *[board_368, color_369], **kwargs_370)
    
    # Testing if the for loop is going to be iterated (line 73)
    # Testing the type of a for loop iterable (line 73)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 73, 4), possible_moves_call_result_371)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 73, 4), possible_moves_call_result_371):
        # Getting the type of the for loop variable (line 73)
        for_loop_var_372 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 73, 4), possible_moves_call_result_371)
        # Assigning a type to the variable 'move' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'move', for_loop_var_372)
        # SSA begins for a for statement (line 73)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'move' (line 74)
        move_373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 11), 'move')
        # Getting the type of 'corners' (line 74)
        corners_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 19), 'corners')
        # Applying the binary operator 'in' (line 74)
        result_contains_375 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 11), 'in', move_373, corners_374)
        
        # Testing if the type of an if condition is none (line 74)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 8), result_contains_375):
            
            # Assigning a ListComp to a Name (line 79):
            
            # Assigning a ListComp to a Name (line 79):
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'board' (line 79)
            board_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 65), 'board')
            comprehension_393 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 25), board_392)
            # Assigning a type to the variable 'line' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 25), 'line', comprehension_393)
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'line' (line 79)
            line_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 47), 'line')
            comprehension_390 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 26), line_389)
            # Assigning a type to the variable 'square' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 26), 'square', comprehension_390)
            # Getting the type of 'square' (line 79)
            square_388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 26), 'square')
            list_391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 26), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 26), list_391, square_388)
            list_394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 25), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 25), list_394, list_391)
            # Assigning a type to the variable 'testboard' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'testboard', list_394)
            
            # Call to flip_stones(...): (line 80)
            # Processing the call arguments (line 80)
            # Getting the type of 'testboard' (line 80)
            testboard_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 24), 'testboard', False)
            # Getting the type of 'move' (line 80)
            move_397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 35), 'move', False)
            # Getting the type of 'color' (line 80)
            color_398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 41), 'color', False)
            # Processing the call keyword arguments (line 80)
            kwargs_399 = {}
            # Getting the type of 'flip_stones' (line 80)
            flip_stones_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'flip_stones', False)
            # Calling flip_stones(args, kwargs) (line 80)
            flip_stones_call_result_400 = invoke(stypy.reporting.localization.Localization(__file__, 80, 12), flip_stones_395, *[testboard_396, move_397, color_398], **kwargs_399)
            
            
            # Getting the type of 'step' (line 81)
            step_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 15), 'step')
            # Getting the type of 'depth' (line 81)
            depth_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 22), 'depth')
            # Applying the binary operator '<' (line 81)
            result_lt_403 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 15), '<', step_401, depth_402)
            
            # Testing if the type of an if condition is none (line 81)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 81, 12), result_lt_403):
                
                # Assigning a Call to a Name (line 84):
                
                # Assigning a Call to a Name (line 84):
                
                # Call to len(...): (line 84)
                # Processing the call arguments (line 84)
                
                # Call to possible_moves(...): (line 84)
                # Processing the call arguments (line 84)
                # Getting the type of 'testboard' (line 84)
                testboard_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 46), 'testboard', False)
                # Getting the type of 'first' (line 84)
                first_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 57), 'first', False)
                # Processing the call keyword arguments (line 84)
                kwargs_431 = {}
                # Getting the type of 'possible_moves' (line 84)
                possible_moves_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 31), 'possible_moves', False)
                # Calling possible_moves(args, kwargs) (line 84)
                possible_moves_call_result_432 = invoke(stypy.reporting.localization.Localization(__file__, 84, 31), possible_moves_428, *[testboard_429, first_430], **kwargs_431)
                
                # Processing the call keyword arguments (line 84)
                kwargs_433 = {}
                # Getting the type of 'len' (line 84)
                len_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 27), 'len', False)
                # Calling len(args, kwargs) (line 84)
                len_call_result_434 = invoke(stypy.reporting.localization.Localization(__file__, 84, 27), len_427, *[possible_moves_call_result_432], **kwargs_433)
                
                # Assigning a type to the variable 'mobility' (line 84)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'mobility', len_call_result_434)
            else:
                
                # Testing the type of an if condition (line 81)
                if_condition_404 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 12), result_lt_403)
                # Assigning a type to the variable 'if_condition_404' (line 81)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'if_condition_404', if_condition_404)
                # SSA begins for if statement (line 81)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Tuple (line 82):
                
                # Assigning a Call to a Name:
                
                # Call to best_move(...): (line 82)
                # Processing the call arguments (line 82)
                # Getting the type of 'testboard' (line 82)
                testboard_406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 48), 'testboard', False)
                
                # Getting the type of 'color' (line 82)
                color_407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 60), 'color', False)
                # Applying the 'usub' unary operator (line 82)
                result___neg___408 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 59), 'usub', color_407)
                
                # Getting the type of 'first' (line 82)
                first_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 67), 'first', False)
                # Getting the type of 'step' (line 82)
                step_410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 74), 'step', False)
                int_411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 81), 'int')
                # Applying the binary operator '+' (line 82)
                result_add_412 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 74), '+', step_410, int_411)
                
                # Processing the call keyword arguments (line 82)
                kwargs_413 = {}
                # Getting the type of 'best_move' (line 82)
                best_move_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 38), 'best_move', False)
                # Calling best_move(args, kwargs) (line 82)
                best_move_call_result_414 = invoke(stypy.reporting.localization.Localization(__file__, 82, 38), best_move_405, *[testboard_406, result___neg___408, first_409, result_add_412], **kwargs_413)
                
                # Assigning a type to the variable 'call_assignment_15' (line 82)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'call_assignment_15', best_move_call_result_414)
                
                # Assigning a Call to a Name (line 82):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 16), 'int')
                # Processing the call keyword arguments
                kwargs_418 = {}
                # Getting the type of 'call_assignment_15' (line 82)
                call_assignment_15_415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'call_assignment_15', False)
                # Obtaining the member '__getitem__' of a type (line 82)
                getitem___416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 16), call_assignment_15_415, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_419 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___416, *[int_417], **kwargs_418)
                
                # Assigning a type to the variable 'call_assignment_16' (line 82)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'call_assignment_16', getitem___call_result_419)
                
                # Assigning a Name to a Name (line 82):
                # Getting the type of 'call_assignment_16' (line 82)
                call_assignment_16_420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'call_assignment_16')
                # Assigning a type to the variable 'next_move' (line 82)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'next_move', call_assignment_16_420)
                
                # Assigning a Call to a Name (line 82):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 16), 'int')
                # Processing the call keyword arguments
                kwargs_424 = {}
                # Getting the type of 'call_assignment_15' (line 82)
                call_assignment_15_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'call_assignment_15', False)
                # Obtaining the member '__getitem__' of a type (line 82)
                getitem___422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 16), call_assignment_15_421, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_425 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___422, *[int_423], **kwargs_424)
                
                # Assigning a type to the variable 'call_assignment_17' (line 82)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'call_assignment_17', getitem___call_result_425)
                
                # Assigning a Name to a Name (line 82):
                # Getting the type of 'call_assignment_17' (line 82)
                call_assignment_17_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'call_assignment_17')
                # Assigning a type to the variable 'mobility' (line 82)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 27), 'mobility', call_assignment_17_426)
                # SSA branch for the else part of an if statement (line 81)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Name (line 84):
                
                # Assigning a Call to a Name (line 84):
                
                # Call to len(...): (line 84)
                # Processing the call arguments (line 84)
                
                # Call to possible_moves(...): (line 84)
                # Processing the call arguments (line 84)
                # Getting the type of 'testboard' (line 84)
                testboard_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 46), 'testboard', False)
                # Getting the type of 'first' (line 84)
                first_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 57), 'first', False)
                # Processing the call keyword arguments (line 84)
                kwargs_431 = {}
                # Getting the type of 'possible_moves' (line 84)
                possible_moves_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 31), 'possible_moves', False)
                # Calling possible_moves(args, kwargs) (line 84)
                possible_moves_call_result_432 = invoke(stypy.reporting.localization.Localization(__file__, 84, 31), possible_moves_428, *[testboard_429, first_430], **kwargs_431)
                
                # Processing the call keyword arguments (line 84)
                kwargs_433 = {}
                # Getting the type of 'len' (line 84)
                len_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 27), 'len', False)
                # Calling len(args, kwargs) (line 84)
                len_call_result_434 = invoke(stypy.reporting.localization.Localization(__file__, 84, 27), len_427, *[possible_moves_call_result_432], **kwargs_433)
                
                # Assigning a type to the variable 'mobility' (line 84)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'mobility', len_call_result_434)
                # SSA join for if statement (line 81)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Name to a Name (line 85):
            
            # Assigning a Name to a Name (line 85):
            # Getting the type of 'mobility' (line 85)
            mobility_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 20), 'mobility')
            # Assigning a type to the variable 'score' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'score', mobility_435)
            
            # Getting the type of 'color' (line 86)
            color_436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'color')
            # Getting the type of 'first' (line 86)
            first_437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 24), 'first')
            # Applying the binary operator '!=' (line 86)
            result_ne_438 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 15), '!=', color_436, first_437)
            
            # Testing if the type of an if condition is none (line 86)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 86, 12), result_ne_438):
                pass
            else:
                
                # Testing the type of an if condition (line 86)
                if_condition_439 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 12), result_ne_438)
                # Assigning a type to the variable 'if_condition_439' (line 86)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'if_condition_439', if_condition_439)
                # SSA begins for if statement (line 86)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a BinOp to a Name (line 87):
                
                # Assigning a BinOp to a Name (line 87):
                int_440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 24), 'int')
                # Getting the type of 'score' (line 87)
                score_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 29), 'score')
                # Applying the binary operator '-' (line 87)
                result_sub_442 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 24), '-', int_440, score_441)
                
                # Assigning a type to the variable 'score' (line 87)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'score', result_sub_442)
                # SSA join for if statement (line 86)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 74)
            if_condition_376 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 8), result_contains_375)
            # Assigning a type to the variable 'if_condition_376' (line 74)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'if_condition_376', if_condition_376)
            # SSA begins for if statement (line 74)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Tuple to a Tuple (line 75):
            
            # Assigning a Num to a Name (line 75):
            int_377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 30), 'int')
            # Assigning a type to the variable 'tuple_assignment_13' (line 75)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'tuple_assignment_13', int_377)
            
            # Assigning a Num to a Name (line 75):
            int_378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 34), 'int')
            # Assigning a type to the variable 'tuple_assignment_14' (line 75)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'tuple_assignment_14', int_378)
            
            # Assigning a Name to a Name (line 75):
            # Getting the type of 'tuple_assignment_13' (line 75)
            tuple_assignment_13_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'tuple_assignment_13')
            # Assigning a type to the variable 'mobility' (line 75)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'mobility', tuple_assignment_13_379)
            
            # Assigning a Name to a Name (line 75):
            # Getting the type of 'tuple_assignment_14' (line 75)
            tuple_assignment_14_380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'tuple_assignment_14')
            # Assigning a type to the variable 'score' (line 75)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 22), 'score', tuple_assignment_14_380)
            
            # Getting the type of 'color' (line 76)
            color_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 15), 'color')
            # Getting the type of 'first' (line 76)
            first_382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 24), 'first')
            # Applying the binary operator '!=' (line 76)
            result_ne_383 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 15), '!=', color_381, first_382)
            
            # Testing if the type of an if condition is none (line 76)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 76, 12), result_ne_383):
                pass
            else:
                
                # Testing the type of an if condition (line 76)
                if_condition_384 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 12), result_ne_383)
                # Assigning a type to the variable 'if_condition_384' (line 76)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'if_condition_384', if_condition_384)
                # SSA begins for if statement (line 76)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a BinOp to a Name (line 77):
                
                # Assigning a BinOp to a Name (line 77):
                int_385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 27), 'int')
                # Getting the type of 'mobility' (line 77)
                mobility_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 32), 'mobility')
                # Applying the binary operator '-' (line 77)
                result_sub_387 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 27), '-', int_385, mobility_386)
                
                # Assigning a type to the variable 'mobility' (line 77)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 16), 'mobility', result_sub_387)
                # SSA join for if statement (line 76)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA branch for the else part of an if statement (line 74)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a ListComp to a Name (line 79):
            
            # Assigning a ListComp to a Name (line 79):
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'board' (line 79)
            board_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 65), 'board')
            comprehension_393 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 25), board_392)
            # Assigning a type to the variable 'line' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 25), 'line', comprehension_393)
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'line' (line 79)
            line_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 47), 'line')
            comprehension_390 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 26), line_389)
            # Assigning a type to the variable 'square' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 26), 'square', comprehension_390)
            # Getting the type of 'square' (line 79)
            square_388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 26), 'square')
            list_391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 26), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 26), list_391, square_388)
            list_394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 25), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 25), list_394, list_391)
            # Assigning a type to the variable 'testboard' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'testboard', list_394)
            
            # Call to flip_stones(...): (line 80)
            # Processing the call arguments (line 80)
            # Getting the type of 'testboard' (line 80)
            testboard_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 24), 'testboard', False)
            # Getting the type of 'move' (line 80)
            move_397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 35), 'move', False)
            # Getting the type of 'color' (line 80)
            color_398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 41), 'color', False)
            # Processing the call keyword arguments (line 80)
            kwargs_399 = {}
            # Getting the type of 'flip_stones' (line 80)
            flip_stones_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'flip_stones', False)
            # Calling flip_stones(args, kwargs) (line 80)
            flip_stones_call_result_400 = invoke(stypy.reporting.localization.Localization(__file__, 80, 12), flip_stones_395, *[testboard_396, move_397, color_398], **kwargs_399)
            
            
            # Getting the type of 'step' (line 81)
            step_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 15), 'step')
            # Getting the type of 'depth' (line 81)
            depth_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 22), 'depth')
            # Applying the binary operator '<' (line 81)
            result_lt_403 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 15), '<', step_401, depth_402)
            
            # Testing if the type of an if condition is none (line 81)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 81, 12), result_lt_403):
                
                # Assigning a Call to a Name (line 84):
                
                # Assigning a Call to a Name (line 84):
                
                # Call to len(...): (line 84)
                # Processing the call arguments (line 84)
                
                # Call to possible_moves(...): (line 84)
                # Processing the call arguments (line 84)
                # Getting the type of 'testboard' (line 84)
                testboard_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 46), 'testboard', False)
                # Getting the type of 'first' (line 84)
                first_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 57), 'first', False)
                # Processing the call keyword arguments (line 84)
                kwargs_431 = {}
                # Getting the type of 'possible_moves' (line 84)
                possible_moves_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 31), 'possible_moves', False)
                # Calling possible_moves(args, kwargs) (line 84)
                possible_moves_call_result_432 = invoke(stypy.reporting.localization.Localization(__file__, 84, 31), possible_moves_428, *[testboard_429, first_430], **kwargs_431)
                
                # Processing the call keyword arguments (line 84)
                kwargs_433 = {}
                # Getting the type of 'len' (line 84)
                len_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 27), 'len', False)
                # Calling len(args, kwargs) (line 84)
                len_call_result_434 = invoke(stypy.reporting.localization.Localization(__file__, 84, 27), len_427, *[possible_moves_call_result_432], **kwargs_433)
                
                # Assigning a type to the variable 'mobility' (line 84)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'mobility', len_call_result_434)
            else:
                
                # Testing the type of an if condition (line 81)
                if_condition_404 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 12), result_lt_403)
                # Assigning a type to the variable 'if_condition_404' (line 81)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'if_condition_404', if_condition_404)
                # SSA begins for if statement (line 81)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Tuple (line 82):
                
                # Assigning a Call to a Name:
                
                # Call to best_move(...): (line 82)
                # Processing the call arguments (line 82)
                # Getting the type of 'testboard' (line 82)
                testboard_406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 48), 'testboard', False)
                
                # Getting the type of 'color' (line 82)
                color_407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 60), 'color', False)
                # Applying the 'usub' unary operator (line 82)
                result___neg___408 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 59), 'usub', color_407)
                
                # Getting the type of 'first' (line 82)
                first_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 67), 'first', False)
                # Getting the type of 'step' (line 82)
                step_410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 74), 'step', False)
                int_411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 81), 'int')
                # Applying the binary operator '+' (line 82)
                result_add_412 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 74), '+', step_410, int_411)
                
                # Processing the call keyword arguments (line 82)
                kwargs_413 = {}
                # Getting the type of 'best_move' (line 82)
                best_move_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 38), 'best_move', False)
                # Calling best_move(args, kwargs) (line 82)
                best_move_call_result_414 = invoke(stypy.reporting.localization.Localization(__file__, 82, 38), best_move_405, *[testboard_406, result___neg___408, first_409, result_add_412], **kwargs_413)
                
                # Assigning a type to the variable 'call_assignment_15' (line 82)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'call_assignment_15', best_move_call_result_414)
                
                # Assigning a Call to a Name (line 82):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 16), 'int')
                # Processing the call keyword arguments
                kwargs_418 = {}
                # Getting the type of 'call_assignment_15' (line 82)
                call_assignment_15_415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'call_assignment_15', False)
                # Obtaining the member '__getitem__' of a type (line 82)
                getitem___416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 16), call_assignment_15_415, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_419 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___416, *[int_417], **kwargs_418)
                
                # Assigning a type to the variable 'call_assignment_16' (line 82)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'call_assignment_16', getitem___call_result_419)
                
                # Assigning a Name to a Name (line 82):
                # Getting the type of 'call_assignment_16' (line 82)
                call_assignment_16_420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'call_assignment_16')
                # Assigning a type to the variable 'next_move' (line 82)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'next_move', call_assignment_16_420)
                
                # Assigning a Call to a Name (line 82):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 16), 'int')
                # Processing the call keyword arguments
                kwargs_424 = {}
                # Getting the type of 'call_assignment_15' (line 82)
                call_assignment_15_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'call_assignment_15', False)
                # Obtaining the member '__getitem__' of a type (line 82)
                getitem___422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 16), call_assignment_15_421, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_425 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___422, *[int_423], **kwargs_424)
                
                # Assigning a type to the variable 'call_assignment_17' (line 82)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'call_assignment_17', getitem___call_result_425)
                
                # Assigning a Name to a Name (line 82):
                # Getting the type of 'call_assignment_17' (line 82)
                call_assignment_17_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'call_assignment_17')
                # Assigning a type to the variable 'mobility' (line 82)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 27), 'mobility', call_assignment_17_426)
                # SSA branch for the else part of an if statement (line 81)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Name (line 84):
                
                # Assigning a Call to a Name (line 84):
                
                # Call to len(...): (line 84)
                # Processing the call arguments (line 84)
                
                # Call to possible_moves(...): (line 84)
                # Processing the call arguments (line 84)
                # Getting the type of 'testboard' (line 84)
                testboard_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 46), 'testboard', False)
                # Getting the type of 'first' (line 84)
                first_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 57), 'first', False)
                # Processing the call keyword arguments (line 84)
                kwargs_431 = {}
                # Getting the type of 'possible_moves' (line 84)
                possible_moves_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 31), 'possible_moves', False)
                # Calling possible_moves(args, kwargs) (line 84)
                possible_moves_call_result_432 = invoke(stypy.reporting.localization.Localization(__file__, 84, 31), possible_moves_428, *[testboard_429, first_430], **kwargs_431)
                
                # Processing the call keyword arguments (line 84)
                kwargs_433 = {}
                # Getting the type of 'len' (line 84)
                len_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 27), 'len', False)
                # Calling len(args, kwargs) (line 84)
                len_call_result_434 = invoke(stypy.reporting.localization.Localization(__file__, 84, 27), len_427, *[possible_moves_call_result_432], **kwargs_433)
                
                # Assigning a type to the variable 'mobility' (line 84)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'mobility', len_call_result_434)
                # SSA join for if statement (line 81)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Name to a Name (line 85):
            
            # Assigning a Name to a Name (line 85):
            # Getting the type of 'mobility' (line 85)
            mobility_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 20), 'mobility')
            # Assigning a type to the variable 'score' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'score', mobility_435)
            
            # Getting the type of 'color' (line 86)
            color_436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'color')
            # Getting the type of 'first' (line 86)
            first_437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 24), 'first')
            # Applying the binary operator '!=' (line 86)
            result_ne_438 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 15), '!=', color_436, first_437)
            
            # Testing if the type of an if condition is none (line 86)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 86, 12), result_ne_438):
                pass
            else:
                
                # Testing the type of an if condition (line 86)
                if_condition_439 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 12), result_ne_438)
                # Assigning a type to the variable 'if_condition_439' (line 86)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'if_condition_439', if_condition_439)
                # SSA begins for if statement (line 86)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a BinOp to a Name (line 87):
                
                # Assigning a BinOp to a Name (line 87):
                int_440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 24), 'int')
                # Getting the type of 'score' (line 87)
                score_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 29), 'score')
                # Applying the binary operator '-' (line 87)
                result_sub_442 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 24), '-', int_440, score_441)
                
                # Assigning a type to the variable 'score' (line 87)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'score', result_sub_442)
                # SSA join for if statement (line 86)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 74)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'score' (line 88)
        score_443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 11), 'score')
        # Getting the type of 'max_score' (line 88)
        max_score_444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 20), 'max_score')
        # Applying the binary operator '>=' (line 88)
        result_ge_445 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 11), '>=', score_443, max_score_444)
        
        # Testing if the type of an if condition is none (line 88)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 88, 8), result_ge_445):
            pass
        else:
            
            # Testing the type of an if condition (line 88)
            if_condition_446 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 8), result_ge_445)
            # Assigning a type to the variable 'if_condition_446' (line 88)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'if_condition_446', if_condition_446)
            # SSA begins for if statement (line 88)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Tuple to a Tuple (line 89):
            
            # Assigning a Name to a Name (line 89):
            # Getting the type of 'move' (line 89)
            move_447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 48), 'move')
            # Assigning a type to the variable 'tuple_assignment_18' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'tuple_assignment_18', move_447)
            
            # Assigning a Name to a Name (line 89):
            # Getting the type of 'mobility' (line 89)
            mobility_448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 54), 'mobility')
            # Assigning a type to the variable 'tuple_assignment_19' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'tuple_assignment_19', mobility_448)
            
            # Assigning a Name to a Name (line 89):
            # Getting the type of 'score' (line 89)
            score_449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 64), 'score')
            # Assigning a type to the variable 'tuple_assignment_20' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'tuple_assignment_20', score_449)
            
            # Assigning a Name to a Name (line 89):
            # Getting the type of 'tuple_assignment_18' (line 89)
            tuple_assignment_18_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'tuple_assignment_18')
            # Assigning a type to the variable 'max_move' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'max_move', tuple_assignment_18_450)
            
            # Assigning a Name to a Name (line 89):
            # Getting the type of 'tuple_assignment_19' (line 89)
            tuple_assignment_19_451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'tuple_assignment_19')
            # Assigning a type to the variable 'max_mobility' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 22), 'max_mobility', tuple_assignment_19_451)
            
            # Assigning a Name to a Name (line 89):
            # Getting the type of 'tuple_assignment_20' (line 89)
            tuple_assignment_20_452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'tuple_assignment_20')
            # Assigning a type to the variable 'max_score' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 36), 'max_score', tuple_assignment_20_452)
            # SSA join for if statement (line 88)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Obtaining an instance of the builtin type 'tuple' (line 90)
    tuple_453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 90)
    # Adding element type (line 90)
    # Getting the type of 'max_move' (line 90)
    max_move_454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 11), 'max_move')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 11), tuple_453, max_move_454)
    # Adding element type (line 90)
    # Getting the type of 'max_mobility' (line 90)
    max_mobility_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 21), 'max_mobility')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 11), tuple_453, max_mobility_455)
    
    # Assigning a type to the variable 'stypy_return_type' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'stypy_return_type', tuple_453)
    
    # ################# End of 'best_move(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'best_move' in the type store
    # Getting the type of 'stypy_return_type' (line 71)
    stypy_return_type_456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_456)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'best_move'
    return stypy_return_type_456

# Assigning a type to the variable 'best_move' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'best_move', best_move)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 93, 0, False)
    
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

    
    # Assigning a Name to a Name (line 94):
    
    # Assigning a Name to a Name (line 94):
    # Getting the type of 'black' (line 94)
    black_457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 11), 'black')
    # Assigning a type to the variable 'turn' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'turn', black_457)
    
    
    # Evaluating a boolean operation
    
    # Call to possible_moves(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'board' (line 95)
    board_459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 25), 'board', False)
    # Getting the type of 'black' (line 95)
    black_460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 32), 'black', False)
    # Processing the call keyword arguments (line 95)
    kwargs_461 = {}
    # Getting the type of 'possible_moves' (line 95)
    possible_moves_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 10), 'possible_moves', False)
    # Calling possible_moves(args, kwargs) (line 95)
    possible_moves_call_result_462 = invoke(stypy.reporting.localization.Localization(__file__, 95, 10), possible_moves_458, *[board_459, black_460], **kwargs_461)
    
    
    # Call to possible_moves(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'board' (line 95)
    board_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 57), 'board', False)
    # Getting the type of 'white' (line 95)
    white_465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 64), 'white', False)
    # Processing the call keyword arguments (line 95)
    kwargs_466 = {}
    # Getting the type of 'possible_moves' (line 95)
    possible_moves_463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 42), 'possible_moves', False)
    # Calling possible_moves(args, kwargs) (line 95)
    possible_moves_call_result_467 = invoke(stypy.reporting.localization.Localization(__file__, 95, 42), possible_moves_463, *[board_464, white_465], **kwargs_466)
    
    # Applying the binary operator 'or' (line 95)
    result_or_keyword_468 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 10), 'or', possible_moves_call_result_462, possible_moves_call_result_467)
    
    # Testing if the while is going to be iterated (line 95)
    # Testing the type of an if condition (line 95)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 4), result_or_keyword_468)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 95, 4), result_or_keyword_468):
        # SSA begins for while statement (line 95)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Call to possible_moves(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'board' (line 96)
        board_470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 26), 'board', False)
        # Getting the type of 'turn' (line 96)
        turn_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 33), 'turn', False)
        # Processing the call keyword arguments (line 96)
        kwargs_472 = {}
        # Getting the type of 'possible_moves' (line 96)
        possible_moves_469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 11), 'possible_moves', False)
        # Calling possible_moves(args, kwargs) (line 96)
        possible_moves_call_result_473 = invoke(stypy.reporting.localization.Localization(__file__, 96, 11), possible_moves_469, *[board_470, turn_471], **kwargs_472)
        
        # Testing if the type of an if condition is none (line 96)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 96, 8), possible_moves_call_result_473):
            pass
        else:
            
            # Testing the type of an if condition (line 96)
            if_condition_474 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 8), possible_moves_call_result_473)
            # Assigning a type to the variable 'if_condition_474' (line 96)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'if_condition_474', if_condition_474)
            # SSA begins for if statement (line 96)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to print_board(...): (line 97)
            # Processing the call arguments (line 97)
            # Getting the type of 'board' (line 97)
            board_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 24), 'board', False)
            # Getting the type of 'turn' (line 97)
            turn_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 31), 'turn', False)
            # Processing the call keyword arguments (line 97)
            kwargs_478 = {}
            # Getting the type of 'print_board' (line 97)
            print_board_475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'print_board', False)
            # Calling print_board(args, kwargs) (line 97)
            print_board_call_result_479 = invoke(stypy.reporting.localization.Localization(__file__, 97, 12), print_board_475, *[board_476, turn_477], **kwargs_478)
            
            
            # Getting the type of 'turn' (line 98)
            turn_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 15), 'turn')
            # Getting the type of 'black' (line 98)
            black_481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 23), 'black')
            # Applying the binary operator '==' (line 98)
            result_eq_482 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 15), '==', turn_480, black_481)
            
            # Testing if the type of an if condition is none (line 98)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 98, 12), result_eq_482):
                
                
                # SSA begins for try-except statement (line 102)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
                
                # Assigning a Call to a Name (line 103):
                
                # Assigning a Call to a Name (line 103):
                
                # Call to coordinates(...): (line 103)
                # Processing the call arguments (line 103)
                
                # Call to raw_input(...): (line 103)
                # Processing the call arguments (line 103)
                str_504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 49), 'str', 'move? ')
                # Processing the call keyword arguments (line 103)
                kwargs_505 = {}
                # Getting the type of 'raw_input' (line 103)
                raw_input_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 39), 'raw_input', False)
                # Calling raw_input(args, kwargs) (line 103)
                raw_input_call_result_506 = invoke(stypy.reporting.localization.Localization(__file__, 103, 39), raw_input_503, *[str_504], **kwargs_505)
                
                # Processing the call keyword arguments (line 103)
                kwargs_507 = {}
                # Getting the type of 'coordinates' (line 103)
                coordinates_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'coordinates', False)
                # Calling coordinates(args, kwargs) (line 103)
                coordinates_call_result_508 = invoke(stypy.reporting.localization.Localization(__file__, 103, 27), coordinates_502, *[raw_input_call_result_506], **kwargs_507)
                
                # Assigning a type to the variable 'move' (line 103)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 20), 'move', coordinates_call_result_508)
                # SSA branch for the except part of a try statement (line 102)
                # SSA branch for the except 'ValueError' branch of a try statement (line 102)
                module_type_store.open_ssa_branch('except')
                # SSA join for try-except statement (line 102)
                module_type_store = module_type_store.join_ssa_context()
                
            else:
                
                # Testing the type of an if condition (line 98)
                if_condition_483 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 12), result_eq_482)
                # Assigning a type to the variable 'if_condition_483' (line 98)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'if_condition_483', if_condition_483)
                # SSA begins for if statement (line 98)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Tuple (line 99):
                
                # Assigning a Call to a Name:
                
                # Call to best_move(...): (line 99)
                # Processing the call arguments (line 99)
                # Getting the type of 'board' (line 99)
                board_485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 43), 'board', False)
                # Getting the type of 'turn' (line 99)
                turn_486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 50), 'turn', False)
                # Getting the type of 'turn' (line 99)
                turn_487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 56), 'turn', False)
                # Processing the call keyword arguments (line 99)
                kwargs_488 = {}
                # Getting the type of 'best_move' (line 99)
                best_move_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 33), 'best_move', False)
                # Calling best_move(args, kwargs) (line 99)
                best_move_call_result_489 = invoke(stypy.reporting.localization.Localization(__file__, 99, 33), best_move_484, *[board_485, turn_486, turn_487], **kwargs_488)
                
                # Assigning a type to the variable 'call_assignment_21' (line 99)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'call_assignment_21', best_move_call_result_489)
                
                # Assigning a Call to a Name (line 99):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 16), 'int')
                # Processing the call keyword arguments
                kwargs_493 = {}
                # Getting the type of 'call_assignment_21' (line 99)
                call_assignment_21_490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'call_assignment_21', False)
                # Obtaining the member '__getitem__' of a type (line 99)
                getitem___491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), call_assignment_21_490, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_494 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___491, *[int_492], **kwargs_493)
                
                # Assigning a type to the variable 'call_assignment_22' (line 99)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'call_assignment_22', getitem___call_result_494)
                
                # Assigning a Name to a Name (line 99):
                # Getting the type of 'call_assignment_22' (line 99)
                call_assignment_22_495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'call_assignment_22')
                # Assigning a type to the variable 'move' (line 99)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'move', call_assignment_22_495)
                
                # Assigning a Call to a Name (line 99):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 16), 'int')
                # Processing the call keyword arguments
                kwargs_499 = {}
                # Getting the type of 'call_assignment_21' (line 99)
                call_assignment_21_496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'call_assignment_21', False)
                # Obtaining the member '__getitem__' of a type (line 99)
                getitem___497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), call_assignment_21_496, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_500 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___497, *[int_498], **kwargs_499)
                
                # Assigning a type to the variable 'call_assignment_23' (line 99)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'call_assignment_23', getitem___call_result_500)
                
                # Assigning a Name to a Name (line 99):
                # Getting the type of 'call_assignment_23' (line 99)
                call_assignment_23_501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'call_assignment_23')
                # Assigning a type to the variable 'mobility' (line 99)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 22), 'mobility', call_assignment_23_501)
                # SSA branch for the else part of an if statement (line 98)
                module_type_store.open_ssa_branch('else')
                
                
                # SSA begins for try-except statement (line 102)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
                
                # Assigning a Call to a Name (line 103):
                
                # Assigning a Call to a Name (line 103):
                
                # Call to coordinates(...): (line 103)
                # Processing the call arguments (line 103)
                
                # Call to raw_input(...): (line 103)
                # Processing the call arguments (line 103)
                str_504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 49), 'str', 'move? ')
                # Processing the call keyword arguments (line 103)
                kwargs_505 = {}
                # Getting the type of 'raw_input' (line 103)
                raw_input_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 39), 'raw_input', False)
                # Calling raw_input(args, kwargs) (line 103)
                raw_input_call_result_506 = invoke(stypy.reporting.localization.Localization(__file__, 103, 39), raw_input_503, *[str_504], **kwargs_505)
                
                # Processing the call keyword arguments (line 103)
                kwargs_507 = {}
                # Getting the type of 'coordinates' (line 103)
                coordinates_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'coordinates', False)
                # Calling coordinates(args, kwargs) (line 103)
                coordinates_call_result_508 = invoke(stypy.reporting.localization.Localization(__file__, 103, 27), coordinates_502, *[raw_input_call_result_506], **kwargs_507)
                
                # Assigning a type to the variable 'move' (line 103)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 20), 'move', coordinates_call_result_508)
                # SSA branch for the except part of a try statement (line 102)
                # SSA branch for the except 'ValueError' branch of a try statement (line 102)
                module_type_store.open_ssa_branch('except')
                # SSA join for try-except statement (line 102)
                module_type_store = module_type_store.join_ssa_context()
                
                # SSA join for if statement (line 98)
                module_type_store = module_type_store.join_ssa_context()
                

            
            
            # Call to possible_move(...): (line 107)
            # Processing the call arguments (line 107)
            # Getting the type of 'board' (line 107)
            board_510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 33), 'board', False)
            
            # Obtaining the type of the subscript
            int_511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 45), 'int')
            # Getting the type of 'move' (line 107)
            move_512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 40), 'move', False)
            # Obtaining the member '__getitem__' of a type (line 107)
            getitem___513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 40), move_512, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 107)
            subscript_call_result_514 = invoke(stypy.reporting.localization.Localization(__file__, 107, 40), getitem___513, int_511)
            
            
            # Obtaining the type of the subscript
            int_515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 54), 'int')
            # Getting the type of 'move' (line 107)
            move_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 49), 'move', False)
            # Obtaining the member '__getitem__' of a type (line 107)
            getitem___517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 49), move_516, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 107)
            subscript_call_result_518 = invoke(stypy.reporting.localization.Localization(__file__, 107, 49), getitem___517, int_515)
            
            # Getting the type of 'turn' (line 107)
            turn_519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 58), 'turn', False)
            # Processing the call keyword arguments (line 107)
            kwargs_520 = {}
            # Getting the type of 'possible_move' (line 107)
            possible_move_509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'possible_move', False)
            # Calling possible_move(args, kwargs) (line 107)
            possible_move_call_result_521 = invoke(stypy.reporting.localization.Localization(__file__, 107, 19), possible_move_509, *[board_510, subscript_call_result_514, subscript_call_result_518, turn_519], **kwargs_520)
            
            # Applying the 'not' unary operator (line 107)
            result_not__522 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 15), 'not', possible_move_call_result_521)
            
            # Testing if the type of an if condition is none (line 107)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 107, 12), result_not__522):
                
                # Call to flip_stones(...): (line 111)
                # Processing the call arguments (line 111)
                # Getting the type of 'board' (line 111)
                board_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 28), 'board', False)
                # Getting the type of 'move' (line 111)
                move_526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 35), 'move', False)
                # Getting the type of 'turn' (line 111)
                turn_527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 41), 'turn', False)
                # Processing the call keyword arguments (line 111)
                kwargs_528 = {}
                # Getting the type of 'flip_stones' (line 111)
                flip_stones_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'flip_stones', False)
                # Calling flip_stones(args, kwargs) (line 111)
                flip_stones_call_result_529 = invoke(stypy.reporting.localization.Localization(__file__, 111, 16), flip_stones_524, *[board_525, move_526, turn_527], **kwargs_528)
                
            else:
                
                # Testing the type of an if condition (line 107)
                if_condition_523 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 12), result_not__522)
                # Assigning a type to the variable 'if_condition_523' (line 107)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'if_condition_523', if_condition_523)
                # SSA begins for if statement (line 107)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA branch for the else part of an if statement (line 107)
                module_type_store.open_ssa_branch('else')
                
                # Call to flip_stones(...): (line 111)
                # Processing the call arguments (line 111)
                # Getting the type of 'board' (line 111)
                board_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 28), 'board', False)
                # Getting the type of 'move' (line 111)
                move_526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 35), 'move', False)
                # Getting the type of 'turn' (line 111)
                turn_527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 41), 'turn', False)
                # Processing the call keyword arguments (line 111)
                kwargs_528 = {}
                # Getting the type of 'flip_stones' (line 111)
                flip_stones_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'flip_stones', False)
                # Calling flip_stones(args, kwargs) (line 111)
                flip_stones_call_result_529 = invoke(stypy.reporting.localization.Localization(__file__, 111, 16), flip_stones_524, *[board_525, move_526, turn_527], **kwargs_528)
                
                # SSA join for if statement (line 107)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 96)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a UnaryOp to a Name (line 113):
        
        # Assigning a UnaryOp to a Name (line 113):
        
        # Getting the type of 'turn' (line 113)
        turn_530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'turn')
        # Applying the 'usub' unary operator (line 113)
        result___neg___531 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 15), 'usub', turn_530)
        
        # Assigning a type to the variable 'turn' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'turn', result___neg___531)
        # SSA join for while statement (line 95)
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Call to print_board(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'board' (line 114)
    board_533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'board', False)
    # Getting the type of 'turn' (line 114)
    turn_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 23), 'turn', False)
    # Processing the call keyword arguments (line 114)
    kwargs_535 = {}
    # Getting the type of 'print_board' (line 114)
    print_board_532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'print_board', False)
    # Calling print_board(args, kwargs) (line 114)
    print_board_call_result_536 = invoke(stypy.reporting.localization.Localization(__file__, 114, 4), print_board_532, *[board_533, turn_534], **kwargs_535)
    
    
    
    # Call to stone_count(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'board' (line 115)
    board_538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), 'board', False)
    # Getting the type of 'black' (line 115)
    black_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 26), 'black', False)
    # Processing the call keyword arguments (line 115)
    kwargs_540 = {}
    # Getting the type of 'stone_count' (line 115)
    stone_count_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 7), 'stone_count', False)
    # Calling stone_count(args, kwargs) (line 115)
    stone_count_call_result_541 = invoke(stypy.reporting.localization.Localization(__file__, 115, 7), stone_count_537, *[board_538, black_539], **kwargs_540)
    
    
    # Call to stone_count(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'board' (line 115)
    board_543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 48), 'board', False)
    # Getting the type of 'white' (line 115)
    white_544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 55), 'white', False)
    # Processing the call keyword arguments (line 115)
    kwargs_545 = {}
    # Getting the type of 'stone_count' (line 115)
    stone_count_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 36), 'stone_count', False)
    # Calling stone_count(args, kwargs) (line 115)
    stone_count_call_result_546 = invoke(stypy.reporting.localization.Localization(__file__, 115, 36), stone_count_542, *[board_543, white_544], **kwargs_545)
    
    # Applying the binary operator '==' (line 115)
    result_eq_547 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 7), '==', stone_count_call_result_541, stone_count_call_result_546)
    
    # Testing if the type of an if condition is none (line 115)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 115, 4), result_eq_547):
        
        
        # Call to stone_count(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'board' (line 119)
        board_550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 23), 'board', False)
        # Getting the type of 'black' (line 119)
        black_551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 30), 'black', False)
        # Processing the call keyword arguments (line 119)
        kwargs_552 = {}
        # Getting the type of 'stone_count' (line 119)
        stone_count_549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 11), 'stone_count', False)
        # Calling stone_count(args, kwargs) (line 119)
        stone_count_call_result_553 = invoke(stypy.reporting.localization.Localization(__file__, 119, 11), stone_count_549, *[board_550, black_551], **kwargs_552)
        
        
        # Call to stone_count(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'board' (line 119)
        board_555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 51), 'board', False)
        # Getting the type of 'white' (line 119)
        white_556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 58), 'white', False)
        # Processing the call keyword arguments (line 119)
        kwargs_557 = {}
        # Getting the type of 'stone_count' (line 119)
        stone_count_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 39), 'stone_count', False)
        # Calling stone_count(args, kwargs) (line 119)
        stone_count_call_result_558 = invoke(stypy.reporting.localization.Localization(__file__, 119, 39), stone_count_554, *[board_555, white_556], **kwargs_557)
        
        # Applying the binary operator '>' (line 119)
        result_gt_559 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 11), '>', stone_count_call_result_553, stone_count_call_result_558)
        
        # Testing if the type of an if condition is none (line 119)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 119, 8), result_gt_559):
            pass
        else:
            
            # Testing the type of an if condition (line 119)
            if_condition_560 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 8), result_gt_559)
            # Assigning a type to the variable 'if_condition_560' (line 119)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'if_condition_560', if_condition_560)
            # SSA begins for if statement (line 119)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            pass
            # SSA branch for the else part of an if statement (line 119)
            module_type_store.open_ssa_branch('else')
            pass
            # SSA join for if statement (line 119)
            module_type_store = module_type_store.join_ssa_context()
            

    else:
        
        # Testing the type of an if condition (line 115)
        if_condition_548 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 115, 4), result_eq_547)
        # Assigning a type to the variable 'if_condition_548' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'if_condition_548', if_condition_548)
        # SSA begins for if statement (line 115)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA branch for the else part of an if statement (line 115)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to stone_count(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'board' (line 119)
        board_550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 23), 'board', False)
        # Getting the type of 'black' (line 119)
        black_551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 30), 'black', False)
        # Processing the call keyword arguments (line 119)
        kwargs_552 = {}
        # Getting the type of 'stone_count' (line 119)
        stone_count_549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 11), 'stone_count', False)
        # Calling stone_count(args, kwargs) (line 119)
        stone_count_call_result_553 = invoke(stypy.reporting.localization.Localization(__file__, 119, 11), stone_count_549, *[board_550, black_551], **kwargs_552)
        
        
        # Call to stone_count(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'board' (line 119)
        board_555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 51), 'board', False)
        # Getting the type of 'white' (line 119)
        white_556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 58), 'white', False)
        # Processing the call keyword arguments (line 119)
        kwargs_557 = {}
        # Getting the type of 'stone_count' (line 119)
        stone_count_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 39), 'stone_count', False)
        # Calling stone_count(args, kwargs) (line 119)
        stone_count_call_result_558 = invoke(stypy.reporting.localization.Localization(__file__, 119, 39), stone_count_554, *[board_555, white_556], **kwargs_557)
        
        # Applying the binary operator '>' (line 119)
        result_gt_559 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 11), '>', stone_count_call_result_553, stone_count_call_result_558)
        
        # Testing if the type of an if condition is none (line 119)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 119, 8), result_gt_559):
            pass
        else:
            
            # Testing the type of an if condition (line 119)
            if_condition_560 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 8), result_gt_559)
            # Assigning a type to the variable 'if_condition_560' (line 119)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'if_condition_560', if_condition_560)
            # SSA begins for if statement (line 119)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            pass
            # SSA branch for the else part of an if statement (line 119)
            module_type_store.open_ssa_branch('else')
            pass
            # SSA join for if statement (line 119)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 115)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'True' (line 123)
    True_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type', True_561)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 93)
    stypy_return_type_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_562)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_562

# Assigning a type to the variable 'run' (line 93)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'run', run)

# Call to run(...): (line 126)
# Processing the call keyword arguments (line 126)
kwargs_564 = {}
# Getting the type of 'run' (line 126)
run_563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 0), 'run', False)
# Calling run(args, kwargs) (line 126)
run_call_result_565 = invoke(stypy.reporting.localization.Localization(__file__, 126, 0), run_563, *[], **kwargs_564)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

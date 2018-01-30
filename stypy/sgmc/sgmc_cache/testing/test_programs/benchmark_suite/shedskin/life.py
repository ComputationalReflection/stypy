
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: # -*- coding: utf-8 -*-
3: #
4: # Copyright 2010 Francesco Frassinelli <fraph24@gmail.com>
5: #
6: #    pylife is free software: you can redistribute it and/or modify
7: #    it under the terms of the GNU General Public License as published by
8: #    the Free Software Foundation, either version 3 of the License, or
9: #    (at your option) any later version.
10: #
11: #    This program is distributed in the hope that it will be useful,
12: #    but WITHOUT ANY WARRANTY; without even the implied warranty of
13: #    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
14: #    GNU General Public License for more details.
15: #
16: #    You should have received a copy of the GNU General Public License
17: #    along with this program.  If not, see <http://www.gnu.org/licenses/>.
18: 
19: ''' Implementation of: http://en.wikipedia.org/wiki/Conway's_Game_of_Life 
20:         Tested on Python 2.6.4 and Python 3.1.1 '''
21: 
22: from collections import defaultdict
23: from itertools import product
24: from sys import argv
25: 
26: 
27: def add(board, pos):
28:     ''' Adds eight cells near current cell '''
29:     row, column = pos
30:     return \
31:         board[row - 1, column - 1] + \
32:         board[row - 1, column] + \
33:         board[row - 1, column + 1] + \
34:         board[row, column - 1] + \
35:         board[row, column + 1] + \
36:         board[row + 1, column - 1] + \
37:         board[row + 1, column] + \
38:         board[row + 1, column + 1]
39: 
40: 
41: def snext(board):
42:     ''' Calculates the next stage '''
43:     new = defaultdict(int, board)
44:     for pos in list(board.keys()):
45:         near = add(board, pos)
46:         item = board[pos]
47:         if near not in (2, 3) and item:
48:             new[pos] = 0
49:         elif near == 3 and not item:
50:             new[pos] = 1
51:     return new
52: 
53: 
54: def process(board):
55:     ''' Finds if this board repeats itself '''
56:     history = [defaultdict(None, board)]
57:     while 1:
58:         board = snext(board)
59:         if board in history:
60:             if board == history[0]:
61:                 return board
62:             return None
63:         history.append(defaultdict(None, board))
64: 
65: 
66: def generator(rows, columns):
67:     ''' Generates a board '''
68:     ppos = [(row, column) for row in range(rows)
69:             for column in range(columns)]
70:     possibilities = product((0, 1), repeat=rows * columns)
71:     for case in possibilities:
72:         board = defaultdict(int)
73:         for pos, value in zip(ppos, case):
74:             board[pos] = value
75:         yield board
76: 
77: 
78: def bruteforce(rows, columns):
79:     global count
80:     count = 0
81:     for board in map(process, generator(rows, columns)):
82:         if board is not None:
83:             count += 1
84:             # print board
85: 
86: 
87: def run():
88:     rows, columns = 4, 3
89:     bruteforce(rows, columns)
90:     # print count
91:     #    try:
92:     #        rows, columns = int(argv[1]), int(argv[2])
93:     #    except IndexError:
94:     #        print("Usage: %s [rows] [columns]" % argv[0])
95:     #    except ValueError:
96:     #        print("Usage: %s [rows] [columns]" % argv[0])
97:     #    else:
98:     #        bruteforce(rows, columns)
99:     return True
100: 
101: 
102: run()
103: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, (-1)), 'str', " Implementation of: http://en.wikipedia.org/wiki/Conway's_Game_of_Life \n        Tested on Python 2.6.4 and Python 3.1.1 ")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from collections import defaultdict' statement (line 22)
try:
    from collections import defaultdict

except:
    defaultdict = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'collections', None, module_type_store, ['defaultdict'], [defaultdict])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from itertools import product' statement (line 23)
try:
    from itertools import product

except:
    product = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'itertools', None, module_type_store, ['product'], [product])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from sys import argv' statement (line 24)
try:
    from sys import argv

except:
    argv = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'sys', None, module_type_store, ['argv'], [argv])


@norecursion
def add(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'add'
    module_type_store = module_type_store.open_function_context('add', 27, 0, False)
    
    # Passed parameters checking function
    add.stypy_localization = localization
    add.stypy_type_of_self = None
    add.stypy_type_store = module_type_store
    add.stypy_function_name = 'add'
    add.stypy_param_names_list = ['board', 'pos']
    add.stypy_varargs_param_name = None
    add.stypy_kwargs_param_name = None
    add.stypy_call_defaults = defaults
    add.stypy_call_varargs = varargs
    add.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'add', ['board', 'pos'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'add', localization, ['board', 'pos'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'add(...)' code ##################

    str_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 4), 'str', ' Adds eight cells near current cell ')
    
    # Assigning a Name to a Tuple (line 29):
    
    # Assigning a Subscript to a Name (line 29):
    
    # Obtaining the type of the subscript
    int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'int')
    # Getting the type of 'pos' (line 29)
    pos_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 18), 'pos')
    # Obtaining the member '__getitem__' of a type (line 29)
    getitem___9 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 4), pos_8, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 29)
    subscript_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 29, 4), getitem___9, int_7)
    
    # Assigning a type to the variable 'tuple_var_assignment_1' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'tuple_var_assignment_1', subscript_call_result_10)
    
    # Assigning a Subscript to a Name (line 29):
    
    # Obtaining the type of the subscript
    int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'int')
    # Getting the type of 'pos' (line 29)
    pos_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 18), 'pos')
    # Obtaining the member '__getitem__' of a type (line 29)
    getitem___13 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 4), pos_12, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 29)
    subscript_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 29, 4), getitem___13, int_11)
    
    # Assigning a type to the variable 'tuple_var_assignment_2' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'tuple_var_assignment_2', subscript_call_result_14)
    
    # Assigning a Name to a Name (line 29):
    # Getting the type of 'tuple_var_assignment_1' (line 29)
    tuple_var_assignment_1_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'tuple_var_assignment_1')
    # Assigning a type to the variable 'row' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'row', tuple_var_assignment_1_15)
    
    # Assigning a Name to a Name (line 29):
    # Getting the type of 'tuple_var_assignment_2' (line 29)
    tuple_var_assignment_2_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'tuple_var_assignment_2')
    # Assigning a type to the variable 'column' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 9), 'column', tuple_var_assignment_2_16)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 31)
    tuple_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 31)
    # Adding element type (line 31)
    # Getting the type of 'row' (line 31)
    row_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'row')
    int_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 20), 'int')
    # Applying the binary operator '-' (line 31)
    result_sub_20 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 14), '-', row_18, int_19)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 14), tuple_17, result_sub_20)
    # Adding element type (line 31)
    # Getting the type of 'column' (line 31)
    column_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 23), 'column')
    int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 32), 'int')
    # Applying the binary operator '-' (line 31)
    result_sub_23 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 23), '-', column_21, int_22)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 14), tuple_17, result_sub_23)
    
    # Getting the type of 'board' (line 31)
    board_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'board')
    # Obtaining the member '__getitem__' of a type (line 31)
    getitem___25 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), board_24, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 31)
    subscript_call_result_26 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), getitem___25, tuple_17)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 32)
    tuple_27 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 32)
    # Adding element type (line 32)
    # Getting the type of 'row' (line 32)
    row_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 14), 'row')
    int_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 20), 'int')
    # Applying the binary operator '-' (line 32)
    result_sub_30 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 14), '-', row_28, int_29)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 14), tuple_27, result_sub_30)
    # Adding element type (line 32)
    # Getting the type of 'column' (line 32)
    column_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 23), 'column')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 14), tuple_27, column_31)
    
    # Getting the type of 'board' (line 32)
    board_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'board')
    # Obtaining the member '__getitem__' of a type (line 32)
    getitem___33 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), board_32, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 32)
    subscript_call_result_34 = invoke(stypy.reporting.localization.Localization(__file__, 32, 8), getitem___33, tuple_27)
    
    # Applying the binary operator '+' (line 31)
    result_add_35 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 8), '+', subscript_call_result_26, subscript_call_result_34)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 33)
    tuple_36 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 33)
    # Adding element type (line 33)
    # Getting the type of 'row' (line 33)
    row_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 14), 'row')
    int_38 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 20), 'int')
    # Applying the binary operator '-' (line 33)
    result_sub_39 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 14), '-', row_37, int_38)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 14), tuple_36, result_sub_39)
    # Adding element type (line 33)
    # Getting the type of 'column' (line 33)
    column_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 23), 'column')
    int_41 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 32), 'int')
    # Applying the binary operator '+' (line 33)
    result_add_42 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 23), '+', column_40, int_41)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 14), tuple_36, result_add_42)
    
    # Getting the type of 'board' (line 33)
    board_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'board')
    # Obtaining the member '__getitem__' of a type (line 33)
    getitem___44 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), board_43, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 33)
    subscript_call_result_45 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), getitem___44, tuple_36)
    
    # Applying the binary operator '+' (line 32)
    result_add_46 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 31), '+', result_add_35, subscript_call_result_45)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 34)
    tuple_47 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 34)
    # Adding element type (line 34)
    # Getting the type of 'row' (line 34)
    row_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 14), 'row')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 14), tuple_47, row_48)
    # Adding element type (line 34)
    # Getting the type of 'column' (line 34)
    column_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'column')
    int_50 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 28), 'int')
    # Applying the binary operator '-' (line 34)
    result_sub_51 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 19), '-', column_49, int_50)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 14), tuple_47, result_sub_51)
    
    # Getting the type of 'board' (line 34)
    board_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'board')
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___53 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), board_52, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_54 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), getitem___53, tuple_47)
    
    # Applying the binary operator '+' (line 33)
    result_add_55 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 35), '+', result_add_46, subscript_call_result_54)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 35)
    tuple_56 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 35)
    # Adding element type (line 35)
    # Getting the type of 'row' (line 35)
    row_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 14), 'row')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 14), tuple_56, row_57)
    # Adding element type (line 35)
    # Getting the type of 'column' (line 35)
    column_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 19), 'column')
    int_59 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 28), 'int')
    # Applying the binary operator '+' (line 35)
    result_add_60 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 19), '+', column_58, int_59)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 14), tuple_56, result_add_60)
    
    # Getting the type of 'board' (line 35)
    board_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'board')
    # Obtaining the member '__getitem__' of a type (line 35)
    getitem___62 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), board_61, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 35)
    subscript_call_result_63 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), getitem___62, tuple_56)
    
    # Applying the binary operator '+' (line 34)
    result_add_64 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 31), '+', result_add_55, subscript_call_result_63)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 36)
    tuple_65 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 36)
    # Adding element type (line 36)
    # Getting the type of 'row' (line 36)
    row_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 14), 'row')
    int_67 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 20), 'int')
    # Applying the binary operator '+' (line 36)
    result_add_68 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 14), '+', row_66, int_67)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 14), tuple_65, result_add_68)
    # Adding element type (line 36)
    # Getting the type of 'column' (line 36)
    column_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 23), 'column')
    int_70 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 32), 'int')
    # Applying the binary operator '-' (line 36)
    result_sub_71 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 23), '-', column_69, int_70)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 14), tuple_65, result_sub_71)
    
    # Getting the type of 'board' (line 36)
    board_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'board')
    # Obtaining the member '__getitem__' of a type (line 36)
    getitem___73 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), board_72, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 36)
    subscript_call_result_74 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), getitem___73, tuple_65)
    
    # Applying the binary operator '+' (line 35)
    result_add_75 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 31), '+', result_add_64, subscript_call_result_74)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 37)
    tuple_76 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 37)
    # Adding element type (line 37)
    # Getting the type of 'row' (line 37)
    row_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 14), 'row')
    int_78 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 20), 'int')
    # Applying the binary operator '+' (line 37)
    result_add_79 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 14), '+', row_77, int_78)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 14), tuple_76, result_add_79)
    # Adding element type (line 37)
    # Getting the type of 'column' (line 37)
    column_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 23), 'column')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 14), tuple_76, column_80)
    
    # Getting the type of 'board' (line 37)
    board_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'board')
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___82 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), board_81, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 37)
    subscript_call_result_83 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), getitem___82, tuple_76)
    
    # Applying the binary operator '+' (line 36)
    result_add_84 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 35), '+', result_add_75, subscript_call_result_83)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 38)
    tuple_85 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 38)
    # Adding element type (line 38)
    # Getting the type of 'row' (line 38)
    row_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 14), 'row')
    int_87 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 20), 'int')
    # Applying the binary operator '+' (line 38)
    result_add_88 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 14), '+', row_86, int_87)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 14), tuple_85, result_add_88)
    # Adding element type (line 38)
    # Getting the type of 'column' (line 38)
    column_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 23), 'column')
    int_90 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 32), 'int')
    # Applying the binary operator '+' (line 38)
    result_add_91 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 23), '+', column_89, int_90)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 14), tuple_85, result_add_91)
    
    # Getting the type of 'board' (line 38)
    board_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'board')
    # Obtaining the member '__getitem__' of a type (line 38)
    getitem___93 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), board_92, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 38)
    subscript_call_result_94 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), getitem___93, tuple_85)
    
    # Applying the binary operator '+' (line 37)
    result_add_95 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 31), '+', result_add_84, subscript_call_result_94)
    
    # Assigning a type to the variable 'stypy_return_type' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type', result_add_95)
    
    # ################# End of 'add(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'add' in the type store
    # Getting the type of 'stypy_return_type' (line 27)
    stypy_return_type_96 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_96)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'add'
    return stypy_return_type_96

# Assigning a type to the variable 'add' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'add', add)

@norecursion
def snext(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'snext'
    module_type_store = module_type_store.open_function_context('snext', 41, 0, False)
    
    # Passed parameters checking function
    snext.stypy_localization = localization
    snext.stypy_type_of_self = None
    snext.stypy_type_store = module_type_store
    snext.stypy_function_name = 'snext'
    snext.stypy_param_names_list = ['board']
    snext.stypy_varargs_param_name = None
    snext.stypy_kwargs_param_name = None
    snext.stypy_call_defaults = defaults
    snext.stypy_call_varargs = varargs
    snext.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'snext', ['board'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'snext', localization, ['board'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'snext(...)' code ##################

    str_97 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 4), 'str', ' Calculates the next stage ')
    
    # Assigning a Call to a Name (line 43):
    
    # Assigning a Call to a Name (line 43):
    
    # Call to defaultdict(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'int' (line 43)
    int_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 22), 'int', False)
    # Getting the type of 'board' (line 43)
    board_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 27), 'board', False)
    # Processing the call keyword arguments (line 43)
    kwargs_101 = {}
    # Getting the type of 'defaultdict' (line 43)
    defaultdict_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 10), 'defaultdict', False)
    # Calling defaultdict(args, kwargs) (line 43)
    defaultdict_call_result_102 = invoke(stypy.reporting.localization.Localization(__file__, 43, 10), defaultdict_98, *[int_99, board_100], **kwargs_101)
    
    # Assigning a type to the variable 'new' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'new', defaultdict_call_result_102)
    
    
    # Call to list(...): (line 44)
    # Processing the call arguments (line 44)
    
    # Call to keys(...): (line 44)
    # Processing the call keyword arguments (line 44)
    kwargs_106 = {}
    # Getting the type of 'board' (line 44)
    board_104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 20), 'board', False)
    # Obtaining the member 'keys' of a type (line 44)
    keys_105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 20), board_104, 'keys')
    # Calling keys(args, kwargs) (line 44)
    keys_call_result_107 = invoke(stypy.reporting.localization.Localization(__file__, 44, 20), keys_105, *[], **kwargs_106)
    
    # Processing the call keyword arguments (line 44)
    kwargs_108 = {}
    # Getting the type of 'list' (line 44)
    list_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'list', False)
    # Calling list(args, kwargs) (line 44)
    list_call_result_109 = invoke(stypy.reporting.localization.Localization(__file__, 44, 15), list_103, *[keys_call_result_107], **kwargs_108)
    
    # Testing if the for loop is going to be iterated (line 44)
    # Testing the type of a for loop iterable (line 44)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 44, 4), list_call_result_109)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 44, 4), list_call_result_109):
        # Getting the type of the for loop variable (line 44)
        for_loop_var_110 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 44, 4), list_call_result_109)
        # Assigning a type to the variable 'pos' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'pos', for_loop_var_110)
        # SSA begins for a for statement (line 44)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 45):
        
        # Assigning a Call to a Name (line 45):
        
        # Call to add(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'board' (line 45)
        board_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 19), 'board', False)
        # Getting the type of 'pos' (line 45)
        pos_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 26), 'pos', False)
        # Processing the call keyword arguments (line 45)
        kwargs_114 = {}
        # Getting the type of 'add' (line 45)
        add_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'add', False)
        # Calling add(args, kwargs) (line 45)
        add_call_result_115 = invoke(stypy.reporting.localization.Localization(__file__, 45, 15), add_111, *[board_112, pos_113], **kwargs_114)
        
        # Assigning a type to the variable 'near' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'near', add_call_result_115)
        
        # Assigning a Subscript to a Name (line 46):
        
        # Assigning a Subscript to a Name (line 46):
        
        # Obtaining the type of the subscript
        # Getting the type of 'pos' (line 46)
        pos_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 21), 'pos')
        # Getting the type of 'board' (line 46)
        board_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 15), 'board')
        # Obtaining the member '__getitem__' of a type (line 46)
        getitem___118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 15), board_117, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 46)
        subscript_call_result_119 = invoke(stypy.reporting.localization.Localization(__file__, 46, 15), getitem___118, pos_116)
        
        # Assigning a type to the variable 'item' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'item', subscript_call_result_119)
        
        # Evaluating a boolean operation
        
        # Getting the type of 'near' (line 47)
        near_120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'near')
        
        # Obtaining an instance of the builtin type 'tuple' (line 47)
        tuple_121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 47)
        # Adding element type (line 47)
        int_122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 24), tuple_121, int_122)
        # Adding element type (line 47)
        int_123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 24), tuple_121, int_123)
        
        # Applying the binary operator 'notin' (line 47)
        result_contains_124 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 11), 'notin', near_120, tuple_121)
        
        # Getting the type of 'item' (line 47)
        item_125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 34), 'item')
        # Applying the binary operator 'and' (line 47)
        result_and_keyword_126 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 11), 'and', result_contains_124, item_125)
        
        # Testing if the type of an if condition is none (line 47)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 47, 8), result_and_keyword_126):
            
            # Evaluating a boolean operation
            
            # Getting the type of 'near' (line 49)
            near_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 13), 'near')
            int_132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 21), 'int')
            # Applying the binary operator '==' (line 49)
            result_eq_133 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 13), '==', near_131, int_132)
            
            
            # Getting the type of 'item' (line 49)
            item_134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 31), 'item')
            # Applying the 'not' unary operator (line 49)
            result_not__135 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 27), 'not', item_134)
            
            # Applying the binary operator 'and' (line 49)
            result_and_keyword_136 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 13), 'and', result_eq_133, result_not__135)
            
            # Testing if the type of an if condition is none (line 49)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 49, 13), result_and_keyword_136):
                pass
            else:
                
                # Testing the type of an if condition (line 49)
                if_condition_137 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 13), result_and_keyword_136)
                # Assigning a type to the variable 'if_condition_137' (line 49)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 13), 'if_condition_137', if_condition_137)
                # SSA begins for if statement (line 49)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Num to a Subscript (line 50):
                
                # Assigning a Num to a Subscript (line 50):
                int_138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 23), 'int')
                # Getting the type of 'new' (line 50)
                new_139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'new')
                # Getting the type of 'pos' (line 50)
                pos_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 16), 'pos')
                # Storing an element on a container (line 50)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 12), new_139, (pos_140, int_138))
                # SSA join for if statement (line 49)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 47)
            if_condition_127 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 8), result_and_keyword_126)
            # Assigning a type to the variable 'if_condition_127' (line 47)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'if_condition_127', if_condition_127)
            # SSA begins for if statement (line 47)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Num to a Subscript (line 48):
            
            # Assigning a Num to a Subscript (line 48):
            int_128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 23), 'int')
            # Getting the type of 'new' (line 48)
            new_129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'new')
            # Getting the type of 'pos' (line 48)
            pos_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'pos')
            # Storing an element on a container (line 48)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 12), new_129, (pos_130, int_128))
            # SSA branch for the else part of an if statement (line 47)
            module_type_store.open_ssa_branch('else')
            
            # Evaluating a boolean operation
            
            # Getting the type of 'near' (line 49)
            near_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 13), 'near')
            int_132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 21), 'int')
            # Applying the binary operator '==' (line 49)
            result_eq_133 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 13), '==', near_131, int_132)
            
            
            # Getting the type of 'item' (line 49)
            item_134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 31), 'item')
            # Applying the 'not' unary operator (line 49)
            result_not__135 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 27), 'not', item_134)
            
            # Applying the binary operator 'and' (line 49)
            result_and_keyword_136 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 13), 'and', result_eq_133, result_not__135)
            
            # Testing if the type of an if condition is none (line 49)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 49, 13), result_and_keyword_136):
                pass
            else:
                
                # Testing the type of an if condition (line 49)
                if_condition_137 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 13), result_and_keyword_136)
                # Assigning a type to the variable 'if_condition_137' (line 49)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 13), 'if_condition_137', if_condition_137)
                # SSA begins for if statement (line 49)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Num to a Subscript (line 50):
                
                # Assigning a Num to a Subscript (line 50):
                int_138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 23), 'int')
                # Getting the type of 'new' (line 50)
                new_139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'new')
                # Getting the type of 'pos' (line 50)
                pos_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 16), 'pos')
                # Storing an element on a container (line 50)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 12), new_139, (pos_140, int_138))
                # SSA join for if statement (line 49)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 47)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'new' (line 51)
    new_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 11), 'new')
    # Assigning a type to the variable 'stypy_return_type' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'stypy_return_type', new_141)
    
    # ################# End of 'snext(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'snext' in the type store
    # Getting the type of 'stypy_return_type' (line 41)
    stypy_return_type_142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_142)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'snext'
    return stypy_return_type_142

# Assigning a type to the variable 'snext' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'snext', snext)

@norecursion
def process(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'process'
    module_type_store = module_type_store.open_function_context('process', 54, 0, False)
    
    # Passed parameters checking function
    process.stypy_localization = localization
    process.stypy_type_of_self = None
    process.stypy_type_store = module_type_store
    process.stypy_function_name = 'process'
    process.stypy_param_names_list = ['board']
    process.stypy_varargs_param_name = None
    process.stypy_kwargs_param_name = None
    process.stypy_call_defaults = defaults
    process.stypy_call_varargs = varargs
    process.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'process', ['board'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'process', localization, ['board'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'process(...)' code ##################

    str_143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 4), 'str', ' Finds if this board repeats itself ')
    
    # Assigning a List to a Name (line 56):
    
    # Assigning a List to a Name (line 56):
    
    # Obtaining an instance of the builtin type 'list' (line 56)
    list_144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 56)
    # Adding element type (line 56)
    
    # Call to defaultdict(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'None' (line 56)
    None_146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 27), 'None', False)
    # Getting the type of 'board' (line 56)
    board_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 33), 'board', False)
    # Processing the call keyword arguments (line 56)
    kwargs_148 = {}
    # Getting the type of 'defaultdict' (line 56)
    defaultdict_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 15), 'defaultdict', False)
    # Calling defaultdict(args, kwargs) (line 56)
    defaultdict_call_result_149 = invoke(stypy.reporting.localization.Localization(__file__, 56, 15), defaultdict_145, *[None_146, board_147], **kwargs_148)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 14), list_144, defaultdict_call_result_149)
    
    # Assigning a type to the variable 'history' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'history', list_144)
    
    int_150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 10), 'int')
    # Testing if the while is going to be iterated (line 57)
    # Testing the type of an if condition (line 57)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 4), int_150)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 57, 4), int_150):
        # SSA begins for while statement (line 57)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 58):
        
        # Assigning a Call to a Name (line 58):
        
        # Call to snext(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'board' (line 58)
        board_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 22), 'board', False)
        # Processing the call keyword arguments (line 58)
        kwargs_153 = {}
        # Getting the type of 'snext' (line 58)
        snext_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'snext', False)
        # Calling snext(args, kwargs) (line 58)
        snext_call_result_154 = invoke(stypy.reporting.localization.Localization(__file__, 58, 16), snext_151, *[board_152], **kwargs_153)
        
        # Assigning a type to the variable 'board' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'board', snext_call_result_154)
        
        # Getting the type of 'board' (line 59)
        board_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 11), 'board')
        # Getting the type of 'history' (line 59)
        history_156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 'history')
        # Applying the binary operator 'in' (line 59)
        result_contains_157 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 11), 'in', board_155, history_156)
        
        # Testing if the type of an if condition is none (line 59)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 59, 8), result_contains_157):
            pass
        else:
            
            # Testing the type of an if condition (line 59)
            if_condition_158 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 8), result_contains_157)
            # Assigning a type to the variable 'if_condition_158' (line 59)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'if_condition_158', if_condition_158)
            # SSA begins for if statement (line 59)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'board' (line 60)
            board_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), 'board')
            
            # Obtaining the type of the subscript
            int_160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 32), 'int')
            # Getting the type of 'history' (line 60)
            history_161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 24), 'history')
            # Obtaining the member '__getitem__' of a type (line 60)
            getitem___162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 24), history_161, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 60)
            subscript_call_result_163 = invoke(stypy.reporting.localization.Localization(__file__, 60, 24), getitem___162, int_160)
            
            # Applying the binary operator '==' (line 60)
            result_eq_164 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 15), '==', board_159, subscript_call_result_163)
            
            # Testing if the type of an if condition is none (line 60)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 60, 12), result_eq_164):
                pass
            else:
                
                # Testing the type of an if condition (line 60)
                if_condition_165 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 12), result_eq_164)
                # Assigning a type to the variable 'if_condition_165' (line 60)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'if_condition_165', if_condition_165)
                # SSA begins for if statement (line 60)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'board' (line 61)
                board_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 23), 'board')
                # Assigning a type to the variable 'stypy_return_type' (line 61)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'stypy_return_type', board_166)
                # SSA join for if statement (line 60)
                module_type_store = module_type_store.join_ssa_context()
                

            # Getting the type of 'None' (line 62)
            None_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 19), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 62)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'stypy_return_type', None_167)
            # SSA join for if statement (line 59)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to append(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Call to defaultdict(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'None' (line 63)
        None_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 35), 'None', False)
        # Getting the type of 'board' (line 63)
        board_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 41), 'board', False)
        # Processing the call keyword arguments (line 63)
        kwargs_173 = {}
        # Getting the type of 'defaultdict' (line 63)
        defaultdict_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 23), 'defaultdict', False)
        # Calling defaultdict(args, kwargs) (line 63)
        defaultdict_call_result_174 = invoke(stypy.reporting.localization.Localization(__file__, 63, 23), defaultdict_170, *[None_171, board_172], **kwargs_173)
        
        # Processing the call keyword arguments (line 63)
        kwargs_175 = {}
        # Getting the type of 'history' (line 63)
        history_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'history', False)
        # Obtaining the member 'append' of a type (line 63)
        append_169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), history_168, 'append')
        # Calling append(args, kwargs) (line 63)
        append_call_result_176 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), append_169, *[defaultdict_call_result_174], **kwargs_175)
        
        # SSA join for while statement (line 57)
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'process(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'process' in the type store
    # Getting the type of 'stypy_return_type' (line 54)
    stypy_return_type_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_177)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'process'
    return stypy_return_type_177

# Assigning a type to the variable 'process' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'process', process)

@norecursion
def generator(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'generator'
    module_type_store = module_type_store.open_function_context('generator', 66, 0, False)
    
    # Passed parameters checking function
    generator.stypy_localization = localization
    generator.stypy_type_of_self = None
    generator.stypy_type_store = module_type_store
    generator.stypy_function_name = 'generator'
    generator.stypy_param_names_list = ['rows', 'columns']
    generator.stypy_varargs_param_name = None
    generator.stypy_kwargs_param_name = None
    generator.stypy_call_defaults = defaults
    generator.stypy_call_varargs = varargs
    generator.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'generator', ['rows', 'columns'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'generator', localization, ['rows', 'columns'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'generator(...)' code ##################

    str_178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 4), 'str', ' Generates a board ')
    
    # Assigning a ListComp to a Name (line 68):
    
    # Assigning a ListComp to a Name (line 68):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'rows' (line 68)
    rows_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 43), 'rows', False)
    # Processing the call keyword arguments (line 68)
    kwargs_184 = {}
    # Getting the type of 'range' (line 68)
    range_182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 37), 'range', False)
    # Calling range(args, kwargs) (line 68)
    range_call_result_185 = invoke(stypy.reporting.localization.Localization(__file__, 68, 37), range_182, *[rows_183], **kwargs_184)
    
    comprehension_186 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 12), range_call_result_185)
    # Assigning a type to the variable 'row' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'row', comprehension_186)
    # Calculating comprehension expression
    
    # Call to range(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'columns' (line 69)
    columns_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 32), 'columns', False)
    # Processing the call keyword arguments (line 69)
    kwargs_189 = {}
    # Getting the type of 'range' (line 69)
    range_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 26), 'range', False)
    # Calling range(args, kwargs) (line 69)
    range_call_result_190 = invoke(stypy.reporting.localization.Localization(__file__, 69, 26), range_187, *[columns_188], **kwargs_189)
    
    comprehension_191 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 12), range_call_result_190)
    # Assigning a type to the variable 'column' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'column', comprehension_191)
    
    # Obtaining an instance of the builtin type 'tuple' (line 68)
    tuple_179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 68)
    # Adding element type (line 68)
    # Getting the type of 'row' (line 68)
    row_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 13), 'row')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 13), tuple_179, row_180)
    # Adding element type (line 68)
    # Getting the type of 'column' (line 68)
    column_181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 18), 'column')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 13), tuple_179, column_181)
    
    list_192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 12), list_192, tuple_179)
    # Assigning a type to the variable 'ppos' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'ppos', list_192)
    
    # Assigning a Call to a Name (line 70):
    
    # Assigning a Call to a Name (line 70):
    
    # Call to product(...): (line 70)
    # Processing the call arguments (line 70)
    
    # Obtaining an instance of the builtin type 'tuple' (line 70)
    tuple_194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 70)
    # Adding element type (line 70)
    int_195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 29), tuple_194, int_195)
    # Adding element type (line 70)
    int_196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 29), tuple_194, int_196)
    
    # Processing the call keyword arguments (line 70)
    # Getting the type of 'rows' (line 70)
    rows_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 43), 'rows', False)
    # Getting the type of 'columns' (line 70)
    columns_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 50), 'columns', False)
    # Applying the binary operator '*' (line 70)
    result_mul_199 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 43), '*', rows_197, columns_198)
    
    keyword_200 = result_mul_199
    kwargs_201 = {'repeat': keyword_200}
    # Getting the type of 'product' (line 70)
    product_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 20), 'product', False)
    # Calling product(args, kwargs) (line 70)
    product_call_result_202 = invoke(stypy.reporting.localization.Localization(__file__, 70, 20), product_193, *[tuple_194], **kwargs_201)
    
    # Assigning a type to the variable 'possibilities' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'possibilities', product_call_result_202)
    
    # Getting the type of 'possibilities' (line 71)
    possibilities_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'possibilities')
    # Testing if the for loop is going to be iterated (line 71)
    # Testing the type of a for loop iterable (line 71)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 71, 4), possibilities_203)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 71, 4), possibilities_203):
        # Getting the type of the for loop variable (line 71)
        for_loop_var_204 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 71, 4), possibilities_203)
        # Assigning a type to the variable 'case' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'case', for_loop_var_204)
        # SSA begins for a for statement (line 71)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 72):
        
        # Assigning a Call to a Name (line 72):
        
        # Call to defaultdict(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'int' (line 72)
        int_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 28), 'int', False)
        # Processing the call keyword arguments (line 72)
        kwargs_207 = {}
        # Getting the type of 'defaultdict' (line 72)
        defaultdict_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'defaultdict', False)
        # Calling defaultdict(args, kwargs) (line 72)
        defaultdict_call_result_208 = invoke(stypy.reporting.localization.Localization(__file__, 72, 16), defaultdict_205, *[int_206], **kwargs_207)
        
        # Assigning a type to the variable 'board' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'board', defaultdict_call_result_208)
        
        
        # Call to zip(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'ppos' (line 73)
        ppos_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 30), 'ppos', False)
        # Getting the type of 'case' (line 73)
        case_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 36), 'case', False)
        # Processing the call keyword arguments (line 73)
        kwargs_212 = {}
        # Getting the type of 'zip' (line 73)
        zip_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 26), 'zip', False)
        # Calling zip(args, kwargs) (line 73)
        zip_call_result_213 = invoke(stypy.reporting.localization.Localization(__file__, 73, 26), zip_209, *[ppos_210, case_211], **kwargs_212)
        
        # Testing if the for loop is going to be iterated (line 73)
        # Testing the type of a for loop iterable (line 73)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 73, 8), zip_call_result_213)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 73, 8), zip_call_result_213):
            # Getting the type of the for loop variable (line 73)
            for_loop_var_214 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 73, 8), zip_call_result_213)
            # Assigning a type to the variable 'pos' (line 73)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'pos', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 8), for_loop_var_214))
            # Assigning a type to the variable 'value' (line 73)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 8), for_loop_var_214))
            # SSA begins for a for statement (line 73)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Name to a Subscript (line 74):
            
            # Assigning a Name to a Subscript (line 74):
            # Getting the type of 'value' (line 74)
            value_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 25), 'value')
            # Getting the type of 'board' (line 74)
            board_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'board')
            # Getting the type of 'pos' (line 74)
            pos_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 18), 'pos')
            # Storing an element on a container (line 74)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 12), board_216, (pos_217, value_215))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Creating a generator
        # Getting the type of 'board' (line 75)
        board_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 14), 'board')
        GeneratorType_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 8), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 8), GeneratorType_219, board_218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'stypy_return_type', GeneratorType_219)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'generator(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'generator' in the type store
    # Getting the type of 'stypy_return_type' (line 66)
    stypy_return_type_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_220)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'generator'
    return stypy_return_type_220

# Assigning a type to the variable 'generator' (line 66)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'generator', generator)

@norecursion
def bruteforce(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'bruteforce'
    module_type_store = module_type_store.open_function_context('bruteforce', 78, 0, False)
    
    # Passed parameters checking function
    bruteforce.stypy_localization = localization
    bruteforce.stypy_type_of_self = None
    bruteforce.stypy_type_store = module_type_store
    bruteforce.stypy_function_name = 'bruteforce'
    bruteforce.stypy_param_names_list = ['rows', 'columns']
    bruteforce.stypy_varargs_param_name = None
    bruteforce.stypy_kwargs_param_name = None
    bruteforce.stypy_call_defaults = defaults
    bruteforce.stypy_call_varargs = varargs
    bruteforce.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'bruteforce', ['rows', 'columns'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'bruteforce', localization, ['rows', 'columns'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'bruteforce(...)' code ##################

    # Marking variables as global (line 79)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 79, 4), 'count')
    
    # Assigning a Num to a Name (line 80):
    
    # Assigning a Num to a Name (line 80):
    int_221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 12), 'int')
    # Assigning a type to the variable 'count' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'count', int_221)
    
    
    # Call to map(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'process' (line 81)
    process_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 21), 'process', False)
    
    # Call to generator(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'rows' (line 81)
    rows_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 40), 'rows', False)
    # Getting the type of 'columns' (line 81)
    columns_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 46), 'columns', False)
    # Processing the call keyword arguments (line 81)
    kwargs_227 = {}
    # Getting the type of 'generator' (line 81)
    generator_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 30), 'generator', False)
    # Calling generator(args, kwargs) (line 81)
    generator_call_result_228 = invoke(stypy.reporting.localization.Localization(__file__, 81, 30), generator_224, *[rows_225, columns_226], **kwargs_227)
    
    # Processing the call keyword arguments (line 81)
    kwargs_229 = {}
    # Getting the type of 'map' (line 81)
    map_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 17), 'map', False)
    # Calling map(args, kwargs) (line 81)
    map_call_result_230 = invoke(stypy.reporting.localization.Localization(__file__, 81, 17), map_222, *[process_223, generator_call_result_228], **kwargs_229)
    
    # Testing if the for loop is going to be iterated (line 81)
    # Testing the type of a for loop iterable (line 81)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 81, 4), map_call_result_230)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 81, 4), map_call_result_230):
        # Getting the type of the for loop variable (line 81)
        for_loop_var_231 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 81, 4), map_call_result_230)
        # Assigning a type to the variable 'board' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'board', for_loop_var_231)
        # SSA begins for a for statement (line 81)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 82)
        # Getting the type of 'board' (line 82)
        board_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'board')
        # Getting the type of 'None' (line 82)
        None_233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 24), 'None')
        
        (may_be_234, more_types_in_union_235) = may_not_be_none(board_232, None_233)

        if may_be_234:

            if more_types_in_union_235:
                # Runtime conditional SSA (line 82)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Getting the type of 'count' (line 83)
            count_236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'count')
            int_237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 21), 'int')
            # Applying the binary operator '+=' (line 83)
            result_iadd_238 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 12), '+=', count_236, int_237)
            # Assigning a type to the variable 'count' (line 83)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'count', result_iadd_238)
            

            if more_types_in_union_235:
                # SSA join for if statement (line 82)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'bruteforce(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'bruteforce' in the type store
    # Getting the type of 'stypy_return_type' (line 78)
    stypy_return_type_239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_239)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'bruteforce'
    return stypy_return_type_239

# Assigning a type to the variable 'bruteforce' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'bruteforce', bruteforce)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 87, 0, False)
    
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

    
    # Assigning a Tuple to a Tuple (line 88):
    
    # Assigning a Num to a Name (line 88):
    int_240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 20), 'int')
    # Assigning a type to the variable 'tuple_assignment_3' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'tuple_assignment_3', int_240)
    
    # Assigning a Num to a Name (line 88):
    int_241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 23), 'int')
    # Assigning a type to the variable 'tuple_assignment_4' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'tuple_assignment_4', int_241)
    
    # Assigning a Name to a Name (line 88):
    # Getting the type of 'tuple_assignment_3' (line 88)
    tuple_assignment_3_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'tuple_assignment_3')
    # Assigning a type to the variable 'rows' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'rows', tuple_assignment_3_242)
    
    # Assigning a Name to a Name (line 88):
    # Getting the type of 'tuple_assignment_4' (line 88)
    tuple_assignment_4_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'tuple_assignment_4')
    # Assigning a type to the variable 'columns' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 10), 'columns', tuple_assignment_4_243)
    
    # Call to bruteforce(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'rows' (line 89)
    rows_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 15), 'rows', False)
    # Getting the type of 'columns' (line 89)
    columns_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 21), 'columns', False)
    # Processing the call keyword arguments (line 89)
    kwargs_247 = {}
    # Getting the type of 'bruteforce' (line 89)
    bruteforce_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'bruteforce', False)
    # Calling bruteforce(args, kwargs) (line 89)
    bruteforce_call_result_248 = invoke(stypy.reporting.localization.Localization(__file__, 89, 4), bruteforce_244, *[rows_245, columns_246], **kwargs_247)
    
    # Getting the type of 'True' (line 99)
    True_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type', True_249)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 87)
    stypy_return_type_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_250)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_250

# Assigning a type to the variable 'run' (line 87)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'run', run)

# Call to run(...): (line 102)
# Processing the call keyword arguments (line 102)
kwargs_252 = {}
# Getting the type of 'run' (line 102)
run_251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), 'run', False)
# Calling run(args, kwargs) (line 102)
run_call_result_253 = invoke(stypy.reporting.localization.Localization(__file__, 102, 0), run_251, *[], **kwargs_252)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

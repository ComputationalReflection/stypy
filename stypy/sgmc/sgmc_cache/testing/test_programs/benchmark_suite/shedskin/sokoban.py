
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from array import array
2: from collections import deque
3: 
4: 
5: class Direction:
6:     def __init__(self, dx, dy, letter):
7:         self.dx, self.dy, self.letter = dx, dy, letter
8: 
9: 
10: class Open:
11:     def __init__(self, cur, csol, x, y):
12:         self.cur, self.csol, self.x, self.y = cur, csol, x, y
13: 
14: 
15: class Board(object):
16:     def __init__(self, board):
17:         data = filter(None, board.splitlines())
18:         self.nrows = max(len(r) for r in data)
19:         self.sdata = ""
20:         self.ddata = ""
21: 
22:         maps = {' ': ' ', '.': '.', '@': ' ', '#': '#', '$': ' '}
23:         mapd = {' ': ' ', '.': ' ', '@': '@', '#': ' ', '$': '*'}
24: 
25:         for r, row in enumerate(data):
26:             for c, ch in enumerate(row):
27:                 self.sdata += maps[ch]
28:                 self.ddata += mapd[ch]
29:                 if ch == '@':
30:                     self.px = c
31:                     self.py = r
32: 
33:     def move(self, x, y, dx, dy, data):
34:         if self.sdata[(y + dy) * self.nrows + x + dx] == '#' or \
35:                 data[(y + dy) * self.nrows + x + dx] != ' ':
36:             return None
37: 
38:         data2 = array("c", data)
39:         data2[y * self.nrows + x] = ' '
40:         data2[(y + dy) * self.nrows + x + dx] = '@'
41:         return data2.tostring()
42: 
43:     def push(self, x, y, dx, dy, data):
44:         if self.sdata[(y + 2 * dy) * self.nrows + x + 2 * dx] == '#' or \
45:                 data[(y + 2 * dy) * self.nrows + x + 2 * dx] != ' ':
46:             return None
47: 
48:         data2 = array("c", data)
49:         data2[y * self.nrows + x] = ' '
50:         data2[(y + dy) * self.nrows + x + dx] = '@'
51:         data2[(y + 2 * dy) * self.nrows + x + 2 * dx] = '*'
52:         return data2.tostring()
53: 
54:     def is_solved(self, data):
55:         for i in xrange(len(data)):
56:             if (self.sdata[i] == '.') != (data[i] == '*'):
57:                 return False
58:         return True
59: 
60:     def solve(self):
61:         open = deque()
62:         open.append(Open(self.ddata, "", self.px, self.py))
63: 
64:         visited = set()
65:         visited.add(self.ddata)
66: 
67:         dirs = (
68:             Direction(0, -1, 'u'),
69:             Direction(1, 0, 'r'),
70:             Direction(0, 1, 'd'),
71:             Direction(-1, 0, 'l'),
72:         )
73: 
74:         while open:
75:             o = open.popleft()
76:             cur, csol, x, y = o.cur, o.csol, o.x, o.y
77: 
78:             for i in xrange(4):
79:                 temp = cur
80:                 dir = dirs[i]
81:                 dx, dy = dir.dx, dir.dy
82: 
83:                 if temp[(y + dy) * self.nrows + x + dx] == '*':
84:                     temp = self.push(x, y, dx, dy, temp)
85:                     if temp and temp not in visited:
86:                         if self.is_solved(temp):
87:                             return csol + dir.letter.upper()
88:                         open.append(Open(temp, csol + dir.letter.upper(), x + dx, y + dy))
89:                         visited.add(temp)
90:                 else:
91:                     temp = self.move(x, y, dx, dy, temp)
92:                     if temp and temp not in visited:
93:                         if self.is_solved(temp):
94:                             return csol + dir.letter
95:                         open.append(Open(temp, csol + dir.letter, x + dx, y + dy))
96:                         visited.add(temp)
97: 
98:         return "No solution"
99: 
100: 
101: def run():
102:     level = '''\
103:     #######
104:     #     #
105:     #     #
106:     #. #  #
107:     #. $$ #
108:     #.$$  #
109:     #.#  @#
110:     #######'''
111: 
112:     ##    print level, "\n"
113:     for i in range(1000):
114:         b = Board(level)
115:     ##    print b.solve()
116:     return True
117: 
118: 
119: run()
120: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from array import array' statement (line 1)
try:
    from array import array

except:
    array = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'array', None, module_type_store, ['array'], [array])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from collections import deque' statement (line 2)
try:
    from collections import deque

except:
    deque = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'collections', None, module_type_store, ['deque'], [deque])

# Declaration of the 'Direction' class

class Direction:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 6, 4, False)
        # Assigning a type to the variable 'self' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Direction.__init__', ['dx', 'dy', 'letter'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['dx', 'dy', 'letter'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 7):
        
        # Assigning a Name to a Name (line 7):
        # Getting the type of 'dx' (line 7)
        dx_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 40), 'dx')
        # Assigning a type to the variable 'tuple_assignment_1' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'tuple_assignment_1', dx_14)
        
        # Assigning a Name to a Name (line 7):
        # Getting the type of 'dy' (line 7)
        dy_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 44), 'dy')
        # Assigning a type to the variable 'tuple_assignment_2' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'tuple_assignment_2', dy_15)
        
        # Assigning a Name to a Name (line 7):
        # Getting the type of 'letter' (line 7)
        letter_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 48), 'letter')
        # Assigning a type to the variable 'tuple_assignment_3' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'tuple_assignment_3', letter_16)
        
        # Assigning a Name to a Attribute (line 7):
        # Getting the type of 'tuple_assignment_1' (line 7)
        tuple_assignment_1_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'tuple_assignment_1')
        # Getting the type of 'self' (line 7)
        self_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'self')
        # Setting the type of the member 'dx' of a type (line 7)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 8), self_18, 'dx', tuple_assignment_1_17)
        
        # Assigning a Name to a Attribute (line 7):
        # Getting the type of 'tuple_assignment_2' (line 7)
        tuple_assignment_2_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'tuple_assignment_2')
        # Getting the type of 'self' (line 7)
        self_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 17), 'self')
        # Setting the type of the member 'dy' of a type (line 7)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 17), self_20, 'dy', tuple_assignment_2_19)
        
        # Assigning a Name to a Attribute (line 7):
        # Getting the type of 'tuple_assignment_3' (line 7)
        tuple_assignment_3_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'tuple_assignment_3')
        # Getting the type of 'self' (line 7)
        self_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 26), 'self')
        # Setting the type of the member 'letter' of a type (line 7)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 26), self_22, 'letter', tuple_assignment_3_21)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Direction' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Direction', Direction)
# Declaration of the 'Open' class

class Open:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 11, 4, False)
        # Assigning a type to the variable 'self' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Open.__init__', ['cur', 'csol', 'x', 'y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['cur', 'csol', 'x', 'y'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 12):
        
        # Assigning a Name to a Name (line 12):
        # Getting the type of 'cur' (line 12)
        cur_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 46), 'cur')
        # Assigning a type to the variable 'tuple_assignment_4' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'tuple_assignment_4', cur_23)
        
        # Assigning a Name to a Name (line 12):
        # Getting the type of 'csol' (line 12)
        csol_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 51), 'csol')
        # Assigning a type to the variable 'tuple_assignment_5' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'tuple_assignment_5', csol_24)
        
        # Assigning a Name to a Name (line 12):
        # Getting the type of 'x' (line 12)
        x_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 57), 'x')
        # Assigning a type to the variable 'tuple_assignment_6' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'tuple_assignment_6', x_25)
        
        # Assigning a Name to a Name (line 12):
        # Getting the type of 'y' (line 12)
        y_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 60), 'y')
        # Assigning a type to the variable 'tuple_assignment_7' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'tuple_assignment_7', y_26)
        
        # Assigning a Name to a Attribute (line 12):
        # Getting the type of 'tuple_assignment_4' (line 12)
        tuple_assignment_4_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'tuple_assignment_4')
        # Getting the type of 'self' (line 12)
        self_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'self')
        # Setting the type of the member 'cur' of a type (line 12)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 8), self_28, 'cur', tuple_assignment_4_27)
        
        # Assigning a Name to a Attribute (line 12):
        # Getting the type of 'tuple_assignment_5' (line 12)
        tuple_assignment_5_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'tuple_assignment_5')
        # Getting the type of 'self' (line 12)
        self_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 18), 'self')
        # Setting the type of the member 'csol' of a type (line 12)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 18), self_30, 'csol', tuple_assignment_5_29)
        
        # Assigning a Name to a Attribute (line 12):
        # Getting the type of 'tuple_assignment_6' (line 12)
        tuple_assignment_6_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'tuple_assignment_6')
        # Getting the type of 'self' (line 12)
        self_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 29), 'self')
        # Setting the type of the member 'x' of a type (line 12)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 29), self_32, 'x', tuple_assignment_6_31)
        
        # Assigning a Name to a Attribute (line 12):
        # Getting the type of 'tuple_assignment_7' (line 12)
        tuple_assignment_7_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'tuple_assignment_7')
        # Getting the type of 'self' (line 12)
        self_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 37), 'self')
        # Setting the type of the member 'y' of a type (line 12)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 37), self_34, 'y', tuple_assignment_7_33)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Open' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'Open', Open)
# Declaration of the 'Board' class

class Board(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 16, 4, False)
        # Assigning a type to the variable 'self' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Board.__init__', ['board'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 17):
        
        # Assigning a Call to a Name (line 17):
        
        # Call to filter(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of 'None' (line 17)
        None_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 22), 'None', False)
        
        # Call to splitlines(...): (line 17)
        # Processing the call keyword arguments (line 17)
        kwargs_39 = {}
        # Getting the type of 'board' (line 17)
        board_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 28), 'board', False)
        # Obtaining the member 'splitlines' of a type (line 17)
        splitlines_38 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 28), board_37, 'splitlines')
        # Calling splitlines(args, kwargs) (line 17)
        splitlines_call_result_40 = invoke(stypy.reporting.localization.Localization(__file__, 17, 28), splitlines_38, *[], **kwargs_39)
        
        # Processing the call keyword arguments (line 17)
        kwargs_41 = {}
        # Getting the type of 'filter' (line 17)
        filter_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 15), 'filter', False)
        # Calling filter(args, kwargs) (line 17)
        filter_call_result_42 = invoke(stypy.reporting.localization.Localization(__file__, 17, 15), filter_35, *[None_36, splitlines_call_result_40], **kwargs_41)
        
        # Assigning a type to the variable 'data' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'data', filter_call_result_42)
        
        # Assigning a Call to a Attribute (line 18):
        
        # Assigning a Call to a Attribute (line 18):
        
        # Call to max(...): (line 18)
        # Processing the call arguments (line 18)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 18, 25, True)
        # Calculating comprehension expression
        # Getting the type of 'data' (line 18)
        data_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 41), 'data', False)
        comprehension_49 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 25), data_48)
        # Assigning a type to the variable 'r' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 25), 'r', comprehension_49)
        
        # Call to len(...): (line 18)
        # Processing the call arguments (line 18)
        # Getting the type of 'r' (line 18)
        r_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 29), 'r', False)
        # Processing the call keyword arguments (line 18)
        kwargs_46 = {}
        # Getting the type of 'len' (line 18)
        len_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 25), 'len', False)
        # Calling len(args, kwargs) (line 18)
        len_call_result_47 = invoke(stypy.reporting.localization.Localization(__file__, 18, 25), len_44, *[r_45], **kwargs_46)
        
        list_50 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 25), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 25), list_50, len_call_result_47)
        # Processing the call keyword arguments (line 18)
        kwargs_51 = {}
        # Getting the type of 'max' (line 18)
        max_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 21), 'max', False)
        # Calling max(args, kwargs) (line 18)
        max_call_result_52 = invoke(stypy.reporting.localization.Localization(__file__, 18, 21), max_43, *[list_50], **kwargs_51)
        
        # Getting the type of 'self' (line 18)
        self_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'self')
        # Setting the type of the member 'nrows' of a type (line 18)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), self_53, 'nrows', max_call_result_52)
        
        # Assigning a Str to a Attribute (line 19):
        
        # Assigning a Str to a Attribute (line 19):
        str_54 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 21), 'str', '')
        # Getting the type of 'self' (line 19)
        self_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'self')
        # Setting the type of the member 'sdata' of a type (line 19)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 8), self_55, 'sdata', str_54)
        
        # Assigning a Str to a Attribute (line 20):
        
        # Assigning a Str to a Attribute (line 20):
        str_56 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 21), 'str', '')
        # Getting the type of 'self' (line 20)
        self_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'self')
        # Setting the type of the member 'ddata' of a type (line 20)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), self_57, 'ddata', str_56)
        
        # Assigning a Dict to a Name (line 22):
        
        # Assigning a Dict to a Name (line 22):
        
        # Obtaining an instance of the builtin type 'dict' (line 22)
        dict_58 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 15), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 22)
        # Adding element type (key, value) (line 22)
        str_59 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'str', ' ')
        str_60 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 21), 'str', ' ')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 15), dict_58, (str_59, str_60))
        # Adding element type (key, value) (line 22)
        str_61 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 26), 'str', '.')
        str_62 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 31), 'str', '.')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 15), dict_58, (str_61, str_62))
        # Adding element type (key, value) (line 22)
        str_63 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 36), 'str', '@')
        str_64 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 41), 'str', ' ')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 15), dict_58, (str_63, str_64))
        # Adding element type (key, value) (line 22)
        str_65 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 46), 'str', '#')
        str_66 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 51), 'str', '#')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 15), dict_58, (str_65, str_66))
        # Adding element type (key, value) (line 22)
        str_67 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 56), 'str', '$')
        str_68 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 61), 'str', ' ')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 15), dict_58, (str_67, str_68))
        
        # Assigning a type to the variable 'maps' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'maps', dict_58)
        
        # Assigning a Dict to a Name (line 23):
        
        # Assigning a Dict to a Name (line 23):
        
        # Obtaining an instance of the builtin type 'dict' (line 23)
        dict_69 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 15), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 23)
        # Adding element type (key, value) (line 23)
        str_70 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 16), 'str', ' ')
        str_71 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 21), 'str', ' ')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 15), dict_69, (str_70, str_71))
        # Adding element type (key, value) (line 23)
        str_72 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 26), 'str', '.')
        str_73 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 31), 'str', ' ')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 15), dict_69, (str_72, str_73))
        # Adding element type (key, value) (line 23)
        str_74 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 36), 'str', '@')
        str_75 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 41), 'str', '@')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 15), dict_69, (str_74, str_75))
        # Adding element type (key, value) (line 23)
        str_76 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 46), 'str', '#')
        str_77 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 51), 'str', ' ')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 15), dict_69, (str_76, str_77))
        # Adding element type (key, value) (line 23)
        str_78 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 56), 'str', '$')
        str_79 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 61), 'str', '*')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 15), dict_69, (str_78, str_79))
        
        # Assigning a type to the variable 'mapd' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'mapd', dict_69)
        
        
        # Call to enumerate(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'data' (line 25)
        data_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 32), 'data', False)
        # Processing the call keyword arguments (line 25)
        kwargs_82 = {}
        # Getting the type of 'enumerate' (line 25)
        enumerate_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 22), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 25)
        enumerate_call_result_83 = invoke(stypy.reporting.localization.Localization(__file__, 25, 22), enumerate_80, *[data_81], **kwargs_82)
        
        # Testing if the for loop is going to be iterated (line 25)
        # Testing the type of a for loop iterable (line 25)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 25, 8), enumerate_call_result_83)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 25, 8), enumerate_call_result_83):
            # Getting the type of the for loop variable (line 25)
            for_loop_var_84 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 25, 8), enumerate_call_result_83)
            # Assigning a type to the variable 'r' (line 25)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'r', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 8), for_loop_var_84))
            # Assigning a type to the variable 'row' (line 25)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'row', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 8), for_loop_var_84))
            # SSA begins for a for statement (line 25)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to enumerate(...): (line 26)
            # Processing the call arguments (line 26)
            # Getting the type of 'row' (line 26)
            row_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 35), 'row', False)
            # Processing the call keyword arguments (line 26)
            kwargs_87 = {}
            # Getting the type of 'enumerate' (line 26)
            enumerate_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 25), 'enumerate', False)
            # Calling enumerate(args, kwargs) (line 26)
            enumerate_call_result_88 = invoke(stypy.reporting.localization.Localization(__file__, 26, 25), enumerate_85, *[row_86], **kwargs_87)
            
            # Testing if the for loop is going to be iterated (line 26)
            # Testing the type of a for loop iterable (line 26)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 26, 12), enumerate_call_result_88)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 26, 12), enumerate_call_result_88):
                # Getting the type of the for loop variable (line 26)
                for_loop_var_89 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 26, 12), enumerate_call_result_88)
                # Assigning a type to the variable 'c' (line 26)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'c', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 12), for_loop_var_89))
                # Assigning a type to the variable 'ch' (line 26)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'ch', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 12), for_loop_var_89))
                # SSA begins for a for statement (line 26)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'self' (line 27)
                self_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 16), 'self')
                # Obtaining the member 'sdata' of a type (line 27)
                sdata_91 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 16), self_90, 'sdata')
                
                # Obtaining the type of the subscript
                # Getting the type of 'ch' (line 27)
                ch_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 35), 'ch')
                # Getting the type of 'maps' (line 27)
                maps_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 30), 'maps')
                # Obtaining the member '__getitem__' of a type (line 27)
                getitem___94 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 30), maps_93, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 27)
                subscript_call_result_95 = invoke(stypy.reporting.localization.Localization(__file__, 27, 30), getitem___94, ch_92)
                
                # Applying the binary operator '+=' (line 27)
                result_iadd_96 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 16), '+=', sdata_91, subscript_call_result_95)
                # Getting the type of 'self' (line 27)
                self_97 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 16), 'self')
                # Setting the type of the member 'sdata' of a type (line 27)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 16), self_97, 'sdata', result_iadd_96)
                
                
                # Getting the type of 'self' (line 28)
                self_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), 'self')
                # Obtaining the member 'ddata' of a type (line 28)
                ddata_99 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 16), self_98, 'ddata')
                
                # Obtaining the type of the subscript
                # Getting the type of 'ch' (line 28)
                ch_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 35), 'ch')
                # Getting the type of 'mapd' (line 28)
                mapd_101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 30), 'mapd')
                # Obtaining the member '__getitem__' of a type (line 28)
                getitem___102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 30), mapd_101, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 28)
                subscript_call_result_103 = invoke(stypy.reporting.localization.Localization(__file__, 28, 30), getitem___102, ch_100)
                
                # Applying the binary operator '+=' (line 28)
                result_iadd_104 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 16), '+=', ddata_99, subscript_call_result_103)
                # Getting the type of 'self' (line 28)
                self_105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), 'self')
                # Setting the type of the member 'ddata' of a type (line 28)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 16), self_105, 'ddata', result_iadd_104)
                
                
                # Getting the type of 'ch' (line 29)
                ch_106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 19), 'ch')
                str_107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 25), 'str', '@')
                # Applying the binary operator '==' (line 29)
                result_eq_108 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 19), '==', ch_106, str_107)
                
                # Testing if the type of an if condition is none (line 29)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 29, 16), result_eq_108):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 29)
                    if_condition_109 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 16), result_eq_108)
                    # Assigning a type to the variable 'if_condition_109' (line 29)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 16), 'if_condition_109', if_condition_109)
                    # SSA begins for if statement (line 29)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Attribute (line 30):
                    
                    # Assigning a Name to a Attribute (line 30):
                    # Getting the type of 'c' (line 30)
                    c_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 30), 'c')
                    # Getting the type of 'self' (line 30)
                    self_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 20), 'self')
                    # Setting the type of the member 'px' of a type (line 30)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 20), self_111, 'px', c_110)
                    
                    # Assigning a Name to a Attribute (line 31):
                    
                    # Assigning a Name to a Attribute (line 31):
                    # Getting the type of 'r' (line 31)
                    r_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 30), 'r')
                    # Getting the type of 'self' (line 31)
                    self_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'self')
                    # Setting the type of the member 'py' of a type (line 31)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 20), self_113, 'py', r_112)
                    # SSA join for if statement (line 29)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def move(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'move'
        module_type_store = module_type_store.open_function_context('move', 33, 4, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Board.move.__dict__.__setitem__('stypy_localization', localization)
        Board.move.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Board.move.__dict__.__setitem__('stypy_type_store', module_type_store)
        Board.move.__dict__.__setitem__('stypy_function_name', 'Board.move')
        Board.move.__dict__.__setitem__('stypy_param_names_list', ['x', 'y', 'dx', 'dy', 'data'])
        Board.move.__dict__.__setitem__('stypy_varargs_param_name', None)
        Board.move.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Board.move.__dict__.__setitem__('stypy_call_defaults', defaults)
        Board.move.__dict__.__setitem__('stypy_call_varargs', varargs)
        Board.move.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Board.move.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Board.move', ['x', 'y', 'dx', 'dy', 'data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'move', localization, ['x', 'y', 'dx', 'dy', 'data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'move(...)' code ##################

        
        # Evaluating a boolean operation
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'y' (line 34)
        y_114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 23), 'y')
        # Getting the type of 'dy' (line 34)
        dy_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 27), 'dy')
        # Applying the binary operator '+' (line 34)
        result_add_116 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 23), '+', y_114, dy_115)
        
        # Getting the type of 'self' (line 34)
        self_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 33), 'self')
        # Obtaining the member 'nrows' of a type (line 34)
        nrows_118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 33), self_117, 'nrows')
        # Applying the binary operator '*' (line 34)
        result_mul_119 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 22), '*', result_add_116, nrows_118)
        
        # Getting the type of 'x' (line 34)
        x_120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 46), 'x')
        # Applying the binary operator '+' (line 34)
        result_add_121 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 22), '+', result_mul_119, x_120)
        
        # Getting the type of 'dx' (line 34)
        dx_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 50), 'dx')
        # Applying the binary operator '+' (line 34)
        result_add_123 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 48), '+', result_add_121, dx_122)
        
        # Getting the type of 'self' (line 34)
        self_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 11), 'self')
        # Obtaining the member 'sdata' of a type (line 34)
        sdata_125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 11), self_124, 'sdata')
        # Obtaining the member '__getitem__' of a type (line 34)
        getitem___126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 11), sdata_125, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 34)
        subscript_call_result_127 = invoke(stypy.reporting.localization.Localization(__file__, 34, 11), getitem___126, result_add_123)
        
        str_128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 57), 'str', '#')
        # Applying the binary operator '==' (line 34)
        result_eq_129 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 11), '==', subscript_call_result_127, str_128)
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'y' (line 35)
        y_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 22), 'y')
        # Getting the type of 'dy' (line 35)
        dy_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 26), 'dy')
        # Applying the binary operator '+' (line 35)
        result_add_132 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 22), '+', y_130, dy_131)
        
        # Getting the type of 'self' (line 35)
        self_133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 32), 'self')
        # Obtaining the member 'nrows' of a type (line 35)
        nrows_134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 32), self_133, 'nrows')
        # Applying the binary operator '*' (line 35)
        result_mul_135 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 21), '*', result_add_132, nrows_134)
        
        # Getting the type of 'x' (line 35)
        x_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 45), 'x')
        # Applying the binary operator '+' (line 35)
        result_add_137 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 21), '+', result_mul_135, x_136)
        
        # Getting the type of 'dx' (line 35)
        dx_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 49), 'dx')
        # Applying the binary operator '+' (line 35)
        result_add_139 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 47), '+', result_add_137, dx_138)
        
        # Getting the type of 'data' (line 35)
        data_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 16), 'data')
        # Obtaining the member '__getitem__' of a type (line 35)
        getitem___141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 16), data_140, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 35)
        subscript_call_result_142 = invoke(stypy.reporting.localization.Localization(__file__, 35, 16), getitem___141, result_add_139)
        
        str_143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 56), 'str', ' ')
        # Applying the binary operator '!=' (line 35)
        result_ne_144 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 16), '!=', subscript_call_result_142, str_143)
        
        # Applying the binary operator 'or' (line 34)
        result_or_keyword_145 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 11), 'or', result_eq_129, result_ne_144)
        
        # Testing if the type of an if condition is none (line 34)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 34, 8), result_or_keyword_145):
            pass
        else:
            
            # Testing the type of an if condition (line 34)
            if_condition_146 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 34, 8), result_or_keyword_145)
            # Assigning a type to the variable 'if_condition_146' (line 34)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'if_condition_146', if_condition_146)
            # SSA begins for if statement (line 34)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'None' (line 36)
            None_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 19), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 36)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'stypy_return_type', None_147)
            # SSA join for if statement (line 34)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 38):
        
        # Assigning a Call to a Name (line 38):
        
        # Call to array(...): (line 38)
        # Processing the call arguments (line 38)
        str_149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 22), 'str', 'c')
        # Getting the type of 'data' (line 38)
        data_150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 27), 'data', False)
        # Processing the call keyword arguments (line 38)
        kwargs_151 = {}
        # Getting the type of 'array' (line 38)
        array_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'array', False)
        # Calling array(args, kwargs) (line 38)
        array_call_result_152 = invoke(stypy.reporting.localization.Localization(__file__, 38, 16), array_148, *[str_149, data_150], **kwargs_151)
        
        # Assigning a type to the variable 'data2' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'data2', array_call_result_152)
        
        # Assigning a Str to a Subscript (line 39):
        
        # Assigning a Str to a Subscript (line 39):
        str_153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 36), 'str', ' ')
        # Getting the type of 'data2' (line 39)
        data2_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'data2')
        # Getting the type of 'y' (line 39)
        y_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 14), 'y')
        # Getting the type of 'self' (line 39)
        self_156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 18), 'self')
        # Obtaining the member 'nrows' of a type (line 39)
        nrows_157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 18), self_156, 'nrows')
        # Applying the binary operator '*' (line 39)
        result_mul_158 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 14), '*', y_155, nrows_157)
        
        # Getting the type of 'x' (line 39)
        x_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 31), 'x')
        # Applying the binary operator '+' (line 39)
        result_add_160 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 14), '+', result_mul_158, x_159)
        
        # Storing an element on a container (line 39)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 8), data2_154, (result_add_160, str_153))
        
        # Assigning a Str to a Subscript (line 40):
        
        # Assigning a Str to a Subscript (line 40):
        str_161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 48), 'str', '@')
        # Getting the type of 'data2' (line 40)
        data2_162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'data2')
        # Getting the type of 'y' (line 40)
        y_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 15), 'y')
        # Getting the type of 'dy' (line 40)
        dy_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'dy')
        # Applying the binary operator '+' (line 40)
        result_add_165 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 15), '+', y_163, dy_164)
        
        # Getting the type of 'self' (line 40)
        self_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 25), 'self')
        # Obtaining the member 'nrows' of a type (line 40)
        nrows_167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 25), self_166, 'nrows')
        # Applying the binary operator '*' (line 40)
        result_mul_168 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 14), '*', result_add_165, nrows_167)
        
        # Getting the type of 'x' (line 40)
        x_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 38), 'x')
        # Applying the binary operator '+' (line 40)
        result_add_170 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 14), '+', result_mul_168, x_169)
        
        # Getting the type of 'dx' (line 40)
        dx_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 42), 'dx')
        # Applying the binary operator '+' (line 40)
        result_add_172 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 40), '+', result_add_170, dx_171)
        
        # Storing an element on a container (line 40)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 8), data2_162, (result_add_172, str_161))
        
        # Call to tostring(...): (line 41)
        # Processing the call keyword arguments (line 41)
        kwargs_175 = {}
        # Getting the type of 'data2' (line 41)
        data2_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 15), 'data2', False)
        # Obtaining the member 'tostring' of a type (line 41)
        tostring_174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 15), data2_173, 'tostring')
        # Calling tostring(args, kwargs) (line 41)
        tostring_call_result_176 = invoke(stypy.reporting.localization.Localization(__file__, 41, 15), tostring_174, *[], **kwargs_175)
        
        # Assigning a type to the variable 'stypy_return_type' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'stypy_return_type', tostring_call_result_176)
        
        # ################# End of 'move(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'move' in the type store
        # Getting the type of 'stypy_return_type' (line 33)
        stypy_return_type_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_177)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'move'
        return stypy_return_type_177


    @norecursion
    def push(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'push'
        module_type_store = module_type_store.open_function_context('push', 43, 4, False)
        # Assigning a type to the variable 'self' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Board.push.__dict__.__setitem__('stypy_localization', localization)
        Board.push.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Board.push.__dict__.__setitem__('stypy_type_store', module_type_store)
        Board.push.__dict__.__setitem__('stypy_function_name', 'Board.push')
        Board.push.__dict__.__setitem__('stypy_param_names_list', ['x', 'y', 'dx', 'dy', 'data'])
        Board.push.__dict__.__setitem__('stypy_varargs_param_name', None)
        Board.push.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Board.push.__dict__.__setitem__('stypy_call_defaults', defaults)
        Board.push.__dict__.__setitem__('stypy_call_varargs', varargs)
        Board.push.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Board.push.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Board.push', ['x', 'y', 'dx', 'dy', 'data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'push', localization, ['x', 'y', 'dx', 'dy', 'data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'push(...)' code ##################

        
        # Evaluating a boolean operation
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'y' (line 44)
        y_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 23), 'y')
        int_179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 27), 'int')
        # Getting the type of 'dy' (line 44)
        dy_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 31), 'dy')
        # Applying the binary operator '*' (line 44)
        result_mul_181 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 27), '*', int_179, dy_180)
        
        # Applying the binary operator '+' (line 44)
        result_add_182 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 23), '+', y_178, result_mul_181)
        
        # Getting the type of 'self' (line 44)
        self_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 37), 'self')
        # Obtaining the member 'nrows' of a type (line 44)
        nrows_184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 37), self_183, 'nrows')
        # Applying the binary operator '*' (line 44)
        result_mul_185 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 22), '*', result_add_182, nrows_184)
        
        # Getting the type of 'x' (line 44)
        x_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 50), 'x')
        # Applying the binary operator '+' (line 44)
        result_add_187 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 22), '+', result_mul_185, x_186)
        
        int_188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 54), 'int')
        # Getting the type of 'dx' (line 44)
        dx_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 58), 'dx')
        # Applying the binary operator '*' (line 44)
        result_mul_190 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 54), '*', int_188, dx_189)
        
        # Applying the binary operator '+' (line 44)
        result_add_191 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 52), '+', result_add_187, result_mul_190)
        
        # Getting the type of 'self' (line 44)
        self_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'self')
        # Obtaining the member 'sdata' of a type (line 44)
        sdata_193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 11), self_192, 'sdata')
        # Obtaining the member '__getitem__' of a type (line 44)
        getitem___194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 11), sdata_193, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 44)
        subscript_call_result_195 = invoke(stypy.reporting.localization.Localization(__file__, 44, 11), getitem___194, result_add_191)
        
        str_196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 65), 'str', '#')
        # Applying the binary operator '==' (line 44)
        result_eq_197 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 11), '==', subscript_call_result_195, str_196)
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'y' (line 45)
        y_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 22), 'y')
        int_199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 26), 'int')
        # Getting the type of 'dy' (line 45)
        dy_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 30), 'dy')
        # Applying the binary operator '*' (line 45)
        result_mul_201 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 26), '*', int_199, dy_200)
        
        # Applying the binary operator '+' (line 45)
        result_add_202 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 22), '+', y_198, result_mul_201)
        
        # Getting the type of 'self' (line 45)
        self_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 36), 'self')
        # Obtaining the member 'nrows' of a type (line 45)
        nrows_204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 36), self_203, 'nrows')
        # Applying the binary operator '*' (line 45)
        result_mul_205 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 21), '*', result_add_202, nrows_204)
        
        # Getting the type of 'x' (line 45)
        x_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 49), 'x')
        # Applying the binary operator '+' (line 45)
        result_add_207 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 21), '+', result_mul_205, x_206)
        
        int_208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 53), 'int')
        # Getting the type of 'dx' (line 45)
        dx_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 57), 'dx')
        # Applying the binary operator '*' (line 45)
        result_mul_210 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 53), '*', int_208, dx_209)
        
        # Applying the binary operator '+' (line 45)
        result_add_211 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 51), '+', result_add_207, result_mul_210)
        
        # Getting the type of 'data' (line 45)
        data_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 16), 'data')
        # Obtaining the member '__getitem__' of a type (line 45)
        getitem___213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 16), data_212, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 45)
        subscript_call_result_214 = invoke(stypy.reporting.localization.Localization(__file__, 45, 16), getitem___213, result_add_211)
        
        str_215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 64), 'str', ' ')
        # Applying the binary operator '!=' (line 45)
        result_ne_216 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 16), '!=', subscript_call_result_214, str_215)
        
        # Applying the binary operator 'or' (line 44)
        result_or_keyword_217 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 11), 'or', result_eq_197, result_ne_216)
        
        # Testing if the type of an if condition is none (line 44)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 44, 8), result_or_keyword_217):
            pass
        else:
            
            # Testing the type of an if condition (line 44)
            if_condition_218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 8), result_or_keyword_217)
            # Assigning a type to the variable 'if_condition_218' (line 44)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'if_condition_218', if_condition_218)
            # SSA begins for if statement (line 44)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'None' (line 46)
            None_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 19), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 46)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'stypy_return_type', None_219)
            # SSA join for if statement (line 44)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 48):
        
        # Assigning a Call to a Name (line 48):
        
        # Call to array(...): (line 48)
        # Processing the call arguments (line 48)
        str_221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 22), 'str', 'c')
        # Getting the type of 'data' (line 48)
        data_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 27), 'data', False)
        # Processing the call keyword arguments (line 48)
        kwargs_223 = {}
        # Getting the type of 'array' (line 48)
        array_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'array', False)
        # Calling array(args, kwargs) (line 48)
        array_call_result_224 = invoke(stypy.reporting.localization.Localization(__file__, 48, 16), array_220, *[str_221, data_222], **kwargs_223)
        
        # Assigning a type to the variable 'data2' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'data2', array_call_result_224)
        
        # Assigning a Str to a Subscript (line 49):
        
        # Assigning a Str to a Subscript (line 49):
        str_225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 36), 'str', ' ')
        # Getting the type of 'data2' (line 49)
        data2_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'data2')
        # Getting the type of 'y' (line 49)
        y_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 14), 'y')
        # Getting the type of 'self' (line 49)
        self_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 18), 'self')
        # Obtaining the member 'nrows' of a type (line 49)
        nrows_229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 18), self_228, 'nrows')
        # Applying the binary operator '*' (line 49)
        result_mul_230 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 14), '*', y_227, nrows_229)
        
        # Getting the type of 'x' (line 49)
        x_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 31), 'x')
        # Applying the binary operator '+' (line 49)
        result_add_232 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 14), '+', result_mul_230, x_231)
        
        # Storing an element on a container (line 49)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 8), data2_226, (result_add_232, str_225))
        
        # Assigning a Str to a Subscript (line 50):
        
        # Assigning a Str to a Subscript (line 50):
        str_233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 48), 'str', '@')
        # Getting the type of 'data2' (line 50)
        data2_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'data2')
        # Getting the type of 'y' (line 50)
        y_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 15), 'y')
        # Getting the type of 'dy' (line 50)
        dy_236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 19), 'dy')
        # Applying the binary operator '+' (line 50)
        result_add_237 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 15), '+', y_235, dy_236)
        
        # Getting the type of 'self' (line 50)
        self_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 25), 'self')
        # Obtaining the member 'nrows' of a type (line 50)
        nrows_239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 25), self_238, 'nrows')
        # Applying the binary operator '*' (line 50)
        result_mul_240 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 14), '*', result_add_237, nrows_239)
        
        # Getting the type of 'x' (line 50)
        x_241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 38), 'x')
        # Applying the binary operator '+' (line 50)
        result_add_242 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 14), '+', result_mul_240, x_241)
        
        # Getting the type of 'dx' (line 50)
        dx_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 42), 'dx')
        # Applying the binary operator '+' (line 50)
        result_add_244 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 40), '+', result_add_242, dx_243)
        
        # Storing an element on a container (line 50)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 8), data2_234, (result_add_244, str_233))
        
        # Assigning a Str to a Subscript (line 51):
        
        # Assigning a Str to a Subscript (line 51):
        str_245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 56), 'str', '*')
        # Getting the type of 'data2' (line 51)
        data2_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'data2')
        # Getting the type of 'y' (line 51)
        y_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'y')
        int_248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 19), 'int')
        # Getting the type of 'dy' (line 51)
        dy_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 23), 'dy')
        # Applying the binary operator '*' (line 51)
        result_mul_250 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 19), '*', int_248, dy_249)
        
        # Applying the binary operator '+' (line 51)
        result_add_251 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 15), '+', y_247, result_mul_250)
        
        # Getting the type of 'self' (line 51)
        self_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 29), 'self')
        # Obtaining the member 'nrows' of a type (line 51)
        nrows_253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 29), self_252, 'nrows')
        # Applying the binary operator '*' (line 51)
        result_mul_254 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 14), '*', result_add_251, nrows_253)
        
        # Getting the type of 'x' (line 51)
        x_255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 42), 'x')
        # Applying the binary operator '+' (line 51)
        result_add_256 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 14), '+', result_mul_254, x_255)
        
        int_257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 46), 'int')
        # Getting the type of 'dx' (line 51)
        dx_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 50), 'dx')
        # Applying the binary operator '*' (line 51)
        result_mul_259 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 46), '*', int_257, dx_258)
        
        # Applying the binary operator '+' (line 51)
        result_add_260 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 44), '+', result_add_256, result_mul_259)
        
        # Storing an element on a container (line 51)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 8), data2_246, (result_add_260, str_245))
        
        # Call to tostring(...): (line 52)
        # Processing the call keyword arguments (line 52)
        kwargs_263 = {}
        # Getting the type of 'data2' (line 52)
        data2_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 15), 'data2', False)
        # Obtaining the member 'tostring' of a type (line 52)
        tostring_262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 15), data2_261, 'tostring')
        # Calling tostring(args, kwargs) (line 52)
        tostring_call_result_264 = invoke(stypy.reporting.localization.Localization(__file__, 52, 15), tostring_262, *[], **kwargs_263)
        
        # Assigning a type to the variable 'stypy_return_type' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'stypy_return_type', tostring_call_result_264)
        
        # ################# End of 'push(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'push' in the type store
        # Getting the type of 'stypy_return_type' (line 43)
        stypy_return_type_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_265)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'push'
        return stypy_return_type_265


    @norecursion
    def is_solved(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'is_solved'
        module_type_store = module_type_store.open_function_context('is_solved', 54, 4, False)
        # Assigning a type to the variable 'self' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Board.is_solved.__dict__.__setitem__('stypy_localization', localization)
        Board.is_solved.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Board.is_solved.__dict__.__setitem__('stypy_type_store', module_type_store)
        Board.is_solved.__dict__.__setitem__('stypy_function_name', 'Board.is_solved')
        Board.is_solved.__dict__.__setitem__('stypy_param_names_list', ['data'])
        Board.is_solved.__dict__.__setitem__('stypy_varargs_param_name', None)
        Board.is_solved.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Board.is_solved.__dict__.__setitem__('stypy_call_defaults', defaults)
        Board.is_solved.__dict__.__setitem__('stypy_call_varargs', varargs)
        Board.is_solved.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Board.is_solved.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Board.is_solved', ['data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'is_solved', localization, ['data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'is_solved(...)' code ##################

        
        
        # Call to xrange(...): (line 55)
        # Processing the call arguments (line 55)
        
        # Call to len(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'data' (line 55)
        data_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 28), 'data', False)
        # Processing the call keyword arguments (line 55)
        kwargs_269 = {}
        # Getting the type of 'len' (line 55)
        len_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 24), 'len', False)
        # Calling len(args, kwargs) (line 55)
        len_call_result_270 = invoke(stypy.reporting.localization.Localization(__file__, 55, 24), len_267, *[data_268], **kwargs_269)
        
        # Processing the call keyword arguments (line 55)
        kwargs_271 = {}
        # Getting the type of 'xrange' (line 55)
        xrange_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 55)
        xrange_call_result_272 = invoke(stypy.reporting.localization.Localization(__file__, 55, 17), xrange_266, *[len_call_result_270], **kwargs_271)
        
        # Testing if the for loop is going to be iterated (line 55)
        # Testing the type of a for loop iterable (line 55)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 55, 8), xrange_call_result_272)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 55, 8), xrange_call_result_272):
            # Getting the type of the for loop variable (line 55)
            for_loop_var_273 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 55, 8), xrange_call_result_272)
            # Assigning a type to the variable 'i' (line 55)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'i', for_loop_var_273)
            # SSA begins for a for statement (line 55)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 56)
            i_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 27), 'i')
            # Getting the type of 'self' (line 56)
            self_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 16), 'self')
            # Obtaining the member 'sdata' of a type (line 56)
            sdata_276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 16), self_275, 'sdata')
            # Obtaining the member '__getitem__' of a type (line 56)
            getitem___277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 16), sdata_276, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 56)
            subscript_call_result_278 = invoke(stypy.reporting.localization.Localization(__file__, 56, 16), getitem___277, i_274)
            
            str_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 33), 'str', '.')
            # Applying the binary operator '==' (line 56)
            result_eq_280 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 16), '==', subscript_call_result_278, str_279)
            
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 56)
            i_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 47), 'i')
            # Getting the type of 'data' (line 56)
            data_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 42), 'data')
            # Obtaining the member '__getitem__' of a type (line 56)
            getitem___283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 42), data_282, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 56)
            subscript_call_result_284 = invoke(stypy.reporting.localization.Localization(__file__, 56, 42), getitem___283, i_281)
            
            str_285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 53), 'str', '*')
            # Applying the binary operator '==' (line 56)
            result_eq_286 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 42), '==', subscript_call_result_284, str_285)
            
            # Applying the binary operator '!=' (line 56)
            result_ne_287 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 15), '!=', result_eq_280, result_eq_286)
            
            # Testing if the type of an if condition is none (line 56)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 56, 12), result_ne_287):
                pass
            else:
                
                # Testing the type of an if condition (line 56)
                if_condition_288 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 56, 12), result_ne_287)
                # Assigning a type to the variable 'if_condition_288' (line 56)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'if_condition_288', if_condition_288)
                # SSA begins for if statement (line 56)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'False' (line 57)
                False_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 23), 'False')
                # Assigning a type to the variable 'stypy_return_type' (line 57)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'stypy_return_type', False_289)
                # SSA join for if statement (line 56)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'True' (line 58)
        True_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'stypy_return_type', True_290)
        
        # ################# End of 'is_solved(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_solved' in the type store
        # Getting the type of 'stypy_return_type' (line 54)
        stypy_return_type_291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_291)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_solved'
        return stypy_return_type_291


    @norecursion
    def solve(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'solve'
        module_type_store = module_type_store.open_function_context('solve', 60, 4, False)
        # Assigning a type to the variable 'self' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Board.solve.__dict__.__setitem__('stypy_localization', localization)
        Board.solve.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Board.solve.__dict__.__setitem__('stypy_type_store', module_type_store)
        Board.solve.__dict__.__setitem__('stypy_function_name', 'Board.solve')
        Board.solve.__dict__.__setitem__('stypy_param_names_list', [])
        Board.solve.__dict__.__setitem__('stypy_varargs_param_name', None)
        Board.solve.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Board.solve.__dict__.__setitem__('stypy_call_defaults', defaults)
        Board.solve.__dict__.__setitem__('stypy_call_varargs', varargs)
        Board.solve.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Board.solve.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Board.solve', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'solve', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'solve(...)' code ##################

        
        # Assigning a Call to a Name (line 61):
        
        # Assigning a Call to a Name (line 61):
        
        # Call to deque(...): (line 61)
        # Processing the call keyword arguments (line 61)
        kwargs_293 = {}
        # Getting the type of 'deque' (line 61)
        deque_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 15), 'deque', False)
        # Calling deque(args, kwargs) (line 61)
        deque_call_result_294 = invoke(stypy.reporting.localization.Localization(__file__, 61, 15), deque_292, *[], **kwargs_293)
        
        # Assigning a type to the variable 'open' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'open', deque_call_result_294)
        
        # Call to append(...): (line 62)
        # Processing the call arguments (line 62)
        
        # Call to Open(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'self' (line 62)
        self_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 25), 'self', False)
        # Obtaining the member 'ddata' of a type (line 62)
        ddata_299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 25), self_298, 'ddata')
        str_300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 37), 'str', '')
        # Getting the type of 'self' (line 62)
        self_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 41), 'self', False)
        # Obtaining the member 'px' of a type (line 62)
        px_302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 41), self_301, 'px')
        # Getting the type of 'self' (line 62)
        self_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 50), 'self', False)
        # Obtaining the member 'py' of a type (line 62)
        py_304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 50), self_303, 'py')
        # Processing the call keyword arguments (line 62)
        kwargs_305 = {}
        # Getting the type of 'Open' (line 62)
        Open_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'Open', False)
        # Calling Open(args, kwargs) (line 62)
        Open_call_result_306 = invoke(stypy.reporting.localization.Localization(__file__, 62, 20), Open_297, *[ddata_299, str_300, px_302, py_304], **kwargs_305)
        
        # Processing the call keyword arguments (line 62)
        kwargs_307 = {}
        # Getting the type of 'open' (line 62)
        open_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'open', False)
        # Obtaining the member 'append' of a type (line 62)
        append_296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), open_295, 'append')
        # Calling append(args, kwargs) (line 62)
        append_call_result_308 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), append_296, *[Open_call_result_306], **kwargs_307)
        
        
        # Assigning a Call to a Name (line 64):
        
        # Assigning a Call to a Name (line 64):
        
        # Call to set(...): (line 64)
        # Processing the call keyword arguments (line 64)
        kwargs_310 = {}
        # Getting the type of 'set' (line 64)
        set_309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 18), 'set', False)
        # Calling set(args, kwargs) (line 64)
        set_call_result_311 = invoke(stypy.reporting.localization.Localization(__file__, 64, 18), set_309, *[], **kwargs_310)
        
        # Assigning a type to the variable 'visited' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'visited', set_call_result_311)
        
        # Call to add(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'self' (line 65)
        self_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 20), 'self', False)
        # Obtaining the member 'ddata' of a type (line 65)
        ddata_315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 20), self_314, 'ddata')
        # Processing the call keyword arguments (line 65)
        kwargs_316 = {}
        # Getting the type of 'visited' (line 65)
        visited_312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'visited', False)
        # Obtaining the member 'add' of a type (line 65)
        add_313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), visited_312, 'add')
        # Calling add(args, kwargs) (line 65)
        add_call_result_317 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), add_313, *[ddata_315], **kwargs_316)
        
        
        # Assigning a Tuple to a Name (line 67):
        
        # Assigning a Tuple to a Name (line 67):
        
        # Obtaining an instance of the builtin type 'tuple' (line 68)
        tuple_318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 12), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 68)
        # Adding element type (line 68)
        
        # Call to Direction(...): (line 68)
        # Processing the call arguments (line 68)
        int_320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 22), 'int')
        int_321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 25), 'int')
        str_322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 29), 'str', 'u')
        # Processing the call keyword arguments (line 68)
        kwargs_323 = {}
        # Getting the type of 'Direction' (line 68)
        Direction_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'Direction', False)
        # Calling Direction(args, kwargs) (line 68)
        Direction_call_result_324 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), Direction_319, *[int_320, int_321, str_322], **kwargs_323)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 12), tuple_318, Direction_call_result_324)
        # Adding element type (line 68)
        
        # Call to Direction(...): (line 69)
        # Processing the call arguments (line 69)
        int_326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 22), 'int')
        int_327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 25), 'int')
        str_328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 28), 'str', 'r')
        # Processing the call keyword arguments (line 69)
        kwargs_329 = {}
        # Getting the type of 'Direction' (line 69)
        Direction_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'Direction', False)
        # Calling Direction(args, kwargs) (line 69)
        Direction_call_result_330 = invoke(stypy.reporting.localization.Localization(__file__, 69, 12), Direction_325, *[int_326, int_327, str_328], **kwargs_329)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 12), tuple_318, Direction_call_result_330)
        # Adding element type (line 68)
        
        # Call to Direction(...): (line 70)
        # Processing the call arguments (line 70)
        int_332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 22), 'int')
        int_333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 25), 'int')
        str_334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 28), 'str', 'd')
        # Processing the call keyword arguments (line 70)
        kwargs_335 = {}
        # Getting the type of 'Direction' (line 70)
        Direction_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'Direction', False)
        # Calling Direction(args, kwargs) (line 70)
        Direction_call_result_336 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), Direction_331, *[int_332, int_333, str_334], **kwargs_335)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 12), tuple_318, Direction_call_result_336)
        # Adding element type (line 68)
        
        # Call to Direction(...): (line 71)
        # Processing the call arguments (line 71)
        int_338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 22), 'int')
        int_339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 26), 'int')
        str_340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 29), 'str', 'l')
        # Processing the call keyword arguments (line 71)
        kwargs_341 = {}
        # Getting the type of 'Direction' (line 71)
        Direction_337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'Direction', False)
        # Calling Direction(args, kwargs) (line 71)
        Direction_call_result_342 = invoke(stypy.reporting.localization.Localization(__file__, 71, 12), Direction_337, *[int_338, int_339, str_340], **kwargs_341)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 12), tuple_318, Direction_call_result_342)
        
        # Assigning a type to the variable 'dirs' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'dirs', tuple_318)
        
        # Getting the type of 'open' (line 74)
        open_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 14), 'open')
        # Testing if the while is going to be iterated (line 74)
        # Testing the type of an if condition (line 74)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 8), open_343)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 74, 8), open_343):
            # SSA begins for while statement (line 74)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a Call to a Name (line 75):
            
            # Assigning a Call to a Name (line 75):
            
            # Call to popleft(...): (line 75)
            # Processing the call keyword arguments (line 75)
            kwargs_346 = {}
            # Getting the type of 'open' (line 75)
            open_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'open', False)
            # Obtaining the member 'popleft' of a type (line 75)
            popleft_345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 16), open_344, 'popleft')
            # Calling popleft(args, kwargs) (line 75)
            popleft_call_result_347 = invoke(stypy.reporting.localization.Localization(__file__, 75, 16), popleft_345, *[], **kwargs_346)
            
            # Assigning a type to the variable 'o' (line 75)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'o', popleft_call_result_347)
            
            # Assigning a Tuple to a Tuple (line 76):
            
            # Assigning a Attribute to a Name (line 76):
            # Getting the type of 'o' (line 76)
            o_348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 30), 'o')
            # Obtaining the member 'cur' of a type (line 76)
            cur_349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 30), o_348, 'cur')
            # Assigning a type to the variable 'tuple_assignment_8' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'tuple_assignment_8', cur_349)
            
            # Assigning a Attribute to a Name (line 76):
            # Getting the type of 'o' (line 76)
            o_350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 37), 'o')
            # Obtaining the member 'csol' of a type (line 76)
            csol_351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 37), o_350, 'csol')
            # Assigning a type to the variable 'tuple_assignment_9' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'tuple_assignment_9', csol_351)
            
            # Assigning a Attribute to a Name (line 76):
            # Getting the type of 'o' (line 76)
            o_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 45), 'o')
            # Obtaining the member 'x' of a type (line 76)
            x_353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 45), o_352, 'x')
            # Assigning a type to the variable 'tuple_assignment_10' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'tuple_assignment_10', x_353)
            
            # Assigning a Attribute to a Name (line 76):
            # Getting the type of 'o' (line 76)
            o_354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 50), 'o')
            # Obtaining the member 'y' of a type (line 76)
            y_355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 50), o_354, 'y')
            # Assigning a type to the variable 'tuple_assignment_11' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'tuple_assignment_11', y_355)
            
            # Assigning a Name to a Name (line 76):
            # Getting the type of 'tuple_assignment_8' (line 76)
            tuple_assignment_8_356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'tuple_assignment_8')
            # Assigning a type to the variable 'cur' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'cur', tuple_assignment_8_356)
            
            # Assigning a Name to a Name (line 76):
            # Getting the type of 'tuple_assignment_9' (line 76)
            tuple_assignment_9_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'tuple_assignment_9')
            # Assigning a type to the variable 'csol' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 17), 'csol', tuple_assignment_9_357)
            
            # Assigning a Name to a Name (line 76):
            # Getting the type of 'tuple_assignment_10' (line 76)
            tuple_assignment_10_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'tuple_assignment_10')
            # Assigning a type to the variable 'x' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'x', tuple_assignment_10_358)
            
            # Assigning a Name to a Name (line 76):
            # Getting the type of 'tuple_assignment_11' (line 76)
            tuple_assignment_11_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'tuple_assignment_11')
            # Assigning a type to the variable 'y' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 26), 'y', tuple_assignment_11_359)
            
            
            # Call to xrange(...): (line 78)
            # Processing the call arguments (line 78)
            int_361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 28), 'int')
            # Processing the call keyword arguments (line 78)
            kwargs_362 = {}
            # Getting the type of 'xrange' (line 78)
            xrange_360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 21), 'xrange', False)
            # Calling xrange(args, kwargs) (line 78)
            xrange_call_result_363 = invoke(stypy.reporting.localization.Localization(__file__, 78, 21), xrange_360, *[int_361], **kwargs_362)
            
            # Testing if the for loop is going to be iterated (line 78)
            # Testing the type of a for loop iterable (line 78)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 78, 12), xrange_call_result_363)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 78, 12), xrange_call_result_363):
                # Getting the type of the for loop variable (line 78)
                for_loop_var_364 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 78, 12), xrange_call_result_363)
                # Assigning a type to the variable 'i' (line 78)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'i', for_loop_var_364)
                # SSA begins for a for statement (line 78)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Name to a Name (line 79):
                
                # Assigning a Name to a Name (line 79):
                # Getting the type of 'cur' (line 79)
                cur_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 23), 'cur')
                # Assigning a type to the variable 'temp' (line 79)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 16), 'temp', cur_365)
                
                # Assigning a Subscript to a Name (line 80):
                
                # Assigning a Subscript to a Name (line 80):
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 80)
                i_366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 27), 'i')
                # Getting the type of 'dirs' (line 80)
                dirs_367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 22), 'dirs')
                # Obtaining the member '__getitem__' of a type (line 80)
                getitem___368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 22), dirs_367, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 80)
                subscript_call_result_369 = invoke(stypy.reporting.localization.Localization(__file__, 80, 22), getitem___368, i_366)
                
                # Assigning a type to the variable 'dir' (line 80)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'dir', subscript_call_result_369)
                
                # Assigning a Tuple to a Tuple (line 81):
                
                # Assigning a Attribute to a Name (line 81):
                # Getting the type of 'dir' (line 81)
                dir_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 25), 'dir')
                # Obtaining the member 'dx' of a type (line 81)
                dx_371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 25), dir_370, 'dx')
                # Assigning a type to the variable 'tuple_assignment_12' (line 81)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'tuple_assignment_12', dx_371)
                
                # Assigning a Attribute to a Name (line 81):
                # Getting the type of 'dir' (line 81)
                dir_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 33), 'dir')
                # Obtaining the member 'dy' of a type (line 81)
                dy_373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 33), dir_372, 'dy')
                # Assigning a type to the variable 'tuple_assignment_13' (line 81)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'tuple_assignment_13', dy_373)
                
                # Assigning a Name to a Name (line 81):
                # Getting the type of 'tuple_assignment_12' (line 81)
                tuple_assignment_12_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'tuple_assignment_12')
                # Assigning a type to the variable 'dx' (line 81)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'dx', tuple_assignment_12_374)
                
                # Assigning a Name to a Name (line 81):
                # Getting the type of 'tuple_assignment_13' (line 81)
                tuple_assignment_13_375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'tuple_assignment_13')
                # Assigning a type to the variable 'dy' (line 81)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 20), 'dy', tuple_assignment_13_375)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'y' (line 83)
                y_376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 25), 'y')
                # Getting the type of 'dy' (line 83)
                dy_377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 29), 'dy')
                # Applying the binary operator '+' (line 83)
                result_add_378 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 25), '+', y_376, dy_377)
                
                # Getting the type of 'self' (line 83)
                self_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 35), 'self')
                # Obtaining the member 'nrows' of a type (line 83)
                nrows_380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 35), self_379, 'nrows')
                # Applying the binary operator '*' (line 83)
                result_mul_381 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 24), '*', result_add_378, nrows_380)
                
                # Getting the type of 'x' (line 83)
                x_382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 48), 'x')
                # Applying the binary operator '+' (line 83)
                result_add_383 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 24), '+', result_mul_381, x_382)
                
                # Getting the type of 'dx' (line 83)
                dx_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 52), 'dx')
                # Applying the binary operator '+' (line 83)
                result_add_385 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 50), '+', result_add_383, dx_384)
                
                # Getting the type of 'temp' (line 83)
                temp_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 19), 'temp')
                # Obtaining the member '__getitem__' of a type (line 83)
                getitem___387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 19), temp_386, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 83)
                subscript_call_result_388 = invoke(stypy.reporting.localization.Localization(__file__, 83, 19), getitem___387, result_add_385)
                
                str_389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 59), 'str', '*')
                # Applying the binary operator '==' (line 83)
                result_eq_390 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 19), '==', subscript_call_result_388, str_389)
                
                # Testing if the type of an if condition is none (line 83)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 83, 16), result_eq_390):
                    
                    # Assigning a Call to a Name (line 91):
                    
                    # Assigning a Call to a Name (line 91):
                    
                    # Call to move(...): (line 91)
                    # Processing the call arguments (line 91)
                    # Getting the type of 'x' (line 91)
                    x_448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 37), 'x', False)
                    # Getting the type of 'y' (line 91)
                    y_449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 40), 'y', False)
                    # Getting the type of 'dx' (line 91)
                    dx_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 43), 'dx', False)
                    # Getting the type of 'dy' (line 91)
                    dy_451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 47), 'dy', False)
                    # Getting the type of 'temp' (line 91)
                    temp_452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 51), 'temp', False)
                    # Processing the call keyword arguments (line 91)
                    kwargs_453 = {}
                    # Getting the type of 'self' (line 91)
                    self_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 27), 'self', False)
                    # Obtaining the member 'move' of a type (line 91)
                    move_447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 27), self_446, 'move')
                    # Calling move(args, kwargs) (line 91)
                    move_call_result_454 = invoke(stypy.reporting.localization.Localization(__file__, 91, 27), move_447, *[x_448, y_449, dx_450, dy_451, temp_452], **kwargs_453)
                    
                    # Assigning a type to the variable 'temp' (line 91)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'temp', move_call_result_454)
                    
                    # Evaluating a boolean operation
                    # Getting the type of 'temp' (line 92)
                    temp_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 23), 'temp')
                    
                    # Getting the type of 'temp' (line 92)
                    temp_456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 32), 'temp')
                    # Getting the type of 'visited' (line 92)
                    visited_457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 44), 'visited')
                    # Applying the binary operator 'notin' (line 92)
                    result_contains_458 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 32), 'notin', temp_456, visited_457)
                    
                    # Applying the binary operator 'and' (line 92)
                    result_and_keyword_459 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 23), 'and', temp_455, result_contains_458)
                    
                    # Testing if the type of an if condition is none (line 92)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 92, 20), result_and_keyword_459):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 92)
                        if_condition_460 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 20), result_and_keyword_459)
                        # Assigning a type to the variable 'if_condition_460' (line 92)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 20), 'if_condition_460', if_condition_460)
                        # SSA begins for if statement (line 92)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to is_solved(...): (line 93)
                        # Processing the call arguments (line 93)
                        # Getting the type of 'temp' (line 93)
                        temp_463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 42), 'temp', False)
                        # Processing the call keyword arguments (line 93)
                        kwargs_464 = {}
                        # Getting the type of 'self' (line 93)
                        self_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 27), 'self', False)
                        # Obtaining the member 'is_solved' of a type (line 93)
                        is_solved_462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 27), self_461, 'is_solved')
                        # Calling is_solved(args, kwargs) (line 93)
                        is_solved_call_result_465 = invoke(stypy.reporting.localization.Localization(__file__, 93, 27), is_solved_462, *[temp_463], **kwargs_464)
                        
                        # Testing if the type of an if condition is none (line 93)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 93, 24), is_solved_call_result_465):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 93)
                            if_condition_466 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 24), is_solved_call_result_465)
                            # Assigning a type to the variable 'if_condition_466' (line 93)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 24), 'if_condition_466', if_condition_466)
                            # SSA begins for if statement (line 93)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            # Getting the type of 'csol' (line 94)
                            csol_467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 35), 'csol')
                            # Getting the type of 'dir' (line 94)
                            dir_468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 42), 'dir')
                            # Obtaining the member 'letter' of a type (line 94)
                            letter_469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 42), dir_468, 'letter')
                            # Applying the binary operator '+' (line 94)
                            result_add_470 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 35), '+', csol_467, letter_469)
                            
                            # Assigning a type to the variable 'stypy_return_type' (line 94)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 28), 'stypy_return_type', result_add_470)
                            # SSA join for if statement (line 93)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        
                        # Call to append(...): (line 95)
                        # Processing the call arguments (line 95)
                        
                        # Call to Open(...): (line 95)
                        # Processing the call arguments (line 95)
                        # Getting the type of 'temp' (line 95)
                        temp_474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 41), 'temp', False)
                        # Getting the type of 'csol' (line 95)
                        csol_475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 47), 'csol', False)
                        # Getting the type of 'dir' (line 95)
                        dir_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 54), 'dir', False)
                        # Obtaining the member 'letter' of a type (line 95)
                        letter_477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 54), dir_476, 'letter')
                        # Applying the binary operator '+' (line 95)
                        result_add_478 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 47), '+', csol_475, letter_477)
                        
                        # Getting the type of 'x' (line 95)
                        x_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 66), 'x', False)
                        # Getting the type of 'dx' (line 95)
                        dx_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 70), 'dx', False)
                        # Applying the binary operator '+' (line 95)
                        result_add_481 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 66), '+', x_479, dx_480)
                        
                        # Getting the type of 'y' (line 95)
                        y_482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 74), 'y', False)
                        # Getting the type of 'dy' (line 95)
                        dy_483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 78), 'dy', False)
                        # Applying the binary operator '+' (line 95)
                        result_add_484 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 74), '+', y_482, dy_483)
                        
                        # Processing the call keyword arguments (line 95)
                        kwargs_485 = {}
                        # Getting the type of 'Open' (line 95)
                        Open_473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 36), 'Open', False)
                        # Calling Open(args, kwargs) (line 95)
                        Open_call_result_486 = invoke(stypy.reporting.localization.Localization(__file__, 95, 36), Open_473, *[temp_474, result_add_478, result_add_481, result_add_484], **kwargs_485)
                        
                        # Processing the call keyword arguments (line 95)
                        kwargs_487 = {}
                        # Getting the type of 'open' (line 95)
                        open_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 24), 'open', False)
                        # Obtaining the member 'append' of a type (line 95)
                        append_472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 24), open_471, 'append')
                        # Calling append(args, kwargs) (line 95)
                        append_call_result_488 = invoke(stypy.reporting.localization.Localization(__file__, 95, 24), append_472, *[Open_call_result_486], **kwargs_487)
                        
                        
                        # Call to add(...): (line 96)
                        # Processing the call arguments (line 96)
                        # Getting the type of 'temp' (line 96)
                        temp_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 36), 'temp', False)
                        # Processing the call keyword arguments (line 96)
                        kwargs_492 = {}
                        # Getting the type of 'visited' (line 96)
                        visited_489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 24), 'visited', False)
                        # Obtaining the member 'add' of a type (line 96)
                        add_490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 24), visited_489, 'add')
                        # Calling add(args, kwargs) (line 96)
                        add_call_result_493 = invoke(stypy.reporting.localization.Localization(__file__, 96, 24), add_490, *[temp_491], **kwargs_492)
                        
                        # SSA join for if statement (line 92)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 83)
                    if_condition_391 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 16), result_eq_390)
                    # Assigning a type to the variable 'if_condition_391' (line 83)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'if_condition_391', if_condition_391)
                    # SSA begins for if statement (line 83)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 84):
                    
                    # Assigning a Call to a Name (line 84):
                    
                    # Call to push(...): (line 84)
                    # Processing the call arguments (line 84)
                    # Getting the type of 'x' (line 84)
                    x_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 37), 'x', False)
                    # Getting the type of 'y' (line 84)
                    y_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 40), 'y', False)
                    # Getting the type of 'dx' (line 84)
                    dx_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 43), 'dx', False)
                    # Getting the type of 'dy' (line 84)
                    dy_397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 47), 'dy', False)
                    # Getting the type of 'temp' (line 84)
                    temp_398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 51), 'temp', False)
                    # Processing the call keyword arguments (line 84)
                    kwargs_399 = {}
                    # Getting the type of 'self' (line 84)
                    self_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 27), 'self', False)
                    # Obtaining the member 'push' of a type (line 84)
                    push_393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 27), self_392, 'push')
                    # Calling push(args, kwargs) (line 84)
                    push_call_result_400 = invoke(stypy.reporting.localization.Localization(__file__, 84, 27), push_393, *[x_394, y_395, dx_396, dy_397, temp_398], **kwargs_399)
                    
                    # Assigning a type to the variable 'temp' (line 84)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 20), 'temp', push_call_result_400)
                    
                    # Evaluating a boolean operation
                    # Getting the type of 'temp' (line 85)
                    temp_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 23), 'temp')
                    
                    # Getting the type of 'temp' (line 85)
                    temp_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 32), 'temp')
                    # Getting the type of 'visited' (line 85)
                    visited_403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 44), 'visited')
                    # Applying the binary operator 'notin' (line 85)
                    result_contains_404 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 32), 'notin', temp_402, visited_403)
                    
                    # Applying the binary operator 'and' (line 85)
                    result_and_keyword_405 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 23), 'and', temp_401, result_contains_404)
                    
                    # Testing if the type of an if condition is none (line 85)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 85, 20), result_and_keyword_405):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 85)
                        if_condition_406 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 20), result_and_keyword_405)
                        # Assigning a type to the variable 'if_condition_406' (line 85)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 20), 'if_condition_406', if_condition_406)
                        # SSA begins for if statement (line 85)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to is_solved(...): (line 86)
                        # Processing the call arguments (line 86)
                        # Getting the type of 'temp' (line 86)
                        temp_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 42), 'temp', False)
                        # Processing the call keyword arguments (line 86)
                        kwargs_410 = {}
                        # Getting the type of 'self' (line 86)
                        self_407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 27), 'self', False)
                        # Obtaining the member 'is_solved' of a type (line 86)
                        is_solved_408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 27), self_407, 'is_solved')
                        # Calling is_solved(args, kwargs) (line 86)
                        is_solved_call_result_411 = invoke(stypy.reporting.localization.Localization(__file__, 86, 27), is_solved_408, *[temp_409], **kwargs_410)
                        
                        # Testing if the type of an if condition is none (line 86)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 86, 24), is_solved_call_result_411):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 86)
                            if_condition_412 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 24), is_solved_call_result_411)
                            # Assigning a type to the variable 'if_condition_412' (line 86)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 24), 'if_condition_412', if_condition_412)
                            # SSA begins for if statement (line 86)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            # Getting the type of 'csol' (line 87)
                            csol_413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 35), 'csol')
                            
                            # Call to upper(...): (line 87)
                            # Processing the call keyword arguments (line 87)
                            kwargs_417 = {}
                            # Getting the type of 'dir' (line 87)
                            dir_414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 42), 'dir', False)
                            # Obtaining the member 'letter' of a type (line 87)
                            letter_415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 42), dir_414, 'letter')
                            # Obtaining the member 'upper' of a type (line 87)
                            upper_416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 42), letter_415, 'upper')
                            # Calling upper(args, kwargs) (line 87)
                            upper_call_result_418 = invoke(stypy.reporting.localization.Localization(__file__, 87, 42), upper_416, *[], **kwargs_417)
                            
                            # Applying the binary operator '+' (line 87)
                            result_add_419 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 35), '+', csol_413, upper_call_result_418)
                            
                            # Assigning a type to the variable 'stypy_return_type' (line 87)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 28), 'stypy_return_type', result_add_419)
                            # SSA join for if statement (line 86)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        
                        # Call to append(...): (line 88)
                        # Processing the call arguments (line 88)
                        
                        # Call to Open(...): (line 88)
                        # Processing the call arguments (line 88)
                        # Getting the type of 'temp' (line 88)
                        temp_423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 41), 'temp', False)
                        # Getting the type of 'csol' (line 88)
                        csol_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 47), 'csol', False)
                        
                        # Call to upper(...): (line 88)
                        # Processing the call keyword arguments (line 88)
                        kwargs_428 = {}
                        # Getting the type of 'dir' (line 88)
                        dir_425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 54), 'dir', False)
                        # Obtaining the member 'letter' of a type (line 88)
                        letter_426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 54), dir_425, 'letter')
                        # Obtaining the member 'upper' of a type (line 88)
                        upper_427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 54), letter_426, 'upper')
                        # Calling upper(args, kwargs) (line 88)
                        upper_call_result_429 = invoke(stypy.reporting.localization.Localization(__file__, 88, 54), upper_427, *[], **kwargs_428)
                        
                        # Applying the binary operator '+' (line 88)
                        result_add_430 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 47), '+', csol_424, upper_call_result_429)
                        
                        # Getting the type of 'x' (line 88)
                        x_431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 74), 'x', False)
                        # Getting the type of 'dx' (line 88)
                        dx_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 78), 'dx', False)
                        # Applying the binary operator '+' (line 88)
                        result_add_433 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 74), '+', x_431, dx_432)
                        
                        # Getting the type of 'y' (line 88)
                        y_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 82), 'y', False)
                        # Getting the type of 'dy' (line 88)
                        dy_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 86), 'dy', False)
                        # Applying the binary operator '+' (line 88)
                        result_add_436 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 82), '+', y_434, dy_435)
                        
                        # Processing the call keyword arguments (line 88)
                        kwargs_437 = {}
                        # Getting the type of 'Open' (line 88)
                        Open_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 36), 'Open', False)
                        # Calling Open(args, kwargs) (line 88)
                        Open_call_result_438 = invoke(stypy.reporting.localization.Localization(__file__, 88, 36), Open_422, *[temp_423, result_add_430, result_add_433, result_add_436], **kwargs_437)
                        
                        # Processing the call keyword arguments (line 88)
                        kwargs_439 = {}
                        # Getting the type of 'open' (line 88)
                        open_420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'open', False)
                        # Obtaining the member 'append' of a type (line 88)
                        append_421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 24), open_420, 'append')
                        # Calling append(args, kwargs) (line 88)
                        append_call_result_440 = invoke(stypy.reporting.localization.Localization(__file__, 88, 24), append_421, *[Open_call_result_438], **kwargs_439)
                        
                        
                        # Call to add(...): (line 89)
                        # Processing the call arguments (line 89)
                        # Getting the type of 'temp' (line 89)
                        temp_443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 36), 'temp', False)
                        # Processing the call keyword arguments (line 89)
                        kwargs_444 = {}
                        # Getting the type of 'visited' (line 89)
                        visited_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 24), 'visited', False)
                        # Obtaining the member 'add' of a type (line 89)
                        add_442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 24), visited_441, 'add')
                        # Calling add(args, kwargs) (line 89)
                        add_call_result_445 = invoke(stypy.reporting.localization.Localization(__file__, 89, 24), add_442, *[temp_443], **kwargs_444)
                        
                        # SSA join for if statement (line 85)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA branch for the else part of an if statement (line 83)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Call to a Name (line 91):
                    
                    # Assigning a Call to a Name (line 91):
                    
                    # Call to move(...): (line 91)
                    # Processing the call arguments (line 91)
                    # Getting the type of 'x' (line 91)
                    x_448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 37), 'x', False)
                    # Getting the type of 'y' (line 91)
                    y_449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 40), 'y', False)
                    # Getting the type of 'dx' (line 91)
                    dx_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 43), 'dx', False)
                    # Getting the type of 'dy' (line 91)
                    dy_451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 47), 'dy', False)
                    # Getting the type of 'temp' (line 91)
                    temp_452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 51), 'temp', False)
                    # Processing the call keyword arguments (line 91)
                    kwargs_453 = {}
                    # Getting the type of 'self' (line 91)
                    self_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 27), 'self', False)
                    # Obtaining the member 'move' of a type (line 91)
                    move_447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 27), self_446, 'move')
                    # Calling move(args, kwargs) (line 91)
                    move_call_result_454 = invoke(stypy.reporting.localization.Localization(__file__, 91, 27), move_447, *[x_448, y_449, dx_450, dy_451, temp_452], **kwargs_453)
                    
                    # Assigning a type to the variable 'temp' (line 91)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'temp', move_call_result_454)
                    
                    # Evaluating a boolean operation
                    # Getting the type of 'temp' (line 92)
                    temp_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 23), 'temp')
                    
                    # Getting the type of 'temp' (line 92)
                    temp_456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 32), 'temp')
                    # Getting the type of 'visited' (line 92)
                    visited_457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 44), 'visited')
                    # Applying the binary operator 'notin' (line 92)
                    result_contains_458 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 32), 'notin', temp_456, visited_457)
                    
                    # Applying the binary operator 'and' (line 92)
                    result_and_keyword_459 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 23), 'and', temp_455, result_contains_458)
                    
                    # Testing if the type of an if condition is none (line 92)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 92, 20), result_and_keyword_459):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 92)
                        if_condition_460 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 20), result_and_keyword_459)
                        # Assigning a type to the variable 'if_condition_460' (line 92)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 20), 'if_condition_460', if_condition_460)
                        # SSA begins for if statement (line 92)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to is_solved(...): (line 93)
                        # Processing the call arguments (line 93)
                        # Getting the type of 'temp' (line 93)
                        temp_463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 42), 'temp', False)
                        # Processing the call keyword arguments (line 93)
                        kwargs_464 = {}
                        # Getting the type of 'self' (line 93)
                        self_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 27), 'self', False)
                        # Obtaining the member 'is_solved' of a type (line 93)
                        is_solved_462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 27), self_461, 'is_solved')
                        # Calling is_solved(args, kwargs) (line 93)
                        is_solved_call_result_465 = invoke(stypy.reporting.localization.Localization(__file__, 93, 27), is_solved_462, *[temp_463], **kwargs_464)
                        
                        # Testing if the type of an if condition is none (line 93)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 93, 24), is_solved_call_result_465):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 93)
                            if_condition_466 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 24), is_solved_call_result_465)
                            # Assigning a type to the variable 'if_condition_466' (line 93)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 24), 'if_condition_466', if_condition_466)
                            # SSA begins for if statement (line 93)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            # Getting the type of 'csol' (line 94)
                            csol_467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 35), 'csol')
                            # Getting the type of 'dir' (line 94)
                            dir_468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 42), 'dir')
                            # Obtaining the member 'letter' of a type (line 94)
                            letter_469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 42), dir_468, 'letter')
                            # Applying the binary operator '+' (line 94)
                            result_add_470 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 35), '+', csol_467, letter_469)
                            
                            # Assigning a type to the variable 'stypy_return_type' (line 94)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 28), 'stypy_return_type', result_add_470)
                            # SSA join for if statement (line 93)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        
                        # Call to append(...): (line 95)
                        # Processing the call arguments (line 95)
                        
                        # Call to Open(...): (line 95)
                        # Processing the call arguments (line 95)
                        # Getting the type of 'temp' (line 95)
                        temp_474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 41), 'temp', False)
                        # Getting the type of 'csol' (line 95)
                        csol_475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 47), 'csol', False)
                        # Getting the type of 'dir' (line 95)
                        dir_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 54), 'dir', False)
                        # Obtaining the member 'letter' of a type (line 95)
                        letter_477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 54), dir_476, 'letter')
                        # Applying the binary operator '+' (line 95)
                        result_add_478 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 47), '+', csol_475, letter_477)
                        
                        # Getting the type of 'x' (line 95)
                        x_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 66), 'x', False)
                        # Getting the type of 'dx' (line 95)
                        dx_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 70), 'dx', False)
                        # Applying the binary operator '+' (line 95)
                        result_add_481 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 66), '+', x_479, dx_480)
                        
                        # Getting the type of 'y' (line 95)
                        y_482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 74), 'y', False)
                        # Getting the type of 'dy' (line 95)
                        dy_483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 78), 'dy', False)
                        # Applying the binary operator '+' (line 95)
                        result_add_484 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 74), '+', y_482, dy_483)
                        
                        # Processing the call keyword arguments (line 95)
                        kwargs_485 = {}
                        # Getting the type of 'Open' (line 95)
                        Open_473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 36), 'Open', False)
                        # Calling Open(args, kwargs) (line 95)
                        Open_call_result_486 = invoke(stypy.reporting.localization.Localization(__file__, 95, 36), Open_473, *[temp_474, result_add_478, result_add_481, result_add_484], **kwargs_485)
                        
                        # Processing the call keyword arguments (line 95)
                        kwargs_487 = {}
                        # Getting the type of 'open' (line 95)
                        open_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 24), 'open', False)
                        # Obtaining the member 'append' of a type (line 95)
                        append_472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 24), open_471, 'append')
                        # Calling append(args, kwargs) (line 95)
                        append_call_result_488 = invoke(stypy.reporting.localization.Localization(__file__, 95, 24), append_472, *[Open_call_result_486], **kwargs_487)
                        
                        
                        # Call to add(...): (line 96)
                        # Processing the call arguments (line 96)
                        # Getting the type of 'temp' (line 96)
                        temp_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 36), 'temp', False)
                        # Processing the call keyword arguments (line 96)
                        kwargs_492 = {}
                        # Getting the type of 'visited' (line 96)
                        visited_489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 24), 'visited', False)
                        # Obtaining the member 'add' of a type (line 96)
                        add_490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 24), visited_489, 'add')
                        # Calling add(args, kwargs) (line 96)
                        add_call_result_493 = invoke(stypy.reporting.localization.Localization(__file__, 96, 24), add_490, *[temp_491], **kwargs_492)
                        
                        # SSA join for if statement (line 92)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 83)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for while statement (line 74)
            module_type_store = module_type_store.join_ssa_context()

        
        str_494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 15), 'str', 'No solution')
        # Assigning a type to the variable 'stypy_return_type' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'stypy_return_type', str_494)
        
        # ################# End of 'solve(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'solve' in the type store
        # Getting the type of 'stypy_return_type' (line 60)
        stypy_return_type_495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_495)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'solve'
        return stypy_return_type_495


# Assigning a type to the variable 'Board' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'Board', Board)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 101, 0, False)
    
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

    
    # Assigning a Str to a Name (line 102):
    
    # Assigning a Str to a Name (line 102):
    str_496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, (-1)), 'str', '    #######\n    #     #\n    #     #\n    #. #  #\n    #. $$ #\n    #.$$  #\n    #.#  @#\n    #######')
    # Assigning a type to the variable 'level' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'level', str_496)
    
    
    # Call to range(...): (line 113)
    # Processing the call arguments (line 113)
    int_498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 19), 'int')
    # Processing the call keyword arguments (line 113)
    kwargs_499 = {}
    # Getting the type of 'range' (line 113)
    range_497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 13), 'range', False)
    # Calling range(args, kwargs) (line 113)
    range_call_result_500 = invoke(stypy.reporting.localization.Localization(__file__, 113, 13), range_497, *[int_498], **kwargs_499)
    
    # Testing if the for loop is going to be iterated (line 113)
    # Testing the type of a for loop iterable (line 113)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 113, 4), range_call_result_500)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 113, 4), range_call_result_500):
        # Getting the type of the for loop variable (line 113)
        for_loop_var_501 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 113, 4), range_call_result_500)
        # Assigning a type to the variable 'i' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'i', for_loop_var_501)
        # SSA begins for a for statement (line 113)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 114):
        
        # Assigning a Call to a Name (line 114):
        
        # Call to Board(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'level' (line 114)
        level_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 18), 'level', False)
        # Processing the call keyword arguments (line 114)
        kwargs_504 = {}
        # Getting the type of 'Board' (line 114)
        Board_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'Board', False)
        # Calling Board(args, kwargs) (line 114)
        Board_call_result_505 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), Board_502, *[level_503], **kwargs_504)
        
        # Assigning a type to the variable 'b' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'b', Board_call_result_505)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'True' (line 116)
    True_506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'stypy_return_type', True_506)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 101)
    stypy_return_type_507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_507)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_507

# Assigning a type to the variable 'run' (line 101)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), 'run', run)

# Call to run(...): (line 119)
# Processing the call keyword arguments (line 119)
kwargs_509 = {}
# Getting the type of 'run' (line 119)
run_508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), 'run', False)
# Calling run(args, kwargs) (line 119)
run_call_result_510 = invoke(stypy.reporting.localization.Localization(__file__, 119, 0), run_508, *[], **kwargs_509)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

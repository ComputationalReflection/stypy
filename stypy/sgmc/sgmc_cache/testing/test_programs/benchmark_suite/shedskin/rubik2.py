
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Rubik's cube solver using Thistlethwaite's algorithm
2: #
3: # Python translation of Stefan Pochmann's C++ implementation
4: # http://www.stefan-pochmann.info/spocc/other_stuff/tools/
5: # by Mark Dufour (mark.dufour@gmail.com)
6: #
7: # cube 'state' is a list with 40 entries, the first 20
8: # are a permutation of {0,...,19} and describe which cubie is at
9: # a certain position (regarding the input ordering). The first
10: # twelve are for edges, the last eight for corners.
11: # 
12: # The last 20 entries are for the orientations, each describing
13: # how often the cubie at a certain position has been turned
14: # counterclockwise away from the correct orientation. Again the
15: # first twelve are edges, the last eight are corners. The values
16: # are 0 or 1 for edges and 0, 1 or 2 for corners.
17: import random
18: 
19: random.seed(1)
20: 
21: facenames = ["U", "D", "F", "B", "L", "R"]
22: affected_cubies = [[0, 1, 2, 3, 0, 1, 2, 3], [4, 7, 6, 5, 4, 5, 6, 7], [0, 9, 4, 8, 0, 3, 5, 4],
23:                    [2, 10, 6, 11, 2, 1, 7, 6], [3, 11, 7, 9, 3, 2, 6, 5], [1, 8, 5, 10, 1, 0, 4, 7]]
24: phase_moves = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
25:                [0, 1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15, 16, 17], [0, 1, 2, 3, 4, 5, 7, 10, 13, 16],
26:                [1, 4, 7, 10, 13, 16]]
27: 
28: 
29: def move_str(move):
30:     return facenames[move / 3] + {1: '', 2: '2', 3: "'"}[move % 3 + 1]
31: 
32: 
33: class cube_state:
34:     def __init__(self, state, route=None):
35:         self.state = state
36:         self.route = route or []
37: 
38:     def id_(self, phase):
39:         if phase == 0:
40:             return tuple(self.state[20:32])
41:         elif phase == 1:
42:             result = self.state[31:40]
43:             for e in range(12):
44:                 result[0] |= (self.state[e] / 8) << e;
45:             return tuple(result)
46:         elif phase == 2:
47:             result = [0, 0, 0]
48:             for e in range(12):
49:                 result[0] |= (2 if (self.state[e] > 7) else (self.state[e] & 1)) << (2 * e)
50:             for c in range(8):
51:                 result[1] |= ((self.state[c + 12] - 12) & 5) << (3 * c)
52:             for i in range(12, 20):
53:                 for j in range(i + 1, 20):
54:                     result[2] ^= int(self.state[i] > self.state[j])
55:             return tuple(result)
56:         else:
57:             return tuple(self.state)
58: 
59:     def apply_move(self, move):
60:         face, turns = move / 3, move % 3 + 1
61:         newstate = self.state[:]
62:         for turn in range(turns):
63:             oldstate = newstate[:]
64:             for i in range(8):
65:                 isCorner = int(i > 3)
66:                 target = affected_cubies[face][i] + isCorner * 12
67:                 killer = affected_cubies[face][(i - 3) if (i & 3) == 3 else i + 1] + isCorner * 12
68:                 orientationDelta = int(1 < face < 4) if i < 4 else (0 if face < 2 else 2 - (i & 1))
69:                 newstate[target] = oldstate[killer]
70:                 newstate[target + 20] = oldstate[killer + 20] + orientationDelta
71:                 if turn == turns - 1:
72:                     newstate[target + 20] %= 2 + isCorner
73:         return cube_state(newstate, self.route + [move])
74: 
75: 
76: def run():
77:     goal_state = cube_state(range(20) + 20 * [0])
78:     state = cube_state(goal_state.state[:])
79:     ##    print '*** randomize ***'
80:     moves = [random.randrange(0, 18) for x in range(30)]
81:     ##    print ','.join([move_str(move) for move in moves])+'\n'
82:     ','.join([move_str(move) for move in moves])
83:     move_str(move)
84:     for move in moves:
85:         state = state.apply_move(move)
86:     state.route = []
87:     ##    print '*** solve ***'
88:     for phase in range(4):
89:         current_id, goal_id = state.id_(phase), goal_state.id_(phase)
90:         states = [state]
91:         state_ids = set([current_id])
92:         if current_id != goal_id:
93:             phase_ok = False
94:             while not phase_ok:
95:                 next_states = []
96:                 for cur_state in states:
97:                     for move in phase_moves[phase]:
98:                         next_state = cur_state.apply_move(move)
99:                         next_id = next_state.id_(phase)
100:                         if next_id == goal_id:
101:                             pass  # print ','.join([move_str(m) for m in next_state.route]) + ' (%d moves)'% len(next_state.route)
102:                             phase_ok = True
103:                             state = next_state
104:                             break
105:                         if next_id not in state_ids:
106:                             state_ids.add(next_id)
107:                             next_states.append(next_state)
108:                     if phase_ok:
109:                         break
110:                 states = next_states
111:     return True
112: 
113: 
114: run()
115: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'import random' statement (line 17)
import random

import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'random', random, module_type_store)


# Call to seed(...): (line 19)
# Processing the call arguments (line 19)
int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 12), 'int')
# Processing the call keyword arguments (line 19)
kwargs_8 = {}
# Getting the type of 'random' (line 19)
random_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'random', False)
# Obtaining the member 'seed' of a type (line 19)
seed_6 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 0), random_5, 'seed')
# Calling seed(args, kwargs) (line 19)
seed_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 19, 0), seed_6, *[int_7], **kwargs_8)


# Assigning a List to a Name (line 21):

# Assigning a List to a Name (line 21):

# Obtaining an instance of the builtin type 'list' (line 21)
list_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 21)
# Adding element type (line 21)
str_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 13), 'str', 'U')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 12), list_10, str_11)
# Adding element type (line 21)
str_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 18), 'str', 'D')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 12), list_10, str_12)
# Adding element type (line 21)
str_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 23), 'str', 'F')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 12), list_10, str_13)
# Adding element type (line 21)
str_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 28), 'str', 'B')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 12), list_10, str_14)
# Adding element type (line 21)
str_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 33), 'str', 'L')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 12), list_10, str_15)
# Adding element type (line 21)
str_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 38), 'str', 'R')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 12), list_10, str_16)

# Assigning a type to the variable 'facenames' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'facenames', list_10)

# Assigning a List to a Name (line 22):

# Assigning a List to a Name (line 22):

# Obtaining an instance of the builtin type 'list' (line 22)
list_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 22)
# Adding element type (line 22)

# Obtaining an instance of the builtin type 'list' (line 22)
list_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 22)
# Adding element type (line 22)
int_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 19), list_18, int_19)
# Adding element type (line 22)
int_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 19), list_18, int_20)
# Adding element type (line 22)
int_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 19), list_18, int_21)
# Adding element type (line 22)
int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 19), list_18, int_22)
# Adding element type (line 22)
int_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 19), list_18, int_23)
# Adding element type (line 22)
int_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 19), list_18, int_24)
# Adding element type (line 22)
int_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 38), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 19), list_18, int_25)
# Adding element type (line 22)
int_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 41), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 19), list_18, int_26)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 18), list_17, list_18)
# Adding element type (line 22)

# Obtaining an instance of the builtin type 'list' (line 22)
list_27 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 45), 'list')
# Adding type elements to the builtin type 'list' instance (line 22)
# Adding element type (line 22)
int_28 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 46), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 45), list_27, int_28)
# Adding element type (line 22)
int_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 49), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 45), list_27, int_29)
# Adding element type (line 22)
int_30 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 52), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 45), list_27, int_30)
# Adding element type (line 22)
int_31 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 55), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 45), list_27, int_31)
# Adding element type (line 22)
int_32 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 58), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 45), list_27, int_32)
# Adding element type (line 22)
int_33 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 45), list_27, int_33)
# Adding element type (line 22)
int_34 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 64), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 45), list_27, int_34)
# Adding element type (line 22)
int_35 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 67), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 45), list_27, int_35)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 18), list_17, list_27)
# Adding element type (line 22)

# Obtaining an instance of the builtin type 'list' (line 22)
list_36 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 71), 'list')
# Adding type elements to the builtin type 'list' instance (line 22)
# Adding element type (line 22)
int_37 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 72), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 71), list_36, int_37)
# Adding element type (line 22)
int_38 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 75), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 71), list_36, int_38)
# Adding element type (line 22)
int_39 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 78), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 71), list_36, int_39)
# Adding element type (line 22)
int_40 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 81), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 71), list_36, int_40)
# Adding element type (line 22)
int_41 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 84), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 71), list_36, int_41)
# Adding element type (line 22)
int_42 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 87), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 71), list_36, int_42)
# Adding element type (line 22)
int_43 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 90), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 71), list_36, int_43)
# Adding element type (line 22)
int_44 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 93), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 71), list_36, int_44)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 18), list_17, list_36)
# Adding element type (line 22)

# Obtaining an instance of the builtin type 'list' (line 23)
list_45 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 23)
# Adding element type (line 23)
int_46 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 19), list_45, int_46)
# Adding element type (line 23)
int_47 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 19), list_45, int_47)
# Adding element type (line 23)
int_48 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 19), list_45, int_48)
# Adding element type (line 23)
int_49 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 19), list_45, int_49)
# Adding element type (line 23)
int_50 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 19), list_45, int_50)
# Adding element type (line 23)
int_51 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 37), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 19), list_45, int_51)
# Adding element type (line 23)
int_52 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 19), list_45, int_52)
# Adding element type (line 23)
int_53 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 19), list_45, int_53)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 18), list_17, list_45)
# Adding element type (line 22)

# Obtaining an instance of the builtin type 'list' (line 23)
list_54 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 47), 'list')
# Adding type elements to the builtin type 'list' instance (line 23)
# Adding element type (line 23)
int_55 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 48), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 47), list_54, int_55)
# Adding element type (line 23)
int_56 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 51), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 47), list_54, int_56)
# Adding element type (line 23)
int_57 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 55), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 47), list_54, int_57)
# Adding element type (line 23)
int_58 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 58), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 47), list_54, int_58)
# Adding element type (line 23)
int_59 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 47), list_54, int_59)
# Adding element type (line 23)
int_60 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 64), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 47), list_54, int_60)
# Adding element type (line 23)
int_61 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 67), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 47), list_54, int_61)
# Adding element type (line 23)
int_62 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 70), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 47), list_54, int_62)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 18), list_17, list_54)
# Adding element type (line 22)

# Obtaining an instance of the builtin type 'list' (line 23)
list_63 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 74), 'list')
# Adding type elements to the builtin type 'list' instance (line 23)
# Adding element type (line 23)
int_64 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 75), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 74), list_63, int_64)
# Adding element type (line 23)
int_65 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 78), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 74), list_63, int_65)
# Adding element type (line 23)
int_66 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 81), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 74), list_63, int_66)
# Adding element type (line 23)
int_67 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 84), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 74), list_63, int_67)
# Adding element type (line 23)
int_68 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 88), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 74), list_63, int_68)
# Adding element type (line 23)
int_69 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 91), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 74), list_63, int_69)
# Adding element type (line 23)
int_70 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 94), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 74), list_63, int_70)
# Adding element type (line 23)
int_71 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 97), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 74), list_63, int_71)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 18), list_17, list_63)

# Assigning a type to the variable 'affected_cubies' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'affected_cubies', list_17)

# Assigning a List to a Name (line 24):

# Assigning a List to a Name (line 24):

# Obtaining an instance of the builtin type 'list' (line 24)
list_72 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 24)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'list' (line 24)
list_73 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 24)
# Adding element type (line 24)
int_74 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 15), list_73, int_74)
# Adding element type (line 24)
int_75 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 15), list_73, int_75)
# Adding element type (line 24)
int_76 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 15), list_73, int_76)
# Adding element type (line 24)
int_77 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 15), list_73, int_77)
# Adding element type (line 24)
int_78 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 15), list_73, int_78)
# Adding element type (line 24)
int_79 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 15), list_73, int_79)
# Adding element type (line 24)
int_80 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 15), list_73, int_80)
# Adding element type (line 24)
int_81 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 37), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 15), list_73, int_81)
# Adding element type (line 24)
int_82 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 15), list_73, int_82)
# Adding element type (line 24)
int_83 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 15), list_73, int_83)
# Adding element type (line 24)
int_84 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 46), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 15), list_73, int_84)
# Adding element type (line 24)
int_85 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 50), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 15), list_73, int_85)
# Adding element type (line 24)
int_86 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 54), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 15), list_73, int_86)
# Adding element type (line 24)
int_87 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 58), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 15), list_73, int_87)
# Adding element type (line 24)
int_88 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 62), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 15), list_73, int_88)
# Adding element type (line 24)
int_89 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 66), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 15), list_73, int_89)
# Adding element type (line 24)
int_90 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 70), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 15), list_73, int_90)
# Adding element type (line 24)
int_91 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 74), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 15), list_73, int_91)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 14), list_72, list_73)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'list' (line 25)
list_92 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 25)
# Adding element type (line 25)
int_93 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_92, int_93)
# Adding element type (line 25)
int_94 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_92, int_94)
# Adding element type (line 25)
int_95 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_92, int_95)
# Adding element type (line 25)
int_96 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_92, int_96)
# Adding element type (line 25)
int_97 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_92, int_97)
# Adding element type (line 25)
int_98 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_92, int_98)
# Adding element type (line 25)
int_99 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_92, int_99)
# Adding element type (line 25)
int_100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 37), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_92, int_100)
# Adding element type (line 25)
int_101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 41), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_92, int_101)
# Adding element type (line 25)
int_102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 45), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_92, int_102)
# Adding element type (line 25)
int_103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 49), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_92, int_103)
# Adding element type (line 25)
int_104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 53), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_92, int_104)
# Adding element type (line 25)
int_105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 57), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_92, int_105)
# Adding element type (line 25)
int_106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_92, int_106)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 14), list_72, list_92)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'list' (line 25)
list_107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 66), 'list')
# Adding type elements to the builtin type 'list' instance (line 25)
# Adding element type (line 25)
int_108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 67), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 66), list_107, int_108)
# Adding element type (line 25)
int_109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 70), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 66), list_107, int_109)
# Adding element type (line 25)
int_110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 73), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 66), list_107, int_110)
# Adding element type (line 25)
int_111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 76), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 66), list_107, int_111)
# Adding element type (line 25)
int_112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 79), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 66), list_107, int_112)
# Adding element type (line 25)
int_113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 82), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 66), list_107, int_113)
# Adding element type (line 25)
int_114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 85), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 66), list_107, int_114)
# Adding element type (line 25)
int_115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 88), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 66), list_107, int_115)
# Adding element type (line 25)
int_116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 92), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 66), list_107, int_116)
# Adding element type (line 25)
int_117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 96), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 66), list_107, int_117)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 14), list_72, list_107)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'list' (line 26)
list_118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 26)
# Adding element type (line 26)
int_119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 15), list_118, int_119)
# Adding element type (line 26)
int_120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 15), list_118, int_120)
# Adding element type (line 26)
int_121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 15), list_118, int_121)
# Adding element type (line 26)
int_122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 15), list_118, int_122)
# Adding element type (line 26)
int_123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 15), list_118, int_123)
# Adding element type (line 26)
int_124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 15), list_118, int_124)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 14), list_72, list_118)

# Assigning a type to the variable 'phase_moves' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'phase_moves', list_72)

@norecursion
def move_str(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'move_str'
    module_type_store = module_type_store.open_function_context('move_str', 29, 0, False)
    
    # Passed parameters checking function
    move_str.stypy_localization = localization
    move_str.stypy_type_of_self = None
    move_str.stypy_type_store = module_type_store
    move_str.stypy_function_name = 'move_str'
    move_str.stypy_param_names_list = ['move']
    move_str.stypy_varargs_param_name = None
    move_str.stypy_kwargs_param_name = None
    move_str.stypy_call_defaults = defaults
    move_str.stypy_call_varargs = varargs
    move_str.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'move_str', ['move'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'move_str', localization, ['move'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'move_str(...)' code ##################

    
    # Obtaining the type of the subscript
    # Getting the type of 'move' (line 30)
    move_125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 21), 'move')
    int_126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 28), 'int')
    # Applying the binary operator 'div' (line 30)
    result_div_127 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 21), 'div', move_125, int_126)
    
    # Getting the type of 'facenames' (line 30)
    facenames_128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 11), 'facenames')
    # Obtaining the member '__getitem__' of a type (line 30)
    getitem___129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 11), facenames_128, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 30)
    subscript_call_result_130 = invoke(stypy.reporting.localization.Localization(__file__, 30, 11), getitem___129, result_div_127)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'move' (line 30)
    move_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 57), 'move')
    int_132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 64), 'int')
    # Applying the binary operator '%' (line 30)
    result_mod_133 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 57), '%', move_131, int_132)
    
    int_134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 68), 'int')
    # Applying the binary operator '+' (line 30)
    result_add_135 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 57), '+', result_mod_133, int_134)
    
    
    # Obtaining an instance of the builtin type 'dict' (line 30)
    dict_136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 33), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 30)
    # Adding element type (key, value) (line 30)
    int_137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 34), 'int')
    str_138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 37), 'str', '')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 33), dict_136, (int_137, str_138))
    # Adding element type (key, value) (line 30)
    int_139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 41), 'int')
    str_140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 44), 'str', '2')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 33), dict_136, (int_139, str_140))
    # Adding element type (key, value) (line 30)
    int_141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 49), 'int')
    str_142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 52), 'str', "'")
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 33), dict_136, (int_141, str_142))
    
    # Obtaining the member '__getitem__' of a type (line 30)
    getitem___143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 33), dict_136, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 30)
    subscript_call_result_144 = invoke(stypy.reporting.localization.Localization(__file__, 30, 33), getitem___143, result_add_135)
    
    # Applying the binary operator '+' (line 30)
    result_add_145 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 11), '+', subscript_call_result_130, subscript_call_result_144)
    
    # Assigning a type to the variable 'stypy_return_type' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type', result_add_145)
    
    # ################# End of 'move_str(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'move_str' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_146)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'move_str'
    return stypy_return_type_146

# Assigning a type to the variable 'move_str' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'move_str', move_str)
# Declaration of the 'cube_state' class

class cube_state:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 34)
        None_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 36), 'None')
        defaults = [None_147]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 34, 4, False)
        # Assigning a type to the variable 'self' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'cube_state.__init__', ['state', 'route'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['state', 'route'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 35):
        
        # Assigning a Name to a Attribute (line 35):
        # Getting the type of 'state' (line 35)
        state_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 21), 'state')
        # Getting the type of 'self' (line 35)
        self_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'self')
        # Setting the type of the member 'state' of a type (line 35)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), self_149, 'state', state_148)
        
        # Assigning a BoolOp to a Attribute (line 36):
        
        # Assigning a BoolOp to a Attribute (line 36):
        
        # Evaluating a boolean operation
        # Getting the type of 'route' (line 36)
        route_150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 21), 'route')
        
        # Obtaining an instance of the builtin type 'list' (line 36)
        list_151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 36)
        
        # Applying the binary operator 'or' (line 36)
        result_or_keyword_152 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 21), 'or', route_150, list_151)
        
        # Getting the type of 'self' (line 36)
        self_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'self')
        # Setting the type of the member 'route' of a type (line 36)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), self_153, 'route', result_or_keyword_152)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def id_(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'id_'
        module_type_store = module_type_store.open_function_context('id_', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        cube_state.id_.__dict__.__setitem__('stypy_localization', localization)
        cube_state.id_.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        cube_state.id_.__dict__.__setitem__('stypy_type_store', module_type_store)
        cube_state.id_.__dict__.__setitem__('stypy_function_name', 'cube_state.id_')
        cube_state.id_.__dict__.__setitem__('stypy_param_names_list', ['phase'])
        cube_state.id_.__dict__.__setitem__('stypy_varargs_param_name', None)
        cube_state.id_.__dict__.__setitem__('stypy_kwargs_param_name', None)
        cube_state.id_.__dict__.__setitem__('stypy_call_defaults', defaults)
        cube_state.id_.__dict__.__setitem__('stypy_call_varargs', varargs)
        cube_state.id_.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        cube_state.id_.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'cube_state.id_', ['phase'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'id_', localization, ['phase'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'id_(...)' code ##################

        
        # Getting the type of 'phase' (line 39)
        phase_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 11), 'phase')
        int_155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 20), 'int')
        # Applying the binary operator '==' (line 39)
        result_eq_156 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 11), '==', phase_154, int_155)
        
        # Testing if the type of an if condition is none (line 39)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 39, 8), result_eq_156):
            
            # Getting the type of 'phase' (line 41)
            phase_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 13), 'phase')
            int_169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 22), 'int')
            # Applying the binary operator '==' (line 41)
            result_eq_170 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 13), '==', phase_168, int_169)
            
            # Testing if the type of an if condition is none (line 41)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 41, 13), result_eq_170):
                
                # Getting the type of 'phase' (line 46)
                phase_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 13), 'phase')
                int_206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 22), 'int')
                # Applying the binary operator '==' (line 46)
                result_eq_207 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 13), '==', phase_205, int_206)
                
                # Testing if the type of an if condition is none (line 46)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 46, 13), result_eq_207):
                    
                    # Call to tuple(...): (line 57)
                    # Processing the call arguments (line 57)
                    # Getting the type of 'self' (line 57)
                    self_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'self', False)
                    # Obtaining the member 'state' of a type (line 57)
                    state_316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 25), self_315, 'state')
                    # Processing the call keyword arguments (line 57)
                    kwargs_317 = {}
                    # Getting the type of 'tuple' (line 57)
                    tuple_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 19), 'tuple', False)
                    # Calling tuple(args, kwargs) (line 57)
                    tuple_call_result_318 = invoke(stypy.reporting.localization.Localization(__file__, 57, 19), tuple_314, *[state_316], **kwargs_317)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 57)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'stypy_return_type', tuple_call_result_318)
                else:
                    
                    # Testing the type of an if condition (line 46)
                    if_condition_208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 46, 13), result_eq_207)
                    # Assigning a type to the variable 'if_condition_208' (line 46)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 13), 'if_condition_208', if_condition_208)
                    # SSA begins for if statement (line 46)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a List to a Name (line 47):
                    
                    # Assigning a List to a Name (line 47):
                    
                    # Obtaining an instance of the builtin type 'list' (line 47)
                    list_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 21), 'list')
                    # Adding type elements to the builtin type 'list' instance (line 47)
                    # Adding element type (line 47)
                    int_210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 22), 'int')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 21), list_209, int_210)
                    # Adding element type (line 47)
                    int_211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 25), 'int')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 21), list_209, int_211)
                    # Adding element type (line 47)
                    int_212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 28), 'int')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 21), list_209, int_212)
                    
                    # Assigning a type to the variable 'result' (line 47)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'result', list_209)
                    
                    
                    # Call to range(...): (line 48)
                    # Processing the call arguments (line 48)
                    int_214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 27), 'int')
                    # Processing the call keyword arguments (line 48)
                    kwargs_215 = {}
                    # Getting the type of 'range' (line 48)
                    range_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 21), 'range', False)
                    # Calling range(args, kwargs) (line 48)
                    range_call_result_216 = invoke(stypy.reporting.localization.Localization(__file__, 48, 21), range_213, *[int_214], **kwargs_215)
                    
                    # Testing if the for loop is going to be iterated (line 48)
                    # Testing the type of a for loop iterable (line 48)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 48, 12), range_call_result_216)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 48, 12), range_call_result_216):
                        # Getting the type of the for loop variable (line 48)
                        for_loop_var_217 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 48, 12), range_call_result_216)
                        # Assigning a type to the variable 'e' (line 48)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'e', for_loop_var_217)
                        # SSA begins for a for statement (line 48)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Getting the type of 'result' (line 49)
                        result_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'result')
                        
                        # Obtaining the type of the subscript
                        int_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 23), 'int')
                        # Getting the type of 'result' (line 49)
                        result_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'result')
                        # Obtaining the member '__getitem__' of a type (line 49)
                        getitem___221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 16), result_220, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
                        subscript_call_result_222 = invoke(stypy.reporting.localization.Localization(__file__, 49, 16), getitem___221, int_219)
                        
                        
                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'e' (line 49)
                        e_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 47), 'e')
                        # Getting the type of 'self' (line 49)
                        self_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 36), 'self')
                        # Obtaining the member 'state' of a type (line 49)
                        state_225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 36), self_224, 'state')
                        # Obtaining the member '__getitem__' of a type (line 49)
                        getitem___226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 36), state_225, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
                        subscript_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 49, 36), getitem___226, e_223)
                        
                        int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 52), 'int')
                        # Applying the binary operator '>' (line 49)
                        result_gt_229 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 36), '>', subscript_call_result_227, int_228)
                        
                        # Testing the type of an if expression (line 49)
                        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 30), result_gt_229)
                        # SSA begins for if expression (line 49)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
                        int_230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 30), 'int')
                        # SSA branch for the else part of an if expression (line 49)
                        module_type_store.open_ssa_branch('if expression else')
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'e' (line 49)
                        e_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 72), 'e')
                        # Getting the type of 'self' (line 49)
                        self_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 61), 'self')
                        # Obtaining the member 'state' of a type (line 49)
                        state_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 61), self_232, 'state')
                        # Obtaining the member '__getitem__' of a type (line 49)
                        getitem___234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 61), state_233, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
                        subscript_call_result_235 = invoke(stypy.reporting.localization.Localization(__file__, 49, 61), getitem___234, e_231)
                        
                        int_236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 77), 'int')
                        # Applying the binary operator '&' (line 49)
                        result_and__237 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 61), '&', subscript_call_result_235, int_236)
                        
                        # SSA join for if expression (line 49)
                        module_type_store = module_type_store.join_ssa_context()
                        if_exp_238 = union_type.UnionType.add(int_230, result_and__237)
                        
                        int_239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 85), 'int')
                        # Getting the type of 'e' (line 49)
                        e_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 89), 'e')
                        # Applying the binary operator '*' (line 49)
                        result_mul_241 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 85), '*', int_239, e_240)
                        
                        # Applying the binary operator '<<' (line 49)
                        result_lshift_242 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 29), '<<', if_exp_238, result_mul_241)
                        
                        # Applying the binary operator '|=' (line 49)
                        result_ior_243 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 16), '|=', subscript_call_result_222, result_lshift_242)
                        # Getting the type of 'result' (line 49)
                        result_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'result')
                        int_245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 23), 'int')
                        # Storing an element on a container (line 49)
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 16), result_244, (int_245, result_ior_243))
                        
                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    
                    
                    # Call to range(...): (line 50)
                    # Processing the call arguments (line 50)
                    int_247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 27), 'int')
                    # Processing the call keyword arguments (line 50)
                    kwargs_248 = {}
                    # Getting the type of 'range' (line 50)
                    range_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 21), 'range', False)
                    # Calling range(args, kwargs) (line 50)
                    range_call_result_249 = invoke(stypy.reporting.localization.Localization(__file__, 50, 21), range_246, *[int_247], **kwargs_248)
                    
                    # Testing if the for loop is going to be iterated (line 50)
                    # Testing the type of a for loop iterable (line 50)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 50, 12), range_call_result_249)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 50, 12), range_call_result_249):
                        # Getting the type of the for loop variable (line 50)
                        for_loop_var_250 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 50, 12), range_call_result_249)
                        # Assigning a type to the variable 'c' (line 50)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'c', for_loop_var_250)
                        # SSA begins for a for statement (line 50)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Getting the type of 'result' (line 51)
                        result_251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'result')
                        
                        # Obtaining the type of the subscript
                        int_252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 23), 'int')
                        # Getting the type of 'result' (line 51)
                        result_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'result')
                        # Obtaining the member '__getitem__' of a type (line 51)
                        getitem___254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 16), result_253, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
                        subscript_call_result_255 = invoke(stypy.reporting.localization.Localization(__file__, 51, 16), getitem___254, int_252)
                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'c' (line 51)
                        c_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 42), 'c')
                        int_257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 46), 'int')
                        # Applying the binary operator '+' (line 51)
                        result_add_258 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 42), '+', c_256, int_257)
                        
                        # Getting the type of 'self' (line 51)
                        self_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 31), 'self')
                        # Obtaining the member 'state' of a type (line 51)
                        state_260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 31), self_259, 'state')
                        # Obtaining the member '__getitem__' of a type (line 51)
                        getitem___261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 31), state_260, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
                        subscript_call_result_262 = invoke(stypy.reporting.localization.Localization(__file__, 51, 31), getitem___261, result_add_258)
                        
                        int_263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 52), 'int')
                        # Applying the binary operator '-' (line 51)
                        result_sub_264 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 31), '-', subscript_call_result_262, int_263)
                        
                        int_265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 58), 'int')
                        # Applying the binary operator '&' (line 51)
                        result_and__266 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 30), '&', result_sub_264, int_265)
                        
                        int_267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 65), 'int')
                        # Getting the type of 'c' (line 51)
                        c_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 69), 'c')
                        # Applying the binary operator '*' (line 51)
                        result_mul_269 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 65), '*', int_267, c_268)
                        
                        # Applying the binary operator '<<' (line 51)
                        result_lshift_270 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 29), '<<', result_and__266, result_mul_269)
                        
                        # Applying the binary operator '|=' (line 51)
                        result_ior_271 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 16), '|=', subscript_call_result_255, result_lshift_270)
                        # Getting the type of 'result' (line 51)
                        result_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'result')
                        int_273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 23), 'int')
                        # Storing an element on a container (line 51)
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 16), result_272, (int_273, result_ior_271))
                        
                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    
                    
                    # Call to range(...): (line 52)
                    # Processing the call arguments (line 52)
                    int_275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 27), 'int')
                    int_276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 31), 'int')
                    # Processing the call keyword arguments (line 52)
                    kwargs_277 = {}
                    # Getting the type of 'range' (line 52)
                    range_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 21), 'range', False)
                    # Calling range(args, kwargs) (line 52)
                    range_call_result_278 = invoke(stypy.reporting.localization.Localization(__file__, 52, 21), range_274, *[int_275, int_276], **kwargs_277)
                    
                    # Testing if the for loop is going to be iterated (line 52)
                    # Testing the type of a for loop iterable (line 52)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 52, 12), range_call_result_278)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 52, 12), range_call_result_278):
                        # Getting the type of the for loop variable (line 52)
                        for_loop_var_279 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 52, 12), range_call_result_278)
                        # Assigning a type to the variable 'i' (line 52)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'i', for_loop_var_279)
                        # SSA begins for a for statement (line 52)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        
                        # Call to range(...): (line 53)
                        # Processing the call arguments (line 53)
                        # Getting the type of 'i' (line 53)
                        i_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 31), 'i', False)
                        int_282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 35), 'int')
                        # Applying the binary operator '+' (line 53)
                        result_add_283 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 31), '+', i_281, int_282)
                        
                        int_284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 38), 'int')
                        # Processing the call keyword arguments (line 53)
                        kwargs_285 = {}
                        # Getting the type of 'range' (line 53)
                        range_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 25), 'range', False)
                        # Calling range(args, kwargs) (line 53)
                        range_call_result_286 = invoke(stypy.reporting.localization.Localization(__file__, 53, 25), range_280, *[result_add_283, int_284], **kwargs_285)
                        
                        # Testing if the for loop is going to be iterated (line 53)
                        # Testing the type of a for loop iterable (line 53)
                        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 53, 16), range_call_result_286)

                        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 53, 16), range_call_result_286):
                            # Getting the type of the for loop variable (line 53)
                            for_loop_var_287 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 53, 16), range_call_result_286)
                            # Assigning a type to the variable 'j' (line 53)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 16), 'j', for_loop_var_287)
                            # SSA begins for a for statement (line 53)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                            
                            # Getting the type of 'result' (line 54)
                            result_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'result')
                            
                            # Obtaining the type of the subscript
                            int_289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 27), 'int')
                            # Getting the type of 'result' (line 54)
                            result_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'result')
                            # Obtaining the member '__getitem__' of a type (line 54)
                            getitem___291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 20), result_290, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 54)
                            subscript_call_result_292 = invoke(stypy.reporting.localization.Localization(__file__, 54, 20), getitem___291, int_289)
                            
                            
                            # Call to int(...): (line 54)
                            # Processing the call arguments (line 54)
                            
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'i' (line 54)
                            i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 48), 'i', False)
                            # Getting the type of 'self' (line 54)
                            self_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 37), 'self', False)
                            # Obtaining the member 'state' of a type (line 54)
                            state_296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 37), self_295, 'state')
                            # Obtaining the member '__getitem__' of a type (line 54)
                            getitem___297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 37), state_296, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 54)
                            subscript_call_result_298 = invoke(stypy.reporting.localization.Localization(__file__, 54, 37), getitem___297, i_294)
                            
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'j' (line 54)
                            j_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 64), 'j', False)
                            # Getting the type of 'self' (line 54)
                            self_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 53), 'self', False)
                            # Obtaining the member 'state' of a type (line 54)
                            state_301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 53), self_300, 'state')
                            # Obtaining the member '__getitem__' of a type (line 54)
                            getitem___302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 53), state_301, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 54)
                            subscript_call_result_303 = invoke(stypy.reporting.localization.Localization(__file__, 54, 53), getitem___302, j_299)
                            
                            # Applying the binary operator '>' (line 54)
                            result_gt_304 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 37), '>', subscript_call_result_298, subscript_call_result_303)
                            
                            # Processing the call keyword arguments (line 54)
                            kwargs_305 = {}
                            # Getting the type of 'int' (line 54)
                            int_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 33), 'int', False)
                            # Calling int(args, kwargs) (line 54)
                            int_call_result_306 = invoke(stypy.reporting.localization.Localization(__file__, 54, 33), int_293, *[result_gt_304], **kwargs_305)
                            
                            # Applying the binary operator '^=' (line 54)
                            result_ixor_307 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 20), '^=', subscript_call_result_292, int_call_result_306)
                            # Getting the type of 'result' (line 54)
                            result_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'result')
                            int_309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 27), 'int')
                            # Storing an element on a container (line 54)
                            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 20), result_308, (int_309, result_ixor_307))
                            
                            # SSA join for a for statement
                            module_type_store = module_type_store.join_ssa_context()

                        
                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    
                    # Call to tuple(...): (line 55)
                    # Processing the call arguments (line 55)
                    # Getting the type of 'result' (line 55)
                    result_311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 25), 'result', False)
                    # Processing the call keyword arguments (line 55)
                    kwargs_312 = {}
                    # Getting the type of 'tuple' (line 55)
                    tuple_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 19), 'tuple', False)
                    # Calling tuple(args, kwargs) (line 55)
                    tuple_call_result_313 = invoke(stypy.reporting.localization.Localization(__file__, 55, 19), tuple_310, *[result_311], **kwargs_312)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 55)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'stypy_return_type', tuple_call_result_313)
                    # SSA branch for the else part of an if statement (line 46)
                    module_type_store.open_ssa_branch('else')
                    
                    # Call to tuple(...): (line 57)
                    # Processing the call arguments (line 57)
                    # Getting the type of 'self' (line 57)
                    self_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'self', False)
                    # Obtaining the member 'state' of a type (line 57)
                    state_316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 25), self_315, 'state')
                    # Processing the call keyword arguments (line 57)
                    kwargs_317 = {}
                    # Getting the type of 'tuple' (line 57)
                    tuple_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 19), 'tuple', False)
                    # Calling tuple(args, kwargs) (line 57)
                    tuple_call_result_318 = invoke(stypy.reporting.localization.Localization(__file__, 57, 19), tuple_314, *[state_316], **kwargs_317)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 57)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'stypy_return_type', tuple_call_result_318)
                    # SSA join for if statement (line 46)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 41)
                if_condition_171 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 13), result_eq_170)
                # Assigning a type to the variable 'if_condition_171' (line 41)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 13), 'if_condition_171', if_condition_171)
                # SSA begins for if statement (line 41)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Subscript to a Name (line 42):
                
                # Assigning a Subscript to a Name (line 42):
                
                # Obtaining the type of the subscript
                int_172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 32), 'int')
                int_173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 35), 'int')
                slice_174 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 42, 21), int_172, int_173, None)
                # Getting the type of 'self' (line 42)
                self_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 21), 'self')
                # Obtaining the member 'state' of a type (line 42)
                state_176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 21), self_175, 'state')
                # Obtaining the member '__getitem__' of a type (line 42)
                getitem___177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 21), state_176, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 42)
                subscript_call_result_178 = invoke(stypy.reporting.localization.Localization(__file__, 42, 21), getitem___177, slice_174)
                
                # Assigning a type to the variable 'result' (line 42)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'result', subscript_call_result_178)
                
                
                # Call to range(...): (line 43)
                # Processing the call arguments (line 43)
                int_180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 27), 'int')
                # Processing the call keyword arguments (line 43)
                kwargs_181 = {}
                # Getting the type of 'range' (line 43)
                range_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 21), 'range', False)
                # Calling range(args, kwargs) (line 43)
                range_call_result_182 = invoke(stypy.reporting.localization.Localization(__file__, 43, 21), range_179, *[int_180], **kwargs_181)
                
                # Testing if the for loop is going to be iterated (line 43)
                # Testing the type of a for loop iterable (line 43)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 43, 12), range_call_result_182)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 43, 12), range_call_result_182):
                    # Getting the type of the for loop variable (line 43)
                    for_loop_var_183 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 43, 12), range_call_result_182)
                    # Assigning a type to the variable 'e' (line 43)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'e', for_loop_var_183)
                    # SSA begins for a for statement (line 43)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Getting the type of 'result' (line 44)
                    result_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), 'result')
                    
                    # Obtaining the type of the subscript
                    int_185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 23), 'int')
                    # Getting the type of 'result' (line 44)
                    result_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), 'result')
                    # Obtaining the member '__getitem__' of a type (line 44)
                    getitem___187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 16), result_186, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
                    subscript_call_result_188 = invoke(stypy.reporting.localization.Localization(__file__, 44, 16), getitem___187, int_185)
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'e' (line 44)
                    e_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 41), 'e')
                    # Getting the type of 'self' (line 44)
                    self_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 30), 'self')
                    # Obtaining the member 'state' of a type (line 44)
                    state_191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 30), self_190, 'state')
                    # Obtaining the member '__getitem__' of a type (line 44)
                    getitem___192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 30), state_191, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
                    subscript_call_result_193 = invoke(stypy.reporting.localization.Localization(__file__, 44, 30), getitem___192, e_189)
                    
                    int_194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 46), 'int')
                    # Applying the binary operator 'div' (line 44)
                    result_div_195 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 30), 'div', subscript_call_result_193, int_194)
                    
                    # Getting the type of 'e' (line 44)
                    e_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 52), 'e')
                    # Applying the binary operator '<<' (line 44)
                    result_lshift_197 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 29), '<<', result_div_195, e_196)
                    
                    # Applying the binary operator '|=' (line 44)
                    result_ior_198 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 16), '|=', subscript_call_result_188, result_lshift_197)
                    # Getting the type of 'result' (line 44)
                    result_199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), 'result')
                    int_200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 23), 'int')
                    # Storing an element on a container (line 44)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 16), result_199, (int_200, result_ior_198))
                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Call to tuple(...): (line 45)
                # Processing the call arguments (line 45)
                # Getting the type of 'result' (line 45)
                result_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 25), 'result', False)
                # Processing the call keyword arguments (line 45)
                kwargs_203 = {}
                # Getting the type of 'tuple' (line 45)
                tuple_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 19), 'tuple', False)
                # Calling tuple(args, kwargs) (line 45)
                tuple_call_result_204 = invoke(stypy.reporting.localization.Localization(__file__, 45, 19), tuple_201, *[result_202], **kwargs_203)
                
                # Assigning a type to the variable 'stypy_return_type' (line 45)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'stypy_return_type', tuple_call_result_204)
                # SSA branch for the else part of an if statement (line 41)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'phase' (line 46)
                phase_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 13), 'phase')
                int_206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 22), 'int')
                # Applying the binary operator '==' (line 46)
                result_eq_207 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 13), '==', phase_205, int_206)
                
                # Testing if the type of an if condition is none (line 46)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 46, 13), result_eq_207):
                    
                    # Call to tuple(...): (line 57)
                    # Processing the call arguments (line 57)
                    # Getting the type of 'self' (line 57)
                    self_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'self', False)
                    # Obtaining the member 'state' of a type (line 57)
                    state_316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 25), self_315, 'state')
                    # Processing the call keyword arguments (line 57)
                    kwargs_317 = {}
                    # Getting the type of 'tuple' (line 57)
                    tuple_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 19), 'tuple', False)
                    # Calling tuple(args, kwargs) (line 57)
                    tuple_call_result_318 = invoke(stypy.reporting.localization.Localization(__file__, 57, 19), tuple_314, *[state_316], **kwargs_317)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 57)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'stypy_return_type', tuple_call_result_318)
                else:
                    
                    # Testing the type of an if condition (line 46)
                    if_condition_208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 46, 13), result_eq_207)
                    # Assigning a type to the variable 'if_condition_208' (line 46)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 13), 'if_condition_208', if_condition_208)
                    # SSA begins for if statement (line 46)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a List to a Name (line 47):
                    
                    # Assigning a List to a Name (line 47):
                    
                    # Obtaining an instance of the builtin type 'list' (line 47)
                    list_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 21), 'list')
                    # Adding type elements to the builtin type 'list' instance (line 47)
                    # Adding element type (line 47)
                    int_210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 22), 'int')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 21), list_209, int_210)
                    # Adding element type (line 47)
                    int_211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 25), 'int')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 21), list_209, int_211)
                    # Adding element type (line 47)
                    int_212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 28), 'int')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 21), list_209, int_212)
                    
                    # Assigning a type to the variable 'result' (line 47)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'result', list_209)
                    
                    
                    # Call to range(...): (line 48)
                    # Processing the call arguments (line 48)
                    int_214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 27), 'int')
                    # Processing the call keyword arguments (line 48)
                    kwargs_215 = {}
                    # Getting the type of 'range' (line 48)
                    range_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 21), 'range', False)
                    # Calling range(args, kwargs) (line 48)
                    range_call_result_216 = invoke(stypy.reporting.localization.Localization(__file__, 48, 21), range_213, *[int_214], **kwargs_215)
                    
                    # Testing if the for loop is going to be iterated (line 48)
                    # Testing the type of a for loop iterable (line 48)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 48, 12), range_call_result_216)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 48, 12), range_call_result_216):
                        # Getting the type of the for loop variable (line 48)
                        for_loop_var_217 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 48, 12), range_call_result_216)
                        # Assigning a type to the variable 'e' (line 48)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'e', for_loop_var_217)
                        # SSA begins for a for statement (line 48)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Getting the type of 'result' (line 49)
                        result_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'result')
                        
                        # Obtaining the type of the subscript
                        int_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 23), 'int')
                        # Getting the type of 'result' (line 49)
                        result_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'result')
                        # Obtaining the member '__getitem__' of a type (line 49)
                        getitem___221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 16), result_220, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
                        subscript_call_result_222 = invoke(stypy.reporting.localization.Localization(__file__, 49, 16), getitem___221, int_219)
                        
                        
                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'e' (line 49)
                        e_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 47), 'e')
                        # Getting the type of 'self' (line 49)
                        self_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 36), 'self')
                        # Obtaining the member 'state' of a type (line 49)
                        state_225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 36), self_224, 'state')
                        # Obtaining the member '__getitem__' of a type (line 49)
                        getitem___226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 36), state_225, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
                        subscript_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 49, 36), getitem___226, e_223)
                        
                        int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 52), 'int')
                        # Applying the binary operator '>' (line 49)
                        result_gt_229 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 36), '>', subscript_call_result_227, int_228)
                        
                        # Testing the type of an if expression (line 49)
                        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 30), result_gt_229)
                        # SSA begins for if expression (line 49)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
                        int_230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 30), 'int')
                        # SSA branch for the else part of an if expression (line 49)
                        module_type_store.open_ssa_branch('if expression else')
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'e' (line 49)
                        e_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 72), 'e')
                        # Getting the type of 'self' (line 49)
                        self_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 61), 'self')
                        # Obtaining the member 'state' of a type (line 49)
                        state_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 61), self_232, 'state')
                        # Obtaining the member '__getitem__' of a type (line 49)
                        getitem___234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 61), state_233, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
                        subscript_call_result_235 = invoke(stypy.reporting.localization.Localization(__file__, 49, 61), getitem___234, e_231)
                        
                        int_236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 77), 'int')
                        # Applying the binary operator '&' (line 49)
                        result_and__237 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 61), '&', subscript_call_result_235, int_236)
                        
                        # SSA join for if expression (line 49)
                        module_type_store = module_type_store.join_ssa_context()
                        if_exp_238 = union_type.UnionType.add(int_230, result_and__237)
                        
                        int_239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 85), 'int')
                        # Getting the type of 'e' (line 49)
                        e_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 89), 'e')
                        # Applying the binary operator '*' (line 49)
                        result_mul_241 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 85), '*', int_239, e_240)
                        
                        # Applying the binary operator '<<' (line 49)
                        result_lshift_242 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 29), '<<', if_exp_238, result_mul_241)
                        
                        # Applying the binary operator '|=' (line 49)
                        result_ior_243 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 16), '|=', subscript_call_result_222, result_lshift_242)
                        # Getting the type of 'result' (line 49)
                        result_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'result')
                        int_245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 23), 'int')
                        # Storing an element on a container (line 49)
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 16), result_244, (int_245, result_ior_243))
                        
                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    
                    
                    # Call to range(...): (line 50)
                    # Processing the call arguments (line 50)
                    int_247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 27), 'int')
                    # Processing the call keyword arguments (line 50)
                    kwargs_248 = {}
                    # Getting the type of 'range' (line 50)
                    range_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 21), 'range', False)
                    # Calling range(args, kwargs) (line 50)
                    range_call_result_249 = invoke(stypy.reporting.localization.Localization(__file__, 50, 21), range_246, *[int_247], **kwargs_248)
                    
                    # Testing if the for loop is going to be iterated (line 50)
                    # Testing the type of a for loop iterable (line 50)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 50, 12), range_call_result_249)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 50, 12), range_call_result_249):
                        # Getting the type of the for loop variable (line 50)
                        for_loop_var_250 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 50, 12), range_call_result_249)
                        # Assigning a type to the variable 'c' (line 50)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'c', for_loop_var_250)
                        # SSA begins for a for statement (line 50)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Getting the type of 'result' (line 51)
                        result_251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'result')
                        
                        # Obtaining the type of the subscript
                        int_252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 23), 'int')
                        # Getting the type of 'result' (line 51)
                        result_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'result')
                        # Obtaining the member '__getitem__' of a type (line 51)
                        getitem___254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 16), result_253, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
                        subscript_call_result_255 = invoke(stypy.reporting.localization.Localization(__file__, 51, 16), getitem___254, int_252)
                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'c' (line 51)
                        c_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 42), 'c')
                        int_257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 46), 'int')
                        # Applying the binary operator '+' (line 51)
                        result_add_258 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 42), '+', c_256, int_257)
                        
                        # Getting the type of 'self' (line 51)
                        self_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 31), 'self')
                        # Obtaining the member 'state' of a type (line 51)
                        state_260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 31), self_259, 'state')
                        # Obtaining the member '__getitem__' of a type (line 51)
                        getitem___261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 31), state_260, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
                        subscript_call_result_262 = invoke(stypy.reporting.localization.Localization(__file__, 51, 31), getitem___261, result_add_258)
                        
                        int_263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 52), 'int')
                        # Applying the binary operator '-' (line 51)
                        result_sub_264 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 31), '-', subscript_call_result_262, int_263)
                        
                        int_265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 58), 'int')
                        # Applying the binary operator '&' (line 51)
                        result_and__266 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 30), '&', result_sub_264, int_265)
                        
                        int_267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 65), 'int')
                        # Getting the type of 'c' (line 51)
                        c_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 69), 'c')
                        # Applying the binary operator '*' (line 51)
                        result_mul_269 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 65), '*', int_267, c_268)
                        
                        # Applying the binary operator '<<' (line 51)
                        result_lshift_270 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 29), '<<', result_and__266, result_mul_269)
                        
                        # Applying the binary operator '|=' (line 51)
                        result_ior_271 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 16), '|=', subscript_call_result_255, result_lshift_270)
                        # Getting the type of 'result' (line 51)
                        result_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'result')
                        int_273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 23), 'int')
                        # Storing an element on a container (line 51)
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 16), result_272, (int_273, result_ior_271))
                        
                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    
                    
                    # Call to range(...): (line 52)
                    # Processing the call arguments (line 52)
                    int_275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 27), 'int')
                    int_276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 31), 'int')
                    # Processing the call keyword arguments (line 52)
                    kwargs_277 = {}
                    # Getting the type of 'range' (line 52)
                    range_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 21), 'range', False)
                    # Calling range(args, kwargs) (line 52)
                    range_call_result_278 = invoke(stypy.reporting.localization.Localization(__file__, 52, 21), range_274, *[int_275, int_276], **kwargs_277)
                    
                    # Testing if the for loop is going to be iterated (line 52)
                    # Testing the type of a for loop iterable (line 52)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 52, 12), range_call_result_278)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 52, 12), range_call_result_278):
                        # Getting the type of the for loop variable (line 52)
                        for_loop_var_279 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 52, 12), range_call_result_278)
                        # Assigning a type to the variable 'i' (line 52)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'i', for_loop_var_279)
                        # SSA begins for a for statement (line 52)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        
                        # Call to range(...): (line 53)
                        # Processing the call arguments (line 53)
                        # Getting the type of 'i' (line 53)
                        i_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 31), 'i', False)
                        int_282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 35), 'int')
                        # Applying the binary operator '+' (line 53)
                        result_add_283 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 31), '+', i_281, int_282)
                        
                        int_284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 38), 'int')
                        # Processing the call keyword arguments (line 53)
                        kwargs_285 = {}
                        # Getting the type of 'range' (line 53)
                        range_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 25), 'range', False)
                        # Calling range(args, kwargs) (line 53)
                        range_call_result_286 = invoke(stypy.reporting.localization.Localization(__file__, 53, 25), range_280, *[result_add_283, int_284], **kwargs_285)
                        
                        # Testing if the for loop is going to be iterated (line 53)
                        # Testing the type of a for loop iterable (line 53)
                        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 53, 16), range_call_result_286)

                        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 53, 16), range_call_result_286):
                            # Getting the type of the for loop variable (line 53)
                            for_loop_var_287 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 53, 16), range_call_result_286)
                            # Assigning a type to the variable 'j' (line 53)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 16), 'j', for_loop_var_287)
                            # SSA begins for a for statement (line 53)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                            
                            # Getting the type of 'result' (line 54)
                            result_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'result')
                            
                            # Obtaining the type of the subscript
                            int_289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 27), 'int')
                            # Getting the type of 'result' (line 54)
                            result_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'result')
                            # Obtaining the member '__getitem__' of a type (line 54)
                            getitem___291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 20), result_290, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 54)
                            subscript_call_result_292 = invoke(stypy.reporting.localization.Localization(__file__, 54, 20), getitem___291, int_289)
                            
                            
                            # Call to int(...): (line 54)
                            # Processing the call arguments (line 54)
                            
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'i' (line 54)
                            i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 48), 'i', False)
                            # Getting the type of 'self' (line 54)
                            self_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 37), 'self', False)
                            # Obtaining the member 'state' of a type (line 54)
                            state_296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 37), self_295, 'state')
                            # Obtaining the member '__getitem__' of a type (line 54)
                            getitem___297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 37), state_296, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 54)
                            subscript_call_result_298 = invoke(stypy.reporting.localization.Localization(__file__, 54, 37), getitem___297, i_294)
                            
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'j' (line 54)
                            j_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 64), 'j', False)
                            # Getting the type of 'self' (line 54)
                            self_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 53), 'self', False)
                            # Obtaining the member 'state' of a type (line 54)
                            state_301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 53), self_300, 'state')
                            # Obtaining the member '__getitem__' of a type (line 54)
                            getitem___302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 53), state_301, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 54)
                            subscript_call_result_303 = invoke(stypy.reporting.localization.Localization(__file__, 54, 53), getitem___302, j_299)
                            
                            # Applying the binary operator '>' (line 54)
                            result_gt_304 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 37), '>', subscript_call_result_298, subscript_call_result_303)
                            
                            # Processing the call keyword arguments (line 54)
                            kwargs_305 = {}
                            # Getting the type of 'int' (line 54)
                            int_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 33), 'int', False)
                            # Calling int(args, kwargs) (line 54)
                            int_call_result_306 = invoke(stypy.reporting.localization.Localization(__file__, 54, 33), int_293, *[result_gt_304], **kwargs_305)
                            
                            # Applying the binary operator '^=' (line 54)
                            result_ixor_307 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 20), '^=', subscript_call_result_292, int_call_result_306)
                            # Getting the type of 'result' (line 54)
                            result_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'result')
                            int_309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 27), 'int')
                            # Storing an element on a container (line 54)
                            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 20), result_308, (int_309, result_ixor_307))
                            
                            # SSA join for a for statement
                            module_type_store = module_type_store.join_ssa_context()

                        
                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    
                    # Call to tuple(...): (line 55)
                    # Processing the call arguments (line 55)
                    # Getting the type of 'result' (line 55)
                    result_311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 25), 'result', False)
                    # Processing the call keyword arguments (line 55)
                    kwargs_312 = {}
                    # Getting the type of 'tuple' (line 55)
                    tuple_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 19), 'tuple', False)
                    # Calling tuple(args, kwargs) (line 55)
                    tuple_call_result_313 = invoke(stypy.reporting.localization.Localization(__file__, 55, 19), tuple_310, *[result_311], **kwargs_312)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 55)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'stypy_return_type', tuple_call_result_313)
                    # SSA branch for the else part of an if statement (line 46)
                    module_type_store.open_ssa_branch('else')
                    
                    # Call to tuple(...): (line 57)
                    # Processing the call arguments (line 57)
                    # Getting the type of 'self' (line 57)
                    self_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'self', False)
                    # Obtaining the member 'state' of a type (line 57)
                    state_316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 25), self_315, 'state')
                    # Processing the call keyword arguments (line 57)
                    kwargs_317 = {}
                    # Getting the type of 'tuple' (line 57)
                    tuple_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 19), 'tuple', False)
                    # Calling tuple(args, kwargs) (line 57)
                    tuple_call_result_318 = invoke(stypy.reporting.localization.Localization(__file__, 57, 19), tuple_314, *[state_316], **kwargs_317)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 57)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'stypy_return_type', tuple_call_result_318)
                    # SSA join for if statement (line 46)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 41)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 39)
            if_condition_157 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 8), result_eq_156)
            # Assigning a type to the variable 'if_condition_157' (line 39)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'if_condition_157', if_condition_157)
            # SSA begins for if statement (line 39)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to tuple(...): (line 40)
            # Processing the call arguments (line 40)
            
            # Obtaining the type of the subscript
            int_159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 36), 'int')
            int_160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 39), 'int')
            slice_161 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 40, 25), int_159, int_160, None)
            # Getting the type of 'self' (line 40)
            self_162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 25), 'self', False)
            # Obtaining the member 'state' of a type (line 40)
            state_163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 25), self_162, 'state')
            # Obtaining the member '__getitem__' of a type (line 40)
            getitem___164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 25), state_163, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 40)
            subscript_call_result_165 = invoke(stypy.reporting.localization.Localization(__file__, 40, 25), getitem___164, slice_161)
            
            # Processing the call keyword arguments (line 40)
            kwargs_166 = {}
            # Getting the type of 'tuple' (line 40)
            tuple_158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'tuple', False)
            # Calling tuple(args, kwargs) (line 40)
            tuple_call_result_167 = invoke(stypy.reporting.localization.Localization(__file__, 40, 19), tuple_158, *[subscript_call_result_165], **kwargs_166)
            
            # Assigning a type to the variable 'stypy_return_type' (line 40)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'stypy_return_type', tuple_call_result_167)
            # SSA branch for the else part of an if statement (line 39)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'phase' (line 41)
            phase_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 13), 'phase')
            int_169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 22), 'int')
            # Applying the binary operator '==' (line 41)
            result_eq_170 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 13), '==', phase_168, int_169)
            
            # Testing if the type of an if condition is none (line 41)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 41, 13), result_eq_170):
                
                # Getting the type of 'phase' (line 46)
                phase_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 13), 'phase')
                int_206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 22), 'int')
                # Applying the binary operator '==' (line 46)
                result_eq_207 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 13), '==', phase_205, int_206)
                
                # Testing if the type of an if condition is none (line 46)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 46, 13), result_eq_207):
                    
                    # Call to tuple(...): (line 57)
                    # Processing the call arguments (line 57)
                    # Getting the type of 'self' (line 57)
                    self_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'self', False)
                    # Obtaining the member 'state' of a type (line 57)
                    state_316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 25), self_315, 'state')
                    # Processing the call keyword arguments (line 57)
                    kwargs_317 = {}
                    # Getting the type of 'tuple' (line 57)
                    tuple_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 19), 'tuple', False)
                    # Calling tuple(args, kwargs) (line 57)
                    tuple_call_result_318 = invoke(stypy.reporting.localization.Localization(__file__, 57, 19), tuple_314, *[state_316], **kwargs_317)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 57)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'stypy_return_type', tuple_call_result_318)
                else:
                    
                    # Testing the type of an if condition (line 46)
                    if_condition_208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 46, 13), result_eq_207)
                    # Assigning a type to the variable 'if_condition_208' (line 46)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 13), 'if_condition_208', if_condition_208)
                    # SSA begins for if statement (line 46)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a List to a Name (line 47):
                    
                    # Assigning a List to a Name (line 47):
                    
                    # Obtaining an instance of the builtin type 'list' (line 47)
                    list_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 21), 'list')
                    # Adding type elements to the builtin type 'list' instance (line 47)
                    # Adding element type (line 47)
                    int_210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 22), 'int')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 21), list_209, int_210)
                    # Adding element type (line 47)
                    int_211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 25), 'int')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 21), list_209, int_211)
                    # Adding element type (line 47)
                    int_212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 28), 'int')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 21), list_209, int_212)
                    
                    # Assigning a type to the variable 'result' (line 47)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'result', list_209)
                    
                    
                    # Call to range(...): (line 48)
                    # Processing the call arguments (line 48)
                    int_214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 27), 'int')
                    # Processing the call keyword arguments (line 48)
                    kwargs_215 = {}
                    # Getting the type of 'range' (line 48)
                    range_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 21), 'range', False)
                    # Calling range(args, kwargs) (line 48)
                    range_call_result_216 = invoke(stypy.reporting.localization.Localization(__file__, 48, 21), range_213, *[int_214], **kwargs_215)
                    
                    # Testing if the for loop is going to be iterated (line 48)
                    # Testing the type of a for loop iterable (line 48)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 48, 12), range_call_result_216)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 48, 12), range_call_result_216):
                        # Getting the type of the for loop variable (line 48)
                        for_loop_var_217 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 48, 12), range_call_result_216)
                        # Assigning a type to the variable 'e' (line 48)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'e', for_loop_var_217)
                        # SSA begins for a for statement (line 48)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Getting the type of 'result' (line 49)
                        result_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'result')
                        
                        # Obtaining the type of the subscript
                        int_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 23), 'int')
                        # Getting the type of 'result' (line 49)
                        result_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'result')
                        # Obtaining the member '__getitem__' of a type (line 49)
                        getitem___221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 16), result_220, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
                        subscript_call_result_222 = invoke(stypy.reporting.localization.Localization(__file__, 49, 16), getitem___221, int_219)
                        
                        
                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'e' (line 49)
                        e_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 47), 'e')
                        # Getting the type of 'self' (line 49)
                        self_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 36), 'self')
                        # Obtaining the member 'state' of a type (line 49)
                        state_225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 36), self_224, 'state')
                        # Obtaining the member '__getitem__' of a type (line 49)
                        getitem___226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 36), state_225, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
                        subscript_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 49, 36), getitem___226, e_223)
                        
                        int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 52), 'int')
                        # Applying the binary operator '>' (line 49)
                        result_gt_229 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 36), '>', subscript_call_result_227, int_228)
                        
                        # Testing the type of an if expression (line 49)
                        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 30), result_gt_229)
                        # SSA begins for if expression (line 49)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
                        int_230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 30), 'int')
                        # SSA branch for the else part of an if expression (line 49)
                        module_type_store.open_ssa_branch('if expression else')
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'e' (line 49)
                        e_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 72), 'e')
                        # Getting the type of 'self' (line 49)
                        self_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 61), 'self')
                        # Obtaining the member 'state' of a type (line 49)
                        state_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 61), self_232, 'state')
                        # Obtaining the member '__getitem__' of a type (line 49)
                        getitem___234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 61), state_233, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
                        subscript_call_result_235 = invoke(stypy.reporting.localization.Localization(__file__, 49, 61), getitem___234, e_231)
                        
                        int_236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 77), 'int')
                        # Applying the binary operator '&' (line 49)
                        result_and__237 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 61), '&', subscript_call_result_235, int_236)
                        
                        # SSA join for if expression (line 49)
                        module_type_store = module_type_store.join_ssa_context()
                        if_exp_238 = union_type.UnionType.add(int_230, result_and__237)
                        
                        int_239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 85), 'int')
                        # Getting the type of 'e' (line 49)
                        e_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 89), 'e')
                        # Applying the binary operator '*' (line 49)
                        result_mul_241 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 85), '*', int_239, e_240)
                        
                        # Applying the binary operator '<<' (line 49)
                        result_lshift_242 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 29), '<<', if_exp_238, result_mul_241)
                        
                        # Applying the binary operator '|=' (line 49)
                        result_ior_243 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 16), '|=', subscript_call_result_222, result_lshift_242)
                        # Getting the type of 'result' (line 49)
                        result_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'result')
                        int_245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 23), 'int')
                        # Storing an element on a container (line 49)
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 16), result_244, (int_245, result_ior_243))
                        
                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    
                    
                    # Call to range(...): (line 50)
                    # Processing the call arguments (line 50)
                    int_247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 27), 'int')
                    # Processing the call keyword arguments (line 50)
                    kwargs_248 = {}
                    # Getting the type of 'range' (line 50)
                    range_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 21), 'range', False)
                    # Calling range(args, kwargs) (line 50)
                    range_call_result_249 = invoke(stypy.reporting.localization.Localization(__file__, 50, 21), range_246, *[int_247], **kwargs_248)
                    
                    # Testing if the for loop is going to be iterated (line 50)
                    # Testing the type of a for loop iterable (line 50)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 50, 12), range_call_result_249)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 50, 12), range_call_result_249):
                        # Getting the type of the for loop variable (line 50)
                        for_loop_var_250 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 50, 12), range_call_result_249)
                        # Assigning a type to the variable 'c' (line 50)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'c', for_loop_var_250)
                        # SSA begins for a for statement (line 50)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Getting the type of 'result' (line 51)
                        result_251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'result')
                        
                        # Obtaining the type of the subscript
                        int_252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 23), 'int')
                        # Getting the type of 'result' (line 51)
                        result_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'result')
                        # Obtaining the member '__getitem__' of a type (line 51)
                        getitem___254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 16), result_253, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
                        subscript_call_result_255 = invoke(stypy.reporting.localization.Localization(__file__, 51, 16), getitem___254, int_252)
                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'c' (line 51)
                        c_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 42), 'c')
                        int_257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 46), 'int')
                        # Applying the binary operator '+' (line 51)
                        result_add_258 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 42), '+', c_256, int_257)
                        
                        # Getting the type of 'self' (line 51)
                        self_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 31), 'self')
                        # Obtaining the member 'state' of a type (line 51)
                        state_260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 31), self_259, 'state')
                        # Obtaining the member '__getitem__' of a type (line 51)
                        getitem___261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 31), state_260, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
                        subscript_call_result_262 = invoke(stypy.reporting.localization.Localization(__file__, 51, 31), getitem___261, result_add_258)
                        
                        int_263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 52), 'int')
                        # Applying the binary operator '-' (line 51)
                        result_sub_264 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 31), '-', subscript_call_result_262, int_263)
                        
                        int_265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 58), 'int')
                        # Applying the binary operator '&' (line 51)
                        result_and__266 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 30), '&', result_sub_264, int_265)
                        
                        int_267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 65), 'int')
                        # Getting the type of 'c' (line 51)
                        c_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 69), 'c')
                        # Applying the binary operator '*' (line 51)
                        result_mul_269 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 65), '*', int_267, c_268)
                        
                        # Applying the binary operator '<<' (line 51)
                        result_lshift_270 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 29), '<<', result_and__266, result_mul_269)
                        
                        # Applying the binary operator '|=' (line 51)
                        result_ior_271 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 16), '|=', subscript_call_result_255, result_lshift_270)
                        # Getting the type of 'result' (line 51)
                        result_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'result')
                        int_273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 23), 'int')
                        # Storing an element on a container (line 51)
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 16), result_272, (int_273, result_ior_271))
                        
                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    
                    
                    # Call to range(...): (line 52)
                    # Processing the call arguments (line 52)
                    int_275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 27), 'int')
                    int_276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 31), 'int')
                    # Processing the call keyword arguments (line 52)
                    kwargs_277 = {}
                    # Getting the type of 'range' (line 52)
                    range_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 21), 'range', False)
                    # Calling range(args, kwargs) (line 52)
                    range_call_result_278 = invoke(stypy.reporting.localization.Localization(__file__, 52, 21), range_274, *[int_275, int_276], **kwargs_277)
                    
                    # Testing if the for loop is going to be iterated (line 52)
                    # Testing the type of a for loop iterable (line 52)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 52, 12), range_call_result_278)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 52, 12), range_call_result_278):
                        # Getting the type of the for loop variable (line 52)
                        for_loop_var_279 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 52, 12), range_call_result_278)
                        # Assigning a type to the variable 'i' (line 52)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'i', for_loop_var_279)
                        # SSA begins for a for statement (line 52)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        
                        # Call to range(...): (line 53)
                        # Processing the call arguments (line 53)
                        # Getting the type of 'i' (line 53)
                        i_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 31), 'i', False)
                        int_282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 35), 'int')
                        # Applying the binary operator '+' (line 53)
                        result_add_283 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 31), '+', i_281, int_282)
                        
                        int_284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 38), 'int')
                        # Processing the call keyword arguments (line 53)
                        kwargs_285 = {}
                        # Getting the type of 'range' (line 53)
                        range_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 25), 'range', False)
                        # Calling range(args, kwargs) (line 53)
                        range_call_result_286 = invoke(stypy.reporting.localization.Localization(__file__, 53, 25), range_280, *[result_add_283, int_284], **kwargs_285)
                        
                        # Testing if the for loop is going to be iterated (line 53)
                        # Testing the type of a for loop iterable (line 53)
                        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 53, 16), range_call_result_286)

                        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 53, 16), range_call_result_286):
                            # Getting the type of the for loop variable (line 53)
                            for_loop_var_287 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 53, 16), range_call_result_286)
                            # Assigning a type to the variable 'j' (line 53)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 16), 'j', for_loop_var_287)
                            # SSA begins for a for statement (line 53)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                            
                            # Getting the type of 'result' (line 54)
                            result_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'result')
                            
                            # Obtaining the type of the subscript
                            int_289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 27), 'int')
                            # Getting the type of 'result' (line 54)
                            result_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'result')
                            # Obtaining the member '__getitem__' of a type (line 54)
                            getitem___291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 20), result_290, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 54)
                            subscript_call_result_292 = invoke(stypy.reporting.localization.Localization(__file__, 54, 20), getitem___291, int_289)
                            
                            
                            # Call to int(...): (line 54)
                            # Processing the call arguments (line 54)
                            
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'i' (line 54)
                            i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 48), 'i', False)
                            # Getting the type of 'self' (line 54)
                            self_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 37), 'self', False)
                            # Obtaining the member 'state' of a type (line 54)
                            state_296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 37), self_295, 'state')
                            # Obtaining the member '__getitem__' of a type (line 54)
                            getitem___297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 37), state_296, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 54)
                            subscript_call_result_298 = invoke(stypy.reporting.localization.Localization(__file__, 54, 37), getitem___297, i_294)
                            
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'j' (line 54)
                            j_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 64), 'j', False)
                            # Getting the type of 'self' (line 54)
                            self_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 53), 'self', False)
                            # Obtaining the member 'state' of a type (line 54)
                            state_301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 53), self_300, 'state')
                            # Obtaining the member '__getitem__' of a type (line 54)
                            getitem___302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 53), state_301, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 54)
                            subscript_call_result_303 = invoke(stypy.reporting.localization.Localization(__file__, 54, 53), getitem___302, j_299)
                            
                            # Applying the binary operator '>' (line 54)
                            result_gt_304 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 37), '>', subscript_call_result_298, subscript_call_result_303)
                            
                            # Processing the call keyword arguments (line 54)
                            kwargs_305 = {}
                            # Getting the type of 'int' (line 54)
                            int_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 33), 'int', False)
                            # Calling int(args, kwargs) (line 54)
                            int_call_result_306 = invoke(stypy.reporting.localization.Localization(__file__, 54, 33), int_293, *[result_gt_304], **kwargs_305)
                            
                            # Applying the binary operator '^=' (line 54)
                            result_ixor_307 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 20), '^=', subscript_call_result_292, int_call_result_306)
                            # Getting the type of 'result' (line 54)
                            result_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'result')
                            int_309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 27), 'int')
                            # Storing an element on a container (line 54)
                            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 20), result_308, (int_309, result_ixor_307))
                            
                            # SSA join for a for statement
                            module_type_store = module_type_store.join_ssa_context()

                        
                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    
                    # Call to tuple(...): (line 55)
                    # Processing the call arguments (line 55)
                    # Getting the type of 'result' (line 55)
                    result_311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 25), 'result', False)
                    # Processing the call keyword arguments (line 55)
                    kwargs_312 = {}
                    # Getting the type of 'tuple' (line 55)
                    tuple_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 19), 'tuple', False)
                    # Calling tuple(args, kwargs) (line 55)
                    tuple_call_result_313 = invoke(stypy.reporting.localization.Localization(__file__, 55, 19), tuple_310, *[result_311], **kwargs_312)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 55)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'stypy_return_type', tuple_call_result_313)
                    # SSA branch for the else part of an if statement (line 46)
                    module_type_store.open_ssa_branch('else')
                    
                    # Call to tuple(...): (line 57)
                    # Processing the call arguments (line 57)
                    # Getting the type of 'self' (line 57)
                    self_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'self', False)
                    # Obtaining the member 'state' of a type (line 57)
                    state_316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 25), self_315, 'state')
                    # Processing the call keyword arguments (line 57)
                    kwargs_317 = {}
                    # Getting the type of 'tuple' (line 57)
                    tuple_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 19), 'tuple', False)
                    # Calling tuple(args, kwargs) (line 57)
                    tuple_call_result_318 = invoke(stypy.reporting.localization.Localization(__file__, 57, 19), tuple_314, *[state_316], **kwargs_317)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 57)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'stypy_return_type', tuple_call_result_318)
                    # SSA join for if statement (line 46)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 41)
                if_condition_171 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 13), result_eq_170)
                # Assigning a type to the variable 'if_condition_171' (line 41)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 13), 'if_condition_171', if_condition_171)
                # SSA begins for if statement (line 41)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Subscript to a Name (line 42):
                
                # Assigning a Subscript to a Name (line 42):
                
                # Obtaining the type of the subscript
                int_172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 32), 'int')
                int_173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 35), 'int')
                slice_174 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 42, 21), int_172, int_173, None)
                # Getting the type of 'self' (line 42)
                self_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 21), 'self')
                # Obtaining the member 'state' of a type (line 42)
                state_176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 21), self_175, 'state')
                # Obtaining the member '__getitem__' of a type (line 42)
                getitem___177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 21), state_176, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 42)
                subscript_call_result_178 = invoke(stypy.reporting.localization.Localization(__file__, 42, 21), getitem___177, slice_174)
                
                # Assigning a type to the variable 'result' (line 42)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'result', subscript_call_result_178)
                
                
                # Call to range(...): (line 43)
                # Processing the call arguments (line 43)
                int_180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 27), 'int')
                # Processing the call keyword arguments (line 43)
                kwargs_181 = {}
                # Getting the type of 'range' (line 43)
                range_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 21), 'range', False)
                # Calling range(args, kwargs) (line 43)
                range_call_result_182 = invoke(stypy.reporting.localization.Localization(__file__, 43, 21), range_179, *[int_180], **kwargs_181)
                
                # Testing if the for loop is going to be iterated (line 43)
                # Testing the type of a for loop iterable (line 43)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 43, 12), range_call_result_182)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 43, 12), range_call_result_182):
                    # Getting the type of the for loop variable (line 43)
                    for_loop_var_183 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 43, 12), range_call_result_182)
                    # Assigning a type to the variable 'e' (line 43)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'e', for_loop_var_183)
                    # SSA begins for a for statement (line 43)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Getting the type of 'result' (line 44)
                    result_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), 'result')
                    
                    # Obtaining the type of the subscript
                    int_185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 23), 'int')
                    # Getting the type of 'result' (line 44)
                    result_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), 'result')
                    # Obtaining the member '__getitem__' of a type (line 44)
                    getitem___187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 16), result_186, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
                    subscript_call_result_188 = invoke(stypy.reporting.localization.Localization(__file__, 44, 16), getitem___187, int_185)
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'e' (line 44)
                    e_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 41), 'e')
                    # Getting the type of 'self' (line 44)
                    self_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 30), 'self')
                    # Obtaining the member 'state' of a type (line 44)
                    state_191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 30), self_190, 'state')
                    # Obtaining the member '__getitem__' of a type (line 44)
                    getitem___192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 30), state_191, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
                    subscript_call_result_193 = invoke(stypy.reporting.localization.Localization(__file__, 44, 30), getitem___192, e_189)
                    
                    int_194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 46), 'int')
                    # Applying the binary operator 'div' (line 44)
                    result_div_195 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 30), 'div', subscript_call_result_193, int_194)
                    
                    # Getting the type of 'e' (line 44)
                    e_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 52), 'e')
                    # Applying the binary operator '<<' (line 44)
                    result_lshift_197 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 29), '<<', result_div_195, e_196)
                    
                    # Applying the binary operator '|=' (line 44)
                    result_ior_198 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 16), '|=', subscript_call_result_188, result_lshift_197)
                    # Getting the type of 'result' (line 44)
                    result_199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), 'result')
                    int_200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 23), 'int')
                    # Storing an element on a container (line 44)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 16), result_199, (int_200, result_ior_198))
                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Call to tuple(...): (line 45)
                # Processing the call arguments (line 45)
                # Getting the type of 'result' (line 45)
                result_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 25), 'result', False)
                # Processing the call keyword arguments (line 45)
                kwargs_203 = {}
                # Getting the type of 'tuple' (line 45)
                tuple_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 19), 'tuple', False)
                # Calling tuple(args, kwargs) (line 45)
                tuple_call_result_204 = invoke(stypy.reporting.localization.Localization(__file__, 45, 19), tuple_201, *[result_202], **kwargs_203)
                
                # Assigning a type to the variable 'stypy_return_type' (line 45)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'stypy_return_type', tuple_call_result_204)
                # SSA branch for the else part of an if statement (line 41)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'phase' (line 46)
                phase_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 13), 'phase')
                int_206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 22), 'int')
                # Applying the binary operator '==' (line 46)
                result_eq_207 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 13), '==', phase_205, int_206)
                
                # Testing if the type of an if condition is none (line 46)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 46, 13), result_eq_207):
                    
                    # Call to tuple(...): (line 57)
                    # Processing the call arguments (line 57)
                    # Getting the type of 'self' (line 57)
                    self_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'self', False)
                    # Obtaining the member 'state' of a type (line 57)
                    state_316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 25), self_315, 'state')
                    # Processing the call keyword arguments (line 57)
                    kwargs_317 = {}
                    # Getting the type of 'tuple' (line 57)
                    tuple_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 19), 'tuple', False)
                    # Calling tuple(args, kwargs) (line 57)
                    tuple_call_result_318 = invoke(stypy.reporting.localization.Localization(__file__, 57, 19), tuple_314, *[state_316], **kwargs_317)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 57)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'stypy_return_type', tuple_call_result_318)
                else:
                    
                    # Testing the type of an if condition (line 46)
                    if_condition_208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 46, 13), result_eq_207)
                    # Assigning a type to the variable 'if_condition_208' (line 46)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 13), 'if_condition_208', if_condition_208)
                    # SSA begins for if statement (line 46)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a List to a Name (line 47):
                    
                    # Assigning a List to a Name (line 47):
                    
                    # Obtaining an instance of the builtin type 'list' (line 47)
                    list_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 21), 'list')
                    # Adding type elements to the builtin type 'list' instance (line 47)
                    # Adding element type (line 47)
                    int_210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 22), 'int')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 21), list_209, int_210)
                    # Adding element type (line 47)
                    int_211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 25), 'int')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 21), list_209, int_211)
                    # Adding element type (line 47)
                    int_212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 28), 'int')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 21), list_209, int_212)
                    
                    # Assigning a type to the variable 'result' (line 47)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'result', list_209)
                    
                    
                    # Call to range(...): (line 48)
                    # Processing the call arguments (line 48)
                    int_214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 27), 'int')
                    # Processing the call keyword arguments (line 48)
                    kwargs_215 = {}
                    # Getting the type of 'range' (line 48)
                    range_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 21), 'range', False)
                    # Calling range(args, kwargs) (line 48)
                    range_call_result_216 = invoke(stypy.reporting.localization.Localization(__file__, 48, 21), range_213, *[int_214], **kwargs_215)
                    
                    # Testing if the for loop is going to be iterated (line 48)
                    # Testing the type of a for loop iterable (line 48)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 48, 12), range_call_result_216)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 48, 12), range_call_result_216):
                        # Getting the type of the for loop variable (line 48)
                        for_loop_var_217 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 48, 12), range_call_result_216)
                        # Assigning a type to the variable 'e' (line 48)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'e', for_loop_var_217)
                        # SSA begins for a for statement (line 48)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Getting the type of 'result' (line 49)
                        result_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'result')
                        
                        # Obtaining the type of the subscript
                        int_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 23), 'int')
                        # Getting the type of 'result' (line 49)
                        result_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'result')
                        # Obtaining the member '__getitem__' of a type (line 49)
                        getitem___221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 16), result_220, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
                        subscript_call_result_222 = invoke(stypy.reporting.localization.Localization(__file__, 49, 16), getitem___221, int_219)
                        
                        
                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'e' (line 49)
                        e_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 47), 'e')
                        # Getting the type of 'self' (line 49)
                        self_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 36), 'self')
                        # Obtaining the member 'state' of a type (line 49)
                        state_225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 36), self_224, 'state')
                        # Obtaining the member '__getitem__' of a type (line 49)
                        getitem___226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 36), state_225, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
                        subscript_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 49, 36), getitem___226, e_223)
                        
                        int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 52), 'int')
                        # Applying the binary operator '>' (line 49)
                        result_gt_229 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 36), '>', subscript_call_result_227, int_228)
                        
                        # Testing the type of an if expression (line 49)
                        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 30), result_gt_229)
                        # SSA begins for if expression (line 49)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
                        int_230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 30), 'int')
                        # SSA branch for the else part of an if expression (line 49)
                        module_type_store.open_ssa_branch('if expression else')
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'e' (line 49)
                        e_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 72), 'e')
                        # Getting the type of 'self' (line 49)
                        self_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 61), 'self')
                        # Obtaining the member 'state' of a type (line 49)
                        state_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 61), self_232, 'state')
                        # Obtaining the member '__getitem__' of a type (line 49)
                        getitem___234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 61), state_233, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
                        subscript_call_result_235 = invoke(stypy.reporting.localization.Localization(__file__, 49, 61), getitem___234, e_231)
                        
                        int_236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 77), 'int')
                        # Applying the binary operator '&' (line 49)
                        result_and__237 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 61), '&', subscript_call_result_235, int_236)
                        
                        # SSA join for if expression (line 49)
                        module_type_store = module_type_store.join_ssa_context()
                        if_exp_238 = union_type.UnionType.add(int_230, result_and__237)
                        
                        int_239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 85), 'int')
                        # Getting the type of 'e' (line 49)
                        e_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 89), 'e')
                        # Applying the binary operator '*' (line 49)
                        result_mul_241 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 85), '*', int_239, e_240)
                        
                        # Applying the binary operator '<<' (line 49)
                        result_lshift_242 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 29), '<<', if_exp_238, result_mul_241)
                        
                        # Applying the binary operator '|=' (line 49)
                        result_ior_243 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 16), '|=', subscript_call_result_222, result_lshift_242)
                        # Getting the type of 'result' (line 49)
                        result_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'result')
                        int_245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 23), 'int')
                        # Storing an element on a container (line 49)
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 16), result_244, (int_245, result_ior_243))
                        
                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    
                    
                    # Call to range(...): (line 50)
                    # Processing the call arguments (line 50)
                    int_247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 27), 'int')
                    # Processing the call keyword arguments (line 50)
                    kwargs_248 = {}
                    # Getting the type of 'range' (line 50)
                    range_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 21), 'range', False)
                    # Calling range(args, kwargs) (line 50)
                    range_call_result_249 = invoke(stypy.reporting.localization.Localization(__file__, 50, 21), range_246, *[int_247], **kwargs_248)
                    
                    # Testing if the for loop is going to be iterated (line 50)
                    # Testing the type of a for loop iterable (line 50)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 50, 12), range_call_result_249)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 50, 12), range_call_result_249):
                        # Getting the type of the for loop variable (line 50)
                        for_loop_var_250 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 50, 12), range_call_result_249)
                        # Assigning a type to the variable 'c' (line 50)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'c', for_loop_var_250)
                        # SSA begins for a for statement (line 50)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Getting the type of 'result' (line 51)
                        result_251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'result')
                        
                        # Obtaining the type of the subscript
                        int_252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 23), 'int')
                        # Getting the type of 'result' (line 51)
                        result_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'result')
                        # Obtaining the member '__getitem__' of a type (line 51)
                        getitem___254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 16), result_253, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
                        subscript_call_result_255 = invoke(stypy.reporting.localization.Localization(__file__, 51, 16), getitem___254, int_252)
                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'c' (line 51)
                        c_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 42), 'c')
                        int_257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 46), 'int')
                        # Applying the binary operator '+' (line 51)
                        result_add_258 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 42), '+', c_256, int_257)
                        
                        # Getting the type of 'self' (line 51)
                        self_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 31), 'self')
                        # Obtaining the member 'state' of a type (line 51)
                        state_260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 31), self_259, 'state')
                        # Obtaining the member '__getitem__' of a type (line 51)
                        getitem___261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 31), state_260, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
                        subscript_call_result_262 = invoke(stypy.reporting.localization.Localization(__file__, 51, 31), getitem___261, result_add_258)
                        
                        int_263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 52), 'int')
                        # Applying the binary operator '-' (line 51)
                        result_sub_264 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 31), '-', subscript_call_result_262, int_263)
                        
                        int_265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 58), 'int')
                        # Applying the binary operator '&' (line 51)
                        result_and__266 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 30), '&', result_sub_264, int_265)
                        
                        int_267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 65), 'int')
                        # Getting the type of 'c' (line 51)
                        c_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 69), 'c')
                        # Applying the binary operator '*' (line 51)
                        result_mul_269 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 65), '*', int_267, c_268)
                        
                        # Applying the binary operator '<<' (line 51)
                        result_lshift_270 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 29), '<<', result_and__266, result_mul_269)
                        
                        # Applying the binary operator '|=' (line 51)
                        result_ior_271 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 16), '|=', subscript_call_result_255, result_lshift_270)
                        # Getting the type of 'result' (line 51)
                        result_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'result')
                        int_273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 23), 'int')
                        # Storing an element on a container (line 51)
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 16), result_272, (int_273, result_ior_271))
                        
                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    
                    
                    # Call to range(...): (line 52)
                    # Processing the call arguments (line 52)
                    int_275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 27), 'int')
                    int_276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 31), 'int')
                    # Processing the call keyword arguments (line 52)
                    kwargs_277 = {}
                    # Getting the type of 'range' (line 52)
                    range_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 21), 'range', False)
                    # Calling range(args, kwargs) (line 52)
                    range_call_result_278 = invoke(stypy.reporting.localization.Localization(__file__, 52, 21), range_274, *[int_275, int_276], **kwargs_277)
                    
                    # Testing if the for loop is going to be iterated (line 52)
                    # Testing the type of a for loop iterable (line 52)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 52, 12), range_call_result_278)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 52, 12), range_call_result_278):
                        # Getting the type of the for loop variable (line 52)
                        for_loop_var_279 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 52, 12), range_call_result_278)
                        # Assigning a type to the variable 'i' (line 52)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'i', for_loop_var_279)
                        # SSA begins for a for statement (line 52)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        
                        # Call to range(...): (line 53)
                        # Processing the call arguments (line 53)
                        # Getting the type of 'i' (line 53)
                        i_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 31), 'i', False)
                        int_282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 35), 'int')
                        # Applying the binary operator '+' (line 53)
                        result_add_283 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 31), '+', i_281, int_282)
                        
                        int_284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 38), 'int')
                        # Processing the call keyword arguments (line 53)
                        kwargs_285 = {}
                        # Getting the type of 'range' (line 53)
                        range_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 25), 'range', False)
                        # Calling range(args, kwargs) (line 53)
                        range_call_result_286 = invoke(stypy.reporting.localization.Localization(__file__, 53, 25), range_280, *[result_add_283, int_284], **kwargs_285)
                        
                        # Testing if the for loop is going to be iterated (line 53)
                        # Testing the type of a for loop iterable (line 53)
                        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 53, 16), range_call_result_286)

                        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 53, 16), range_call_result_286):
                            # Getting the type of the for loop variable (line 53)
                            for_loop_var_287 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 53, 16), range_call_result_286)
                            # Assigning a type to the variable 'j' (line 53)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 16), 'j', for_loop_var_287)
                            # SSA begins for a for statement (line 53)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                            
                            # Getting the type of 'result' (line 54)
                            result_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'result')
                            
                            # Obtaining the type of the subscript
                            int_289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 27), 'int')
                            # Getting the type of 'result' (line 54)
                            result_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'result')
                            # Obtaining the member '__getitem__' of a type (line 54)
                            getitem___291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 20), result_290, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 54)
                            subscript_call_result_292 = invoke(stypy.reporting.localization.Localization(__file__, 54, 20), getitem___291, int_289)
                            
                            
                            # Call to int(...): (line 54)
                            # Processing the call arguments (line 54)
                            
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'i' (line 54)
                            i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 48), 'i', False)
                            # Getting the type of 'self' (line 54)
                            self_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 37), 'self', False)
                            # Obtaining the member 'state' of a type (line 54)
                            state_296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 37), self_295, 'state')
                            # Obtaining the member '__getitem__' of a type (line 54)
                            getitem___297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 37), state_296, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 54)
                            subscript_call_result_298 = invoke(stypy.reporting.localization.Localization(__file__, 54, 37), getitem___297, i_294)
                            
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'j' (line 54)
                            j_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 64), 'j', False)
                            # Getting the type of 'self' (line 54)
                            self_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 53), 'self', False)
                            # Obtaining the member 'state' of a type (line 54)
                            state_301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 53), self_300, 'state')
                            # Obtaining the member '__getitem__' of a type (line 54)
                            getitem___302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 53), state_301, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 54)
                            subscript_call_result_303 = invoke(stypy.reporting.localization.Localization(__file__, 54, 53), getitem___302, j_299)
                            
                            # Applying the binary operator '>' (line 54)
                            result_gt_304 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 37), '>', subscript_call_result_298, subscript_call_result_303)
                            
                            # Processing the call keyword arguments (line 54)
                            kwargs_305 = {}
                            # Getting the type of 'int' (line 54)
                            int_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 33), 'int', False)
                            # Calling int(args, kwargs) (line 54)
                            int_call_result_306 = invoke(stypy.reporting.localization.Localization(__file__, 54, 33), int_293, *[result_gt_304], **kwargs_305)
                            
                            # Applying the binary operator '^=' (line 54)
                            result_ixor_307 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 20), '^=', subscript_call_result_292, int_call_result_306)
                            # Getting the type of 'result' (line 54)
                            result_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'result')
                            int_309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 27), 'int')
                            # Storing an element on a container (line 54)
                            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 20), result_308, (int_309, result_ixor_307))
                            
                            # SSA join for a for statement
                            module_type_store = module_type_store.join_ssa_context()

                        
                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    
                    # Call to tuple(...): (line 55)
                    # Processing the call arguments (line 55)
                    # Getting the type of 'result' (line 55)
                    result_311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 25), 'result', False)
                    # Processing the call keyword arguments (line 55)
                    kwargs_312 = {}
                    # Getting the type of 'tuple' (line 55)
                    tuple_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 19), 'tuple', False)
                    # Calling tuple(args, kwargs) (line 55)
                    tuple_call_result_313 = invoke(stypy.reporting.localization.Localization(__file__, 55, 19), tuple_310, *[result_311], **kwargs_312)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 55)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'stypy_return_type', tuple_call_result_313)
                    # SSA branch for the else part of an if statement (line 46)
                    module_type_store.open_ssa_branch('else')
                    
                    # Call to tuple(...): (line 57)
                    # Processing the call arguments (line 57)
                    # Getting the type of 'self' (line 57)
                    self_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'self', False)
                    # Obtaining the member 'state' of a type (line 57)
                    state_316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 25), self_315, 'state')
                    # Processing the call keyword arguments (line 57)
                    kwargs_317 = {}
                    # Getting the type of 'tuple' (line 57)
                    tuple_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 19), 'tuple', False)
                    # Calling tuple(args, kwargs) (line 57)
                    tuple_call_result_318 = invoke(stypy.reporting.localization.Localization(__file__, 57, 19), tuple_314, *[state_316], **kwargs_317)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 57)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'stypy_return_type', tuple_call_result_318)
                    # SSA join for if statement (line 46)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 41)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 39)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'id_(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'id_' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_319)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'id_'
        return stypy_return_type_319


    @norecursion
    def apply_move(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'apply_move'
        module_type_store = module_type_store.open_function_context('apply_move', 59, 4, False)
        # Assigning a type to the variable 'self' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        cube_state.apply_move.__dict__.__setitem__('stypy_localization', localization)
        cube_state.apply_move.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        cube_state.apply_move.__dict__.__setitem__('stypy_type_store', module_type_store)
        cube_state.apply_move.__dict__.__setitem__('stypy_function_name', 'cube_state.apply_move')
        cube_state.apply_move.__dict__.__setitem__('stypy_param_names_list', ['move'])
        cube_state.apply_move.__dict__.__setitem__('stypy_varargs_param_name', None)
        cube_state.apply_move.__dict__.__setitem__('stypy_kwargs_param_name', None)
        cube_state.apply_move.__dict__.__setitem__('stypy_call_defaults', defaults)
        cube_state.apply_move.__dict__.__setitem__('stypy_call_varargs', varargs)
        cube_state.apply_move.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        cube_state.apply_move.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'cube_state.apply_move', ['move'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'apply_move', localization, ['move'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'apply_move(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 60):
        
        # Assigning a BinOp to a Name (line 60):
        # Getting the type of 'move' (line 60)
        move_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 22), 'move')
        int_321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 29), 'int')
        # Applying the binary operator 'div' (line 60)
        result_div_322 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 22), 'div', move_320, int_321)
        
        # Assigning a type to the variable 'tuple_assignment_1' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'tuple_assignment_1', result_div_322)
        
        # Assigning a BinOp to a Name (line 60):
        # Getting the type of 'move' (line 60)
        move_323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 32), 'move')
        int_324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 39), 'int')
        # Applying the binary operator '%' (line 60)
        result_mod_325 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 32), '%', move_323, int_324)
        
        int_326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 43), 'int')
        # Applying the binary operator '+' (line 60)
        result_add_327 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 32), '+', result_mod_325, int_326)
        
        # Assigning a type to the variable 'tuple_assignment_2' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'tuple_assignment_2', result_add_327)
        
        # Assigning a Name to a Name (line 60):
        # Getting the type of 'tuple_assignment_1' (line 60)
        tuple_assignment_1_328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'tuple_assignment_1')
        # Assigning a type to the variable 'face' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'face', tuple_assignment_1_328)
        
        # Assigning a Name to a Name (line 60):
        # Getting the type of 'tuple_assignment_2' (line 60)
        tuple_assignment_2_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'tuple_assignment_2')
        # Assigning a type to the variable 'turns' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 14), 'turns', tuple_assignment_2_329)
        
        # Assigning a Subscript to a Name (line 61):
        
        # Assigning a Subscript to a Name (line 61):
        
        # Obtaining the type of the subscript
        slice_330 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 61, 19), None, None, None)
        # Getting the type of 'self' (line 61)
        self_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 19), 'self')
        # Obtaining the member 'state' of a type (line 61)
        state_332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 19), self_331, 'state')
        # Obtaining the member '__getitem__' of a type (line 61)
        getitem___333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 19), state_332, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 61)
        subscript_call_result_334 = invoke(stypy.reporting.localization.Localization(__file__, 61, 19), getitem___333, slice_330)
        
        # Assigning a type to the variable 'newstate' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'newstate', subscript_call_result_334)
        
        
        # Call to range(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'turns' (line 62)
        turns_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 26), 'turns', False)
        # Processing the call keyword arguments (line 62)
        kwargs_337 = {}
        # Getting the type of 'range' (line 62)
        range_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'range', False)
        # Calling range(args, kwargs) (line 62)
        range_call_result_338 = invoke(stypy.reporting.localization.Localization(__file__, 62, 20), range_335, *[turns_336], **kwargs_337)
        
        # Testing if the for loop is going to be iterated (line 62)
        # Testing the type of a for loop iterable (line 62)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 62, 8), range_call_result_338)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 62, 8), range_call_result_338):
            # Getting the type of the for loop variable (line 62)
            for_loop_var_339 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 62, 8), range_call_result_338)
            # Assigning a type to the variable 'turn' (line 62)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'turn', for_loop_var_339)
            # SSA begins for a for statement (line 62)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Name (line 63):
            
            # Assigning a Subscript to a Name (line 63):
            
            # Obtaining the type of the subscript
            slice_340 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 63, 23), None, None, None)
            # Getting the type of 'newstate' (line 63)
            newstate_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 23), 'newstate')
            # Obtaining the member '__getitem__' of a type (line 63)
            getitem___342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 23), newstate_341, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 63)
            subscript_call_result_343 = invoke(stypy.reporting.localization.Localization(__file__, 63, 23), getitem___342, slice_340)
            
            # Assigning a type to the variable 'oldstate' (line 63)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'oldstate', subscript_call_result_343)
            
            
            # Call to range(...): (line 64)
            # Processing the call arguments (line 64)
            int_345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 27), 'int')
            # Processing the call keyword arguments (line 64)
            kwargs_346 = {}
            # Getting the type of 'range' (line 64)
            range_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 21), 'range', False)
            # Calling range(args, kwargs) (line 64)
            range_call_result_347 = invoke(stypy.reporting.localization.Localization(__file__, 64, 21), range_344, *[int_345], **kwargs_346)
            
            # Testing if the for loop is going to be iterated (line 64)
            # Testing the type of a for loop iterable (line 64)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 64, 12), range_call_result_347)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 64, 12), range_call_result_347):
                # Getting the type of the for loop variable (line 64)
                for_loop_var_348 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 64, 12), range_call_result_347)
                # Assigning a type to the variable 'i' (line 64)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'i', for_loop_var_348)
                # SSA begins for a for statement (line 64)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Call to a Name (line 65):
                
                # Assigning a Call to a Name (line 65):
                
                # Call to int(...): (line 65)
                # Processing the call arguments (line 65)
                
                # Getting the type of 'i' (line 65)
                i_350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 31), 'i', False)
                int_351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 35), 'int')
                # Applying the binary operator '>' (line 65)
                result_gt_352 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 31), '>', i_350, int_351)
                
                # Processing the call keyword arguments (line 65)
                kwargs_353 = {}
                # Getting the type of 'int' (line 65)
                int_349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 27), 'int', False)
                # Calling int(args, kwargs) (line 65)
                int_call_result_354 = invoke(stypy.reporting.localization.Localization(__file__, 65, 27), int_349, *[result_gt_352], **kwargs_353)
                
                # Assigning a type to the variable 'isCorner' (line 65)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'isCorner', int_call_result_354)
                
                # Assigning a BinOp to a Name (line 66):
                
                # Assigning a BinOp to a Name (line 66):
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 66)
                i_355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 47), 'i')
                
                # Obtaining the type of the subscript
                # Getting the type of 'face' (line 66)
                face_356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 41), 'face')
                # Getting the type of 'affected_cubies' (line 66)
                affected_cubies_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'affected_cubies')
                # Obtaining the member '__getitem__' of a type (line 66)
                getitem___358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 25), affected_cubies_357, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 66)
                subscript_call_result_359 = invoke(stypy.reporting.localization.Localization(__file__, 66, 25), getitem___358, face_356)
                
                # Obtaining the member '__getitem__' of a type (line 66)
                getitem___360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 25), subscript_call_result_359, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 66)
                subscript_call_result_361 = invoke(stypy.reporting.localization.Localization(__file__, 66, 25), getitem___360, i_355)
                
                # Getting the type of 'isCorner' (line 66)
                isCorner_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 52), 'isCorner')
                int_363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 63), 'int')
                # Applying the binary operator '*' (line 66)
                result_mul_364 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 52), '*', isCorner_362, int_363)
                
                # Applying the binary operator '+' (line 66)
                result_add_365 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 25), '+', subscript_call_result_361, result_mul_364)
                
                # Assigning a type to the variable 'target' (line 66)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 16), 'target', result_add_365)
                
                # Assigning a BinOp to a Name (line 67):
                
                # Assigning a BinOp to a Name (line 67):
                
                # Obtaining the type of the subscript
                
                
                # Getting the type of 'i' (line 67)
                i_366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 59), 'i')
                int_367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 63), 'int')
                # Applying the binary operator '&' (line 67)
                result_and__368 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 59), '&', i_366, int_367)
                
                int_369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 69), 'int')
                # Applying the binary operator '==' (line 67)
                result_eq_370 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 58), '==', result_and__368, int_369)
                
                # Testing the type of an if expression (line 67)
                is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 47), result_eq_370)
                # SSA begins for if expression (line 67)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
                # Getting the type of 'i' (line 67)
                i_371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 48), 'i')
                int_372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 52), 'int')
                # Applying the binary operator '-' (line 67)
                result_sub_373 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 48), '-', i_371, int_372)
                
                # SSA branch for the else part of an if expression (line 67)
                module_type_store.open_ssa_branch('if expression else')
                # Getting the type of 'i' (line 67)
                i_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 76), 'i')
                int_375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 80), 'int')
                # Applying the binary operator '+' (line 67)
                result_add_376 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 76), '+', i_374, int_375)
                
                # SSA join for if expression (line 67)
                module_type_store = module_type_store.join_ssa_context()
                if_exp_377 = union_type.UnionType.add(result_sub_373, result_add_376)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'face' (line 67)
                face_378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 41), 'face')
                # Getting the type of 'affected_cubies' (line 67)
                affected_cubies_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 25), 'affected_cubies')
                # Obtaining the member '__getitem__' of a type (line 67)
                getitem___380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 25), affected_cubies_379, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 67)
                subscript_call_result_381 = invoke(stypy.reporting.localization.Localization(__file__, 67, 25), getitem___380, face_378)
                
                # Obtaining the member '__getitem__' of a type (line 67)
                getitem___382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 25), subscript_call_result_381, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 67)
                subscript_call_result_383 = invoke(stypy.reporting.localization.Localization(__file__, 67, 25), getitem___382, if_exp_377)
                
                # Getting the type of 'isCorner' (line 67)
                isCorner_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 85), 'isCorner')
                int_385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 96), 'int')
                # Applying the binary operator '*' (line 67)
                result_mul_386 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 85), '*', isCorner_384, int_385)
                
                # Applying the binary operator '+' (line 67)
                result_add_387 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 25), '+', subscript_call_result_383, result_mul_386)
                
                # Assigning a type to the variable 'killer' (line 67)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 16), 'killer', result_add_387)
                
                # Assigning a IfExp to a Name (line 68):
                
                # Assigning a IfExp to a Name (line 68):
                
                
                # Getting the type of 'i' (line 68)
                i_388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 56), 'i')
                int_389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 60), 'int')
                # Applying the binary operator '<' (line 68)
                result_lt_390 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 56), '<', i_388, int_389)
                
                # Testing the type of an if expression (line 68)
                is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 35), result_lt_390)
                # SSA begins for if expression (line 68)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
                
                # Call to int(...): (line 68)
                # Processing the call arguments (line 68)
                
                int_392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 39), 'int')
                # Getting the type of 'face' (line 68)
                face_393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 43), 'face', False)
                # Applying the binary operator '<' (line 68)
                result_lt_394 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 39), '<', int_392, face_393)
                int_395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 50), 'int')
                # Applying the binary operator '<' (line 68)
                result_lt_396 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 39), '<', face_393, int_395)
                # Applying the binary operator '&' (line 68)
                result_and__397 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 39), '&', result_lt_394, result_lt_396)
                
                # Processing the call keyword arguments (line 68)
                kwargs_398 = {}
                # Getting the type of 'int' (line 68)
                int_391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 35), 'int', False)
                # Calling int(args, kwargs) (line 68)
                int_call_result_399 = invoke(stypy.reporting.localization.Localization(__file__, 68, 35), int_391, *[result_and__397], **kwargs_398)
                
                # SSA branch for the else part of an if expression (line 68)
                module_type_store.open_ssa_branch('if expression else')
                
                
                # Getting the type of 'face' (line 68)
                face_400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 73), 'face')
                int_401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 80), 'int')
                # Applying the binary operator '<' (line 68)
                result_lt_402 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 73), '<', face_400, int_401)
                
                # Testing the type of an if expression (line 68)
                is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 68), result_lt_402)
                # SSA begins for if expression (line 68)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
                int_403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 68), 'int')
                # SSA branch for the else part of an if expression (line 68)
                module_type_store.open_ssa_branch('if expression else')
                int_404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 87), 'int')
                # Getting the type of 'i' (line 68)
                i_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 92), 'i')
                int_406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 96), 'int')
                # Applying the binary operator '&' (line 68)
                result_and__407 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 92), '&', i_405, int_406)
                
                # Applying the binary operator '-' (line 68)
                result_sub_408 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 87), '-', int_404, result_and__407)
                
                # SSA join for if expression (line 68)
                module_type_store = module_type_store.join_ssa_context()
                if_exp_409 = union_type.UnionType.add(int_403, result_sub_408)
                
                # SSA join for if expression (line 68)
                module_type_store = module_type_store.join_ssa_context()
                if_exp_410 = union_type.UnionType.add(int_call_result_399, if_exp_409)
                
                # Assigning a type to the variable 'orientationDelta' (line 68)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'orientationDelta', if_exp_410)
                
                # Assigning a Subscript to a Subscript (line 69):
                
                # Assigning a Subscript to a Subscript (line 69):
                
                # Obtaining the type of the subscript
                # Getting the type of 'killer' (line 69)
                killer_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 44), 'killer')
                # Getting the type of 'oldstate' (line 69)
                oldstate_412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 35), 'oldstate')
                # Obtaining the member '__getitem__' of a type (line 69)
                getitem___413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 35), oldstate_412, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 69)
                subscript_call_result_414 = invoke(stypy.reporting.localization.Localization(__file__, 69, 35), getitem___413, killer_411)
                
                # Getting the type of 'newstate' (line 69)
                newstate_415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'newstate')
                # Getting the type of 'target' (line 69)
                target_416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 25), 'target')
                # Storing an element on a container (line 69)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 16), newstate_415, (target_416, subscript_call_result_414))
                
                # Assigning a BinOp to a Subscript (line 70):
                
                # Assigning a BinOp to a Subscript (line 70):
                
                # Obtaining the type of the subscript
                # Getting the type of 'killer' (line 70)
                killer_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 49), 'killer')
                int_418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 58), 'int')
                # Applying the binary operator '+' (line 70)
                result_add_419 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 49), '+', killer_417, int_418)
                
                # Getting the type of 'oldstate' (line 70)
                oldstate_420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 40), 'oldstate')
                # Obtaining the member '__getitem__' of a type (line 70)
                getitem___421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 40), oldstate_420, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 70)
                subscript_call_result_422 = invoke(stypy.reporting.localization.Localization(__file__, 70, 40), getitem___421, result_add_419)
                
                # Getting the type of 'orientationDelta' (line 70)
                orientationDelta_423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 64), 'orientationDelta')
                # Applying the binary operator '+' (line 70)
                result_add_424 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 40), '+', subscript_call_result_422, orientationDelta_423)
                
                # Getting the type of 'newstate' (line 70)
                newstate_425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'newstate')
                # Getting the type of 'target' (line 70)
                target_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 25), 'target')
                int_427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 34), 'int')
                # Applying the binary operator '+' (line 70)
                result_add_428 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 25), '+', target_426, int_427)
                
                # Storing an element on a container (line 70)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 16), newstate_425, (result_add_428, result_add_424))
                
                # Getting the type of 'turn' (line 71)
                turn_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'turn')
                # Getting the type of 'turns' (line 71)
                turns_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 27), 'turns')
                int_431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 35), 'int')
                # Applying the binary operator '-' (line 71)
                result_sub_432 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 27), '-', turns_430, int_431)
                
                # Applying the binary operator '==' (line 71)
                result_eq_433 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 19), '==', turn_429, result_sub_432)
                
                # Testing if the type of an if condition is none (line 71)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 71, 16), result_eq_433):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 71)
                    if_condition_434 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 16), result_eq_433)
                    # Assigning a type to the variable 'if_condition_434' (line 71)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'if_condition_434', if_condition_434)
                    # SSA begins for if statement (line 71)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'newstate' (line 72)
                    newstate_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 20), 'newstate')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'target' (line 72)
                    target_436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 29), 'target')
                    int_437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 38), 'int')
                    # Applying the binary operator '+' (line 72)
                    result_add_438 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 29), '+', target_436, int_437)
                    
                    # Getting the type of 'newstate' (line 72)
                    newstate_439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 20), 'newstate')
                    # Obtaining the member '__getitem__' of a type (line 72)
                    getitem___440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 20), newstate_439, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 72)
                    subscript_call_result_441 = invoke(stypy.reporting.localization.Localization(__file__, 72, 20), getitem___440, result_add_438)
                    
                    int_442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 45), 'int')
                    # Getting the type of 'isCorner' (line 72)
                    isCorner_443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 49), 'isCorner')
                    # Applying the binary operator '+' (line 72)
                    result_add_444 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 45), '+', int_442, isCorner_443)
                    
                    # Applying the binary operator '%=' (line 72)
                    result_imod_445 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 20), '%=', subscript_call_result_441, result_add_444)
                    # Getting the type of 'newstate' (line 72)
                    newstate_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 20), 'newstate')
                    # Getting the type of 'target' (line 72)
                    target_447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 29), 'target')
                    int_448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 38), 'int')
                    # Applying the binary operator '+' (line 72)
                    result_add_449 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 29), '+', target_447, int_448)
                    
                    # Storing an element on a container (line 72)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 20), newstate_446, (result_add_449, result_imod_445))
                    
                    # SSA join for if statement (line 71)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to cube_state(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'newstate' (line 73)
        newstate_451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 26), 'newstate', False)
        # Getting the type of 'self' (line 73)
        self_452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 36), 'self', False)
        # Obtaining the member 'route' of a type (line 73)
        route_453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 36), self_452, 'route')
        
        # Obtaining an instance of the builtin type 'list' (line 73)
        list_454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 73)
        # Adding element type (line 73)
        # Getting the type of 'move' (line 73)
        move_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 50), 'move', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 49), list_454, move_455)
        
        # Applying the binary operator '+' (line 73)
        result_add_456 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 36), '+', route_453, list_454)
        
        # Processing the call keyword arguments (line 73)
        kwargs_457 = {}
        # Getting the type of 'cube_state' (line 73)
        cube_state_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 15), 'cube_state', False)
        # Calling cube_state(args, kwargs) (line 73)
        cube_state_call_result_458 = invoke(stypy.reporting.localization.Localization(__file__, 73, 15), cube_state_450, *[newstate_451, result_add_456], **kwargs_457)
        
        # Assigning a type to the variable 'stypy_return_type' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'stypy_return_type', cube_state_call_result_458)
        
        # ################# End of 'apply_move(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'apply_move' in the type store
        # Getting the type of 'stypy_return_type' (line 59)
        stypy_return_type_459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_459)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'apply_move'
        return stypy_return_type_459


# Assigning a type to the variable 'cube_state' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'cube_state', cube_state)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 76, 0, False)
    
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

    
    # Assigning a Call to a Name (line 77):
    
    # Assigning a Call to a Name (line 77):
    
    # Call to cube_state(...): (line 77)
    # Processing the call arguments (line 77)
    
    # Call to range(...): (line 77)
    # Processing the call arguments (line 77)
    int_462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 34), 'int')
    # Processing the call keyword arguments (line 77)
    kwargs_463 = {}
    # Getting the type of 'range' (line 77)
    range_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 28), 'range', False)
    # Calling range(args, kwargs) (line 77)
    range_call_result_464 = invoke(stypy.reporting.localization.Localization(__file__, 77, 28), range_461, *[int_462], **kwargs_463)
    
    int_465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 40), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 77)
    list_466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 45), 'list')
    # Adding type elements to the builtin type 'list' instance (line 77)
    # Adding element type (line 77)
    int_467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 45), list_466, int_467)
    
    # Applying the binary operator '*' (line 77)
    result_mul_468 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 40), '*', int_465, list_466)
    
    # Applying the binary operator '+' (line 77)
    result_add_469 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 28), '+', range_call_result_464, result_mul_468)
    
    # Processing the call keyword arguments (line 77)
    kwargs_470 = {}
    # Getting the type of 'cube_state' (line 77)
    cube_state_460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 17), 'cube_state', False)
    # Calling cube_state(args, kwargs) (line 77)
    cube_state_call_result_471 = invoke(stypy.reporting.localization.Localization(__file__, 77, 17), cube_state_460, *[result_add_469], **kwargs_470)
    
    # Assigning a type to the variable 'goal_state' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'goal_state', cube_state_call_result_471)
    
    # Assigning a Call to a Name (line 78):
    
    # Assigning a Call to a Name (line 78):
    
    # Call to cube_state(...): (line 78)
    # Processing the call arguments (line 78)
    
    # Obtaining the type of the subscript
    slice_473 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 78, 23), None, None, None)
    # Getting the type of 'goal_state' (line 78)
    goal_state_474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'goal_state', False)
    # Obtaining the member 'state' of a type (line 78)
    state_475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), goal_state_474, 'state')
    # Obtaining the member '__getitem__' of a type (line 78)
    getitem___476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), state_475, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 78)
    subscript_call_result_477 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___476, slice_473)
    
    # Processing the call keyword arguments (line 78)
    kwargs_478 = {}
    # Getting the type of 'cube_state' (line 78)
    cube_state_472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'cube_state', False)
    # Calling cube_state(args, kwargs) (line 78)
    cube_state_call_result_479 = invoke(stypy.reporting.localization.Localization(__file__, 78, 12), cube_state_472, *[subscript_call_result_477], **kwargs_478)
    
    # Assigning a type to the variable 'state' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'state', cube_state_call_result_479)
    
    # Assigning a ListComp to a Name (line 80):
    
    # Assigning a ListComp to a Name (line 80):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 80)
    # Processing the call arguments (line 80)
    int_487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 52), 'int')
    # Processing the call keyword arguments (line 80)
    kwargs_488 = {}
    # Getting the type of 'range' (line 80)
    range_486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 46), 'range', False)
    # Calling range(args, kwargs) (line 80)
    range_call_result_489 = invoke(stypy.reporting.localization.Localization(__file__, 80, 46), range_486, *[int_487], **kwargs_488)
    
    comprehension_490 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 13), range_call_result_489)
    # Assigning a type to the variable 'x' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 13), 'x', comprehension_490)
    
    # Call to randrange(...): (line 80)
    # Processing the call arguments (line 80)
    int_482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 30), 'int')
    int_483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 33), 'int')
    # Processing the call keyword arguments (line 80)
    kwargs_484 = {}
    # Getting the type of 'random' (line 80)
    random_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 13), 'random', False)
    # Obtaining the member 'randrange' of a type (line 80)
    randrange_481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 13), random_480, 'randrange')
    # Calling randrange(args, kwargs) (line 80)
    randrange_call_result_485 = invoke(stypy.reporting.localization.Localization(__file__, 80, 13), randrange_481, *[int_482, int_483], **kwargs_484)
    
    list_491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 13), list_491, randrange_call_result_485)
    # Assigning a type to the variable 'moves' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'moves', list_491)
    
    # Call to join(...): (line 82)
    # Processing the call arguments (line 82)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'moves' (line 82)
    moves_498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 41), 'moves', False)
    comprehension_499 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 14), moves_498)
    # Assigning a type to the variable 'move' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 14), 'move', comprehension_499)
    
    # Call to move_str(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'move' (line 82)
    move_495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 23), 'move', False)
    # Processing the call keyword arguments (line 82)
    kwargs_496 = {}
    # Getting the type of 'move_str' (line 82)
    move_str_494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 14), 'move_str', False)
    # Calling move_str(args, kwargs) (line 82)
    move_str_call_result_497 = invoke(stypy.reporting.localization.Localization(__file__, 82, 14), move_str_494, *[move_495], **kwargs_496)
    
    list_500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 14), list_500, move_str_call_result_497)
    # Processing the call keyword arguments (line 82)
    kwargs_501 = {}
    str_492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 4), 'str', ',')
    # Obtaining the member 'join' of a type (line 82)
    join_493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 4), str_492, 'join')
    # Calling join(args, kwargs) (line 82)
    join_call_result_502 = invoke(stypy.reporting.localization.Localization(__file__, 82, 4), join_493, *[list_500], **kwargs_501)
    
    
    # Call to move_str(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'move' (line 83)
    move_504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 13), 'move', False)
    # Processing the call keyword arguments (line 83)
    kwargs_505 = {}
    # Getting the type of 'move_str' (line 83)
    move_str_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'move_str', False)
    # Calling move_str(args, kwargs) (line 83)
    move_str_call_result_506 = invoke(stypy.reporting.localization.Localization(__file__, 83, 4), move_str_503, *[move_504], **kwargs_505)
    
    
    # Getting the type of 'moves' (line 84)
    moves_507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'moves')
    # Testing if the for loop is going to be iterated (line 84)
    # Testing the type of a for loop iterable (line 84)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 84, 4), moves_507)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 84, 4), moves_507):
        # Getting the type of the for loop variable (line 84)
        for_loop_var_508 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 84, 4), moves_507)
        # Assigning a type to the variable 'move' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'move', for_loop_var_508)
        # SSA begins for a for statement (line 84)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 85):
        
        # Assigning a Call to a Name (line 85):
        
        # Call to apply_move(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'move' (line 85)
        move_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 33), 'move', False)
        # Processing the call keyword arguments (line 85)
        kwargs_512 = {}
        # Getting the type of 'state' (line 85)
        state_509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'state', False)
        # Obtaining the member 'apply_move' of a type (line 85)
        apply_move_510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 16), state_509, 'apply_move')
        # Calling apply_move(args, kwargs) (line 85)
        apply_move_call_result_513 = invoke(stypy.reporting.localization.Localization(__file__, 85, 16), apply_move_510, *[move_511], **kwargs_512)
        
        # Assigning a type to the variable 'state' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'state', apply_move_call_result_513)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a List to a Attribute (line 86):
    
    # Assigning a List to a Attribute (line 86):
    
    # Obtaining an instance of the builtin type 'list' (line 86)
    list_514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 86)
    
    # Getting the type of 'state' (line 86)
    state_515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'state')
    # Setting the type of the member 'route' of a type (line 86)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 4), state_515, 'route', list_514)
    
    
    # Call to range(...): (line 88)
    # Processing the call arguments (line 88)
    int_517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 23), 'int')
    # Processing the call keyword arguments (line 88)
    kwargs_518 = {}
    # Getting the type of 'range' (line 88)
    range_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 17), 'range', False)
    # Calling range(args, kwargs) (line 88)
    range_call_result_519 = invoke(stypy.reporting.localization.Localization(__file__, 88, 17), range_516, *[int_517], **kwargs_518)
    
    # Testing if the for loop is going to be iterated (line 88)
    # Testing the type of a for loop iterable (line 88)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 88, 4), range_call_result_519)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 88, 4), range_call_result_519):
        # Getting the type of the for loop variable (line 88)
        for_loop_var_520 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 88, 4), range_call_result_519)
        # Assigning a type to the variable 'phase' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'phase', for_loop_var_520)
        # SSA begins for a for statement (line 88)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Tuple to a Tuple (line 89):
        
        # Assigning a Call to a Name (line 89):
        
        # Call to id_(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'phase' (line 89)
        phase_523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 40), 'phase', False)
        # Processing the call keyword arguments (line 89)
        kwargs_524 = {}
        # Getting the type of 'state' (line 89)
        state_521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 30), 'state', False)
        # Obtaining the member 'id_' of a type (line 89)
        id__522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 30), state_521, 'id_')
        # Calling id_(args, kwargs) (line 89)
        id__call_result_525 = invoke(stypy.reporting.localization.Localization(__file__, 89, 30), id__522, *[phase_523], **kwargs_524)
        
        # Assigning a type to the variable 'tuple_assignment_3' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'tuple_assignment_3', id__call_result_525)
        
        # Assigning a Call to a Name (line 89):
        
        # Call to id_(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'phase' (line 89)
        phase_528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 63), 'phase', False)
        # Processing the call keyword arguments (line 89)
        kwargs_529 = {}
        # Getting the type of 'goal_state' (line 89)
        goal_state_526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 48), 'goal_state', False)
        # Obtaining the member 'id_' of a type (line 89)
        id__527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 48), goal_state_526, 'id_')
        # Calling id_(args, kwargs) (line 89)
        id__call_result_530 = invoke(stypy.reporting.localization.Localization(__file__, 89, 48), id__527, *[phase_528], **kwargs_529)
        
        # Assigning a type to the variable 'tuple_assignment_4' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'tuple_assignment_4', id__call_result_530)
        
        # Assigning a Name to a Name (line 89):
        # Getting the type of 'tuple_assignment_3' (line 89)
        tuple_assignment_3_531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'tuple_assignment_3')
        # Assigning a type to the variable 'current_id' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'current_id', tuple_assignment_3_531)
        
        # Assigning a Name to a Name (line 89):
        # Getting the type of 'tuple_assignment_4' (line 89)
        tuple_assignment_4_532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'tuple_assignment_4')
        # Assigning a type to the variable 'goal_id' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 20), 'goal_id', tuple_assignment_4_532)
        
        # Assigning a List to a Name (line 90):
        
        # Assigning a List to a Name (line 90):
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        # Getting the type of 'state' (line 90)
        state_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 18), 'state')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 17), list_533, state_534)
        
        # Assigning a type to the variable 'states' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'states', list_533)
        
        # Assigning a Call to a Name (line 91):
        
        # Assigning a Call to a Name (line 91):
        
        # Call to set(...): (line 91)
        # Processing the call arguments (line 91)
        
        # Obtaining an instance of the builtin type 'list' (line 91)
        list_536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 91)
        # Adding element type (line 91)
        # Getting the type of 'current_id' (line 91)
        current_id_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 25), 'current_id', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 24), list_536, current_id_537)
        
        # Processing the call keyword arguments (line 91)
        kwargs_538 = {}
        # Getting the type of 'set' (line 91)
        set_535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'set', False)
        # Calling set(args, kwargs) (line 91)
        set_call_result_539 = invoke(stypy.reporting.localization.Localization(__file__, 91, 20), set_535, *[list_536], **kwargs_538)
        
        # Assigning a type to the variable 'state_ids' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'state_ids', set_call_result_539)
        
        # Getting the type of 'current_id' (line 92)
        current_id_540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'current_id')
        # Getting the type of 'goal_id' (line 92)
        goal_id_541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), 'goal_id')
        # Applying the binary operator '!=' (line 92)
        result_ne_542 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 11), '!=', current_id_540, goal_id_541)
        
        # Testing if the type of an if condition is none (line 92)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 92, 8), result_ne_542):
            pass
        else:
            
            # Testing the type of an if condition (line 92)
            if_condition_543 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 8), result_ne_542)
            # Assigning a type to the variable 'if_condition_543' (line 92)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'if_condition_543', if_condition_543)
            # SSA begins for if statement (line 92)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 93):
            
            # Assigning a Name to a Name (line 93):
            # Getting the type of 'False' (line 93)
            False_544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 23), 'False')
            # Assigning a type to the variable 'phase_ok' (line 93)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'phase_ok', False_544)
            
            
            # Getting the type of 'phase_ok' (line 94)
            phase_ok_545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 22), 'phase_ok')
            # Applying the 'not' unary operator (line 94)
            result_not__546 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 18), 'not', phase_ok_545)
            
            # Testing if the while is going to be iterated (line 94)
            # Testing the type of an if condition (line 94)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 12), result_not__546)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 94, 12), result_not__546):
                # SSA begins for while statement (line 94)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
                
                # Assigning a List to a Name (line 95):
                
                # Assigning a List to a Name (line 95):
                
                # Obtaining an instance of the builtin type 'list' (line 95)
                list_547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 30), 'list')
                # Adding type elements to the builtin type 'list' instance (line 95)
                
                # Assigning a type to the variable 'next_states' (line 95)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'next_states', list_547)
                
                # Getting the type of 'states' (line 96)
                states_548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 33), 'states')
                # Testing if the for loop is going to be iterated (line 96)
                # Testing the type of a for loop iterable (line 96)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 96, 16), states_548)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 96, 16), states_548):
                    # Getting the type of the for loop variable (line 96)
                    for_loop_var_549 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 96, 16), states_548)
                    # Assigning a type to the variable 'cur_state' (line 96)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'cur_state', for_loop_var_549)
                    # SSA begins for a for statement (line 96)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'phase' (line 97)
                    phase_550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 44), 'phase')
                    # Getting the type of 'phase_moves' (line 97)
                    phase_moves_551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 32), 'phase_moves')
                    # Obtaining the member '__getitem__' of a type (line 97)
                    getitem___552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 32), phase_moves_551, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 97)
                    subscript_call_result_553 = invoke(stypy.reporting.localization.Localization(__file__, 97, 32), getitem___552, phase_550)
                    
                    # Testing if the for loop is going to be iterated (line 97)
                    # Testing the type of a for loop iterable (line 97)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 97, 20), subscript_call_result_553)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 97, 20), subscript_call_result_553):
                        # Getting the type of the for loop variable (line 97)
                        for_loop_var_554 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 97, 20), subscript_call_result_553)
                        # Assigning a type to the variable 'move' (line 97)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 20), 'move', for_loop_var_554)
                        # SSA begins for a for statement (line 97)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Assigning a Call to a Name (line 98):
                        
                        # Assigning a Call to a Name (line 98):
                        
                        # Call to apply_move(...): (line 98)
                        # Processing the call arguments (line 98)
                        # Getting the type of 'move' (line 98)
                        move_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 58), 'move', False)
                        # Processing the call keyword arguments (line 98)
                        kwargs_558 = {}
                        # Getting the type of 'cur_state' (line 98)
                        cur_state_555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 37), 'cur_state', False)
                        # Obtaining the member 'apply_move' of a type (line 98)
                        apply_move_556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 37), cur_state_555, 'apply_move')
                        # Calling apply_move(args, kwargs) (line 98)
                        apply_move_call_result_559 = invoke(stypy.reporting.localization.Localization(__file__, 98, 37), apply_move_556, *[move_557], **kwargs_558)
                        
                        # Assigning a type to the variable 'next_state' (line 98)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 24), 'next_state', apply_move_call_result_559)
                        
                        # Assigning a Call to a Name (line 99):
                        
                        # Assigning a Call to a Name (line 99):
                        
                        # Call to id_(...): (line 99)
                        # Processing the call arguments (line 99)
                        # Getting the type of 'phase' (line 99)
                        phase_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 49), 'phase', False)
                        # Processing the call keyword arguments (line 99)
                        kwargs_563 = {}
                        # Getting the type of 'next_state' (line 99)
                        next_state_560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 34), 'next_state', False)
                        # Obtaining the member 'id_' of a type (line 99)
                        id__561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 34), next_state_560, 'id_')
                        # Calling id_(args, kwargs) (line 99)
                        id__call_result_564 = invoke(stypy.reporting.localization.Localization(__file__, 99, 34), id__561, *[phase_562], **kwargs_563)
                        
                        # Assigning a type to the variable 'next_id' (line 99)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 24), 'next_id', id__call_result_564)
                        
                        # Getting the type of 'next_id' (line 100)
                        next_id_565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 27), 'next_id')
                        # Getting the type of 'goal_id' (line 100)
                        goal_id_566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 38), 'goal_id')
                        # Applying the binary operator '==' (line 100)
                        result_eq_567 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 27), '==', next_id_565, goal_id_566)
                        
                        # Testing if the type of an if condition is none (line 100)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 100, 24), result_eq_567):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 100)
                            if_condition_568 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 24), result_eq_567)
                            # Assigning a type to the variable 'if_condition_568' (line 100)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 24), 'if_condition_568', if_condition_568)
                            # SSA begins for if statement (line 100)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            pass
                            
                            # Assigning a Name to a Name (line 102):
                            
                            # Assigning a Name to a Name (line 102):
                            # Getting the type of 'True' (line 102)
                            True_569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 39), 'True')
                            # Assigning a type to the variable 'phase_ok' (line 102)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 28), 'phase_ok', True_569)
                            
                            # Assigning a Name to a Name (line 103):
                            
                            # Assigning a Name to a Name (line 103):
                            # Getting the type of 'next_state' (line 103)
                            next_state_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 36), 'next_state')
                            # Assigning a type to the variable 'state' (line 103)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 28), 'state', next_state_570)
                            # SSA join for if statement (line 100)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        
                        # Getting the type of 'next_id' (line 105)
                        next_id_571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 27), 'next_id')
                        # Getting the type of 'state_ids' (line 105)
                        state_ids_572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 42), 'state_ids')
                        # Applying the binary operator 'notin' (line 105)
                        result_contains_573 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 27), 'notin', next_id_571, state_ids_572)
                        
                        # Testing if the type of an if condition is none (line 105)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 105, 24), result_contains_573):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 105)
                            if_condition_574 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 24), result_contains_573)
                            # Assigning a type to the variable 'if_condition_574' (line 105)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 24), 'if_condition_574', if_condition_574)
                            # SSA begins for if statement (line 105)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to add(...): (line 106)
                            # Processing the call arguments (line 106)
                            # Getting the type of 'next_id' (line 106)
                            next_id_577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 42), 'next_id', False)
                            # Processing the call keyword arguments (line 106)
                            kwargs_578 = {}
                            # Getting the type of 'state_ids' (line 106)
                            state_ids_575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 28), 'state_ids', False)
                            # Obtaining the member 'add' of a type (line 106)
                            add_576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 28), state_ids_575, 'add')
                            # Calling add(args, kwargs) (line 106)
                            add_call_result_579 = invoke(stypy.reporting.localization.Localization(__file__, 106, 28), add_576, *[next_id_577], **kwargs_578)
                            
                            
                            # Call to append(...): (line 107)
                            # Processing the call arguments (line 107)
                            # Getting the type of 'next_state' (line 107)
                            next_state_582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 47), 'next_state', False)
                            # Processing the call keyword arguments (line 107)
                            kwargs_583 = {}
                            # Getting the type of 'next_states' (line 107)
                            next_states_580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 28), 'next_states', False)
                            # Obtaining the member 'append' of a type (line 107)
                            append_581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 28), next_states_580, 'append')
                            # Calling append(args, kwargs) (line 107)
                            append_call_result_584 = invoke(stypy.reporting.localization.Localization(__file__, 107, 28), append_581, *[next_state_582], **kwargs_583)
                            
                            # SSA join for if statement (line 105)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    # Getting the type of 'phase_ok' (line 108)
                    phase_ok_585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 23), 'phase_ok')
                    # Testing if the type of an if condition is none (line 108)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 20), phase_ok_585):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 108)
                        if_condition_586 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 20), phase_ok_585)
                        # Assigning a type to the variable 'if_condition_586' (line 108)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 20), 'if_condition_586', if_condition_586)
                        # SSA begins for if statement (line 108)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # SSA join for if statement (line 108)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Assigning a Name to a Name (line 110):
                
                # Assigning a Name to a Name (line 110):
                # Getting the type of 'next_states' (line 110)
                next_states_587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 25), 'next_states')
                # Assigning a type to the variable 'states' (line 110)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'states', next_states_587)
                # SSA join for while statement (line 94)
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 92)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'True' (line 111)
    True_588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'stypy_return_type', True_588)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 76)
    stypy_return_type_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_589)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_589

# Assigning a type to the variable 'run' (line 76)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'run', run)

# Call to run(...): (line 114)
# Processing the call keyword arguments (line 114)
kwargs_591 = {}
# Getting the type of 'run' (line 114)
run_590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 0), 'run', False)
# Calling run(args, kwargs) (line 114)
run_call_result_592 = invoke(stypy.reporting.localization.Localization(__file__, 114, 0), run_590, *[], **kwargs_591)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

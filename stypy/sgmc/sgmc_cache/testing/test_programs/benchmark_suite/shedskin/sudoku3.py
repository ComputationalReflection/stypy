
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # (c) Peter Cock
2: # --- http://www2.warwick.ac.uk/fac/sci/moac/currentstudents/peter_cock/python/sudoku/
3: #
4: # sudoku solver
5: 
6: TRIPLETS = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
7: 
8: ROW_ITER = [[(row, col) for col in range(0, 9)] for row in range(0, 9)]
9: COL_ITER = [[(row, col) for row in range(0, 9)] for col in range(0, 9)]
10: TxT_ITER = [[(row, col) for row in rows for col in cols] for rows in TRIPLETS for cols in TRIPLETS]
11: 
12: 
13: class soduko:
14:     def __init__(self, start_grid=None):
15:         self.squares = [[range(1, 10) for col in range(0, 9)] for row in range(0, 9)]
16: 
17:         if start_grid is not None:
18:             assert len(start_grid) == 9, "Bad input!"
19:             for row in range(0, 9):
20:                 self.set_row(row, start_grid[row])
21: 
22:         self._changed = False
23: 
24:     def copy(self):
25:         soduko_copy = soduko(None)
26:         for row in range(0, 9):
27:             for col in range(0, 9):
28:                 soduko_copy.squares[row][col] = self.squares[row][col][:]
29:         soduko_copy._changed = False
30:         return soduko_copy
31: 
32:     def set_row(self, row, x_list):
33:         assert len(x_list) == 9, 'not 9'
34:         for col in range(0, 9):
35:             try:
36:                 x = int(x_list[col])
37:             except:
38:                 x = 0
39:             self.set_cell(row, col, x)
40: 
41:     def set_cell(self, row, col, x):
42:         if self.squares[row][col] == [x]:
43:             pass
44:         elif x not in range(1, 9 + 1):
45:             pass
46:         else:
47:             assert x in self.squares[row][col], "bugger2"
48: 
49:             self.squares[row][col] = [x]
50:             self.update_neighbours(row, col, x)
51:             self._changed = True
52: 
53:     def cell_exclude(self, row, col, x):
54:         assert x in range(1, 9 + 1), 'inra'
55:         if x in self.squares[row][col]:
56:             self.squares[row][col].remove(x)
57:             assert len(self.squares[row][col]) > 0, "bugger"
58:             if len(self.squares[row][col]) == 1:
59:                 self._changed = True
60:                 self.update_neighbours(row, col, self.squares[row][col][0])
61:         else:
62:             pass
63:         return
64: 
65:     def update_neighbours(self, set_row, set_col, x):
66:         for row in range(0, 9):
67:             if row <> set_row:
68:                 self.cell_exclude(row, set_col, x)
69:         for col in range(0, 9):
70:             if col <> set_col:
71:                 self.cell_exclude(set_row, col, x)
72:         for triplet in TRIPLETS:
73:             if set_row in triplet: rows = triplet[:]
74:             if set_col in triplet: cols = triplet[:]
75:         rows.remove(set_row)
76:         cols.remove(set_col)
77:         for row in rows:
78:             for col in cols:
79:                 assert row <> set_row or col <> set_col, 'meuh'
80:                 self.cell_exclude(row, col, x)
81: 
82:     def get_cell_digit_str(self, row, col):
83:         if len(self.squares[row][col]) == 1:
84:             return str(self.squares[row][col][0])
85:         else:
86:             return "0"
87: 
88:     def __str__(self):
89:         answer = "   123   456   789\n"
90:         for row in range(0, 9):
91:             answer = answer + str(row + 1) + " [" + "".join(
92:                 [self.get_cell_digit_str(row, col).replace("0", "?") for col in range(0, 3)]) + "] [" + "".join(
93:                 [self.get_cell_digit_str(row, col).replace("0", "?") for col in range(3, 6)]) + "] [" + "".join(
94:                 [self.get_cell_digit_str(row, col).replace("0", "?") for col in range(6, 9)]) + "]\n"
95:             if row + 1 in [3, 6]:
96:                 answer = answer + "   ---   ---   ---\n"
97:         return answer
98: 
99:     def check(self):
100:         self._changed = True
101:         while self._changed:
102:             self._changed = False
103:             self.check_for_single_occurances()
104:             self.check_for_last_in_row_col_3x3()
105:         return
106: 
107:     def check_for_single_occurances(self):
108:         for check_type in [ROW_ITER, COL_ITER, TxT_ITER]:
109:             for check_list in check_type:
110:                 for x in range(1, 9 + 1):  # 1 to 9 inclusive
111:                     x_in_list = []
112:                     for (row, col) in check_list:
113:                         if x in self.squares[row][col]:
114:                             x_in_list.append((row, col))
115:                     if len(x_in_list) == 1:
116:                         (row, col) = x_in_list[0]
117:                         if len(self.squares[row][col]) > 1:
118:                             self.set_cell(row, col, x)
119: 
120:     def check_for_last_in_row_col_3x3(self):
121:         for (type_name, check_type) in [("Row", ROW_ITER), ("Col", COL_ITER), ("3x3", TxT_ITER)]:
122:             for check_list in check_type:
123:                 unknown_entries = []
124:                 unassigned_values = range(1, 9 + 1)  # 1-9 inclusive
125:                 known_values = []
126:                 for (row, col) in check_list:
127:                     if len(self.squares[row][col]) == 1:
128:                         assert self.squares[row][col][0] not in known_values, "bugger3"
129: 
130:                         known_values.append(self.squares[row][col][0])
131: 
132:                         assert self.squares[row][col][0] in unassigned_values, "bugger4"
133: 
134:                         unassigned_values.remove(self.squares[row][col][0])
135:                     else:
136:                         unknown_entries.append((row, col))
137:                 assert len(unknown_entries) + len(known_values) == 9, 'bugger5'
138:                 assert len(unknown_entries) == len(unassigned_values), 'bugger6'
139:                 if len(unknown_entries) == 1:
140:                     x = unassigned_values[0]
141:                     (row, col) = unknown_entries[0]
142:                     self.set_cell(row, col, x)
143:         return
144: 
145:     def one_level_supposition(self):
146:         progress = True
147:         while progress:
148:             progress = False
149:             for row in range(0, 9):
150:                 for col in range(0, 9):
151:                     if len(self.squares[row][col]) > 1:
152:                         bad_x = []
153:                         for x in self.squares[row][col]:
154:                             soduko_copy = self.copy()
155:                             try:
156:                                 soduko_copy.set_cell(row, col, x)
157:                                 soduko_copy.check()
158:                             except AssertionError, e:
159:                                 bad_x.append(x)
160:                             del soduko_copy
161:                         if len(bad_x) == 0:
162:                             pass
163:                         elif len(bad_x) < len(self.squares[row][col]):
164:                             for x in bad_x:
165:                                 self.cell_exclude(row, col, x)
166:                                 self.check()
167:                             progress = True
168:                         else:
169:                             assert False, "bugger7"
170: 
171: 
172: def main():
173:     for x in range(50):
174:         t = soduko(["800000600",
175:                     "040500100",
176:                     "070090000",
177:                     "030020007",
178:                     "600008004",
179:                     "500000090",
180:                     "000030020",
181:                     "001006050",
182:                     "004000003"])
183: 
184:         t.check()
185:         t.one_level_supposition()
186:         t.check()
187:         # print t
188: 
189: 
190: def run():
191:     main()
192:     return True
193: 
194: 
195: run()
196: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a List to a Name (line 6):

# Assigning a List to a Name (line 6):

# Obtaining an instance of the builtin type 'list' (line 6)
list_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 11), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)

# Obtaining an instance of the builtin type 'list' (line 6)
list_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 12), list_6, int_7)
# Adding element type (line 6)
int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 12), list_6, int_8)
# Adding element type (line 6)
int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 12), list_6, int_9)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 11), list_5, list_6)
# Adding element type (line 6)

# Obtaining an instance of the builtin type 'list' (line 6)
list_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 23), list_10, int_11)
# Adding element type (line 6)
int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 23), list_10, int_12)
# Adding element type (line 6)
int_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 23), list_10, int_13)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 11), list_5, list_10)
# Adding element type (line 6)

# Obtaining an instance of the builtin type 'list' (line 6)
list_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 34), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
int_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 34), list_14, int_15)
# Adding element type (line 6)
int_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 38), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 34), list_14, int_16)
# Adding element type (line 6)
int_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 41), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 34), list_14, int_17)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 11), list_5, list_14)

# Assigning a type to the variable 'TRIPLETS' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'TRIPLETS', list_5)

# Assigning a ListComp to a Name (line 8):

# Assigning a ListComp to a Name (line 8):
# Calculating list comprehension
# Calculating comprehension expression

# Call to range(...): (line 8)
# Processing the call arguments (line 8)
int_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 65), 'int')
int_30 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 68), 'int')
# Processing the call keyword arguments (line 8)
kwargs_31 = {}
# Getting the type of 'range' (line 8)
range_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 59), 'range', False)
# Calling range(args, kwargs) (line 8)
range_call_result_32 = invoke(stypy.reporting.localization.Localization(__file__, 8, 59), range_28, *[int_29, int_30], **kwargs_31)

comprehension_33 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 12), range_call_result_32)
# Assigning a type to the variable 'row' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 12), 'row', comprehension_33)
# Calculating list comprehension
# Calculating comprehension expression

# Call to range(...): (line 8)
# Processing the call arguments (line 8)
int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 41), 'int')
int_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 44), 'int')
# Processing the call keyword arguments (line 8)
kwargs_24 = {}
# Getting the type of 'range' (line 8)
range_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 35), 'range', False)
# Calling range(args, kwargs) (line 8)
range_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 8, 35), range_21, *[int_22, int_23], **kwargs_24)

comprehension_26 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 13), range_call_result_25)
# Assigning a type to the variable 'col' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 13), 'col', comprehension_26)

# Obtaining an instance of the builtin type 'tuple' (line 8)
tuple_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 8)
# Adding element type (line 8)
# Getting the type of 'row' (line 8)
row_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 14), 'row')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 14), tuple_18, row_19)
# Adding element type (line 8)
# Getting the type of 'col' (line 8)
col_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 19), 'col')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 14), tuple_18, col_20)

list_27 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 13), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 13), list_27, tuple_18)
list_34 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 12), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 12), list_34, list_27)
# Assigning a type to the variable 'ROW_ITER' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'ROW_ITER', list_34)

# Assigning a ListComp to a Name (line 9):

# Assigning a ListComp to a Name (line 9):
# Calculating list comprehension
# Calculating comprehension expression

# Call to range(...): (line 9)
# Processing the call arguments (line 9)
int_46 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 65), 'int')
int_47 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 68), 'int')
# Processing the call keyword arguments (line 9)
kwargs_48 = {}
# Getting the type of 'range' (line 9)
range_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 59), 'range', False)
# Calling range(args, kwargs) (line 9)
range_call_result_49 = invoke(stypy.reporting.localization.Localization(__file__, 9, 59), range_45, *[int_46, int_47], **kwargs_48)

comprehension_50 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 12), range_call_result_49)
# Assigning a type to the variable 'col' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'col', comprehension_50)
# Calculating list comprehension
# Calculating comprehension expression

# Call to range(...): (line 9)
# Processing the call arguments (line 9)
int_39 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 41), 'int')
int_40 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 44), 'int')
# Processing the call keyword arguments (line 9)
kwargs_41 = {}
# Getting the type of 'range' (line 9)
range_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 35), 'range', False)
# Calling range(args, kwargs) (line 9)
range_call_result_42 = invoke(stypy.reporting.localization.Localization(__file__, 9, 35), range_38, *[int_39, int_40], **kwargs_41)

comprehension_43 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 13), range_call_result_42)
# Assigning a type to the variable 'row' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 13), 'row', comprehension_43)

# Obtaining an instance of the builtin type 'tuple' (line 9)
tuple_35 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 9)
# Adding element type (line 9)
# Getting the type of 'row' (line 9)
row_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 14), 'row')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 14), tuple_35, row_36)
# Adding element type (line 9)
# Getting the type of 'col' (line 9)
col_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 19), 'col')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 14), tuple_35, col_37)

list_44 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 13), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 13), list_44, tuple_35)
list_51 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 12), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 12), list_51, list_44)
# Assigning a type to the variable 'COL_ITER' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'COL_ITER', list_51)

# Assigning a ListComp to a Name (line 10):

# Assigning a ListComp to a Name (line 10):
# Calculating list comprehension
# Calculating comprehension expression
# Getting the type of 'TRIPLETS' (line 10)
TRIPLETS_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 69), 'TRIPLETS')
comprehension_61 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 12), TRIPLETS_60)
# Assigning a type to the variable 'rows' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 'rows', comprehension_61)
# Calculating comprehension expression
# Getting the type of 'TRIPLETS' (line 10)
TRIPLETS_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 90), 'TRIPLETS')
comprehension_63 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 12), TRIPLETS_62)
# Assigning a type to the variable 'cols' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 'cols', comprehension_63)
# Calculating list comprehension
# Calculating comprehension expression
# Getting the type of 'rows' (line 10)
rows_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 35), 'rows')
comprehension_56 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 13), rows_55)
# Assigning a type to the variable 'row' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 13), 'row', comprehension_56)
# Calculating comprehension expression
# Getting the type of 'cols' (line 10)
cols_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 51), 'cols')
comprehension_58 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 13), cols_57)
# Assigning a type to the variable 'col' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 13), 'col', comprehension_58)

# Obtaining an instance of the builtin type 'tuple' (line 10)
tuple_52 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 10)
# Adding element type (line 10)
# Getting the type of 'row' (line 10)
row_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 14), 'row')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 14), tuple_52, row_53)
# Adding element type (line 10)
# Getting the type of 'col' (line 10)
col_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 19), 'col')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 14), tuple_52, col_54)

list_59 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 13), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 13), list_59, tuple_52)
list_64 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 12), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 12), list_64, list_59)
# Assigning a type to the variable 'TxT_ITER' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'TxT_ITER', list_64)
# Declaration of the 'soduko' class

class soduko:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 14)
        None_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 34), 'None')
        defaults = [None_65]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 14, 4, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'soduko.__init__', ['start_grid'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['start_grid'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a ListComp to a Attribute (line 15):
        
        # Assigning a ListComp to a Attribute (line 15):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 15)
        # Processing the call arguments (line 15)
        int_79 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 79), 'int')
        int_80 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 82), 'int')
        # Processing the call keyword arguments (line 15)
        kwargs_81 = {}
        # Getting the type of 'range' (line 15)
        range_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 73), 'range', False)
        # Calling range(args, kwargs) (line 15)
        range_call_result_82 = invoke(stypy.reporting.localization.Localization(__file__, 15, 73), range_78, *[int_79, int_80], **kwargs_81)
        
        comprehension_83 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 24), range_call_result_82)
        # Assigning a type to the variable 'row' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 24), 'row', comprehension_83)
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 15)
        # Processing the call arguments (line 15)
        int_72 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 55), 'int')
        int_73 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 58), 'int')
        # Processing the call keyword arguments (line 15)
        kwargs_74 = {}
        # Getting the type of 'range' (line 15)
        range_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 49), 'range', False)
        # Calling range(args, kwargs) (line 15)
        range_call_result_75 = invoke(stypy.reporting.localization.Localization(__file__, 15, 49), range_71, *[int_72, int_73], **kwargs_74)
        
        comprehension_76 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 25), range_call_result_75)
        # Assigning a type to the variable 'col' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 25), 'col', comprehension_76)
        
        # Call to range(...): (line 15)
        # Processing the call arguments (line 15)
        int_67 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 31), 'int')
        int_68 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 34), 'int')
        # Processing the call keyword arguments (line 15)
        kwargs_69 = {}
        # Getting the type of 'range' (line 15)
        range_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 25), 'range', False)
        # Calling range(args, kwargs) (line 15)
        range_call_result_70 = invoke(stypy.reporting.localization.Localization(__file__, 15, 25), range_66, *[int_67, int_68], **kwargs_69)
        
        list_77 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 25), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 25), list_77, range_call_result_70)
        list_84 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 24), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 24), list_84, list_77)
        # Getting the type of 'self' (line 15)
        self_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'self')
        # Setting the type of the member 'squares' of a type (line 15)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), self_85, 'squares', list_84)
        
        # Type idiom detected: calculating its left and rigth part (line 17)
        # Getting the type of 'start_grid' (line 17)
        start_grid_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'start_grid')
        # Getting the type of 'None' (line 17)
        None_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 29), 'None')
        
        (may_be_88, more_types_in_union_89) = may_not_be_none(start_grid_86, None_87)

        if may_be_88:

            if more_types_in_union_89:
                # Runtime conditional SSA (line 17)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Evaluating assert statement condition
            
            
            # Call to len(...): (line 18)
            # Processing the call arguments (line 18)
            # Getting the type of 'start_grid' (line 18)
            start_grid_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 23), 'start_grid', False)
            # Processing the call keyword arguments (line 18)
            kwargs_92 = {}
            # Getting the type of 'len' (line 18)
            len_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 19), 'len', False)
            # Calling len(args, kwargs) (line 18)
            len_call_result_93 = invoke(stypy.reporting.localization.Localization(__file__, 18, 19), len_90, *[start_grid_91], **kwargs_92)
            
            int_94 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 38), 'int')
            # Applying the binary operator '==' (line 18)
            result_eq_95 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 19), '==', len_call_result_93, int_94)
            
            assert_96 = result_eq_95
            # Assigning a type to the variable 'assert_96' (line 18)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 12), 'assert_96', result_eq_95)
            
            
            # Call to range(...): (line 19)
            # Processing the call arguments (line 19)
            int_98 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 29), 'int')
            int_99 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 32), 'int')
            # Processing the call keyword arguments (line 19)
            kwargs_100 = {}
            # Getting the type of 'range' (line 19)
            range_97 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 23), 'range', False)
            # Calling range(args, kwargs) (line 19)
            range_call_result_101 = invoke(stypy.reporting.localization.Localization(__file__, 19, 23), range_97, *[int_98, int_99], **kwargs_100)
            
            # Assigning a type to the variable 'range_call_result_101' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'range_call_result_101', range_call_result_101)
            # Testing if the for loop is going to be iterated (line 19)
            # Testing the type of a for loop iterable (line 19)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 19, 12), range_call_result_101)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 19, 12), range_call_result_101):
                # Getting the type of the for loop variable (line 19)
                for_loop_var_102 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 19, 12), range_call_result_101)
                # Assigning a type to the variable 'row' (line 19)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'row', for_loop_var_102)
                # SSA begins for a for statement (line 19)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to set_row(...): (line 20)
                # Processing the call arguments (line 20)
                # Getting the type of 'row' (line 20)
                row_105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 29), 'row', False)
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 20)
                row_106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 45), 'row', False)
                # Getting the type of 'start_grid' (line 20)
                start_grid_107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 34), 'start_grid', False)
                # Obtaining the member '__getitem__' of a type (line 20)
                getitem___108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 34), start_grid_107, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 20)
                subscript_call_result_109 = invoke(stypy.reporting.localization.Localization(__file__, 20, 34), getitem___108, row_106)
                
                # Processing the call keyword arguments (line 20)
                kwargs_110 = {}
                # Getting the type of 'self' (line 20)
                self_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'self', False)
                # Obtaining the member 'set_row' of a type (line 20)
                set_row_104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 16), self_103, 'set_row')
                # Calling set_row(args, kwargs) (line 20)
                set_row_call_result_111 = invoke(stypy.reporting.localization.Localization(__file__, 20, 16), set_row_104, *[row_105, subscript_call_result_109], **kwargs_110)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            

            if more_types_in_union_89:
                # SSA join for if statement (line 17)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 22):
        
        # Assigning a Name to a Attribute (line 22):
        # Getting the type of 'False' (line 22)
        False_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 24), 'False')
        # Getting the type of 'self' (line 22)
        self_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self')
        # Setting the type of the member '_changed' of a type (line 22)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), self_113, '_changed', False_112)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def copy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'copy'
        module_type_store = module_type_store.open_function_context('copy', 24, 4, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        soduko.copy.__dict__.__setitem__('stypy_localization', localization)
        soduko.copy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        soduko.copy.__dict__.__setitem__('stypy_type_store', module_type_store)
        soduko.copy.__dict__.__setitem__('stypy_function_name', 'soduko.copy')
        soduko.copy.__dict__.__setitem__('stypy_param_names_list', [])
        soduko.copy.__dict__.__setitem__('stypy_varargs_param_name', None)
        soduko.copy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        soduko.copy.__dict__.__setitem__('stypy_call_defaults', defaults)
        soduko.copy.__dict__.__setitem__('stypy_call_varargs', varargs)
        soduko.copy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        soduko.copy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'soduko.copy', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'copy', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'copy(...)' code ##################

        
        # Assigning a Call to a Name (line 25):
        
        # Assigning a Call to a Name (line 25):
        
        # Call to soduko(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'None' (line 25)
        None_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 29), 'None', False)
        # Processing the call keyword arguments (line 25)
        kwargs_116 = {}
        # Getting the type of 'soduko' (line 25)
        soduko_114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 22), 'soduko', False)
        # Calling soduko(args, kwargs) (line 25)
        soduko_call_result_117 = invoke(stypy.reporting.localization.Localization(__file__, 25, 22), soduko_114, *[None_115], **kwargs_116)
        
        # Assigning a type to the variable 'soduko_copy' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'soduko_copy', soduko_call_result_117)
        
        
        # Call to range(...): (line 26)
        # Processing the call arguments (line 26)
        int_119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'int')
        int_120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 28), 'int')
        # Processing the call keyword arguments (line 26)
        kwargs_121 = {}
        # Getting the type of 'range' (line 26)
        range_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 19), 'range', False)
        # Calling range(args, kwargs) (line 26)
        range_call_result_122 = invoke(stypy.reporting.localization.Localization(__file__, 26, 19), range_118, *[int_119, int_120], **kwargs_121)
        
        # Assigning a type to the variable 'range_call_result_122' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'range_call_result_122', range_call_result_122)
        # Testing if the for loop is going to be iterated (line 26)
        # Testing the type of a for loop iterable (line 26)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 26, 8), range_call_result_122)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 26, 8), range_call_result_122):
            # Getting the type of the for loop variable (line 26)
            for_loop_var_123 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 26, 8), range_call_result_122)
            # Assigning a type to the variable 'row' (line 26)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'row', for_loop_var_123)
            # SSA begins for a for statement (line 26)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to range(...): (line 27)
            # Processing the call arguments (line 27)
            int_125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 29), 'int')
            int_126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 32), 'int')
            # Processing the call keyword arguments (line 27)
            kwargs_127 = {}
            # Getting the type of 'range' (line 27)
            range_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 23), 'range', False)
            # Calling range(args, kwargs) (line 27)
            range_call_result_128 = invoke(stypy.reporting.localization.Localization(__file__, 27, 23), range_124, *[int_125, int_126], **kwargs_127)
            
            # Assigning a type to the variable 'range_call_result_128' (line 27)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'range_call_result_128', range_call_result_128)
            # Testing if the for loop is going to be iterated (line 27)
            # Testing the type of a for loop iterable (line 27)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 27, 12), range_call_result_128)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 27, 12), range_call_result_128):
                # Getting the type of the for loop variable (line 27)
                for_loop_var_129 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 27, 12), range_call_result_128)
                # Assigning a type to the variable 'col' (line 27)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'col', for_loop_var_129)
                # SSA begins for a for statement (line 27)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Subscript to a Subscript (line 28):
                
                # Assigning a Subscript to a Subscript (line 28):
                
                # Obtaining the type of the subscript
                slice_130 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 28, 48), None, None, None)
                
                # Obtaining the type of the subscript
                # Getting the type of 'col' (line 28)
                col_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 66), 'col')
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 28)
                row_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 61), 'row')
                # Getting the type of 'self' (line 28)
                self_133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 48), 'self')
                # Obtaining the member 'squares' of a type (line 28)
                squares_134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 48), self_133, 'squares')
                # Obtaining the member '__getitem__' of a type (line 28)
                getitem___135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 48), squares_134, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 28)
                subscript_call_result_136 = invoke(stypy.reporting.localization.Localization(__file__, 28, 48), getitem___135, row_132)
                
                # Obtaining the member '__getitem__' of a type (line 28)
                getitem___137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 48), subscript_call_result_136, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 28)
                subscript_call_result_138 = invoke(stypy.reporting.localization.Localization(__file__, 28, 48), getitem___137, col_131)
                
                # Obtaining the member '__getitem__' of a type (line 28)
                getitem___139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 48), subscript_call_result_138, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 28)
                subscript_call_result_140 = invoke(stypy.reporting.localization.Localization(__file__, 28, 48), getitem___139, slice_130)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 28)
                row_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 36), 'row')
                # Getting the type of 'soduko_copy' (line 28)
                soduko_copy_142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), 'soduko_copy')
                # Obtaining the member 'squares' of a type (line 28)
                squares_143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 16), soduko_copy_142, 'squares')
                # Obtaining the member '__getitem__' of a type (line 28)
                getitem___144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 16), squares_143, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 28)
                subscript_call_result_145 = invoke(stypy.reporting.localization.Localization(__file__, 28, 16), getitem___144, row_141)
                
                # Getting the type of 'col' (line 28)
                col_146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 41), 'col')
                # Storing an element on a container (line 28)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 16), subscript_call_result_145, (col_146, subscript_call_result_140))
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Name to a Attribute (line 29):
        
        # Assigning a Name to a Attribute (line 29):
        # Getting the type of 'False' (line 29)
        False_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 31), 'False')
        # Getting the type of 'soduko_copy' (line 29)
        soduko_copy_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'soduko_copy')
        # Setting the type of the member '_changed' of a type (line 29)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), soduko_copy_148, '_changed', False_147)
        # Getting the type of 'soduko_copy' (line 30)
        soduko_copy_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 15), 'soduko_copy')
        # Assigning a type to the variable 'stypy_return_type' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'stypy_return_type', soduko_copy_149)
        
        # ################# End of 'copy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'copy' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_150)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'copy'
        return stypy_return_type_150


    @norecursion
    def set_row(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_row'
        module_type_store = module_type_store.open_function_context('set_row', 32, 4, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        soduko.set_row.__dict__.__setitem__('stypy_localization', localization)
        soduko.set_row.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        soduko.set_row.__dict__.__setitem__('stypy_type_store', module_type_store)
        soduko.set_row.__dict__.__setitem__('stypy_function_name', 'soduko.set_row')
        soduko.set_row.__dict__.__setitem__('stypy_param_names_list', ['row', 'x_list'])
        soduko.set_row.__dict__.__setitem__('stypy_varargs_param_name', None)
        soduko.set_row.__dict__.__setitem__('stypy_kwargs_param_name', None)
        soduko.set_row.__dict__.__setitem__('stypy_call_defaults', defaults)
        soduko.set_row.__dict__.__setitem__('stypy_call_varargs', varargs)
        soduko.set_row.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        soduko.set_row.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'soduko.set_row', ['row', 'x_list'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_row', localization, ['row', 'x_list'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_row(...)' code ##################

        # Evaluating assert statement condition
        
        
        # Call to len(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'x_list' (line 33)
        x_list_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 19), 'x_list', False)
        # Processing the call keyword arguments (line 33)
        kwargs_153 = {}
        # Getting the type of 'len' (line 33)
        len_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'len', False)
        # Calling len(args, kwargs) (line 33)
        len_call_result_154 = invoke(stypy.reporting.localization.Localization(__file__, 33, 15), len_151, *[x_list_152], **kwargs_153)
        
        int_155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 30), 'int')
        # Applying the binary operator '==' (line 33)
        result_eq_156 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 15), '==', len_call_result_154, int_155)
        
        assert_157 = result_eq_156
        # Assigning a type to the variable 'assert_157' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'assert_157', result_eq_156)
        
        
        # Call to range(...): (line 34)
        # Processing the call arguments (line 34)
        int_159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 25), 'int')
        int_160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 28), 'int')
        # Processing the call keyword arguments (line 34)
        kwargs_161 = {}
        # Getting the type of 'range' (line 34)
        range_158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'range', False)
        # Calling range(args, kwargs) (line 34)
        range_call_result_162 = invoke(stypy.reporting.localization.Localization(__file__, 34, 19), range_158, *[int_159, int_160], **kwargs_161)
        
        # Assigning a type to the variable 'range_call_result_162' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'range_call_result_162', range_call_result_162)
        # Testing if the for loop is going to be iterated (line 34)
        # Testing the type of a for loop iterable (line 34)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 34, 8), range_call_result_162)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 34, 8), range_call_result_162):
            # Getting the type of the for loop variable (line 34)
            for_loop_var_163 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 34, 8), range_call_result_162)
            # Assigning a type to the variable 'col' (line 34)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'col', for_loop_var_163)
            # SSA begins for a for statement (line 34)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # SSA begins for try-except statement (line 35)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Name (line 36):
            
            # Assigning a Call to a Name (line 36):
            
            # Call to int(...): (line 36)
            # Processing the call arguments (line 36)
            
            # Obtaining the type of the subscript
            # Getting the type of 'col' (line 36)
            col_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 31), 'col', False)
            # Getting the type of 'x_list' (line 36)
            x_list_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 24), 'x_list', False)
            # Obtaining the member '__getitem__' of a type (line 36)
            getitem___167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 24), x_list_166, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 36)
            subscript_call_result_168 = invoke(stypy.reporting.localization.Localization(__file__, 36, 24), getitem___167, col_165)
            
            # Processing the call keyword arguments (line 36)
            kwargs_169 = {}
            # Getting the type of 'int' (line 36)
            int_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 20), 'int', False)
            # Calling int(args, kwargs) (line 36)
            int_call_result_170 = invoke(stypy.reporting.localization.Localization(__file__, 36, 20), int_164, *[subscript_call_result_168], **kwargs_169)
            
            # Assigning a type to the variable 'x' (line 36)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'x', int_call_result_170)
            # SSA branch for the except part of a try statement (line 35)
            # SSA branch for the except '<any exception>' branch of a try statement (line 35)
            module_type_store.open_ssa_branch('except')
            
            # Assigning a Num to a Name (line 38):
            
            # Assigning a Num to a Name (line 38):
            int_171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 20), 'int')
            # Assigning a type to the variable 'x' (line 38)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'x', int_171)
            # SSA join for try-except statement (line 35)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to set_cell(...): (line 39)
            # Processing the call arguments (line 39)
            # Getting the type of 'row' (line 39)
            row_174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 26), 'row', False)
            # Getting the type of 'col' (line 39)
            col_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 31), 'col', False)
            # Getting the type of 'x' (line 39)
            x_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 36), 'x', False)
            # Processing the call keyword arguments (line 39)
            kwargs_177 = {}
            # Getting the type of 'self' (line 39)
            self_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'self', False)
            # Obtaining the member 'set_cell' of a type (line 39)
            set_cell_173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 12), self_172, 'set_cell')
            # Calling set_cell(args, kwargs) (line 39)
            set_cell_call_result_178 = invoke(stypy.reporting.localization.Localization(__file__, 39, 12), set_cell_173, *[row_174, col_175, x_176], **kwargs_177)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'set_row(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_row' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_179)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_row'
        return stypy_return_type_179


    @norecursion
    def set_cell(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_cell'
        module_type_store = module_type_store.open_function_context('set_cell', 41, 4, False)
        # Assigning a type to the variable 'self' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        soduko.set_cell.__dict__.__setitem__('stypy_localization', localization)
        soduko.set_cell.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        soduko.set_cell.__dict__.__setitem__('stypy_type_store', module_type_store)
        soduko.set_cell.__dict__.__setitem__('stypy_function_name', 'soduko.set_cell')
        soduko.set_cell.__dict__.__setitem__('stypy_param_names_list', ['row', 'col', 'x'])
        soduko.set_cell.__dict__.__setitem__('stypy_varargs_param_name', None)
        soduko.set_cell.__dict__.__setitem__('stypy_kwargs_param_name', None)
        soduko.set_cell.__dict__.__setitem__('stypy_call_defaults', defaults)
        soduko.set_cell.__dict__.__setitem__('stypy_call_varargs', varargs)
        soduko.set_cell.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        soduko.set_cell.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'soduko.set_cell', ['row', 'col', 'x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_cell', localization, ['row', 'col', 'x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_cell(...)' code ##################

        
        
        # Obtaining the type of the subscript
        # Getting the type of 'col' (line 42)
        col_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 29), 'col')
        
        # Obtaining the type of the subscript
        # Getting the type of 'row' (line 42)
        row_181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 24), 'row')
        # Getting the type of 'self' (line 42)
        self_182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'self')
        # Obtaining the member 'squares' of a type (line 42)
        squares_183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 11), self_182, 'squares')
        # Obtaining the member '__getitem__' of a type (line 42)
        getitem___184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 11), squares_183, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 42)
        subscript_call_result_185 = invoke(stypy.reporting.localization.Localization(__file__, 42, 11), getitem___184, row_181)
        
        # Obtaining the member '__getitem__' of a type (line 42)
        getitem___186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 11), subscript_call_result_185, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 42)
        subscript_call_result_187 = invoke(stypy.reporting.localization.Localization(__file__, 42, 11), getitem___186, col_180)
        
        
        # Obtaining an instance of the builtin type 'list' (line 42)
        list_188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 42)
        # Adding element type (line 42)
        # Getting the type of 'x' (line 42)
        x_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 38), 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 37), list_188, x_189)
        
        # Applying the binary operator '==' (line 42)
        result_eq_190 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 11), '==', subscript_call_result_187, list_188)
        
        # Testing if the type of an if condition is none (line 42)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 42, 8), result_eq_190):
            
            # Getting the type of 'x' (line 44)
            x_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'x')
            
            # Call to range(...): (line 44)
            # Processing the call arguments (line 44)
            int_194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 28), 'int')
            int_195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 31), 'int')
            int_196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 35), 'int')
            # Applying the binary operator '+' (line 44)
            result_add_197 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 31), '+', int_195, int_196)
            
            # Processing the call keyword arguments (line 44)
            kwargs_198 = {}
            # Getting the type of 'range' (line 44)
            range_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 22), 'range', False)
            # Calling range(args, kwargs) (line 44)
            range_call_result_199 = invoke(stypy.reporting.localization.Localization(__file__, 44, 22), range_193, *[int_194, result_add_197], **kwargs_198)
            
            # Applying the binary operator 'notin' (line 44)
            result_contains_200 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 13), 'notin', x_192, range_call_result_199)
            
            # Testing if the type of an if condition is none (line 44)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 44, 13), result_contains_200):
                # Evaluating assert statement condition
                
                # Getting the type of 'x' (line 47)
                x_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 19), 'x')
                
                # Obtaining the type of the subscript
                # Getting the type of 'col' (line 47)
                col_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 42), 'col')
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 47)
                row_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 37), 'row')
                # Getting the type of 'self' (line 47)
                self_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 24), 'self')
                # Obtaining the member 'squares' of a type (line 47)
                squares_206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), self_205, 'squares')
                # Obtaining the member '__getitem__' of a type (line 47)
                getitem___207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), squares_206, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 47)
                subscript_call_result_208 = invoke(stypy.reporting.localization.Localization(__file__, 47, 24), getitem___207, row_204)
                
                # Obtaining the member '__getitem__' of a type (line 47)
                getitem___209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), subscript_call_result_208, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 47)
                subscript_call_result_210 = invoke(stypy.reporting.localization.Localization(__file__, 47, 24), getitem___209, col_203)
                
                # Applying the binary operator 'in' (line 47)
                result_contains_211 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 19), 'in', x_202, subscript_call_result_210)
                
                assert_212 = result_contains_211
                # Assigning a type to the variable 'assert_212' (line 47)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'assert_212', result_contains_211)
                
                # Assigning a List to a Subscript (line 49):
                
                # Assigning a List to a Subscript (line 49):
                
                # Obtaining an instance of the builtin type 'list' (line 49)
                list_213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 37), 'list')
                # Adding type elements to the builtin type 'list' instance (line 49)
                # Adding element type (line 49)
                # Getting the type of 'x' (line 49)
                x_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 38), 'x')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 37), list_213, x_214)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 49)
                row_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'row')
                # Getting the type of 'self' (line 49)
                self_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'self')
                # Obtaining the member 'squares' of a type (line 49)
                squares_217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), self_216, 'squares')
                # Obtaining the member '__getitem__' of a type (line 49)
                getitem___218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), squares_217, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 49)
                subscript_call_result_219 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), getitem___218, row_215)
                
                # Getting the type of 'col' (line 49)
                col_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 30), 'col')
                # Storing an element on a container (line 49)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 12), subscript_call_result_219, (col_220, list_213))
                
                # Call to update_neighbours(...): (line 50)
                # Processing the call arguments (line 50)
                # Getting the type of 'row' (line 50)
                row_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 35), 'row', False)
                # Getting the type of 'col' (line 50)
                col_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 40), 'col', False)
                # Getting the type of 'x' (line 50)
                x_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 45), 'x', False)
                # Processing the call keyword arguments (line 50)
                kwargs_226 = {}
                # Getting the type of 'self' (line 50)
                self_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'self', False)
                # Obtaining the member 'update_neighbours' of a type (line 50)
                update_neighbours_222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), self_221, 'update_neighbours')
                # Calling update_neighbours(args, kwargs) (line 50)
                update_neighbours_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), update_neighbours_222, *[row_223, col_224, x_225], **kwargs_226)
                
                
                # Assigning a Name to a Attribute (line 51):
                
                # Assigning a Name to a Attribute (line 51):
                # Getting the type of 'True' (line 51)
                True_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 28), 'True')
                # Getting the type of 'self' (line 51)
                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'self')
                # Setting the type of the member '_changed' of a type (line 51)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), self_229, '_changed', True_228)
            else:
                
                # Testing the type of an if condition (line 44)
                if_condition_201 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 13), result_contains_200)
                # Assigning a type to the variable 'if_condition_201' (line 44)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'if_condition_201', if_condition_201)
                # SSA begins for if statement (line 44)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                pass
                # SSA branch for the else part of an if statement (line 44)
                module_type_store.open_ssa_branch('else')
                # Evaluating assert statement condition
                
                # Getting the type of 'x' (line 47)
                x_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 19), 'x')
                
                # Obtaining the type of the subscript
                # Getting the type of 'col' (line 47)
                col_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 42), 'col')
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 47)
                row_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 37), 'row')
                # Getting the type of 'self' (line 47)
                self_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 24), 'self')
                # Obtaining the member 'squares' of a type (line 47)
                squares_206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), self_205, 'squares')
                # Obtaining the member '__getitem__' of a type (line 47)
                getitem___207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), squares_206, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 47)
                subscript_call_result_208 = invoke(stypy.reporting.localization.Localization(__file__, 47, 24), getitem___207, row_204)
                
                # Obtaining the member '__getitem__' of a type (line 47)
                getitem___209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), subscript_call_result_208, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 47)
                subscript_call_result_210 = invoke(stypy.reporting.localization.Localization(__file__, 47, 24), getitem___209, col_203)
                
                # Applying the binary operator 'in' (line 47)
                result_contains_211 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 19), 'in', x_202, subscript_call_result_210)
                
                assert_212 = result_contains_211
                # Assigning a type to the variable 'assert_212' (line 47)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'assert_212', result_contains_211)
                
                # Assigning a List to a Subscript (line 49):
                
                # Assigning a List to a Subscript (line 49):
                
                # Obtaining an instance of the builtin type 'list' (line 49)
                list_213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 37), 'list')
                # Adding type elements to the builtin type 'list' instance (line 49)
                # Adding element type (line 49)
                # Getting the type of 'x' (line 49)
                x_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 38), 'x')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 37), list_213, x_214)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 49)
                row_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'row')
                # Getting the type of 'self' (line 49)
                self_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'self')
                # Obtaining the member 'squares' of a type (line 49)
                squares_217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), self_216, 'squares')
                # Obtaining the member '__getitem__' of a type (line 49)
                getitem___218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), squares_217, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 49)
                subscript_call_result_219 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), getitem___218, row_215)
                
                # Getting the type of 'col' (line 49)
                col_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 30), 'col')
                # Storing an element on a container (line 49)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 12), subscript_call_result_219, (col_220, list_213))
                
                # Call to update_neighbours(...): (line 50)
                # Processing the call arguments (line 50)
                # Getting the type of 'row' (line 50)
                row_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 35), 'row', False)
                # Getting the type of 'col' (line 50)
                col_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 40), 'col', False)
                # Getting the type of 'x' (line 50)
                x_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 45), 'x', False)
                # Processing the call keyword arguments (line 50)
                kwargs_226 = {}
                # Getting the type of 'self' (line 50)
                self_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'self', False)
                # Obtaining the member 'update_neighbours' of a type (line 50)
                update_neighbours_222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), self_221, 'update_neighbours')
                # Calling update_neighbours(args, kwargs) (line 50)
                update_neighbours_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), update_neighbours_222, *[row_223, col_224, x_225], **kwargs_226)
                
                
                # Assigning a Name to a Attribute (line 51):
                
                # Assigning a Name to a Attribute (line 51):
                # Getting the type of 'True' (line 51)
                True_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 28), 'True')
                # Getting the type of 'self' (line 51)
                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'self')
                # Setting the type of the member '_changed' of a type (line 51)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), self_229, '_changed', True_228)
                # SSA join for if statement (line 44)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 42)
            if_condition_191 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 8), result_eq_190)
            # Assigning a type to the variable 'if_condition_191' (line 42)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'if_condition_191', if_condition_191)
            # SSA begins for if statement (line 42)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            pass
            # SSA branch for the else part of an if statement (line 42)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'x' (line 44)
            x_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'x')
            
            # Call to range(...): (line 44)
            # Processing the call arguments (line 44)
            int_194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 28), 'int')
            int_195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 31), 'int')
            int_196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 35), 'int')
            # Applying the binary operator '+' (line 44)
            result_add_197 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 31), '+', int_195, int_196)
            
            # Processing the call keyword arguments (line 44)
            kwargs_198 = {}
            # Getting the type of 'range' (line 44)
            range_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 22), 'range', False)
            # Calling range(args, kwargs) (line 44)
            range_call_result_199 = invoke(stypy.reporting.localization.Localization(__file__, 44, 22), range_193, *[int_194, result_add_197], **kwargs_198)
            
            # Applying the binary operator 'notin' (line 44)
            result_contains_200 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 13), 'notin', x_192, range_call_result_199)
            
            # Testing if the type of an if condition is none (line 44)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 44, 13), result_contains_200):
                # Evaluating assert statement condition
                
                # Getting the type of 'x' (line 47)
                x_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 19), 'x')
                
                # Obtaining the type of the subscript
                # Getting the type of 'col' (line 47)
                col_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 42), 'col')
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 47)
                row_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 37), 'row')
                # Getting the type of 'self' (line 47)
                self_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 24), 'self')
                # Obtaining the member 'squares' of a type (line 47)
                squares_206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), self_205, 'squares')
                # Obtaining the member '__getitem__' of a type (line 47)
                getitem___207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), squares_206, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 47)
                subscript_call_result_208 = invoke(stypy.reporting.localization.Localization(__file__, 47, 24), getitem___207, row_204)
                
                # Obtaining the member '__getitem__' of a type (line 47)
                getitem___209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), subscript_call_result_208, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 47)
                subscript_call_result_210 = invoke(stypy.reporting.localization.Localization(__file__, 47, 24), getitem___209, col_203)
                
                # Applying the binary operator 'in' (line 47)
                result_contains_211 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 19), 'in', x_202, subscript_call_result_210)
                
                assert_212 = result_contains_211
                # Assigning a type to the variable 'assert_212' (line 47)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'assert_212', result_contains_211)
                
                # Assigning a List to a Subscript (line 49):
                
                # Assigning a List to a Subscript (line 49):
                
                # Obtaining an instance of the builtin type 'list' (line 49)
                list_213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 37), 'list')
                # Adding type elements to the builtin type 'list' instance (line 49)
                # Adding element type (line 49)
                # Getting the type of 'x' (line 49)
                x_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 38), 'x')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 37), list_213, x_214)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 49)
                row_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'row')
                # Getting the type of 'self' (line 49)
                self_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'self')
                # Obtaining the member 'squares' of a type (line 49)
                squares_217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), self_216, 'squares')
                # Obtaining the member '__getitem__' of a type (line 49)
                getitem___218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), squares_217, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 49)
                subscript_call_result_219 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), getitem___218, row_215)
                
                # Getting the type of 'col' (line 49)
                col_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 30), 'col')
                # Storing an element on a container (line 49)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 12), subscript_call_result_219, (col_220, list_213))
                
                # Call to update_neighbours(...): (line 50)
                # Processing the call arguments (line 50)
                # Getting the type of 'row' (line 50)
                row_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 35), 'row', False)
                # Getting the type of 'col' (line 50)
                col_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 40), 'col', False)
                # Getting the type of 'x' (line 50)
                x_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 45), 'x', False)
                # Processing the call keyword arguments (line 50)
                kwargs_226 = {}
                # Getting the type of 'self' (line 50)
                self_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'self', False)
                # Obtaining the member 'update_neighbours' of a type (line 50)
                update_neighbours_222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), self_221, 'update_neighbours')
                # Calling update_neighbours(args, kwargs) (line 50)
                update_neighbours_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), update_neighbours_222, *[row_223, col_224, x_225], **kwargs_226)
                
                
                # Assigning a Name to a Attribute (line 51):
                
                # Assigning a Name to a Attribute (line 51):
                # Getting the type of 'True' (line 51)
                True_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 28), 'True')
                # Getting the type of 'self' (line 51)
                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'self')
                # Setting the type of the member '_changed' of a type (line 51)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), self_229, '_changed', True_228)
            else:
                
                # Testing the type of an if condition (line 44)
                if_condition_201 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 13), result_contains_200)
                # Assigning a type to the variable 'if_condition_201' (line 44)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'if_condition_201', if_condition_201)
                # SSA begins for if statement (line 44)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                pass
                # SSA branch for the else part of an if statement (line 44)
                module_type_store.open_ssa_branch('else')
                # Evaluating assert statement condition
                
                # Getting the type of 'x' (line 47)
                x_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 19), 'x')
                
                # Obtaining the type of the subscript
                # Getting the type of 'col' (line 47)
                col_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 42), 'col')
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 47)
                row_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 37), 'row')
                # Getting the type of 'self' (line 47)
                self_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 24), 'self')
                # Obtaining the member 'squares' of a type (line 47)
                squares_206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), self_205, 'squares')
                # Obtaining the member '__getitem__' of a type (line 47)
                getitem___207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), squares_206, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 47)
                subscript_call_result_208 = invoke(stypy.reporting.localization.Localization(__file__, 47, 24), getitem___207, row_204)
                
                # Obtaining the member '__getitem__' of a type (line 47)
                getitem___209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), subscript_call_result_208, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 47)
                subscript_call_result_210 = invoke(stypy.reporting.localization.Localization(__file__, 47, 24), getitem___209, col_203)
                
                # Applying the binary operator 'in' (line 47)
                result_contains_211 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 19), 'in', x_202, subscript_call_result_210)
                
                assert_212 = result_contains_211
                # Assigning a type to the variable 'assert_212' (line 47)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'assert_212', result_contains_211)
                
                # Assigning a List to a Subscript (line 49):
                
                # Assigning a List to a Subscript (line 49):
                
                # Obtaining an instance of the builtin type 'list' (line 49)
                list_213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 37), 'list')
                # Adding type elements to the builtin type 'list' instance (line 49)
                # Adding element type (line 49)
                # Getting the type of 'x' (line 49)
                x_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 38), 'x')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 37), list_213, x_214)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 49)
                row_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'row')
                # Getting the type of 'self' (line 49)
                self_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'self')
                # Obtaining the member 'squares' of a type (line 49)
                squares_217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), self_216, 'squares')
                # Obtaining the member '__getitem__' of a type (line 49)
                getitem___218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), squares_217, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 49)
                subscript_call_result_219 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), getitem___218, row_215)
                
                # Getting the type of 'col' (line 49)
                col_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 30), 'col')
                # Storing an element on a container (line 49)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 12), subscript_call_result_219, (col_220, list_213))
                
                # Call to update_neighbours(...): (line 50)
                # Processing the call arguments (line 50)
                # Getting the type of 'row' (line 50)
                row_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 35), 'row', False)
                # Getting the type of 'col' (line 50)
                col_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 40), 'col', False)
                # Getting the type of 'x' (line 50)
                x_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 45), 'x', False)
                # Processing the call keyword arguments (line 50)
                kwargs_226 = {}
                # Getting the type of 'self' (line 50)
                self_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'self', False)
                # Obtaining the member 'update_neighbours' of a type (line 50)
                update_neighbours_222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), self_221, 'update_neighbours')
                # Calling update_neighbours(args, kwargs) (line 50)
                update_neighbours_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), update_neighbours_222, *[row_223, col_224, x_225], **kwargs_226)
                
                
                # Assigning a Name to a Attribute (line 51):
                
                # Assigning a Name to a Attribute (line 51):
                # Getting the type of 'True' (line 51)
                True_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 28), 'True')
                # Getting the type of 'self' (line 51)
                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'self')
                # Setting the type of the member '_changed' of a type (line 51)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), self_229, '_changed', True_228)
                # SSA join for if statement (line 44)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 42)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'set_cell(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_cell' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_230)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_cell'
        return stypy_return_type_230


    @norecursion
    def cell_exclude(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'cell_exclude'
        module_type_store = module_type_store.open_function_context('cell_exclude', 53, 4, False)
        # Assigning a type to the variable 'self' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        soduko.cell_exclude.__dict__.__setitem__('stypy_localization', localization)
        soduko.cell_exclude.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        soduko.cell_exclude.__dict__.__setitem__('stypy_type_store', module_type_store)
        soduko.cell_exclude.__dict__.__setitem__('stypy_function_name', 'soduko.cell_exclude')
        soduko.cell_exclude.__dict__.__setitem__('stypy_param_names_list', ['row', 'col', 'x'])
        soduko.cell_exclude.__dict__.__setitem__('stypy_varargs_param_name', None)
        soduko.cell_exclude.__dict__.__setitem__('stypy_kwargs_param_name', None)
        soduko.cell_exclude.__dict__.__setitem__('stypy_call_defaults', defaults)
        soduko.cell_exclude.__dict__.__setitem__('stypy_call_varargs', varargs)
        soduko.cell_exclude.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        soduko.cell_exclude.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'soduko.cell_exclude', ['row', 'col', 'x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'cell_exclude', localization, ['row', 'col', 'x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'cell_exclude(...)' code ##################

        # Evaluating assert statement condition
        
        # Getting the type of 'x' (line 54)
        x_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 15), 'x')
        
        # Call to range(...): (line 54)
        # Processing the call arguments (line 54)
        int_233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 26), 'int')
        int_234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 29), 'int')
        int_235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 33), 'int')
        # Applying the binary operator '+' (line 54)
        result_add_236 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 29), '+', int_234, int_235)
        
        # Processing the call keyword arguments (line 54)
        kwargs_237 = {}
        # Getting the type of 'range' (line 54)
        range_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'range', False)
        # Calling range(args, kwargs) (line 54)
        range_call_result_238 = invoke(stypy.reporting.localization.Localization(__file__, 54, 20), range_232, *[int_233, result_add_236], **kwargs_237)
        
        # Applying the binary operator 'in' (line 54)
        result_contains_239 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 15), 'in', x_231, range_call_result_238)
        
        assert_240 = result_contains_239
        # Assigning a type to the variable 'assert_240' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'assert_240', result_contains_239)
        
        # Getting the type of 'x' (line 55)
        x_241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 11), 'x')
        
        # Obtaining the type of the subscript
        # Getting the type of 'col' (line 55)
        col_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 34), 'col')
        
        # Obtaining the type of the subscript
        # Getting the type of 'row' (line 55)
        row_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 29), 'row')
        # Getting the type of 'self' (line 55)
        self_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'self')
        # Obtaining the member 'squares' of a type (line 55)
        squares_245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 16), self_244, 'squares')
        # Obtaining the member '__getitem__' of a type (line 55)
        getitem___246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 16), squares_245, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 55)
        subscript_call_result_247 = invoke(stypy.reporting.localization.Localization(__file__, 55, 16), getitem___246, row_243)
        
        # Obtaining the member '__getitem__' of a type (line 55)
        getitem___248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 16), subscript_call_result_247, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 55)
        subscript_call_result_249 = invoke(stypy.reporting.localization.Localization(__file__, 55, 16), getitem___248, col_242)
        
        # Applying the binary operator 'in' (line 55)
        result_contains_250 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 11), 'in', x_241, subscript_call_result_249)
        
        # Testing if the type of an if condition is none (line 55)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 55, 8), result_contains_250):
            pass
        else:
            
            # Testing the type of an if condition (line 55)
            if_condition_251 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 55, 8), result_contains_250)
            # Assigning a type to the variable 'if_condition_251' (line 55)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'if_condition_251', if_condition_251)
            # SSA begins for if statement (line 55)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to remove(...): (line 56)
            # Processing the call arguments (line 56)
            # Getting the type of 'x' (line 56)
            x_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 42), 'x', False)
            # Processing the call keyword arguments (line 56)
            kwargs_262 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'col' (line 56)
            col_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 30), 'col', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row' (line 56)
            row_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 25), 'row', False)
            # Getting the type of 'self' (line 56)
            self_254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'self', False)
            # Obtaining the member 'squares' of a type (line 56)
            squares_255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 12), self_254, 'squares')
            # Obtaining the member '__getitem__' of a type (line 56)
            getitem___256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 12), squares_255, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 56)
            subscript_call_result_257 = invoke(stypy.reporting.localization.Localization(__file__, 56, 12), getitem___256, row_253)
            
            # Obtaining the member '__getitem__' of a type (line 56)
            getitem___258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 12), subscript_call_result_257, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 56)
            subscript_call_result_259 = invoke(stypy.reporting.localization.Localization(__file__, 56, 12), getitem___258, col_252)
            
            # Obtaining the member 'remove' of a type (line 56)
            remove_260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 12), subscript_call_result_259, 'remove')
            # Calling remove(args, kwargs) (line 56)
            remove_call_result_263 = invoke(stypy.reporting.localization.Localization(__file__, 56, 12), remove_260, *[x_261], **kwargs_262)
            
            # Evaluating assert statement condition
            
            
            # Call to len(...): (line 57)
            # Processing the call arguments (line 57)
            
            # Obtaining the type of the subscript
            # Getting the type of 'col' (line 57)
            col_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 41), 'col', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row' (line 57)
            row_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 36), 'row', False)
            # Getting the type of 'self' (line 57)
            self_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 23), 'self', False)
            # Obtaining the member 'squares' of a type (line 57)
            squares_268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 23), self_267, 'squares')
            # Obtaining the member '__getitem__' of a type (line 57)
            getitem___269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 23), squares_268, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 57)
            subscript_call_result_270 = invoke(stypy.reporting.localization.Localization(__file__, 57, 23), getitem___269, row_266)
            
            # Obtaining the member '__getitem__' of a type (line 57)
            getitem___271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 23), subscript_call_result_270, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 57)
            subscript_call_result_272 = invoke(stypy.reporting.localization.Localization(__file__, 57, 23), getitem___271, col_265)
            
            # Processing the call keyword arguments (line 57)
            kwargs_273 = {}
            # Getting the type of 'len' (line 57)
            len_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 19), 'len', False)
            # Calling len(args, kwargs) (line 57)
            len_call_result_274 = invoke(stypy.reporting.localization.Localization(__file__, 57, 19), len_264, *[subscript_call_result_272], **kwargs_273)
            
            int_275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 49), 'int')
            # Applying the binary operator '>' (line 57)
            result_gt_276 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 19), '>', len_call_result_274, int_275)
            
            assert_277 = result_gt_276
            # Assigning a type to the variable 'assert_277' (line 57)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'assert_277', result_gt_276)
            
            
            # Call to len(...): (line 58)
            # Processing the call arguments (line 58)
            
            # Obtaining the type of the subscript
            # Getting the type of 'col' (line 58)
            col_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 37), 'col', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row' (line 58)
            row_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 32), 'row', False)
            # Getting the type of 'self' (line 58)
            self_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 19), 'self', False)
            # Obtaining the member 'squares' of a type (line 58)
            squares_282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 19), self_281, 'squares')
            # Obtaining the member '__getitem__' of a type (line 58)
            getitem___283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 19), squares_282, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 58)
            subscript_call_result_284 = invoke(stypy.reporting.localization.Localization(__file__, 58, 19), getitem___283, row_280)
            
            # Obtaining the member '__getitem__' of a type (line 58)
            getitem___285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 19), subscript_call_result_284, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 58)
            subscript_call_result_286 = invoke(stypy.reporting.localization.Localization(__file__, 58, 19), getitem___285, col_279)
            
            # Processing the call keyword arguments (line 58)
            kwargs_287 = {}
            # Getting the type of 'len' (line 58)
            len_278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 15), 'len', False)
            # Calling len(args, kwargs) (line 58)
            len_call_result_288 = invoke(stypy.reporting.localization.Localization(__file__, 58, 15), len_278, *[subscript_call_result_286], **kwargs_287)
            
            int_289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 46), 'int')
            # Applying the binary operator '==' (line 58)
            result_eq_290 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 15), '==', len_call_result_288, int_289)
            
            # Testing if the type of an if condition is none (line 58)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 58, 12), result_eq_290):
                pass
            else:
                
                # Testing the type of an if condition (line 58)
                if_condition_291 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 12), result_eq_290)
                # Assigning a type to the variable 'if_condition_291' (line 58)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'if_condition_291', if_condition_291)
                # SSA begins for if statement (line 58)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Attribute (line 59):
                
                # Assigning a Name to a Attribute (line 59):
                # Getting the type of 'True' (line 59)
                True_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 32), 'True')
                # Getting the type of 'self' (line 59)
                self_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'self')
                # Setting the type of the member '_changed' of a type (line 59)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 16), self_293, '_changed', True_292)
                
                # Call to update_neighbours(...): (line 60)
                # Processing the call arguments (line 60)
                # Getting the type of 'row' (line 60)
                row_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 39), 'row', False)
                # Getting the type of 'col' (line 60)
                col_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 44), 'col', False)
                
                # Obtaining the type of the subscript
                int_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 72), 'int')
                
                # Obtaining the type of the subscript
                # Getting the type of 'col' (line 60)
                col_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 67), 'col', False)
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 60)
                row_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 62), 'row', False)
                # Getting the type of 'self' (line 60)
                self_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 49), 'self', False)
                # Obtaining the member 'squares' of a type (line 60)
                squares_302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 49), self_301, 'squares')
                # Obtaining the member '__getitem__' of a type (line 60)
                getitem___303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 49), squares_302, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 60)
                subscript_call_result_304 = invoke(stypy.reporting.localization.Localization(__file__, 60, 49), getitem___303, row_300)
                
                # Obtaining the member '__getitem__' of a type (line 60)
                getitem___305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 49), subscript_call_result_304, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 60)
                subscript_call_result_306 = invoke(stypy.reporting.localization.Localization(__file__, 60, 49), getitem___305, col_299)
                
                # Obtaining the member '__getitem__' of a type (line 60)
                getitem___307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 49), subscript_call_result_306, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 60)
                subscript_call_result_308 = invoke(stypy.reporting.localization.Localization(__file__, 60, 49), getitem___307, int_298)
                
                # Processing the call keyword arguments (line 60)
                kwargs_309 = {}
                # Getting the type of 'self' (line 60)
                self_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'self', False)
                # Obtaining the member 'update_neighbours' of a type (line 60)
                update_neighbours_295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 16), self_294, 'update_neighbours')
                # Calling update_neighbours(args, kwargs) (line 60)
                update_neighbours_call_result_310 = invoke(stypy.reporting.localization.Localization(__file__, 60, 16), update_neighbours_295, *[row_296, col_297, subscript_call_result_308], **kwargs_309)
                
                # SSA join for if statement (line 58)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA branch for the else part of an if statement (line 55)
            module_type_store.open_ssa_branch('else')
            pass
            # SSA join for if statement (line 55)
            module_type_store = module_type_store.join_ssa_context()
            

        # Assigning a type to the variable 'stypy_return_type' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of 'cell_exclude(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cell_exclude' in the type store
        # Getting the type of 'stypy_return_type' (line 53)
        stypy_return_type_311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_311)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cell_exclude'
        return stypy_return_type_311


    @norecursion
    def update_neighbours(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update_neighbours'
        module_type_store = module_type_store.open_function_context('update_neighbours', 65, 4, False)
        # Assigning a type to the variable 'self' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        soduko.update_neighbours.__dict__.__setitem__('stypy_localization', localization)
        soduko.update_neighbours.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        soduko.update_neighbours.__dict__.__setitem__('stypy_type_store', module_type_store)
        soduko.update_neighbours.__dict__.__setitem__('stypy_function_name', 'soduko.update_neighbours')
        soduko.update_neighbours.__dict__.__setitem__('stypy_param_names_list', ['set_row', 'set_col', 'x'])
        soduko.update_neighbours.__dict__.__setitem__('stypy_varargs_param_name', None)
        soduko.update_neighbours.__dict__.__setitem__('stypy_kwargs_param_name', None)
        soduko.update_neighbours.__dict__.__setitem__('stypy_call_defaults', defaults)
        soduko.update_neighbours.__dict__.__setitem__('stypy_call_varargs', varargs)
        soduko.update_neighbours.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        soduko.update_neighbours.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'soduko.update_neighbours', ['set_row', 'set_col', 'x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update_neighbours', localization, ['set_row', 'set_col', 'x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update_neighbours(...)' code ##################

        
        
        # Call to range(...): (line 66)
        # Processing the call arguments (line 66)
        int_313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 25), 'int')
        int_314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 28), 'int')
        # Processing the call keyword arguments (line 66)
        kwargs_315 = {}
        # Getting the type of 'range' (line 66)
        range_312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 19), 'range', False)
        # Calling range(args, kwargs) (line 66)
        range_call_result_316 = invoke(stypy.reporting.localization.Localization(__file__, 66, 19), range_312, *[int_313, int_314], **kwargs_315)
        
        # Assigning a type to the variable 'range_call_result_316' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'range_call_result_316', range_call_result_316)
        # Testing if the for loop is going to be iterated (line 66)
        # Testing the type of a for loop iterable (line 66)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 66, 8), range_call_result_316)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 66, 8), range_call_result_316):
            # Getting the type of the for loop variable (line 66)
            for_loop_var_317 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 66, 8), range_call_result_316)
            # Assigning a type to the variable 'row' (line 66)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'row', for_loop_var_317)
            # SSA begins for a for statement (line 66)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'row' (line 67)
            row_318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 15), 'row')
            # Getting the type of 'set_row' (line 67)
            set_row_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 22), 'set_row')
            # Applying the binary operator '!=' (line 67)
            result_ne_320 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 15), '!=', row_318, set_row_319)
            
            # Testing if the type of an if condition is none (line 67)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 67, 12), result_ne_320):
                pass
            else:
                
                # Testing the type of an if condition (line 67)
                if_condition_321 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 12), result_ne_320)
                # Assigning a type to the variable 'if_condition_321' (line 67)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'if_condition_321', if_condition_321)
                # SSA begins for if statement (line 67)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to cell_exclude(...): (line 68)
                # Processing the call arguments (line 68)
                # Getting the type of 'row' (line 68)
                row_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 34), 'row', False)
                # Getting the type of 'set_col' (line 68)
                set_col_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 39), 'set_col', False)
                # Getting the type of 'x' (line 68)
                x_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 48), 'x', False)
                # Processing the call keyword arguments (line 68)
                kwargs_327 = {}
                # Getting the type of 'self' (line 68)
                self_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'self', False)
                # Obtaining the member 'cell_exclude' of a type (line 68)
                cell_exclude_323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 16), self_322, 'cell_exclude')
                # Calling cell_exclude(args, kwargs) (line 68)
                cell_exclude_call_result_328 = invoke(stypy.reporting.localization.Localization(__file__, 68, 16), cell_exclude_323, *[row_324, set_col_325, x_326], **kwargs_327)
                
                # SSA join for if statement (line 67)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to range(...): (line 69)
        # Processing the call arguments (line 69)
        int_330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 25), 'int')
        int_331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 28), 'int')
        # Processing the call keyword arguments (line 69)
        kwargs_332 = {}
        # Getting the type of 'range' (line 69)
        range_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 19), 'range', False)
        # Calling range(args, kwargs) (line 69)
        range_call_result_333 = invoke(stypy.reporting.localization.Localization(__file__, 69, 19), range_329, *[int_330, int_331], **kwargs_332)
        
        # Assigning a type to the variable 'range_call_result_333' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'range_call_result_333', range_call_result_333)
        # Testing if the for loop is going to be iterated (line 69)
        # Testing the type of a for loop iterable (line 69)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 69, 8), range_call_result_333)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 69, 8), range_call_result_333):
            # Getting the type of the for loop variable (line 69)
            for_loop_var_334 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 69, 8), range_call_result_333)
            # Assigning a type to the variable 'col' (line 69)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'col', for_loop_var_334)
            # SSA begins for a for statement (line 69)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'col' (line 70)
            col_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 15), 'col')
            # Getting the type of 'set_col' (line 70)
            set_col_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 22), 'set_col')
            # Applying the binary operator '!=' (line 70)
            result_ne_337 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 15), '!=', col_335, set_col_336)
            
            # Testing if the type of an if condition is none (line 70)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 70, 12), result_ne_337):
                pass
            else:
                
                # Testing the type of an if condition (line 70)
                if_condition_338 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 12), result_ne_337)
                # Assigning a type to the variable 'if_condition_338' (line 70)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'if_condition_338', if_condition_338)
                # SSA begins for if statement (line 70)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to cell_exclude(...): (line 71)
                # Processing the call arguments (line 71)
                # Getting the type of 'set_row' (line 71)
                set_row_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 34), 'set_row', False)
                # Getting the type of 'col' (line 71)
                col_342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 43), 'col', False)
                # Getting the type of 'x' (line 71)
                x_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 48), 'x', False)
                # Processing the call keyword arguments (line 71)
                kwargs_344 = {}
                # Getting the type of 'self' (line 71)
                self_339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'self', False)
                # Obtaining the member 'cell_exclude' of a type (line 71)
                cell_exclude_340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 16), self_339, 'cell_exclude')
                # Calling cell_exclude(args, kwargs) (line 71)
                cell_exclude_call_result_345 = invoke(stypy.reporting.localization.Localization(__file__, 71, 16), cell_exclude_340, *[set_row_341, col_342, x_343], **kwargs_344)
                
                # SSA join for if statement (line 70)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 'TRIPLETS' (line 72)
        TRIPLETS_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 23), 'TRIPLETS')
        # Assigning a type to the variable 'TRIPLETS_346' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'TRIPLETS_346', TRIPLETS_346)
        # Testing if the for loop is going to be iterated (line 72)
        # Testing the type of a for loop iterable (line 72)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 72, 8), TRIPLETS_346)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 72, 8), TRIPLETS_346):
            # Getting the type of the for loop variable (line 72)
            for_loop_var_347 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 72, 8), TRIPLETS_346)
            # Assigning a type to the variable 'triplet' (line 72)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'triplet', for_loop_var_347)
            # SSA begins for a for statement (line 72)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'set_row' (line 73)
            set_row_348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 15), 'set_row')
            # Getting the type of 'triplet' (line 73)
            triplet_349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 26), 'triplet')
            # Applying the binary operator 'in' (line 73)
            result_contains_350 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 15), 'in', set_row_348, triplet_349)
            
            # Testing if the type of an if condition is none (line 73)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 73, 12), result_contains_350):
                pass
            else:
                
                # Testing the type of an if condition (line 73)
                if_condition_351 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 12), result_contains_350)
                # Assigning a type to the variable 'if_condition_351' (line 73)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'if_condition_351', if_condition_351)
                # SSA begins for if statement (line 73)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Subscript to a Name (line 73):
                
                # Assigning a Subscript to a Name (line 73):
                
                # Obtaining the type of the subscript
                slice_352 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 73, 42), None, None, None)
                # Getting the type of 'triplet' (line 73)
                triplet_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 42), 'triplet')
                # Obtaining the member '__getitem__' of a type (line 73)
                getitem___354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 42), triplet_353, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 73)
                subscript_call_result_355 = invoke(stypy.reporting.localization.Localization(__file__, 73, 42), getitem___354, slice_352)
                
                # Assigning a type to the variable 'rows' (line 73)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 35), 'rows', subscript_call_result_355)
                # SSA join for if statement (line 73)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'set_col' (line 74)
            set_col_356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 15), 'set_col')
            # Getting the type of 'triplet' (line 74)
            triplet_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 26), 'triplet')
            # Applying the binary operator 'in' (line 74)
            result_contains_358 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 15), 'in', set_col_356, triplet_357)
            
            # Testing if the type of an if condition is none (line 74)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 12), result_contains_358):
                pass
            else:
                
                # Testing the type of an if condition (line 74)
                if_condition_359 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 12), result_contains_358)
                # Assigning a type to the variable 'if_condition_359' (line 74)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'if_condition_359', if_condition_359)
                # SSA begins for if statement (line 74)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Subscript to a Name (line 74):
                
                # Assigning a Subscript to a Name (line 74):
                
                # Obtaining the type of the subscript
                slice_360 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 74, 42), None, None, None)
                # Getting the type of 'triplet' (line 74)
                triplet_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 42), 'triplet')
                # Obtaining the member '__getitem__' of a type (line 74)
                getitem___362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 42), triplet_361, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 74)
                subscript_call_result_363 = invoke(stypy.reporting.localization.Localization(__file__, 74, 42), getitem___362, slice_360)
                
                # Assigning a type to the variable 'cols' (line 74)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 35), 'cols', subscript_call_result_363)
                # SSA join for if statement (line 74)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to remove(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'set_row' (line 75)
        set_row_366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 20), 'set_row', False)
        # Processing the call keyword arguments (line 75)
        kwargs_367 = {}
        # Getting the type of 'rows' (line 75)
        rows_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'rows', False)
        # Obtaining the member 'remove' of a type (line 75)
        remove_365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), rows_364, 'remove')
        # Calling remove(args, kwargs) (line 75)
        remove_call_result_368 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), remove_365, *[set_row_366], **kwargs_367)
        
        
        # Call to remove(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'set_col' (line 76)
        set_col_371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 20), 'set_col', False)
        # Processing the call keyword arguments (line 76)
        kwargs_372 = {}
        # Getting the type of 'cols' (line 76)
        cols_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'cols', False)
        # Obtaining the member 'remove' of a type (line 76)
        remove_370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), cols_369, 'remove')
        # Calling remove(args, kwargs) (line 76)
        remove_call_result_373 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), remove_370, *[set_col_371], **kwargs_372)
        
        
        # Getting the type of 'rows' (line 77)
        rows_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 19), 'rows')
        # Assigning a type to the variable 'rows_374' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'rows_374', rows_374)
        # Testing if the for loop is going to be iterated (line 77)
        # Testing the type of a for loop iterable (line 77)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 77, 8), rows_374)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 77, 8), rows_374):
            # Getting the type of the for loop variable (line 77)
            for_loop_var_375 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 77, 8), rows_374)
            # Assigning a type to the variable 'row' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'row', for_loop_var_375)
            # SSA begins for a for statement (line 77)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'cols' (line 78)
            cols_376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'cols')
            # Assigning a type to the variable 'cols_376' (line 78)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'cols_376', cols_376)
            # Testing if the for loop is going to be iterated (line 78)
            # Testing the type of a for loop iterable (line 78)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 78, 12), cols_376)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 78, 12), cols_376):
                # Getting the type of the for loop variable (line 78)
                for_loop_var_377 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 78, 12), cols_376)
                # Assigning a type to the variable 'col' (line 78)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'col', for_loop_var_377)
                # SSA begins for a for statement (line 78)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                # Evaluating assert statement condition
                
                # Evaluating a boolean operation
                
                # Getting the type of 'row' (line 79)
                row_378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 23), 'row')
                # Getting the type of 'set_row' (line 79)
                set_row_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 30), 'set_row')
                # Applying the binary operator '!=' (line 79)
                result_ne_380 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 23), '!=', row_378, set_row_379)
                
                
                # Getting the type of 'col' (line 79)
                col_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 41), 'col')
                # Getting the type of 'set_col' (line 79)
                set_col_382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 48), 'set_col')
                # Applying the binary operator '!=' (line 79)
                result_ne_383 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 41), '!=', col_381, set_col_382)
                
                # Applying the binary operator 'or' (line 79)
                result_or_keyword_384 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 23), 'or', result_ne_380, result_ne_383)
                
                assert_385 = result_or_keyword_384
                # Assigning a type to the variable 'assert_385' (line 79)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 16), 'assert_385', result_or_keyword_384)
                
                # Call to cell_exclude(...): (line 80)
                # Processing the call arguments (line 80)
                # Getting the type of 'row' (line 80)
                row_388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 34), 'row', False)
                # Getting the type of 'col' (line 80)
                col_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 39), 'col', False)
                # Getting the type of 'x' (line 80)
                x_390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 44), 'x', False)
                # Processing the call keyword arguments (line 80)
                kwargs_391 = {}
                # Getting the type of 'self' (line 80)
                self_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'self', False)
                # Obtaining the member 'cell_exclude' of a type (line 80)
                cell_exclude_387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 16), self_386, 'cell_exclude')
                # Calling cell_exclude(args, kwargs) (line 80)
                cell_exclude_call_result_392 = invoke(stypy.reporting.localization.Localization(__file__, 80, 16), cell_exclude_387, *[row_388, col_389, x_390], **kwargs_391)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'update_neighbours(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update_neighbours' in the type store
        # Getting the type of 'stypy_return_type' (line 65)
        stypy_return_type_393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_393)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update_neighbours'
        return stypy_return_type_393


    @norecursion
    def get_cell_digit_str(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_cell_digit_str'
        module_type_store = module_type_store.open_function_context('get_cell_digit_str', 82, 4, False)
        # Assigning a type to the variable 'self' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        soduko.get_cell_digit_str.__dict__.__setitem__('stypy_localization', localization)
        soduko.get_cell_digit_str.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        soduko.get_cell_digit_str.__dict__.__setitem__('stypy_type_store', module_type_store)
        soduko.get_cell_digit_str.__dict__.__setitem__('stypy_function_name', 'soduko.get_cell_digit_str')
        soduko.get_cell_digit_str.__dict__.__setitem__('stypy_param_names_list', ['row', 'col'])
        soduko.get_cell_digit_str.__dict__.__setitem__('stypy_varargs_param_name', None)
        soduko.get_cell_digit_str.__dict__.__setitem__('stypy_kwargs_param_name', None)
        soduko.get_cell_digit_str.__dict__.__setitem__('stypy_call_defaults', defaults)
        soduko.get_cell_digit_str.__dict__.__setitem__('stypy_call_varargs', varargs)
        soduko.get_cell_digit_str.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        soduko.get_cell_digit_str.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'soduko.get_cell_digit_str', ['row', 'col'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_cell_digit_str', localization, ['row', 'col'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_cell_digit_str(...)' code ##################

        
        
        # Call to len(...): (line 83)
        # Processing the call arguments (line 83)
        
        # Obtaining the type of the subscript
        # Getting the type of 'col' (line 83)
        col_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 33), 'col', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'row' (line 83)
        row_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 28), 'row', False)
        # Getting the type of 'self' (line 83)
        self_397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 15), 'self', False)
        # Obtaining the member 'squares' of a type (line 83)
        squares_398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 15), self_397, 'squares')
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 15), squares_398, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_400 = invoke(stypy.reporting.localization.Localization(__file__, 83, 15), getitem___399, row_396)
        
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 15), subscript_call_result_400, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_402 = invoke(stypy.reporting.localization.Localization(__file__, 83, 15), getitem___401, col_395)
        
        # Processing the call keyword arguments (line 83)
        kwargs_403 = {}
        # Getting the type of 'len' (line 83)
        len_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 11), 'len', False)
        # Calling len(args, kwargs) (line 83)
        len_call_result_404 = invoke(stypy.reporting.localization.Localization(__file__, 83, 11), len_394, *[subscript_call_result_402], **kwargs_403)
        
        int_405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 42), 'int')
        # Applying the binary operator '==' (line 83)
        result_eq_406 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 11), '==', len_call_result_404, int_405)
        
        # Testing if the type of an if condition is none (line 83)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 83, 8), result_eq_406):
            str_422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 19), 'str', '0')
            # Assigning a type to the variable 'stypy_return_type' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'stypy_return_type', str_422)
        else:
            
            # Testing the type of an if condition (line 83)
            if_condition_407 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 8), result_eq_406)
            # Assigning a type to the variable 'if_condition_407' (line 83)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'if_condition_407', if_condition_407)
            # SSA begins for if statement (line 83)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to str(...): (line 84)
            # Processing the call arguments (line 84)
            
            # Obtaining the type of the subscript
            int_409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 46), 'int')
            
            # Obtaining the type of the subscript
            # Getting the type of 'col' (line 84)
            col_410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 41), 'col', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row' (line 84)
            row_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 36), 'row', False)
            # Getting the type of 'self' (line 84)
            self_412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 23), 'self', False)
            # Obtaining the member 'squares' of a type (line 84)
            squares_413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 23), self_412, 'squares')
            # Obtaining the member '__getitem__' of a type (line 84)
            getitem___414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 23), squares_413, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 84)
            subscript_call_result_415 = invoke(stypy.reporting.localization.Localization(__file__, 84, 23), getitem___414, row_411)
            
            # Obtaining the member '__getitem__' of a type (line 84)
            getitem___416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 23), subscript_call_result_415, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 84)
            subscript_call_result_417 = invoke(stypy.reporting.localization.Localization(__file__, 84, 23), getitem___416, col_410)
            
            # Obtaining the member '__getitem__' of a type (line 84)
            getitem___418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 23), subscript_call_result_417, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 84)
            subscript_call_result_419 = invoke(stypy.reporting.localization.Localization(__file__, 84, 23), getitem___418, int_409)
            
            # Processing the call keyword arguments (line 84)
            kwargs_420 = {}
            # Getting the type of 'str' (line 84)
            str_408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 19), 'str', False)
            # Calling str(args, kwargs) (line 84)
            str_call_result_421 = invoke(stypy.reporting.localization.Localization(__file__, 84, 19), str_408, *[subscript_call_result_419], **kwargs_420)
            
            # Assigning a type to the variable 'stypy_return_type' (line 84)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'stypy_return_type', str_call_result_421)
            # SSA branch for the else part of an if statement (line 83)
            module_type_store.open_ssa_branch('else')
            str_422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 19), 'str', '0')
            # Assigning a type to the variable 'stypy_return_type' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'stypy_return_type', str_422)
            # SSA join for if statement (line 83)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'get_cell_digit_str(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_cell_digit_str' in the type store
        # Getting the type of 'stypy_return_type' (line 82)
        stypy_return_type_423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_423)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_cell_digit_str'
        return stypy_return_type_423


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 88, 4, False)
        # Assigning a type to the variable 'self' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        soduko.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        soduko.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        soduko.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        soduko.stypy__str__.__dict__.__setitem__('stypy_function_name', 'soduko.stypy__str__')
        soduko.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        soduko.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        soduko.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        soduko.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        soduko.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        soduko.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        soduko.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'soduko.stypy__str__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Str to a Name (line 89):
        
        # Assigning a Str to a Name (line 89):
        str_424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 17), 'str', '   123   456   789\n')
        # Assigning a type to the variable 'answer' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'answer', str_424)
        
        
        # Call to range(...): (line 90)
        # Processing the call arguments (line 90)
        int_426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 25), 'int')
        int_427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 28), 'int')
        # Processing the call keyword arguments (line 90)
        kwargs_428 = {}
        # Getting the type of 'range' (line 90)
        range_425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 19), 'range', False)
        # Calling range(args, kwargs) (line 90)
        range_call_result_429 = invoke(stypy.reporting.localization.Localization(__file__, 90, 19), range_425, *[int_426, int_427], **kwargs_428)
        
        # Assigning a type to the variable 'range_call_result_429' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'range_call_result_429', range_call_result_429)
        # Testing if the for loop is going to be iterated (line 90)
        # Testing the type of a for loop iterable (line 90)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 90, 8), range_call_result_429)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 90, 8), range_call_result_429):
            # Getting the type of the for loop variable (line 90)
            for_loop_var_430 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 90, 8), range_call_result_429)
            # Assigning a type to the variable 'row' (line 90)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'row', for_loop_var_430)
            # SSA begins for a for statement (line 90)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a BinOp to a Name (line 91):
            
            # Assigning a BinOp to a Name (line 91):
            # Getting the type of 'answer' (line 91)
            answer_431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 21), 'answer')
            
            # Call to str(...): (line 91)
            # Processing the call arguments (line 91)
            # Getting the type of 'row' (line 91)
            row_433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 34), 'row', False)
            int_434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 40), 'int')
            # Applying the binary operator '+' (line 91)
            result_add_435 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 34), '+', row_433, int_434)
            
            # Processing the call keyword arguments (line 91)
            kwargs_436 = {}
            # Getting the type of 'str' (line 91)
            str_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 30), 'str', False)
            # Calling str(args, kwargs) (line 91)
            str_call_result_437 = invoke(stypy.reporting.localization.Localization(__file__, 91, 30), str_432, *[result_add_435], **kwargs_436)
            
            # Applying the binary operator '+' (line 91)
            result_add_438 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 21), '+', answer_431, str_call_result_437)
            
            str_439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 45), 'str', ' [')
            # Applying the binary operator '+' (line 91)
            result_add_440 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 43), '+', result_add_438, str_439)
            
            
            # Call to join(...): (line 91)
            # Processing the call arguments (line 91)
            # Calculating list comprehension
            # Calculating comprehension expression
            
            # Call to range(...): (line 92)
            # Processing the call arguments (line 92)
            int_455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 86), 'int')
            int_456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 89), 'int')
            # Processing the call keyword arguments (line 92)
            kwargs_457 = {}
            # Getting the type of 'range' (line 92)
            range_454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 80), 'range', False)
            # Calling range(args, kwargs) (line 92)
            range_call_result_458 = invoke(stypy.reporting.localization.Localization(__file__, 92, 80), range_454, *[int_455, int_456], **kwargs_457)
            
            comprehension_459 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 17), range_call_result_458)
            # Assigning a type to the variable 'col' (line 92)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 17), 'col', comprehension_459)
            
            # Call to replace(...): (line 92)
            # Processing the call arguments (line 92)
            str_450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 59), 'str', '0')
            str_451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 64), 'str', '?')
            # Processing the call keyword arguments (line 92)
            kwargs_452 = {}
            
            # Call to get_cell_digit_str(...): (line 92)
            # Processing the call arguments (line 92)
            # Getting the type of 'row' (line 92)
            row_445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 41), 'row', False)
            # Getting the type of 'col' (line 92)
            col_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 46), 'col', False)
            # Processing the call keyword arguments (line 92)
            kwargs_447 = {}
            # Getting the type of 'self' (line 92)
            self_443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 17), 'self', False)
            # Obtaining the member 'get_cell_digit_str' of a type (line 92)
            get_cell_digit_str_444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 17), self_443, 'get_cell_digit_str')
            # Calling get_cell_digit_str(args, kwargs) (line 92)
            get_cell_digit_str_call_result_448 = invoke(stypy.reporting.localization.Localization(__file__, 92, 17), get_cell_digit_str_444, *[row_445, col_446], **kwargs_447)
            
            # Obtaining the member 'replace' of a type (line 92)
            replace_449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 17), get_cell_digit_str_call_result_448, 'replace')
            # Calling replace(args, kwargs) (line 92)
            replace_call_result_453 = invoke(stypy.reporting.localization.Localization(__file__, 92, 17), replace_449, *[str_450, str_451], **kwargs_452)
            
            list_460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 17), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 17), list_460, replace_call_result_453)
            # Processing the call keyword arguments (line 91)
            kwargs_461 = {}
            str_441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 52), 'str', '')
            # Obtaining the member 'join' of a type (line 91)
            join_442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 52), str_441, 'join')
            # Calling join(args, kwargs) (line 91)
            join_call_result_462 = invoke(stypy.reporting.localization.Localization(__file__, 91, 52), join_442, *[list_460], **kwargs_461)
            
            # Applying the binary operator '+' (line 91)
            result_add_463 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 50), '+', result_add_440, join_call_result_462)
            
            str_464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 96), 'str', '] [')
            # Applying the binary operator '+' (line 92)
            result_add_465 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 94), '+', result_add_463, str_464)
            
            
            # Call to join(...): (line 92)
            # Processing the call arguments (line 92)
            # Calculating list comprehension
            # Calculating comprehension expression
            
            # Call to range(...): (line 93)
            # Processing the call arguments (line 93)
            int_480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 86), 'int')
            int_481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 89), 'int')
            # Processing the call keyword arguments (line 93)
            kwargs_482 = {}
            # Getting the type of 'range' (line 93)
            range_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 80), 'range', False)
            # Calling range(args, kwargs) (line 93)
            range_call_result_483 = invoke(stypy.reporting.localization.Localization(__file__, 93, 80), range_479, *[int_480, int_481], **kwargs_482)
            
            comprehension_484 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 17), range_call_result_483)
            # Assigning a type to the variable 'col' (line 93)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 17), 'col', comprehension_484)
            
            # Call to replace(...): (line 93)
            # Processing the call arguments (line 93)
            str_475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 59), 'str', '0')
            str_476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 64), 'str', '?')
            # Processing the call keyword arguments (line 93)
            kwargs_477 = {}
            
            # Call to get_cell_digit_str(...): (line 93)
            # Processing the call arguments (line 93)
            # Getting the type of 'row' (line 93)
            row_470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 41), 'row', False)
            # Getting the type of 'col' (line 93)
            col_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 46), 'col', False)
            # Processing the call keyword arguments (line 93)
            kwargs_472 = {}
            # Getting the type of 'self' (line 93)
            self_468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 17), 'self', False)
            # Obtaining the member 'get_cell_digit_str' of a type (line 93)
            get_cell_digit_str_469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 17), self_468, 'get_cell_digit_str')
            # Calling get_cell_digit_str(args, kwargs) (line 93)
            get_cell_digit_str_call_result_473 = invoke(stypy.reporting.localization.Localization(__file__, 93, 17), get_cell_digit_str_469, *[row_470, col_471], **kwargs_472)
            
            # Obtaining the member 'replace' of a type (line 93)
            replace_474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 17), get_cell_digit_str_call_result_473, 'replace')
            # Calling replace(args, kwargs) (line 93)
            replace_call_result_478 = invoke(stypy.reporting.localization.Localization(__file__, 93, 17), replace_474, *[str_475, str_476], **kwargs_477)
            
            list_485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 17), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 17), list_485, replace_call_result_478)
            # Processing the call keyword arguments (line 92)
            kwargs_486 = {}
            str_466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 104), 'str', '')
            # Obtaining the member 'join' of a type (line 92)
            join_467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 104), str_466, 'join')
            # Calling join(args, kwargs) (line 92)
            join_call_result_487 = invoke(stypy.reporting.localization.Localization(__file__, 92, 104), join_467, *[list_485], **kwargs_486)
            
            # Applying the binary operator '+' (line 92)
            result_add_488 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 102), '+', result_add_465, join_call_result_487)
            
            str_489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 96), 'str', '] [')
            # Applying the binary operator '+' (line 93)
            result_add_490 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 94), '+', result_add_488, str_489)
            
            
            # Call to join(...): (line 93)
            # Processing the call arguments (line 93)
            # Calculating list comprehension
            # Calculating comprehension expression
            
            # Call to range(...): (line 94)
            # Processing the call arguments (line 94)
            int_505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 86), 'int')
            int_506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 89), 'int')
            # Processing the call keyword arguments (line 94)
            kwargs_507 = {}
            # Getting the type of 'range' (line 94)
            range_504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 80), 'range', False)
            # Calling range(args, kwargs) (line 94)
            range_call_result_508 = invoke(stypy.reporting.localization.Localization(__file__, 94, 80), range_504, *[int_505, int_506], **kwargs_507)
            
            comprehension_509 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 17), range_call_result_508)
            # Assigning a type to the variable 'col' (line 94)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 17), 'col', comprehension_509)
            
            # Call to replace(...): (line 94)
            # Processing the call arguments (line 94)
            str_500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 59), 'str', '0')
            str_501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 64), 'str', '?')
            # Processing the call keyword arguments (line 94)
            kwargs_502 = {}
            
            # Call to get_cell_digit_str(...): (line 94)
            # Processing the call arguments (line 94)
            # Getting the type of 'row' (line 94)
            row_495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 41), 'row', False)
            # Getting the type of 'col' (line 94)
            col_496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 46), 'col', False)
            # Processing the call keyword arguments (line 94)
            kwargs_497 = {}
            # Getting the type of 'self' (line 94)
            self_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 17), 'self', False)
            # Obtaining the member 'get_cell_digit_str' of a type (line 94)
            get_cell_digit_str_494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 17), self_493, 'get_cell_digit_str')
            # Calling get_cell_digit_str(args, kwargs) (line 94)
            get_cell_digit_str_call_result_498 = invoke(stypy.reporting.localization.Localization(__file__, 94, 17), get_cell_digit_str_494, *[row_495, col_496], **kwargs_497)
            
            # Obtaining the member 'replace' of a type (line 94)
            replace_499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 17), get_cell_digit_str_call_result_498, 'replace')
            # Calling replace(args, kwargs) (line 94)
            replace_call_result_503 = invoke(stypy.reporting.localization.Localization(__file__, 94, 17), replace_499, *[str_500, str_501], **kwargs_502)
            
            list_510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 17), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 17), list_510, replace_call_result_503)
            # Processing the call keyword arguments (line 93)
            kwargs_511 = {}
            str_491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 104), 'str', '')
            # Obtaining the member 'join' of a type (line 93)
            join_492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 104), str_491, 'join')
            # Calling join(args, kwargs) (line 93)
            join_call_result_512 = invoke(stypy.reporting.localization.Localization(__file__, 93, 104), join_492, *[list_510], **kwargs_511)
            
            # Applying the binary operator '+' (line 93)
            result_add_513 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 102), '+', result_add_490, join_call_result_512)
            
            str_514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 96), 'str', ']\n')
            # Applying the binary operator '+' (line 94)
            result_add_515 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 94), '+', result_add_513, str_514)
            
            # Assigning a type to the variable 'answer' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'answer', result_add_515)
            
            # Getting the type of 'row' (line 95)
            row_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'row')
            int_517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 21), 'int')
            # Applying the binary operator '+' (line 95)
            result_add_518 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 15), '+', row_516, int_517)
            
            
            # Obtaining an instance of the builtin type 'list' (line 95)
            list_519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 26), 'list')
            # Adding type elements to the builtin type 'list' instance (line 95)
            # Adding element type (line 95)
            int_520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 27), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 26), list_519, int_520)
            # Adding element type (line 95)
            int_521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 30), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 26), list_519, int_521)
            
            # Applying the binary operator 'in' (line 95)
            result_contains_522 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 15), 'in', result_add_518, list_519)
            
            # Testing if the type of an if condition is none (line 95)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 95, 12), result_contains_522):
                pass
            else:
                
                # Testing the type of an if condition (line 95)
                if_condition_523 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 12), result_contains_522)
                # Assigning a type to the variable 'if_condition_523' (line 95)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'if_condition_523', if_condition_523)
                # SSA begins for if statement (line 95)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a BinOp to a Name (line 96):
                
                # Assigning a BinOp to a Name (line 96):
                # Getting the type of 'answer' (line 96)
                answer_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 25), 'answer')
                str_525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 34), 'str', '   ---   ---   ---\n')
                # Applying the binary operator '+' (line 96)
                result_add_526 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 25), '+', answer_524, str_525)
                
                # Assigning a type to the variable 'answer' (line 96)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'answer', result_add_526)
                # SSA join for if statement (line 95)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'answer' (line 97)
        answer_527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 15), 'answer')
        # Assigning a type to the variable 'stypy_return_type' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'stypy_return_type', answer_527)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 88)
        stypy_return_type_528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_528)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_528


    @norecursion
    def check(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check'
        module_type_store = module_type_store.open_function_context('check', 99, 4, False)
        # Assigning a type to the variable 'self' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        soduko.check.__dict__.__setitem__('stypy_localization', localization)
        soduko.check.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        soduko.check.__dict__.__setitem__('stypy_type_store', module_type_store)
        soduko.check.__dict__.__setitem__('stypy_function_name', 'soduko.check')
        soduko.check.__dict__.__setitem__('stypy_param_names_list', [])
        soduko.check.__dict__.__setitem__('stypy_varargs_param_name', None)
        soduko.check.__dict__.__setitem__('stypy_kwargs_param_name', None)
        soduko.check.__dict__.__setitem__('stypy_call_defaults', defaults)
        soduko.check.__dict__.__setitem__('stypy_call_varargs', varargs)
        soduko.check.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        soduko.check.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'soduko.check', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check(...)' code ##################

        
        # Assigning a Name to a Attribute (line 100):
        
        # Assigning a Name to a Attribute (line 100):
        # Getting the type of 'True' (line 100)
        True_529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 24), 'True')
        # Getting the type of 'self' (line 100)
        self_530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'self')
        # Setting the type of the member '_changed' of a type (line 100)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), self_530, '_changed', True_529)
        
        # Getting the type of 'self' (line 101)
        self_531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 14), 'self')
        # Obtaining the member '_changed' of a type (line 101)
        _changed_532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 14), self_531, '_changed')
        # Assigning a type to the variable '_changed_532' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), '_changed_532', _changed_532)
        # Testing if the while is going to be iterated (line 101)
        # Testing the type of an if condition (line 101)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 8), _changed_532)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 101, 8), _changed_532):
            # SSA begins for while statement (line 101)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a Name to a Attribute (line 102):
            
            # Assigning a Name to a Attribute (line 102):
            # Getting the type of 'False' (line 102)
            False_533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 28), 'False')
            # Getting the type of 'self' (line 102)
            self_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'self')
            # Setting the type of the member '_changed' of a type (line 102)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), self_534, '_changed', False_533)
            
            # Call to check_for_single_occurances(...): (line 103)
            # Processing the call keyword arguments (line 103)
            kwargs_537 = {}
            # Getting the type of 'self' (line 103)
            self_535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'self', False)
            # Obtaining the member 'check_for_single_occurances' of a type (line 103)
            check_for_single_occurances_536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), self_535, 'check_for_single_occurances')
            # Calling check_for_single_occurances(args, kwargs) (line 103)
            check_for_single_occurances_call_result_538 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), check_for_single_occurances_536, *[], **kwargs_537)
            
            
            # Call to check_for_last_in_row_col_3x3(...): (line 104)
            # Processing the call keyword arguments (line 104)
            kwargs_541 = {}
            # Getting the type of 'self' (line 104)
            self_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'self', False)
            # Obtaining the member 'check_for_last_in_row_col_3x3' of a type (line 104)
            check_for_last_in_row_col_3x3_540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 12), self_539, 'check_for_last_in_row_col_3x3')
            # Calling check_for_last_in_row_col_3x3(args, kwargs) (line 104)
            check_for_last_in_row_col_3x3_call_result_542 = invoke(stypy.reporting.localization.Localization(__file__, 104, 12), check_for_last_in_row_col_3x3_540, *[], **kwargs_541)
            
            # SSA join for while statement (line 101)
            module_type_store = module_type_store.join_ssa_context()

        
        # Assigning a type to the variable 'stypy_return_type' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of 'check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check' in the type store
        # Getting the type of 'stypy_return_type' (line 99)
        stypy_return_type_543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_543)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check'
        return stypy_return_type_543


    @norecursion
    def check_for_single_occurances(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_for_single_occurances'
        module_type_store = module_type_store.open_function_context('check_for_single_occurances', 107, 4, False)
        # Assigning a type to the variable 'self' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        soduko.check_for_single_occurances.__dict__.__setitem__('stypy_localization', localization)
        soduko.check_for_single_occurances.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        soduko.check_for_single_occurances.__dict__.__setitem__('stypy_type_store', module_type_store)
        soduko.check_for_single_occurances.__dict__.__setitem__('stypy_function_name', 'soduko.check_for_single_occurances')
        soduko.check_for_single_occurances.__dict__.__setitem__('stypy_param_names_list', [])
        soduko.check_for_single_occurances.__dict__.__setitem__('stypy_varargs_param_name', None)
        soduko.check_for_single_occurances.__dict__.__setitem__('stypy_kwargs_param_name', None)
        soduko.check_for_single_occurances.__dict__.__setitem__('stypy_call_defaults', defaults)
        soduko.check_for_single_occurances.__dict__.__setitem__('stypy_call_varargs', varargs)
        soduko.check_for_single_occurances.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        soduko.check_for_single_occurances.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'soduko.check_for_single_occurances', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_for_single_occurances', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_for_single_occurances(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        # Adding element type (line 108)
        # Getting the type of 'ROW_ITER' (line 108)
        ROW_ITER_545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 27), 'ROW_ITER')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 26), list_544, ROW_ITER_545)
        # Adding element type (line 108)
        # Getting the type of 'COL_ITER' (line 108)
        COL_ITER_546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 37), 'COL_ITER')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 26), list_544, COL_ITER_546)
        # Adding element type (line 108)
        # Getting the type of 'TxT_ITER' (line 108)
        TxT_ITER_547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 47), 'TxT_ITER')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 26), list_544, TxT_ITER_547)
        
        # Assigning a type to the variable 'list_544' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'list_544', list_544)
        # Testing if the for loop is going to be iterated (line 108)
        # Testing the type of a for loop iterable (line 108)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 108, 8), list_544)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 108, 8), list_544):
            # Getting the type of the for loop variable (line 108)
            for_loop_var_548 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 108, 8), list_544)
            # Assigning a type to the variable 'check_type' (line 108)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'check_type', for_loop_var_548)
            # SSA begins for a for statement (line 108)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'check_type' (line 109)
            check_type_549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 30), 'check_type')
            # Assigning a type to the variable 'check_type_549' (line 109)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'check_type_549', check_type_549)
            # Testing if the for loop is going to be iterated (line 109)
            # Testing the type of a for loop iterable (line 109)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 109, 12), check_type_549)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 109, 12), check_type_549):
                # Getting the type of the for loop variable (line 109)
                for_loop_var_550 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 109, 12), check_type_549)
                # Assigning a type to the variable 'check_list' (line 109)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'check_list', for_loop_var_550)
                # SSA begins for a for statement (line 109)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Call to range(...): (line 110)
                # Processing the call arguments (line 110)
                int_552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 31), 'int')
                int_553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 34), 'int')
                int_554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 38), 'int')
                # Applying the binary operator '+' (line 110)
                result_add_555 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 34), '+', int_553, int_554)
                
                # Processing the call keyword arguments (line 110)
                kwargs_556 = {}
                # Getting the type of 'range' (line 110)
                range_551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 25), 'range', False)
                # Calling range(args, kwargs) (line 110)
                range_call_result_557 = invoke(stypy.reporting.localization.Localization(__file__, 110, 25), range_551, *[int_552, result_add_555], **kwargs_556)
                
                # Assigning a type to the variable 'range_call_result_557' (line 110)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'range_call_result_557', range_call_result_557)
                # Testing if the for loop is going to be iterated (line 110)
                # Testing the type of a for loop iterable (line 110)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 110, 16), range_call_result_557)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 110, 16), range_call_result_557):
                    # Getting the type of the for loop variable (line 110)
                    for_loop_var_558 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 110, 16), range_call_result_557)
                    # Assigning a type to the variable 'x' (line 110)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'x', for_loop_var_558)
                    # SSA begins for a for statement (line 110)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Assigning a List to a Name (line 111):
                    
                    # Assigning a List to a Name (line 111):
                    
                    # Obtaining an instance of the builtin type 'list' (line 111)
                    list_559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 32), 'list')
                    # Adding type elements to the builtin type 'list' instance (line 111)
                    
                    # Assigning a type to the variable 'x_in_list' (line 111)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 20), 'x_in_list', list_559)
                    
                    # Getting the type of 'check_list' (line 112)
                    check_list_560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 38), 'check_list')
                    # Assigning a type to the variable 'check_list_560' (line 112)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'check_list_560', check_list_560)
                    # Testing if the for loop is going to be iterated (line 112)
                    # Testing the type of a for loop iterable (line 112)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 112, 20), check_list_560)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 112, 20), check_list_560):
                        # Getting the type of the for loop variable (line 112)
                        for_loop_var_561 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 112, 20), check_list_560)
                        # Assigning a type to the variable 'row' (line 112)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'row', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 20), for_loop_var_561, 2, 0))
                        # Assigning a type to the variable 'col' (line 112)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'col', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 20), for_loop_var_561, 2, 1))
                        # SSA begins for a for statement (line 112)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Getting the type of 'x' (line 113)
                        x_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 27), 'x')
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'col' (line 113)
                        col_563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 50), 'col')
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'row' (line 113)
                        row_564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 45), 'row')
                        # Getting the type of 'self' (line 113)
                        self_565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 32), 'self')
                        # Obtaining the member 'squares' of a type (line 113)
                        squares_566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 32), self_565, 'squares')
                        # Obtaining the member '__getitem__' of a type (line 113)
                        getitem___567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 32), squares_566, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 113)
                        subscript_call_result_568 = invoke(stypy.reporting.localization.Localization(__file__, 113, 32), getitem___567, row_564)
                        
                        # Obtaining the member '__getitem__' of a type (line 113)
                        getitem___569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 32), subscript_call_result_568, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 113)
                        subscript_call_result_570 = invoke(stypy.reporting.localization.Localization(__file__, 113, 32), getitem___569, col_563)
                        
                        # Applying the binary operator 'in' (line 113)
                        result_contains_571 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 27), 'in', x_562, subscript_call_result_570)
                        
                        # Testing if the type of an if condition is none (line 113)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 113, 24), result_contains_571):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 113)
                            if_condition_572 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 24), result_contains_571)
                            # Assigning a type to the variable 'if_condition_572' (line 113)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 24), 'if_condition_572', if_condition_572)
                            # SSA begins for if statement (line 113)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to append(...): (line 114)
                            # Processing the call arguments (line 114)
                            
                            # Obtaining an instance of the builtin type 'tuple' (line 114)
                            tuple_575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 46), 'tuple')
                            # Adding type elements to the builtin type 'tuple' instance (line 114)
                            # Adding element type (line 114)
                            # Getting the type of 'row' (line 114)
                            row_576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 46), 'row', False)
                            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 46), tuple_575, row_576)
                            # Adding element type (line 114)
                            # Getting the type of 'col' (line 114)
                            col_577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 51), 'col', False)
                            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 46), tuple_575, col_577)
                            
                            # Processing the call keyword arguments (line 114)
                            kwargs_578 = {}
                            # Getting the type of 'x_in_list' (line 114)
                            x_in_list_573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 28), 'x_in_list', False)
                            # Obtaining the member 'append' of a type (line 114)
                            append_574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 28), x_in_list_573, 'append')
                            # Calling append(args, kwargs) (line 114)
                            append_call_result_579 = invoke(stypy.reporting.localization.Localization(__file__, 114, 28), append_574, *[tuple_575], **kwargs_578)
                            
                            # SSA join for if statement (line 113)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    
                    
                    # Call to len(...): (line 115)
                    # Processing the call arguments (line 115)
                    # Getting the type of 'x_in_list' (line 115)
                    x_in_list_581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 27), 'x_in_list', False)
                    # Processing the call keyword arguments (line 115)
                    kwargs_582 = {}
                    # Getting the type of 'len' (line 115)
                    len_580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 23), 'len', False)
                    # Calling len(args, kwargs) (line 115)
                    len_call_result_583 = invoke(stypy.reporting.localization.Localization(__file__, 115, 23), len_580, *[x_in_list_581], **kwargs_582)
                    
                    int_584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 41), 'int')
                    # Applying the binary operator '==' (line 115)
                    result_eq_585 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 23), '==', len_call_result_583, int_584)
                    
                    # Testing if the type of an if condition is none (line 115)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 115, 20), result_eq_585):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 115)
                        if_condition_586 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 115, 20), result_eq_585)
                        # Assigning a type to the variable 'if_condition_586' (line 115)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 20), 'if_condition_586', if_condition_586)
                        # SSA begins for if statement (line 115)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Subscript to a Tuple (line 116):
                        
                        # Assigning a Subscript to a Name (line 116):
                        
                        # Obtaining the type of the subscript
                        int_587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 24), 'int')
                        
                        # Obtaining the type of the subscript
                        int_588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 47), 'int')
                        # Getting the type of 'x_in_list' (line 116)
                        x_in_list_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 37), 'x_in_list')
                        # Obtaining the member '__getitem__' of a type (line 116)
                        getitem___590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 37), x_in_list_589, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 116)
                        subscript_call_result_591 = invoke(stypy.reporting.localization.Localization(__file__, 116, 37), getitem___590, int_588)
                        
                        # Obtaining the member '__getitem__' of a type (line 116)
                        getitem___592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 24), subscript_call_result_591, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 116)
                        subscript_call_result_593 = invoke(stypy.reporting.localization.Localization(__file__, 116, 24), getitem___592, int_587)
                        
                        # Assigning a type to the variable 'tuple_var_assignment_1' (line 116)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 24), 'tuple_var_assignment_1', subscript_call_result_593)
                        
                        # Assigning a Subscript to a Name (line 116):
                        
                        # Obtaining the type of the subscript
                        int_594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 24), 'int')
                        
                        # Obtaining the type of the subscript
                        int_595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 47), 'int')
                        # Getting the type of 'x_in_list' (line 116)
                        x_in_list_596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 37), 'x_in_list')
                        # Obtaining the member '__getitem__' of a type (line 116)
                        getitem___597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 37), x_in_list_596, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 116)
                        subscript_call_result_598 = invoke(stypy.reporting.localization.Localization(__file__, 116, 37), getitem___597, int_595)
                        
                        # Obtaining the member '__getitem__' of a type (line 116)
                        getitem___599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 24), subscript_call_result_598, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 116)
                        subscript_call_result_600 = invoke(stypy.reporting.localization.Localization(__file__, 116, 24), getitem___599, int_594)
                        
                        # Assigning a type to the variable 'tuple_var_assignment_2' (line 116)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 24), 'tuple_var_assignment_2', subscript_call_result_600)
                        
                        # Assigning a Name to a Name (line 116):
                        # Getting the type of 'tuple_var_assignment_1' (line 116)
                        tuple_var_assignment_1_601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 24), 'tuple_var_assignment_1')
                        # Assigning a type to the variable 'row' (line 116)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 25), 'row', tuple_var_assignment_1_601)
                        
                        # Assigning a Name to a Name (line 116):
                        # Getting the type of 'tuple_var_assignment_2' (line 116)
                        tuple_var_assignment_2_602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 24), 'tuple_var_assignment_2')
                        # Assigning a type to the variable 'col' (line 116)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 30), 'col', tuple_var_assignment_2_602)
                        
                        
                        # Call to len(...): (line 117)
                        # Processing the call arguments (line 117)
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'col' (line 117)
                        col_604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 49), 'col', False)
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'row' (line 117)
                        row_605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 44), 'row', False)
                        # Getting the type of 'self' (line 117)
                        self_606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 31), 'self', False)
                        # Obtaining the member 'squares' of a type (line 117)
                        squares_607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 31), self_606, 'squares')
                        # Obtaining the member '__getitem__' of a type (line 117)
                        getitem___608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 31), squares_607, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 117)
                        subscript_call_result_609 = invoke(stypy.reporting.localization.Localization(__file__, 117, 31), getitem___608, row_605)
                        
                        # Obtaining the member '__getitem__' of a type (line 117)
                        getitem___610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 31), subscript_call_result_609, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 117)
                        subscript_call_result_611 = invoke(stypy.reporting.localization.Localization(__file__, 117, 31), getitem___610, col_604)
                        
                        # Processing the call keyword arguments (line 117)
                        kwargs_612 = {}
                        # Getting the type of 'len' (line 117)
                        len_603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 27), 'len', False)
                        # Calling len(args, kwargs) (line 117)
                        len_call_result_613 = invoke(stypy.reporting.localization.Localization(__file__, 117, 27), len_603, *[subscript_call_result_611], **kwargs_612)
                        
                        int_614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 57), 'int')
                        # Applying the binary operator '>' (line 117)
                        result_gt_615 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 27), '>', len_call_result_613, int_614)
                        
                        # Testing if the type of an if condition is none (line 117)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 117, 24), result_gt_615):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 117)
                            if_condition_616 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 117, 24), result_gt_615)
                            # Assigning a type to the variable 'if_condition_616' (line 117)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 24), 'if_condition_616', if_condition_616)
                            # SSA begins for if statement (line 117)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to set_cell(...): (line 118)
                            # Processing the call arguments (line 118)
                            # Getting the type of 'row' (line 118)
                            row_619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 42), 'row', False)
                            # Getting the type of 'col' (line 118)
                            col_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 47), 'col', False)
                            # Getting the type of 'x' (line 118)
                            x_621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 52), 'x', False)
                            # Processing the call keyword arguments (line 118)
                            kwargs_622 = {}
                            # Getting the type of 'self' (line 118)
                            self_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 28), 'self', False)
                            # Obtaining the member 'set_cell' of a type (line 118)
                            set_cell_618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 28), self_617, 'set_cell')
                            # Calling set_cell(args, kwargs) (line 118)
                            set_cell_call_result_623 = invoke(stypy.reporting.localization.Localization(__file__, 118, 28), set_cell_618, *[row_619, col_620, x_621], **kwargs_622)
                            
                            # SSA join for if statement (line 117)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 115)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'check_for_single_occurances(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_for_single_occurances' in the type store
        # Getting the type of 'stypy_return_type' (line 107)
        stypy_return_type_624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_624)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_for_single_occurances'
        return stypy_return_type_624


    @norecursion
    def check_for_last_in_row_col_3x3(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_for_last_in_row_col_3x3'
        module_type_store = module_type_store.open_function_context('check_for_last_in_row_col_3x3', 120, 4, False)
        # Assigning a type to the variable 'self' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        soduko.check_for_last_in_row_col_3x3.__dict__.__setitem__('stypy_localization', localization)
        soduko.check_for_last_in_row_col_3x3.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        soduko.check_for_last_in_row_col_3x3.__dict__.__setitem__('stypy_type_store', module_type_store)
        soduko.check_for_last_in_row_col_3x3.__dict__.__setitem__('stypy_function_name', 'soduko.check_for_last_in_row_col_3x3')
        soduko.check_for_last_in_row_col_3x3.__dict__.__setitem__('stypy_param_names_list', [])
        soduko.check_for_last_in_row_col_3x3.__dict__.__setitem__('stypy_varargs_param_name', None)
        soduko.check_for_last_in_row_col_3x3.__dict__.__setitem__('stypy_kwargs_param_name', None)
        soduko.check_for_last_in_row_col_3x3.__dict__.__setitem__('stypy_call_defaults', defaults)
        soduko.check_for_last_in_row_col_3x3.__dict__.__setitem__('stypy_call_varargs', varargs)
        soduko.check_for_last_in_row_col_3x3.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        soduko.check_for_last_in_row_col_3x3.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'soduko.check_for_last_in_row_col_3x3', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_for_last_in_row_col_3x3', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_for_last_in_row_col_3x3(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 121)
        list_625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 121)
        # Adding element type (line 121)
        
        # Obtaining an instance of the builtin type 'tuple' (line 121)
        tuple_626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 121)
        # Adding element type (line 121)
        str_627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 41), 'str', 'Row')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 41), tuple_626, str_627)
        # Adding element type (line 121)
        # Getting the type of 'ROW_ITER' (line 121)
        ROW_ITER_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 48), 'ROW_ITER')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 41), tuple_626, ROW_ITER_628)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 39), list_625, tuple_626)
        # Adding element type (line 121)
        
        # Obtaining an instance of the builtin type 'tuple' (line 121)
        tuple_629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 60), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 121)
        # Adding element type (line 121)
        str_630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 60), 'str', 'Col')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 60), tuple_629, str_630)
        # Adding element type (line 121)
        # Getting the type of 'COL_ITER' (line 121)
        COL_ITER_631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 67), 'COL_ITER')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 60), tuple_629, COL_ITER_631)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 39), list_625, tuple_629)
        # Adding element type (line 121)
        
        # Obtaining an instance of the builtin type 'tuple' (line 121)
        tuple_632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 79), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 121)
        # Adding element type (line 121)
        str_633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 79), 'str', '3x3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 79), tuple_632, str_633)
        # Adding element type (line 121)
        # Getting the type of 'TxT_ITER' (line 121)
        TxT_ITER_634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 86), 'TxT_ITER')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 79), tuple_632, TxT_ITER_634)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 39), list_625, tuple_632)
        
        # Assigning a type to the variable 'list_625' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'list_625', list_625)
        # Testing if the for loop is going to be iterated (line 121)
        # Testing the type of a for loop iterable (line 121)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 121, 8), list_625)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 121, 8), list_625):
            # Getting the type of the for loop variable (line 121)
            for_loop_var_635 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 121, 8), list_625)
            # Assigning a type to the variable 'type_name' (line 121)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'type_name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 8), for_loop_var_635, 2, 0))
            # Assigning a type to the variable 'check_type' (line 121)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'check_type', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 8), for_loop_var_635, 2, 1))
            # SSA begins for a for statement (line 121)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'check_type' (line 122)
            check_type_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 30), 'check_type')
            # Assigning a type to the variable 'check_type_636' (line 122)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'check_type_636', check_type_636)
            # Testing if the for loop is going to be iterated (line 122)
            # Testing the type of a for loop iterable (line 122)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 122, 12), check_type_636)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 122, 12), check_type_636):
                # Getting the type of the for loop variable (line 122)
                for_loop_var_637 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 122, 12), check_type_636)
                # Assigning a type to the variable 'check_list' (line 122)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'check_list', for_loop_var_637)
                # SSA begins for a for statement (line 122)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a List to a Name (line 123):
                
                # Assigning a List to a Name (line 123):
                
                # Obtaining an instance of the builtin type 'list' (line 123)
                list_638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 34), 'list')
                # Adding type elements to the builtin type 'list' instance (line 123)
                
                # Assigning a type to the variable 'unknown_entries' (line 123)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'unknown_entries', list_638)
                
                # Assigning a Call to a Name (line 124):
                
                # Assigning a Call to a Name (line 124):
                
                # Call to range(...): (line 124)
                # Processing the call arguments (line 124)
                int_640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 42), 'int')
                int_641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 45), 'int')
                int_642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 49), 'int')
                # Applying the binary operator '+' (line 124)
                result_add_643 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 45), '+', int_641, int_642)
                
                # Processing the call keyword arguments (line 124)
                kwargs_644 = {}
                # Getting the type of 'range' (line 124)
                range_639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 36), 'range', False)
                # Calling range(args, kwargs) (line 124)
                range_call_result_645 = invoke(stypy.reporting.localization.Localization(__file__, 124, 36), range_639, *[int_640, result_add_643], **kwargs_644)
                
                # Assigning a type to the variable 'unassigned_values' (line 124)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'unassigned_values', range_call_result_645)
                
                # Assigning a List to a Name (line 125):
                
                # Assigning a List to a Name (line 125):
                
                # Obtaining an instance of the builtin type 'list' (line 125)
                list_646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 31), 'list')
                # Adding type elements to the builtin type 'list' instance (line 125)
                
                # Assigning a type to the variable 'known_values' (line 125)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'known_values', list_646)
                
                # Getting the type of 'check_list' (line 126)
                check_list_647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 34), 'check_list')
                # Assigning a type to the variable 'check_list_647' (line 126)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'check_list_647', check_list_647)
                # Testing if the for loop is going to be iterated (line 126)
                # Testing the type of a for loop iterable (line 126)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 126, 16), check_list_647)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 126, 16), check_list_647):
                    # Getting the type of the for loop variable (line 126)
                    for_loop_var_648 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 126, 16), check_list_647)
                    # Assigning a type to the variable 'row' (line 126)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'row', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 16), for_loop_var_648, 2, 0))
                    # Assigning a type to the variable 'col' (line 126)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'col', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 16), for_loop_var_648, 2, 1))
                    # SSA begins for a for statement (line 126)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    
                    # Call to len(...): (line 127)
                    # Processing the call arguments (line 127)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'col' (line 127)
                    col_650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 45), 'col', False)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'row' (line 127)
                    row_651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 40), 'row', False)
                    # Getting the type of 'self' (line 127)
                    self_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 27), 'self', False)
                    # Obtaining the member 'squares' of a type (line 127)
                    squares_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 27), self_652, 'squares')
                    # Obtaining the member '__getitem__' of a type (line 127)
                    getitem___654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 27), squares_653, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                    subscript_call_result_655 = invoke(stypy.reporting.localization.Localization(__file__, 127, 27), getitem___654, row_651)
                    
                    # Obtaining the member '__getitem__' of a type (line 127)
                    getitem___656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 27), subscript_call_result_655, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                    subscript_call_result_657 = invoke(stypy.reporting.localization.Localization(__file__, 127, 27), getitem___656, col_650)
                    
                    # Processing the call keyword arguments (line 127)
                    kwargs_658 = {}
                    # Getting the type of 'len' (line 127)
                    len_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'len', False)
                    # Calling len(args, kwargs) (line 127)
                    len_call_result_659 = invoke(stypy.reporting.localization.Localization(__file__, 127, 23), len_649, *[subscript_call_result_657], **kwargs_658)
                    
                    int_660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 54), 'int')
                    # Applying the binary operator '==' (line 127)
                    result_eq_661 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 23), '==', len_call_result_659, int_660)
                    
                    # Testing if the type of an if condition is none (line 127)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 127, 20), result_eq_661):
                        
                        # Call to append(...): (line 136)
                        # Processing the call arguments (line 136)
                        
                        # Obtaining an instance of the builtin type 'tuple' (line 136)
                        tuple_723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 48), 'tuple')
                        # Adding type elements to the builtin type 'tuple' instance (line 136)
                        # Adding element type (line 136)
                        # Getting the type of 'row' (line 136)
                        row_724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 48), 'row', False)
                        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 48), tuple_723, row_724)
                        # Adding element type (line 136)
                        # Getting the type of 'col' (line 136)
                        col_725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 53), 'col', False)
                        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 48), tuple_723, col_725)
                        
                        # Processing the call keyword arguments (line 136)
                        kwargs_726 = {}
                        # Getting the type of 'unknown_entries' (line 136)
                        unknown_entries_721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 24), 'unknown_entries', False)
                        # Obtaining the member 'append' of a type (line 136)
                        append_722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 24), unknown_entries_721, 'append')
                        # Calling append(args, kwargs) (line 136)
                        append_call_result_727 = invoke(stypy.reporting.localization.Localization(__file__, 136, 24), append_722, *[tuple_723], **kwargs_726)
                        
                    else:
                        
                        # Testing the type of an if condition (line 127)
                        if_condition_662 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 127, 20), result_eq_661)
                        # Assigning a type to the variable 'if_condition_662' (line 127)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 20), 'if_condition_662', if_condition_662)
                        # SSA begins for if statement (line 127)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Evaluating assert statement condition
                        
                        
                        # Obtaining the type of the subscript
                        int_663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 54), 'int')
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'col' (line 128)
                        col_664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 49), 'col')
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'row' (line 128)
                        row_665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 44), 'row')
                        # Getting the type of 'self' (line 128)
                        self_666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 31), 'self')
                        # Obtaining the member 'squares' of a type (line 128)
                        squares_667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 31), self_666, 'squares')
                        # Obtaining the member '__getitem__' of a type (line 128)
                        getitem___668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 31), squares_667, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
                        subscript_call_result_669 = invoke(stypy.reporting.localization.Localization(__file__, 128, 31), getitem___668, row_665)
                        
                        # Obtaining the member '__getitem__' of a type (line 128)
                        getitem___670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 31), subscript_call_result_669, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
                        subscript_call_result_671 = invoke(stypy.reporting.localization.Localization(__file__, 128, 31), getitem___670, col_664)
                        
                        # Obtaining the member '__getitem__' of a type (line 128)
                        getitem___672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 31), subscript_call_result_671, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
                        subscript_call_result_673 = invoke(stypy.reporting.localization.Localization(__file__, 128, 31), getitem___672, int_663)
                        
                        # Getting the type of 'known_values' (line 128)
                        known_values_674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 64), 'known_values')
                        # Applying the binary operator 'notin' (line 128)
                        result_contains_675 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 31), 'notin', subscript_call_result_673, known_values_674)
                        
                        assert_676 = result_contains_675
                        # Assigning a type to the variable 'assert_676' (line 128)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 24), 'assert_676', result_contains_675)
                        
                        # Call to append(...): (line 130)
                        # Processing the call arguments (line 130)
                        
                        # Obtaining the type of the subscript
                        int_679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 67), 'int')
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'col' (line 130)
                        col_680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 62), 'col', False)
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'row' (line 130)
                        row_681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 57), 'row', False)
                        # Getting the type of 'self' (line 130)
                        self_682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 44), 'self', False)
                        # Obtaining the member 'squares' of a type (line 130)
                        squares_683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 44), self_682, 'squares')
                        # Obtaining the member '__getitem__' of a type (line 130)
                        getitem___684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 44), squares_683, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                        subscript_call_result_685 = invoke(stypy.reporting.localization.Localization(__file__, 130, 44), getitem___684, row_681)
                        
                        # Obtaining the member '__getitem__' of a type (line 130)
                        getitem___686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 44), subscript_call_result_685, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                        subscript_call_result_687 = invoke(stypy.reporting.localization.Localization(__file__, 130, 44), getitem___686, col_680)
                        
                        # Obtaining the member '__getitem__' of a type (line 130)
                        getitem___688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 44), subscript_call_result_687, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                        subscript_call_result_689 = invoke(stypy.reporting.localization.Localization(__file__, 130, 44), getitem___688, int_679)
                        
                        # Processing the call keyword arguments (line 130)
                        kwargs_690 = {}
                        # Getting the type of 'known_values' (line 130)
                        known_values_677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 24), 'known_values', False)
                        # Obtaining the member 'append' of a type (line 130)
                        append_678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 24), known_values_677, 'append')
                        # Calling append(args, kwargs) (line 130)
                        append_call_result_691 = invoke(stypy.reporting.localization.Localization(__file__, 130, 24), append_678, *[subscript_call_result_689], **kwargs_690)
                        
                        # Evaluating assert statement condition
                        
                        
                        # Obtaining the type of the subscript
                        int_692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 54), 'int')
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'col' (line 132)
                        col_693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 49), 'col')
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'row' (line 132)
                        row_694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 44), 'row')
                        # Getting the type of 'self' (line 132)
                        self_695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 31), 'self')
                        # Obtaining the member 'squares' of a type (line 132)
                        squares_696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 31), self_695, 'squares')
                        # Obtaining the member '__getitem__' of a type (line 132)
                        getitem___697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 31), squares_696, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 132)
                        subscript_call_result_698 = invoke(stypy.reporting.localization.Localization(__file__, 132, 31), getitem___697, row_694)
                        
                        # Obtaining the member '__getitem__' of a type (line 132)
                        getitem___699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 31), subscript_call_result_698, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 132)
                        subscript_call_result_700 = invoke(stypy.reporting.localization.Localization(__file__, 132, 31), getitem___699, col_693)
                        
                        # Obtaining the member '__getitem__' of a type (line 132)
                        getitem___701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 31), subscript_call_result_700, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 132)
                        subscript_call_result_702 = invoke(stypy.reporting.localization.Localization(__file__, 132, 31), getitem___701, int_692)
                        
                        # Getting the type of 'unassigned_values' (line 132)
                        unassigned_values_703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 60), 'unassigned_values')
                        # Applying the binary operator 'in' (line 132)
                        result_contains_704 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 31), 'in', subscript_call_result_702, unassigned_values_703)
                        
                        assert_705 = result_contains_704
                        # Assigning a type to the variable 'assert_705' (line 132)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 24), 'assert_705', result_contains_704)
                        
                        # Call to remove(...): (line 134)
                        # Processing the call arguments (line 134)
                        
                        # Obtaining the type of the subscript
                        int_708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 72), 'int')
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'col' (line 134)
                        col_709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 67), 'col', False)
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'row' (line 134)
                        row_710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 62), 'row', False)
                        # Getting the type of 'self' (line 134)
                        self_711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 49), 'self', False)
                        # Obtaining the member 'squares' of a type (line 134)
                        squares_712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 49), self_711, 'squares')
                        # Obtaining the member '__getitem__' of a type (line 134)
                        getitem___713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 49), squares_712, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 134)
                        subscript_call_result_714 = invoke(stypy.reporting.localization.Localization(__file__, 134, 49), getitem___713, row_710)
                        
                        # Obtaining the member '__getitem__' of a type (line 134)
                        getitem___715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 49), subscript_call_result_714, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 134)
                        subscript_call_result_716 = invoke(stypy.reporting.localization.Localization(__file__, 134, 49), getitem___715, col_709)
                        
                        # Obtaining the member '__getitem__' of a type (line 134)
                        getitem___717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 49), subscript_call_result_716, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 134)
                        subscript_call_result_718 = invoke(stypy.reporting.localization.Localization(__file__, 134, 49), getitem___717, int_708)
                        
                        # Processing the call keyword arguments (line 134)
                        kwargs_719 = {}
                        # Getting the type of 'unassigned_values' (line 134)
                        unassigned_values_706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 24), 'unassigned_values', False)
                        # Obtaining the member 'remove' of a type (line 134)
                        remove_707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 24), unassigned_values_706, 'remove')
                        # Calling remove(args, kwargs) (line 134)
                        remove_call_result_720 = invoke(stypy.reporting.localization.Localization(__file__, 134, 24), remove_707, *[subscript_call_result_718], **kwargs_719)
                        
                        # SSA branch for the else part of an if statement (line 127)
                        module_type_store.open_ssa_branch('else')
                        
                        # Call to append(...): (line 136)
                        # Processing the call arguments (line 136)
                        
                        # Obtaining an instance of the builtin type 'tuple' (line 136)
                        tuple_723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 48), 'tuple')
                        # Adding type elements to the builtin type 'tuple' instance (line 136)
                        # Adding element type (line 136)
                        # Getting the type of 'row' (line 136)
                        row_724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 48), 'row', False)
                        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 48), tuple_723, row_724)
                        # Adding element type (line 136)
                        # Getting the type of 'col' (line 136)
                        col_725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 53), 'col', False)
                        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 48), tuple_723, col_725)
                        
                        # Processing the call keyword arguments (line 136)
                        kwargs_726 = {}
                        # Getting the type of 'unknown_entries' (line 136)
                        unknown_entries_721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 24), 'unknown_entries', False)
                        # Obtaining the member 'append' of a type (line 136)
                        append_722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 24), unknown_entries_721, 'append')
                        # Calling append(args, kwargs) (line 136)
                        append_call_result_727 = invoke(stypy.reporting.localization.Localization(__file__, 136, 24), append_722, *[tuple_723], **kwargs_726)
                        
                        # SSA join for if statement (line 127)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # Evaluating assert statement condition
                
                
                # Call to len(...): (line 137)
                # Processing the call arguments (line 137)
                # Getting the type of 'unknown_entries' (line 137)
                unknown_entries_729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 27), 'unknown_entries', False)
                # Processing the call keyword arguments (line 137)
                kwargs_730 = {}
                # Getting the type of 'len' (line 137)
                len_728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 23), 'len', False)
                # Calling len(args, kwargs) (line 137)
                len_call_result_731 = invoke(stypy.reporting.localization.Localization(__file__, 137, 23), len_728, *[unknown_entries_729], **kwargs_730)
                
                
                # Call to len(...): (line 137)
                # Processing the call arguments (line 137)
                # Getting the type of 'known_values' (line 137)
                known_values_733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 50), 'known_values', False)
                # Processing the call keyword arguments (line 137)
                kwargs_734 = {}
                # Getting the type of 'len' (line 137)
                len_732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 46), 'len', False)
                # Calling len(args, kwargs) (line 137)
                len_call_result_735 = invoke(stypy.reporting.localization.Localization(__file__, 137, 46), len_732, *[known_values_733], **kwargs_734)
                
                # Applying the binary operator '+' (line 137)
                result_add_736 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 23), '+', len_call_result_731, len_call_result_735)
                
                int_737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 67), 'int')
                # Applying the binary operator '==' (line 137)
                result_eq_738 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 23), '==', result_add_736, int_737)
                
                assert_739 = result_eq_738
                # Assigning a type to the variable 'assert_739' (line 137)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'assert_739', result_eq_738)
                # Evaluating assert statement condition
                
                
                # Call to len(...): (line 138)
                # Processing the call arguments (line 138)
                # Getting the type of 'unknown_entries' (line 138)
                unknown_entries_741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 27), 'unknown_entries', False)
                # Processing the call keyword arguments (line 138)
                kwargs_742 = {}
                # Getting the type of 'len' (line 138)
                len_740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 23), 'len', False)
                # Calling len(args, kwargs) (line 138)
                len_call_result_743 = invoke(stypy.reporting.localization.Localization(__file__, 138, 23), len_740, *[unknown_entries_741], **kwargs_742)
                
                
                # Call to len(...): (line 138)
                # Processing the call arguments (line 138)
                # Getting the type of 'unassigned_values' (line 138)
                unassigned_values_745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 51), 'unassigned_values', False)
                # Processing the call keyword arguments (line 138)
                kwargs_746 = {}
                # Getting the type of 'len' (line 138)
                len_744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 47), 'len', False)
                # Calling len(args, kwargs) (line 138)
                len_call_result_747 = invoke(stypy.reporting.localization.Localization(__file__, 138, 47), len_744, *[unassigned_values_745], **kwargs_746)
                
                # Applying the binary operator '==' (line 138)
                result_eq_748 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 23), '==', len_call_result_743, len_call_result_747)
                
                assert_749 = result_eq_748
                # Assigning a type to the variable 'assert_749' (line 138)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'assert_749', result_eq_748)
                
                
                # Call to len(...): (line 139)
                # Processing the call arguments (line 139)
                # Getting the type of 'unknown_entries' (line 139)
                unknown_entries_751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 23), 'unknown_entries', False)
                # Processing the call keyword arguments (line 139)
                kwargs_752 = {}
                # Getting the type of 'len' (line 139)
                len_750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 19), 'len', False)
                # Calling len(args, kwargs) (line 139)
                len_call_result_753 = invoke(stypy.reporting.localization.Localization(__file__, 139, 19), len_750, *[unknown_entries_751], **kwargs_752)
                
                int_754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 43), 'int')
                # Applying the binary operator '==' (line 139)
                result_eq_755 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 19), '==', len_call_result_753, int_754)
                
                # Testing if the type of an if condition is none (line 139)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 139, 16), result_eq_755):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 139)
                    if_condition_756 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 16), result_eq_755)
                    # Assigning a type to the variable 'if_condition_756' (line 139)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 'if_condition_756', if_condition_756)
                    # SSA begins for if statement (line 139)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Subscript to a Name (line 140):
                    
                    # Assigning a Subscript to a Name (line 140):
                    
                    # Obtaining the type of the subscript
                    int_757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 42), 'int')
                    # Getting the type of 'unassigned_values' (line 140)
                    unassigned_values_758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 24), 'unassigned_values')
                    # Obtaining the member '__getitem__' of a type (line 140)
                    getitem___759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 24), unassigned_values_758, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 140)
                    subscript_call_result_760 = invoke(stypy.reporting.localization.Localization(__file__, 140, 24), getitem___759, int_757)
                    
                    # Assigning a type to the variable 'x' (line 140)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 20), 'x', subscript_call_result_760)
                    
                    # Assigning a Subscript to a Tuple (line 141):
                    
                    # Assigning a Subscript to a Name (line 141):
                    
                    # Obtaining the type of the subscript
                    int_761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 20), 'int')
                    
                    # Obtaining the type of the subscript
                    int_762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 49), 'int')
                    # Getting the type of 'unknown_entries' (line 141)
                    unknown_entries_763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 33), 'unknown_entries')
                    # Obtaining the member '__getitem__' of a type (line 141)
                    getitem___764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 33), unknown_entries_763, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 141)
                    subscript_call_result_765 = invoke(stypy.reporting.localization.Localization(__file__, 141, 33), getitem___764, int_762)
                    
                    # Obtaining the member '__getitem__' of a type (line 141)
                    getitem___766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 20), subscript_call_result_765, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 141)
                    subscript_call_result_767 = invoke(stypy.reporting.localization.Localization(__file__, 141, 20), getitem___766, int_761)
                    
                    # Assigning a type to the variable 'tuple_var_assignment_3' (line 141)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 20), 'tuple_var_assignment_3', subscript_call_result_767)
                    
                    # Assigning a Subscript to a Name (line 141):
                    
                    # Obtaining the type of the subscript
                    int_768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 20), 'int')
                    
                    # Obtaining the type of the subscript
                    int_769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 49), 'int')
                    # Getting the type of 'unknown_entries' (line 141)
                    unknown_entries_770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 33), 'unknown_entries')
                    # Obtaining the member '__getitem__' of a type (line 141)
                    getitem___771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 33), unknown_entries_770, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 141)
                    subscript_call_result_772 = invoke(stypy.reporting.localization.Localization(__file__, 141, 33), getitem___771, int_769)
                    
                    # Obtaining the member '__getitem__' of a type (line 141)
                    getitem___773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 20), subscript_call_result_772, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 141)
                    subscript_call_result_774 = invoke(stypy.reporting.localization.Localization(__file__, 141, 20), getitem___773, int_768)
                    
                    # Assigning a type to the variable 'tuple_var_assignment_4' (line 141)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 20), 'tuple_var_assignment_4', subscript_call_result_774)
                    
                    # Assigning a Name to a Name (line 141):
                    # Getting the type of 'tuple_var_assignment_3' (line 141)
                    tuple_var_assignment_3_775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 20), 'tuple_var_assignment_3')
                    # Assigning a type to the variable 'row' (line 141)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 21), 'row', tuple_var_assignment_3_775)
                    
                    # Assigning a Name to a Name (line 141):
                    # Getting the type of 'tuple_var_assignment_4' (line 141)
                    tuple_var_assignment_4_776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 20), 'tuple_var_assignment_4')
                    # Assigning a type to the variable 'col' (line 141)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 26), 'col', tuple_var_assignment_4_776)
                    
                    # Call to set_cell(...): (line 142)
                    # Processing the call arguments (line 142)
                    # Getting the type of 'row' (line 142)
                    row_779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 34), 'row', False)
                    # Getting the type of 'col' (line 142)
                    col_780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 39), 'col', False)
                    # Getting the type of 'x' (line 142)
                    x_781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 44), 'x', False)
                    # Processing the call keyword arguments (line 142)
                    kwargs_782 = {}
                    # Getting the type of 'self' (line 142)
                    self_777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 20), 'self', False)
                    # Obtaining the member 'set_cell' of a type (line 142)
                    set_cell_778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 20), self_777, 'set_cell')
                    # Calling set_cell(args, kwargs) (line 142)
                    set_cell_call_result_783 = invoke(stypy.reporting.localization.Localization(__file__, 142, 20), set_cell_778, *[row_779, col_780, x_781], **kwargs_782)
                    
                    # SSA join for if statement (line 139)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Assigning a type to the variable 'stypy_return_type' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of 'check_for_last_in_row_col_3x3(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_for_last_in_row_col_3x3' in the type store
        # Getting the type of 'stypy_return_type' (line 120)
        stypy_return_type_784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_784)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_for_last_in_row_col_3x3'
        return stypy_return_type_784


    @norecursion
    def one_level_supposition(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'one_level_supposition'
        module_type_store = module_type_store.open_function_context('one_level_supposition', 145, 4, False)
        # Assigning a type to the variable 'self' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        soduko.one_level_supposition.__dict__.__setitem__('stypy_localization', localization)
        soduko.one_level_supposition.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        soduko.one_level_supposition.__dict__.__setitem__('stypy_type_store', module_type_store)
        soduko.one_level_supposition.__dict__.__setitem__('stypy_function_name', 'soduko.one_level_supposition')
        soduko.one_level_supposition.__dict__.__setitem__('stypy_param_names_list', [])
        soduko.one_level_supposition.__dict__.__setitem__('stypy_varargs_param_name', None)
        soduko.one_level_supposition.__dict__.__setitem__('stypy_kwargs_param_name', None)
        soduko.one_level_supposition.__dict__.__setitem__('stypy_call_defaults', defaults)
        soduko.one_level_supposition.__dict__.__setitem__('stypy_call_varargs', varargs)
        soduko.one_level_supposition.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        soduko.one_level_supposition.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'soduko.one_level_supposition', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'one_level_supposition', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'one_level_supposition(...)' code ##################

        
        # Assigning a Name to a Name (line 146):
        
        # Assigning a Name to a Name (line 146):
        # Getting the type of 'True' (line 146)
        True_785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 19), 'True')
        # Assigning a type to the variable 'progress' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'progress', True_785)
        
        # Getting the type of 'progress' (line 147)
        progress_786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 14), 'progress')
        # Assigning a type to the variable 'progress_786' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'progress_786', progress_786)
        # Testing if the while is going to be iterated (line 147)
        # Testing the type of an if condition (line 147)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 8), progress_786)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 147, 8), progress_786):
            # SSA begins for while statement (line 147)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a Name to a Name (line 148):
            
            # Assigning a Name to a Name (line 148):
            # Getting the type of 'False' (line 148)
            False_787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 23), 'False')
            # Assigning a type to the variable 'progress' (line 148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'progress', False_787)
            
            
            # Call to range(...): (line 149)
            # Processing the call arguments (line 149)
            int_789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 29), 'int')
            int_790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 32), 'int')
            # Processing the call keyword arguments (line 149)
            kwargs_791 = {}
            # Getting the type of 'range' (line 149)
            range_788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 23), 'range', False)
            # Calling range(args, kwargs) (line 149)
            range_call_result_792 = invoke(stypy.reporting.localization.Localization(__file__, 149, 23), range_788, *[int_789, int_790], **kwargs_791)
            
            # Assigning a type to the variable 'range_call_result_792' (line 149)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'range_call_result_792', range_call_result_792)
            # Testing if the for loop is going to be iterated (line 149)
            # Testing the type of a for loop iterable (line 149)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 149, 12), range_call_result_792)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 149, 12), range_call_result_792):
                # Getting the type of the for loop variable (line 149)
                for_loop_var_793 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 149, 12), range_call_result_792)
                # Assigning a type to the variable 'row' (line 149)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'row', for_loop_var_793)
                # SSA begins for a for statement (line 149)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Call to range(...): (line 150)
                # Processing the call arguments (line 150)
                int_795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 33), 'int')
                int_796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 36), 'int')
                # Processing the call keyword arguments (line 150)
                kwargs_797 = {}
                # Getting the type of 'range' (line 150)
                range_794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 27), 'range', False)
                # Calling range(args, kwargs) (line 150)
                range_call_result_798 = invoke(stypy.reporting.localization.Localization(__file__, 150, 27), range_794, *[int_795, int_796], **kwargs_797)
                
                # Assigning a type to the variable 'range_call_result_798' (line 150)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'range_call_result_798', range_call_result_798)
                # Testing if the for loop is going to be iterated (line 150)
                # Testing the type of a for loop iterable (line 150)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 150, 16), range_call_result_798)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 150, 16), range_call_result_798):
                    # Getting the type of the for loop variable (line 150)
                    for_loop_var_799 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 150, 16), range_call_result_798)
                    # Assigning a type to the variable 'col' (line 150)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'col', for_loop_var_799)
                    # SSA begins for a for statement (line 150)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    
                    # Call to len(...): (line 151)
                    # Processing the call arguments (line 151)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'col' (line 151)
                    col_801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 45), 'col', False)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'row' (line 151)
                    row_802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 40), 'row', False)
                    # Getting the type of 'self' (line 151)
                    self_803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 27), 'self', False)
                    # Obtaining the member 'squares' of a type (line 151)
                    squares_804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 27), self_803, 'squares')
                    # Obtaining the member '__getitem__' of a type (line 151)
                    getitem___805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 27), squares_804, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 151)
                    subscript_call_result_806 = invoke(stypy.reporting.localization.Localization(__file__, 151, 27), getitem___805, row_802)
                    
                    # Obtaining the member '__getitem__' of a type (line 151)
                    getitem___807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 27), subscript_call_result_806, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 151)
                    subscript_call_result_808 = invoke(stypy.reporting.localization.Localization(__file__, 151, 27), getitem___807, col_801)
                    
                    # Processing the call keyword arguments (line 151)
                    kwargs_809 = {}
                    # Getting the type of 'len' (line 151)
                    len_800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 23), 'len', False)
                    # Calling len(args, kwargs) (line 151)
                    len_call_result_810 = invoke(stypy.reporting.localization.Localization(__file__, 151, 23), len_800, *[subscript_call_result_808], **kwargs_809)
                    
                    int_811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 53), 'int')
                    # Applying the binary operator '>' (line 151)
                    result_gt_812 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 23), '>', len_call_result_810, int_811)
                    
                    # Testing if the type of an if condition is none (line 151)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 151, 20), result_gt_812):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 151)
                        if_condition_813 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 20), result_gt_812)
                        # Assigning a type to the variable 'if_condition_813' (line 151)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 20), 'if_condition_813', if_condition_813)
                        # SSA begins for if statement (line 151)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a List to a Name (line 152):
                        
                        # Assigning a List to a Name (line 152):
                        
                        # Obtaining an instance of the builtin type 'list' (line 152)
                        list_814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 32), 'list')
                        # Adding type elements to the builtin type 'list' instance (line 152)
                        
                        # Assigning a type to the variable 'bad_x' (line 152)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 24), 'bad_x', list_814)
                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'col' (line 153)
                        col_815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 51), 'col')
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'row' (line 153)
                        row_816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 46), 'row')
                        # Getting the type of 'self' (line 153)
                        self_817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 33), 'self')
                        # Obtaining the member 'squares' of a type (line 153)
                        squares_818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 33), self_817, 'squares')
                        # Obtaining the member '__getitem__' of a type (line 153)
                        getitem___819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 33), squares_818, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 153)
                        subscript_call_result_820 = invoke(stypy.reporting.localization.Localization(__file__, 153, 33), getitem___819, row_816)
                        
                        # Obtaining the member '__getitem__' of a type (line 153)
                        getitem___821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 33), subscript_call_result_820, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 153)
                        subscript_call_result_822 = invoke(stypy.reporting.localization.Localization(__file__, 153, 33), getitem___821, col_815)
                        
                        # Assigning a type to the variable 'subscript_call_result_822' (line 153)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 24), 'subscript_call_result_822', subscript_call_result_822)
                        # Testing if the for loop is going to be iterated (line 153)
                        # Testing the type of a for loop iterable (line 153)
                        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 153, 24), subscript_call_result_822)

                        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 153, 24), subscript_call_result_822):
                            # Getting the type of the for loop variable (line 153)
                            for_loop_var_823 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 153, 24), subscript_call_result_822)
                            # Assigning a type to the variable 'x' (line 153)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 24), 'x', for_loop_var_823)
                            # SSA begins for a for statement (line 153)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                            
                            # Assigning a Call to a Name (line 154):
                            
                            # Assigning a Call to a Name (line 154):
                            
                            # Call to copy(...): (line 154)
                            # Processing the call keyword arguments (line 154)
                            kwargs_826 = {}
                            # Getting the type of 'self' (line 154)
                            self_824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 42), 'self', False)
                            # Obtaining the member 'copy' of a type (line 154)
                            copy_825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 42), self_824, 'copy')
                            # Calling copy(args, kwargs) (line 154)
                            copy_call_result_827 = invoke(stypy.reporting.localization.Localization(__file__, 154, 42), copy_825, *[], **kwargs_826)
                            
                            # Assigning a type to the variable 'soduko_copy' (line 154)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 28), 'soduko_copy', copy_call_result_827)
                            
                            
                            # SSA begins for try-except statement (line 155)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
                            
                            # Call to set_cell(...): (line 156)
                            # Processing the call arguments (line 156)
                            # Getting the type of 'row' (line 156)
                            row_830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 53), 'row', False)
                            # Getting the type of 'col' (line 156)
                            col_831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 58), 'col', False)
                            # Getting the type of 'x' (line 156)
                            x_832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 63), 'x', False)
                            # Processing the call keyword arguments (line 156)
                            kwargs_833 = {}
                            # Getting the type of 'soduko_copy' (line 156)
                            soduko_copy_828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 32), 'soduko_copy', False)
                            # Obtaining the member 'set_cell' of a type (line 156)
                            set_cell_829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 32), soduko_copy_828, 'set_cell')
                            # Calling set_cell(args, kwargs) (line 156)
                            set_cell_call_result_834 = invoke(stypy.reporting.localization.Localization(__file__, 156, 32), set_cell_829, *[row_830, col_831, x_832], **kwargs_833)
                            
                            
                            # Call to check(...): (line 157)
                            # Processing the call keyword arguments (line 157)
                            kwargs_837 = {}
                            # Getting the type of 'soduko_copy' (line 157)
                            soduko_copy_835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 32), 'soduko_copy', False)
                            # Obtaining the member 'check' of a type (line 157)
                            check_836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 32), soduko_copy_835, 'check')
                            # Calling check(args, kwargs) (line 157)
                            check_call_result_838 = invoke(stypy.reporting.localization.Localization(__file__, 157, 32), check_836, *[], **kwargs_837)
                            
                            # SSA branch for the except part of a try statement (line 155)
                            # SSA branch for the except 'AssertionError' branch of a try statement (line 155)
                            # Storing handler type
                            module_type_store.open_ssa_branch('except')
                            # Getting the type of 'AssertionError' (line 158)
                            AssertionError_839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 35), 'AssertionError')
                            # Assigning a type to the variable 'e' (line 158)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 28), 'e', AssertionError_839)
                            
                            # Call to append(...): (line 159)
                            # Processing the call arguments (line 159)
                            # Getting the type of 'x' (line 159)
                            x_842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 45), 'x', False)
                            # Processing the call keyword arguments (line 159)
                            kwargs_843 = {}
                            # Getting the type of 'bad_x' (line 159)
                            bad_x_840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 32), 'bad_x', False)
                            # Obtaining the member 'append' of a type (line 159)
                            append_841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 32), bad_x_840, 'append')
                            # Calling append(args, kwargs) (line 159)
                            append_call_result_844 = invoke(stypy.reporting.localization.Localization(__file__, 159, 32), append_841, *[x_842], **kwargs_843)
                            
                            # SSA join for try-except statement (line 155)
                            module_type_store = module_type_store.join_ssa_context()
                            
                            # Deleting a member
                            module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 160, 28), module_type_store, 'soduko_copy')
                            # SSA join for a for statement
                            module_type_store = module_type_store.join_ssa_context()

                        
                        
                        
                        # Call to len(...): (line 161)
                        # Processing the call arguments (line 161)
                        # Getting the type of 'bad_x' (line 161)
                        bad_x_846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 31), 'bad_x', False)
                        # Processing the call keyword arguments (line 161)
                        kwargs_847 = {}
                        # Getting the type of 'len' (line 161)
                        len_845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 27), 'len', False)
                        # Calling len(args, kwargs) (line 161)
                        len_call_result_848 = invoke(stypy.reporting.localization.Localization(__file__, 161, 27), len_845, *[bad_x_846], **kwargs_847)
                        
                        int_849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 41), 'int')
                        # Applying the binary operator '==' (line 161)
                        result_eq_850 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 27), '==', len_call_result_848, int_849)
                        
                        # Testing if the type of an if condition is none (line 161)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 161, 24), result_eq_850):
                            
                            
                            # Call to len(...): (line 163)
                            # Processing the call arguments (line 163)
                            # Getting the type of 'bad_x' (line 163)
                            bad_x_853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 33), 'bad_x', False)
                            # Processing the call keyword arguments (line 163)
                            kwargs_854 = {}
                            # Getting the type of 'len' (line 163)
                            len_852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 29), 'len', False)
                            # Calling len(args, kwargs) (line 163)
                            len_call_result_855 = invoke(stypy.reporting.localization.Localization(__file__, 163, 29), len_852, *[bad_x_853], **kwargs_854)
                            
                            
                            # Call to len(...): (line 163)
                            # Processing the call arguments (line 163)
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'col' (line 163)
                            col_857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 64), 'col', False)
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'row' (line 163)
                            row_858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 59), 'row', False)
                            # Getting the type of 'self' (line 163)
                            self_859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 46), 'self', False)
                            # Obtaining the member 'squares' of a type (line 163)
                            squares_860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 46), self_859, 'squares')
                            # Obtaining the member '__getitem__' of a type (line 163)
                            getitem___861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 46), squares_860, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 163)
                            subscript_call_result_862 = invoke(stypy.reporting.localization.Localization(__file__, 163, 46), getitem___861, row_858)
                            
                            # Obtaining the member '__getitem__' of a type (line 163)
                            getitem___863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 46), subscript_call_result_862, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 163)
                            subscript_call_result_864 = invoke(stypy.reporting.localization.Localization(__file__, 163, 46), getitem___863, col_857)
                            
                            # Processing the call keyword arguments (line 163)
                            kwargs_865 = {}
                            # Getting the type of 'len' (line 163)
                            len_856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 42), 'len', False)
                            # Calling len(args, kwargs) (line 163)
                            len_call_result_866 = invoke(stypy.reporting.localization.Localization(__file__, 163, 42), len_856, *[subscript_call_result_864], **kwargs_865)
                            
                            # Applying the binary operator '<' (line 163)
                            result_lt_867 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 29), '<', len_call_result_855, len_call_result_866)
                            
                            # Testing if the type of an if condition is none (line 163)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 163, 29), result_lt_867):
                                # Evaluating assert statement condition
                                # Getting the type of 'False' (line 169)
                                False_883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 35), 'False')
                                assert_884 = False_883
                                # Assigning a type to the variable 'assert_884' (line 169)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 28), 'assert_884', False_883)
                            else:
                                
                                # Testing the type of an if condition (line 163)
                                if_condition_868 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 29), result_lt_867)
                                # Assigning a type to the variable 'if_condition_868' (line 163)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 29), 'if_condition_868', if_condition_868)
                                # SSA begins for if statement (line 163)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Getting the type of 'bad_x' (line 164)
                                bad_x_869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 37), 'bad_x')
                                # Assigning a type to the variable 'bad_x_869' (line 164)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'bad_x_869', bad_x_869)
                                # Testing if the for loop is going to be iterated (line 164)
                                # Testing the type of a for loop iterable (line 164)
                                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 164, 28), bad_x_869)

                                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 164, 28), bad_x_869):
                                    # Getting the type of the for loop variable (line 164)
                                    for_loop_var_870 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 164, 28), bad_x_869)
                                    # Assigning a type to the variable 'x' (line 164)
                                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'x', for_loop_var_870)
                                    # SSA begins for a for statement (line 164)
                                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                                    
                                    # Call to cell_exclude(...): (line 165)
                                    # Processing the call arguments (line 165)
                                    # Getting the type of 'row' (line 165)
                                    row_873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 50), 'row', False)
                                    # Getting the type of 'col' (line 165)
                                    col_874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 55), 'col', False)
                                    # Getting the type of 'x' (line 165)
                                    x_875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 60), 'x', False)
                                    # Processing the call keyword arguments (line 165)
                                    kwargs_876 = {}
                                    # Getting the type of 'self' (line 165)
                                    self_871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 32), 'self', False)
                                    # Obtaining the member 'cell_exclude' of a type (line 165)
                                    cell_exclude_872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 32), self_871, 'cell_exclude')
                                    # Calling cell_exclude(args, kwargs) (line 165)
                                    cell_exclude_call_result_877 = invoke(stypy.reporting.localization.Localization(__file__, 165, 32), cell_exclude_872, *[row_873, col_874, x_875], **kwargs_876)
                                    
                                    
                                    # Call to check(...): (line 166)
                                    # Processing the call keyword arguments (line 166)
                                    kwargs_880 = {}
                                    # Getting the type of 'self' (line 166)
                                    self_878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 32), 'self', False)
                                    # Obtaining the member 'check' of a type (line 166)
                                    check_879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 32), self_878, 'check')
                                    # Calling check(args, kwargs) (line 166)
                                    check_call_result_881 = invoke(stypy.reporting.localization.Localization(__file__, 166, 32), check_879, *[], **kwargs_880)
                                    
                                    # SSA join for a for statement
                                    module_type_store = module_type_store.join_ssa_context()

                                
                                
                                # Assigning a Name to a Name (line 167):
                                
                                # Assigning a Name to a Name (line 167):
                                # Getting the type of 'True' (line 167)
                                True_882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 39), 'True')
                                # Assigning a type to the variable 'progress' (line 167)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 28), 'progress', True_882)
                                # SSA branch for the else part of an if statement (line 163)
                                module_type_store.open_ssa_branch('else')
                                # Evaluating assert statement condition
                                # Getting the type of 'False' (line 169)
                                False_883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 35), 'False')
                                assert_884 = False_883
                                # Assigning a type to the variable 'assert_884' (line 169)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 28), 'assert_884', False_883)
                                # SSA join for if statement (line 163)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 161)
                            if_condition_851 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 24), result_eq_850)
                            # Assigning a type to the variable 'if_condition_851' (line 161)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 24), 'if_condition_851', if_condition_851)
                            # SSA begins for if statement (line 161)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            pass
                            # SSA branch for the else part of an if statement (line 161)
                            module_type_store.open_ssa_branch('else')
                            
                            
                            # Call to len(...): (line 163)
                            # Processing the call arguments (line 163)
                            # Getting the type of 'bad_x' (line 163)
                            bad_x_853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 33), 'bad_x', False)
                            # Processing the call keyword arguments (line 163)
                            kwargs_854 = {}
                            # Getting the type of 'len' (line 163)
                            len_852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 29), 'len', False)
                            # Calling len(args, kwargs) (line 163)
                            len_call_result_855 = invoke(stypy.reporting.localization.Localization(__file__, 163, 29), len_852, *[bad_x_853], **kwargs_854)
                            
                            
                            # Call to len(...): (line 163)
                            # Processing the call arguments (line 163)
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'col' (line 163)
                            col_857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 64), 'col', False)
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'row' (line 163)
                            row_858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 59), 'row', False)
                            # Getting the type of 'self' (line 163)
                            self_859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 46), 'self', False)
                            # Obtaining the member 'squares' of a type (line 163)
                            squares_860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 46), self_859, 'squares')
                            # Obtaining the member '__getitem__' of a type (line 163)
                            getitem___861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 46), squares_860, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 163)
                            subscript_call_result_862 = invoke(stypy.reporting.localization.Localization(__file__, 163, 46), getitem___861, row_858)
                            
                            # Obtaining the member '__getitem__' of a type (line 163)
                            getitem___863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 46), subscript_call_result_862, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 163)
                            subscript_call_result_864 = invoke(stypy.reporting.localization.Localization(__file__, 163, 46), getitem___863, col_857)
                            
                            # Processing the call keyword arguments (line 163)
                            kwargs_865 = {}
                            # Getting the type of 'len' (line 163)
                            len_856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 42), 'len', False)
                            # Calling len(args, kwargs) (line 163)
                            len_call_result_866 = invoke(stypy.reporting.localization.Localization(__file__, 163, 42), len_856, *[subscript_call_result_864], **kwargs_865)
                            
                            # Applying the binary operator '<' (line 163)
                            result_lt_867 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 29), '<', len_call_result_855, len_call_result_866)
                            
                            # Testing if the type of an if condition is none (line 163)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 163, 29), result_lt_867):
                                # Evaluating assert statement condition
                                # Getting the type of 'False' (line 169)
                                False_883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 35), 'False')
                                assert_884 = False_883
                                # Assigning a type to the variable 'assert_884' (line 169)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 28), 'assert_884', False_883)
                            else:
                                
                                # Testing the type of an if condition (line 163)
                                if_condition_868 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 29), result_lt_867)
                                # Assigning a type to the variable 'if_condition_868' (line 163)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 29), 'if_condition_868', if_condition_868)
                                # SSA begins for if statement (line 163)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Getting the type of 'bad_x' (line 164)
                                bad_x_869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 37), 'bad_x')
                                # Assigning a type to the variable 'bad_x_869' (line 164)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'bad_x_869', bad_x_869)
                                # Testing if the for loop is going to be iterated (line 164)
                                # Testing the type of a for loop iterable (line 164)
                                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 164, 28), bad_x_869)

                                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 164, 28), bad_x_869):
                                    # Getting the type of the for loop variable (line 164)
                                    for_loop_var_870 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 164, 28), bad_x_869)
                                    # Assigning a type to the variable 'x' (line 164)
                                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'x', for_loop_var_870)
                                    # SSA begins for a for statement (line 164)
                                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                                    
                                    # Call to cell_exclude(...): (line 165)
                                    # Processing the call arguments (line 165)
                                    # Getting the type of 'row' (line 165)
                                    row_873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 50), 'row', False)
                                    # Getting the type of 'col' (line 165)
                                    col_874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 55), 'col', False)
                                    # Getting the type of 'x' (line 165)
                                    x_875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 60), 'x', False)
                                    # Processing the call keyword arguments (line 165)
                                    kwargs_876 = {}
                                    # Getting the type of 'self' (line 165)
                                    self_871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 32), 'self', False)
                                    # Obtaining the member 'cell_exclude' of a type (line 165)
                                    cell_exclude_872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 32), self_871, 'cell_exclude')
                                    # Calling cell_exclude(args, kwargs) (line 165)
                                    cell_exclude_call_result_877 = invoke(stypy.reporting.localization.Localization(__file__, 165, 32), cell_exclude_872, *[row_873, col_874, x_875], **kwargs_876)
                                    
                                    
                                    # Call to check(...): (line 166)
                                    # Processing the call keyword arguments (line 166)
                                    kwargs_880 = {}
                                    # Getting the type of 'self' (line 166)
                                    self_878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 32), 'self', False)
                                    # Obtaining the member 'check' of a type (line 166)
                                    check_879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 32), self_878, 'check')
                                    # Calling check(args, kwargs) (line 166)
                                    check_call_result_881 = invoke(stypy.reporting.localization.Localization(__file__, 166, 32), check_879, *[], **kwargs_880)
                                    
                                    # SSA join for a for statement
                                    module_type_store = module_type_store.join_ssa_context()

                                
                                
                                # Assigning a Name to a Name (line 167):
                                
                                # Assigning a Name to a Name (line 167):
                                # Getting the type of 'True' (line 167)
                                True_882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 39), 'True')
                                # Assigning a type to the variable 'progress' (line 167)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 28), 'progress', True_882)
                                # SSA branch for the else part of an if statement (line 163)
                                module_type_store.open_ssa_branch('else')
                                # Evaluating assert statement condition
                                # Getting the type of 'False' (line 169)
                                False_883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 35), 'False')
                                assert_884 = False_883
                                # Assigning a type to the variable 'assert_884' (line 169)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 28), 'assert_884', False_883)
                                # SSA join for if statement (line 163)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 161)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 151)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for while statement (line 147)
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'one_level_supposition(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'one_level_supposition' in the type store
        # Getting the type of 'stypy_return_type' (line 145)
        stypy_return_type_885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_885)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'one_level_supposition'
        return stypy_return_type_885


# Assigning a type to the variable 'soduko' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'soduko', soduko)

@norecursion
def main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'main'
    module_type_store = module_type_store.open_function_context('main', 172, 0, False)
    
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

    
    
    # Call to range(...): (line 173)
    # Processing the call arguments (line 173)
    int_887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 19), 'int')
    # Processing the call keyword arguments (line 173)
    kwargs_888 = {}
    # Getting the type of 'range' (line 173)
    range_886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 13), 'range', False)
    # Calling range(args, kwargs) (line 173)
    range_call_result_889 = invoke(stypy.reporting.localization.Localization(__file__, 173, 13), range_886, *[int_887], **kwargs_888)
    
    # Assigning a type to the variable 'range_call_result_889' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'range_call_result_889', range_call_result_889)
    # Testing if the for loop is going to be iterated (line 173)
    # Testing the type of a for loop iterable (line 173)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 173, 4), range_call_result_889)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 173, 4), range_call_result_889):
        # Getting the type of the for loop variable (line 173)
        for_loop_var_890 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 173, 4), range_call_result_889)
        # Assigning a type to the variable 'x' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'x', for_loop_var_890)
        # SSA begins for a for statement (line 173)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 174):
        
        # Assigning a Call to a Name (line 174):
        
        # Call to soduko(...): (line 174)
        # Processing the call arguments (line 174)
        
        # Obtaining an instance of the builtin type 'list' (line 174)
        list_892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 174)
        # Adding element type (line 174)
        str_893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 20), 'str', '800000600')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 19), list_892, str_893)
        # Adding element type (line 174)
        str_894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 20), 'str', '040500100')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 19), list_892, str_894)
        # Adding element type (line 174)
        str_895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 20), 'str', '070090000')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 19), list_892, str_895)
        # Adding element type (line 174)
        str_896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 20), 'str', '030020007')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 19), list_892, str_896)
        # Adding element type (line 174)
        str_897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 20), 'str', '600008004')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 19), list_892, str_897)
        # Adding element type (line 174)
        str_898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 20), 'str', '500000090')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 19), list_892, str_898)
        # Adding element type (line 174)
        str_899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 20), 'str', '000030020')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 19), list_892, str_899)
        # Adding element type (line 174)
        str_900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 20), 'str', '001006050')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 19), list_892, str_900)
        # Adding element type (line 174)
        str_901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 20), 'str', '004000003')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 19), list_892, str_901)
        
        # Processing the call keyword arguments (line 174)
        kwargs_902 = {}
        # Getting the type of 'soduko' (line 174)
        soduko_891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'soduko', False)
        # Calling soduko(args, kwargs) (line 174)
        soduko_call_result_903 = invoke(stypy.reporting.localization.Localization(__file__, 174, 12), soduko_891, *[list_892], **kwargs_902)
        
        # Assigning a type to the variable 't' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 't', soduko_call_result_903)
        
        # Call to check(...): (line 184)
        # Processing the call keyword arguments (line 184)
        kwargs_906 = {}
        # Getting the type of 't' (line 184)
        t_904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 't', False)
        # Obtaining the member 'check' of a type (line 184)
        check_905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), t_904, 'check')
        # Calling check(args, kwargs) (line 184)
        check_call_result_907 = invoke(stypy.reporting.localization.Localization(__file__, 184, 8), check_905, *[], **kwargs_906)
        
        
        # Call to one_level_supposition(...): (line 185)
        # Processing the call keyword arguments (line 185)
        kwargs_910 = {}
        # Getting the type of 't' (line 185)
        t_908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 't', False)
        # Obtaining the member 'one_level_supposition' of a type (line 185)
        one_level_supposition_909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 8), t_908, 'one_level_supposition')
        # Calling one_level_supposition(args, kwargs) (line 185)
        one_level_supposition_call_result_911 = invoke(stypy.reporting.localization.Localization(__file__, 185, 8), one_level_supposition_909, *[], **kwargs_910)
        
        
        # Call to check(...): (line 186)
        # Processing the call keyword arguments (line 186)
        kwargs_914 = {}
        # Getting the type of 't' (line 186)
        t_912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 't', False)
        # Obtaining the member 'check' of a type (line 186)
        check_913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), t_912, 'check')
        # Calling check(args, kwargs) (line 186)
        check_call_result_915 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), check_913, *[], **kwargs_914)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 172)
    stypy_return_type_916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_916)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_916

# Assigning a type to the variable 'main' (line 172)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 0), 'main', main)

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

    
    # Call to main(...): (line 191)
    # Processing the call keyword arguments (line 191)
    kwargs_918 = {}
    # Getting the type of 'main' (line 191)
    main_917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'main', False)
    # Calling main(args, kwargs) (line 191)
    main_call_result_919 = invoke(stypy.reporting.localization.Localization(__file__, 191, 4), main_917, *[], **kwargs_918)
    
    # Getting the type of 'True' (line 192)
    True_920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'stypy_return_type', True_920)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 190)
    stypy_return_type_921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_921)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_921

# Assigning a type to the variable 'run' (line 190)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'run', run)

# Call to run(...): (line 195)
# Processing the call keyword arguments (line 195)
kwargs_923 = {}
# Getting the type of 'run' (line 195)
run_922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 0), 'run', False)
# Calling run(args, kwargs) (line 195)
run_call_result_924 = invoke(stypy.reporting.localization.Localization(__file__, 195, 0), run_922, *[], **kwargs_923)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

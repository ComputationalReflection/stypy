
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
            
            
            
            # Call to range(...): (line 19)
            # Processing the call arguments (line 19)
            int_97 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 29), 'int')
            int_98 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 32), 'int')
            # Processing the call keyword arguments (line 19)
            kwargs_99 = {}
            # Getting the type of 'range' (line 19)
            range_96 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 23), 'range', False)
            # Calling range(args, kwargs) (line 19)
            range_call_result_100 = invoke(stypy.reporting.localization.Localization(__file__, 19, 23), range_96, *[int_97, int_98], **kwargs_99)
            
            # Testing if the for loop is going to be iterated (line 19)
            # Testing the type of a for loop iterable (line 19)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 19, 12), range_call_result_100)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 19, 12), range_call_result_100):
                # Getting the type of the for loop variable (line 19)
                for_loop_var_101 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 19, 12), range_call_result_100)
                # Assigning a type to the variable 'row' (line 19)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'row', for_loop_var_101)
                # SSA begins for a for statement (line 19)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to set_row(...): (line 20)
                # Processing the call arguments (line 20)
                # Getting the type of 'row' (line 20)
                row_104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 29), 'row', False)
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 20)
                row_105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 45), 'row', False)
                # Getting the type of 'start_grid' (line 20)
                start_grid_106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 34), 'start_grid', False)
                # Obtaining the member '__getitem__' of a type (line 20)
                getitem___107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 34), start_grid_106, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 20)
                subscript_call_result_108 = invoke(stypy.reporting.localization.Localization(__file__, 20, 34), getitem___107, row_105)
                
                # Processing the call keyword arguments (line 20)
                kwargs_109 = {}
                # Getting the type of 'self' (line 20)
                self_102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'self', False)
                # Obtaining the member 'set_row' of a type (line 20)
                set_row_103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 16), self_102, 'set_row')
                # Calling set_row(args, kwargs) (line 20)
                set_row_call_result_110 = invoke(stypy.reporting.localization.Localization(__file__, 20, 16), set_row_103, *[row_104, subscript_call_result_108], **kwargs_109)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            

            if more_types_in_union_89:
                # SSA join for if statement (line 17)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 22):
        
        # Assigning a Name to a Attribute (line 22):
        # Getting the type of 'False' (line 22)
        False_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 24), 'False')
        # Getting the type of 'self' (line 22)
        self_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self')
        # Setting the type of the member '_changed' of a type (line 22)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), self_112, '_changed', False_111)
        
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
        None_114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 29), 'None', False)
        # Processing the call keyword arguments (line 25)
        kwargs_115 = {}
        # Getting the type of 'soduko' (line 25)
        soduko_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 22), 'soduko', False)
        # Calling soduko(args, kwargs) (line 25)
        soduko_call_result_116 = invoke(stypy.reporting.localization.Localization(__file__, 25, 22), soduko_113, *[None_114], **kwargs_115)
        
        # Assigning a type to the variable 'soduko_copy' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'soduko_copy', soduko_call_result_116)
        
        
        # Call to range(...): (line 26)
        # Processing the call arguments (line 26)
        int_118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'int')
        int_119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 28), 'int')
        # Processing the call keyword arguments (line 26)
        kwargs_120 = {}
        # Getting the type of 'range' (line 26)
        range_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 19), 'range', False)
        # Calling range(args, kwargs) (line 26)
        range_call_result_121 = invoke(stypy.reporting.localization.Localization(__file__, 26, 19), range_117, *[int_118, int_119], **kwargs_120)
        
        # Testing if the for loop is going to be iterated (line 26)
        # Testing the type of a for loop iterable (line 26)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 26, 8), range_call_result_121)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 26, 8), range_call_result_121):
            # Getting the type of the for loop variable (line 26)
            for_loop_var_122 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 26, 8), range_call_result_121)
            # Assigning a type to the variable 'row' (line 26)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'row', for_loop_var_122)
            # SSA begins for a for statement (line 26)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to range(...): (line 27)
            # Processing the call arguments (line 27)
            int_124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 29), 'int')
            int_125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 32), 'int')
            # Processing the call keyword arguments (line 27)
            kwargs_126 = {}
            # Getting the type of 'range' (line 27)
            range_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 23), 'range', False)
            # Calling range(args, kwargs) (line 27)
            range_call_result_127 = invoke(stypy.reporting.localization.Localization(__file__, 27, 23), range_123, *[int_124, int_125], **kwargs_126)
            
            # Testing if the for loop is going to be iterated (line 27)
            # Testing the type of a for loop iterable (line 27)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 27, 12), range_call_result_127)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 27, 12), range_call_result_127):
                # Getting the type of the for loop variable (line 27)
                for_loop_var_128 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 27, 12), range_call_result_127)
                # Assigning a type to the variable 'col' (line 27)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'col', for_loop_var_128)
                # SSA begins for a for statement (line 27)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Subscript to a Subscript (line 28):
                
                # Assigning a Subscript to a Subscript (line 28):
                
                # Obtaining the type of the subscript
                slice_129 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 28, 48), None, None, None)
                
                # Obtaining the type of the subscript
                # Getting the type of 'col' (line 28)
                col_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 66), 'col')
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 28)
                row_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 61), 'row')
                # Getting the type of 'self' (line 28)
                self_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 48), 'self')
                # Obtaining the member 'squares' of a type (line 28)
                squares_133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 48), self_132, 'squares')
                # Obtaining the member '__getitem__' of a type (line 28)
                getitem___134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 48), squares_133, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 28)
                subscript_call_result_135 = invoke(stypy.reporting.localization.Localization(__file__, 28, 48), getitem___134, row_131)
                
                # Obtaining the member '__getitem__' of a type (line 28)
                getitem___136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 48), subscript_call_result_135, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 28)
                subscript_call_result_137 = invoke(stypy.reporting.localization.Localization(__file__, 28, 48), getitem___136, col_130)
                
                # Obtaining the member '__getitem__' of a type (line 28)
                getitem___138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 48), subscript_call_result_137, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 28)
                subscript_call_result_139 = invoke(stypy.reporting.localization.Localization(__file__, 28, 48), getitem___138, slice_129)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 28)
                row_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 36), 'row')
                # Getting the type of 'soduko_copy' (line 28)
                soduko_copy_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), 'soduko_copy')
                # Obtaining the member 'squares' of a type (line 28)
                squares_142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 16), soduko_copy_141, 'squares')
                # Obtaining the member '__getitem__' of a type (line 28)
                getitem___143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 16), squares_142, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 28)
                subscript_call_result_144 = invoke(stypy.reporting.localization.Localization(__file__, 28, 16), getitem___143, row_140)
                
                # Getting the type of 'col' (line 28)
                col_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 41), 'col')
                # Storing an element on a container (line 28)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 16), subscript_call_result_144, (col_145, subscript_call_result_139))
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Name to a Attribute (line 29):
        
        # Assigning a Name to a Attribute (line 29):
        # Getting the type of 'False' (line 29)
        False_146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 31), 'False')
        # Getting the type of 'soduko_copy' (line 29)
        soduko_copy_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'soduko_copy')
        # Setting the type of the member '_changed' of a type (line 29)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), soduko_copy_147, '_changed', False_146)
        # Getting the type of 'soduko_copy' (line 30)
        soduko_copy_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 15), 'soduko_copy')
        # Assigning a type to the variable 'stypy_return_type' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'stypy_return_type', soduko_copy_148)
        
        # ################# End of 'copy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'copy' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_149)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'copy'
        return stypy_return_type_149


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
        x_list_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 19), 'x_list', False)
        # Processing the call keyword arguments (line 33)
        kwargs_152 = {}
        # Getting the type of 'len' (line 33)
        len_150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'len', False)
        # Calling len(args, kwargs) (line 33)
        len_call_result_153 = invoke(stypy.reporting.localization.Localization(__file__, 33, 15), len_150, *[x_list_151], **kwargs_152)
        
        int_154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 30), 'int')
        # Applying the binary operator '==' (line 33)
        result_eq_155 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 15), '==', len_call_result_153, int_154)
        
        
        
        # Call to range(...): (line 34)
        # Processing the call arguments (line 34)
        int_157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 25), 'int')
        int_158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 28), 'int')
        # Processing the call keyword arguments (line 34)
        kwargs_159 = {}
        # Getting the type of 'range' (line 34)
        range_156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'range', False)
        # Calling range(args, kwargs) (line 34)
        range_call_result_160 = invoke(stypy.reporting.localization.Localization(__file__, 34, 19), range_156, *[int_157, int_158], **kwargs_159)
        
        # Testing if the for loop is going to be iterated (line 34)
        # Testing the type of a for loop iterable (line 34)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 34, 8), range_call_result_160)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 34, 8), range_call_result_160):
            # Getting the type of the for loop variable (line 34)
            for_loop_var_161 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 34, 8), range_call_result_160)
            # Assigning a type to the variable 'col' (line 34)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'col', for_loop_var_161)
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
            col_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 31), 'col', False)
            # Getting the type of 'x_list' (line 36)
            x_list_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 24), 'x_list', False)
            # Obtaining the member '__getitem__' of a type (line 36)
            getitem___165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 24), x_list_164, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 36)
            subscript_call_result_166 = invoke(stypy.reporting.localization.Localization(__file__, 36, 24), getitem___165, col_163)
            
            # Processing the call keyword arguments (line 36)
            kwargs_167 = {}
            # Getting the type of 'int' (line 36)
            int_162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 20), 'int', False)
            # Calling int(args, kwargs) (line 36)
            int_call_result_168 = invoke(stypy.reporting.localization.Localization(__file__, 36, 20), int_162, *[subscript_call_result_166], **kwargs_167)
            
            # Assigning a type to the variable 'x' (line 36)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'x', int_call_result_168)
            # SSA branch for the except part of a try statement (line 35)
            # SSA branch for the except '<any exception>' branch of a try statement (line 35)
            module_type_store.open_ssa_branch('except')
            
            # Assigning a Num to a Name (line 38):
            
            # Assigning a Num to a Name (line 38):
            int_169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 20), 'int')
            # Assigning a type to the variable 'x' (line 38)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'x', int_169)
            # SSA join for try-except statement (line 35)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to set_cell(...): (line 39)
            # Processing the call arguments (line 39)
            # Getting the type of 'row' (line 39)
            row_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 26), 'row', False)
            # Getting the type of 'col' (line 39)
            col_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 31), 'col', False)
            # Getting the type of 'x' (line 39)
            x_174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 36), 'x', False)
            # Processing the call keyword arguments (line 39)
            kwargs_175 = {}
            # Getting the type of 'self' (line 39)
            self_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'self', False)
            # Obtaining the member 'set_cell' of a type (line 39)
            set_cell_171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 12), self_170, 'set_cell')
            # Calling set_cell(args, kwargs) (line 39)
            set_cell_call_result_176 = invoke(stypy.reporting.localization.Localization(__file__, 39, 12), set_cell_171, *[row_172, col_173, x_174], **kwargs_175)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'set_row(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_row' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_177)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_row'
        return stypy_return_type_177


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
        col_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 29), 'col')
        
        # Obtaining the type of the subscript
        # Getting the type of 'row' (line 42)
        row_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 24), 'row')
        # Getting the type of 'self' (line 42)
        self_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'self')
        # Obtaining the member 'squares' of a type (line 42)
        squares_181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 11), self_180, 'squares')
        # Obtaining the member '__getitem__' of a type (line 42)
        getitem___182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 11), squares_181, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 42)
        subscript_call_result_183 = invoke(stypy.reporting.localization.Localization(__file__, 42, 11), getitem___182, row_179)
        
        # Obtaining the member '__getitem__' of a type (line 42)
        getitem___184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 11), subscript_call_result_183, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 42)
        subscript_call_result_185 = invoke(stypy.reporting.localization.Localization(__file__, 42, 11), getitem___184, col_178)
        
        
        # Obtaining an instance of the builtin type 'list' (line 42)
        list_186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 42)
        # Adding element type (line 42)
        # Getting the type of 'x' (line 42)
        x_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 38), 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 37), list_186, x_187)
        
        # Applying the binary operator '==' (line 42)
        result_eq_188 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 11), '==', subscript_call_result_185, list_186)
        
        # Testing if the type of an if condition is none (line 42)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 42, 8), result_eq_188):
            
            # Getting the type of 'x' (line 44)
            x_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'x')
            
            # Call to range(...): (line 44)
            # Processing the call arguments (line 44)
            int_192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 28), 'int')
            int_193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 31), 'int')
            int_194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 35), 'int')
            # Applying the binary operator '+' (line 44)
            result_add_195 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 31), '+', int_193, int_194)
            
            # Processing the call keyword arguments (line 44)
            kwargs_196 = {}
            # Getting the type of 'range' (line 44)
            range_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 22), 'range', False)
            # Calling range(args, kwargs) (line 44)
            range_call_result_197 = invoke(stypy.reporting.localization.Localization(__file__, 44, 22), range_191, *[int_192, result_add_195], **kwargs_196)
            
            # Applying the binary operator 'notin' (line 44)
            result_contains_198 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 13), 'notin', x_190, range_call_result_197)
            
            # Testing if the type of an if condition is none (line 44)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 44, 13), result_contains_198):
                # Evaluating assert statement condition
                
                # Getting the type of 'x' (line 47)
                x_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 19), 'x')
                
                # Obtaining the type of the subscript
                # Getting the type of 'col' (line 47)
                col_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 42), 'col')
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 47)
                row_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 37), 'row')
                # Getting the type of 'self' (line 47)
                self_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 24), 'self')
                # Obtaining the member 'squares' of a type (line 47)
                squares_204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), self_203, 'squares')
                # Obtaining the member '__getitem__' of a type (line 47)
                getitem___205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), squares_204, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 47)
                subscript_call_result_206 = invoke(stypy.reporting.localization.Localization(__file__, 47, 24), getitem___205, row_202)
                
                # Obtaining the member '__getitem__' of a type (line 47)
                getitem___207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), subscript_call_result_206, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 47)
                subscript_call_result_208 = invoke(stypy.reporting.localization.Localization(__file__, 47, 24), getitem___207, col_201)
                
                # Applying the binary operator 'in' (line 47)
                result_contains_209 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 19), 'in', x_200, subscript_call_result_208)
                
                
                # Assigning a List to a Subscript (line 49):
                
                # Assigning a List to a Subscript (line 49):
                
                # Obtaining an instance of the builtin type 'list' (line 49)
                list_210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 37), 'list')
                # Adding type elements to the builtin type 'list' instance (line 49)
                # Adding element type (line 49)
                # Getting the type of 'x' (line 49)
                x_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 38), 'x')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 37), list_210, x_211)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 49)
                row_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'row')
                # Getting the type of 'self' (line 49)
                self_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'self')
                # Obtaining the member 'squares' of a type (line 49)
                squares_214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), self_213, 'squares')
                # Obtaining the member '__getitem__' of a type (line 49)
                getitem___215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), squares_214, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 49)
                subscript_call_result_216 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), getitem___215, row_212)
                
                # Getting the type of 'col' (line 49)
                col_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 30), 'col')
                # Storing an element on a container (line 49)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 12), subscript_call_result_216, (col_217, list_210))
                
                # Call to update_neighbours(...): (line 50)
                # Processing the call arguments (line 50)
                # Getting the type of 'row' (line 50)
                row_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 35), 'row', False)
                # Getting the type of 'col' (line 50)
                col_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 40), 'col', False)
                # Getting the type of 'x' (line 50)
                x_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 45), 'x', False)
                # Processing the call keyword arguments (line 50)
                kwargs_223 = {}
                # Getting the type of 'self' (line 50)
                self_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'self', False)
                # Obtaining the member 'update_neighbours' of a type (line 50)
                update_neighbours_219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), self_218, 'update_neighbours')
                # Calling update_neighbours(args, kwargs) (line 50)
                update_neighbours_call_result_224 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), update_neighbours_219, *[row_220, col_221, x_222], **kwargs_223)
                
                
                # Assigning a Name to a Attribute (line 51):
                
                # Assigning a Name to a Attribute (line 51):
                # Getting the type of 'True' (line 51)
                True_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 28), 'True')
                # Getting the type of 'self' (line 51)
                self_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'self')
                # Setting the type of the member '_changed' of a type (line 51)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), self_226, '_changed', True_225)
            else:
                
                # Testing the type of an if condition (line 44)
                if_condition_199 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 13), result_contains_198)
                # Assigning a type to the variable 'if_condition_199' (line 44)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'if_condition_199', if_condition_199)
                # SSA begins for if statement (line 44)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                pass
                # SSA branch for the else part of an if statement (line 44)
                module_type_store.open_ssa_branch('else')
                # Evaluating assert statement condition
                
                # Getting the type of 'x' (line 47)
                x_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 19), 'x')
                
                # Obtaining the type of the subscript
                # Getting the type of 'col' (line 47)
                col_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 42), 'col')
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 47)
                row_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 37), 'row')
                # Getting the type of 'self' (line 47)
                self_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 24), 'self')
                # Obtaining the member 'squares' of a type (line 47)
                squares_204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), self_203, 'squares')
                # Obtaining the member '__getitem__' of a type (line 47)
                getitem___205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), squares_204, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 47)
                subscript_call_result_206 = invoke(stypy.reporting.localization.Localization(__file__, 47, 24), getitem___205, row_202)
                
                # Obtaining the member '__getitem__' of a type (line 47)
                getitem___207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), subscript_call_result_206, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 47)
                subscript_call_result_208 = invoke(stypy.reporting.localization.Localization(__file__, 47, 24), getitem___207, col_201)
                
                # Applying the binary operator 'in' (line 47)
                result_contains_209 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 19), 'in', x_200, subscript_call_result_208)
                
                
                # Assigning a List to a Subscript (line 49):
                
                # Assigning a List to a Subscript (line 49):
                
                # Obtaining an instance of the builtin type 'list' (line 49)
                list_210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 37), 'list')
                # Adding type elements to the builtin type 'list' instance (line 49)
                # Adding element type (line 49)
                # Getting the type of 'x' (line 49)
                x_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 38), 'x')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 37), list_210, x_211)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 49)
                row_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'row')
                # Getting the type of 'self' (line 49)
                self_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'self')
                # Obtaining the member 'squares' of a type (line 49)
                squares_214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), self_213, 'squares')
                # Obtaining the member '__getitem__' of a type (line 49)
                getitem___215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), squares_214, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 49)
                subscript_call_result_216 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), getitem___215, row_212)
                
                # Getting the type of 'col' (line 49)
                col_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 30), 'col')
                # Storing an element on a container (line 49)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 12), subscript_call_result_216, (col_217, list_210))
                
                # Call to update_neighbours(...): (line 50)
                # Processing the call arguments (line 50)
                # Getting the type of 'row' (line 50)
                row_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 35), 'row', False)
                # Getting the type of 'col' (line 50)
                col_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 40), 'col', False)
                # Getting the type of 'x' (line 50)
                x_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 45), 'x', False)
                # Processing the call keyword arguments (line 50)
                kwargs_223 = {}
                # Getting the type of 'self' (line 50)
                self_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'self', False)
                # Obtaining the member 'update_neighbours' of a type (line 50)
                update_neighbours_219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), self_218, 'update_neighbours')
                # Calling update_neighbours(args, kwargs) (line 50)
                update_neighbours_call_result_224 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), update_neighbours_219, *[row_220, col_221, x_222], **kwargs_223)
                
                
                # Assigning a Name to a Attribute (line 51):
                
                # Assigning a Name to a Attribute (line 51):
                # Getting the type of 'True' (line 51)
                True_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 28), 'True')
                # Getting the type of 'self' (line 51)
                self_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'self')
                # Setting the type of the member '_changed' of a type (line 51)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), self_226, '_changed', True_225)
                # SSA join for if statement (line 44)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 42)
            if_condition_189 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 8), result_eq_188)
            # Assigning a type to the variable 'if_condition_189' (line 42)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'if_condition_189', if_condition_189)
            # SSA begins for if statement (line 42)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            pass
            # SSA branch for the else part of an if statement (line 42)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'x' (line 44)
            x_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'x')
            
            # Call to range(...): (line 44)
            # Processing the call arguments (line 44)
            int_192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 28), 'int')
            int_193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 31), 'int')
            int_194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 35), 'int')
            # Applying the binary operator '+' (line 44)
            result_add_195 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 31), '+', int_193, int_194)
            
            # Processing the call keyword arguments (line 44)
            kwargs_196 = {}
            # Getting the type of 'range' (line 44)
            range_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 22), 'range', False)
            # Calling range(args, kwargs) (line 44)
            range_call_result_197 = invoke(stypy.reporting.localization.Localization(__file__, 44, 22), range_191, *[int_192, result_add_195], **kwargs_196)
            
            # Applying the binary operator 'notin' (line 44)
            result_contains_198 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 13), 'notin', x_190, range_call_result_197)
            
            # Testing if the type of an if condition is none (line 44)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 44, 13), result_contains_198):
                # Evaluating assert statement condition
                
                # Getting the type of 'x' (line 47)
                x_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 19), 'x')
                
                # Obtaining the type of the subscript
                # Getting the type of 'col' (line 47)
                col_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 42), 'col')
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 47)
                row_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 37), 'row')
                # Getting the type of 'self' (line 47)
                self_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 24), 'self')
                # Obtaining the member 'squares' of a type (line 47)
                squares_204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), self_203, 'squares')
                # Obtaining the member '__getitem__' of a type (line 47)
                getitem___205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), squares_204, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 47)
                subscript_call_result_206 = invoke(stypy.reporting.localization.Localization(__file__, 47, 24), getitem___205, row_202)
                
                # Obtaining the member '__getitem__' of a type (line 47)
                getitem___207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), subscript_call_result_206, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 47)
                subscript_call_result_208 = invoke(stypy.reporting.localization.Localization(__file__, 47, 24), getitem___207, col_201)
                
                # Applying the binary operator 'in' (line 47)
                result_contains_209 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 19), 'in', x_200, subscript_call_result_208)
                
                
                # Assigning a List to a Subscript (line 49):
                
                # Assigning a List to a Subscript (line 49):
                
                # Obtaining an instance of the builtin type 'list' (line 49)
                list_210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 37), 'list')
                # Adding type elements to the builtin type 'list' instance (line 49)
                # Adding element type (line 49)
                # Getting the type of 'x' (line 49)
                x_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 38), 'x')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 37), list_210, x_211)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 49)
                row_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'row')
                # Getting the type of 'self' (line 49)
                self_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'self')
                # Obtaining the member 'squares' of a type (line 49)
                squares_214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), self_213, 'squares')
                # Obtaining the member '__getitem__' of a type (line 49)
                getitem___215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), squares_214, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 49)
                subscript_call_result_216 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), getitem___215, row_212)
                
                # Getting the type of 'col' (line 49)
                col_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 30), 'col')
                # Storing an element on a container (line 49)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 12), subscript_call_result_216, (col_217, list_210))
                
                # Call to update_neighbours(...): (line 50)
                # Processing the call arguments (line 50)
                # Getting the type of 'row' (line 50)
                row_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 35), 'row', False)
                # Getting the type of 'col' (line 50)
                col_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 40), 'col', False)
                # Getting the type of 'x' (line 50)
                x_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 45), 'x', False)
                # Processing the call keyword arguments (line 50)
                kwargs_223 = {}
                # Getting the type of 'self' (line 50)
                self_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'self', False)
                # Obtaining the member 'update_neighbours' of a type (line 50)
                update_neighbours_219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), self_218, 'update_neighbours')
                # Calling update_neighbours(args, kwargs) (line 50)
                update_neighbours_call_result_224 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), update_neighbours_219, *[row_220, col_221, x_222], **kwargs_223)
                
                
                # Assigning a Name to a Attribute (line 51):
                
                # Assigning a Name to a Attribute (line 51):
                # Getting the type of 'True' (line 51)
                True_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 28), 'True')
                # Getting the type of 'self' (line 51)
                self_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'self')
                # Setting the type of the member '_changed' of a type (line 51)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), self_226, '_changed', True_225)
            else:
                
                # Testing the type of an if condition (line 44)
                if_condition_199 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 13), result_contains_198)
                # Assigning a type to the variable 'if_condition_199' (line 44)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'if_condition_199', if_condition_199)
                # SSA begins for if statement (line 44)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                pass
                # SSA branch for the else part of an if statement (line 44)
                module_type_store.open_ssa_branch('else')
                # Evaluating assert statement condition
                
                # Getting the type of 'x' (line 47)
                x_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 19), 'x')
                
                # Obtaining the type of the subscript
                # Getting the type of 'col' (line 47)
                col_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 42), 'col')
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 47)
                row_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 37), 'row')
                # Getting the type of 'self' (line 47)
                self_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 24), 'self')
                # Obtaining the member 'squares' of a type (line 47)
                squares_204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), self_203, 'squares')
                # Obtaining the member '__getitem__' of a type (line 47)
                getitem___205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), squares_204, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 47)
                subscript_call_result_206 = invoke(stypy.reporting.localization.Localization(__file__, 47, 24), getitem___205, row_202)
                
                # Obtaining the member '__getitem__' of a type (line 47)
                getitem___207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), subscript_call_result_206, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 47)
                subscript_call_result_208 = invoke(stypy.reporting.localization.Localization(__file__, 47, 24), getitem___207, col_201)
                
                # Applying the binary operator 'in' (line 47)
                result_contains_209 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 19), 'in', x_200, subscript_call_result_208)
                
                
                # Assigning a List to a Subscript (line 49):
                
                # Assigning a List to a Subscript (line 49):
                
                # Obtaining an instance of the builtin type 'list' (line 49)
                list_210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 37), 'list')
                # Adding type elements to the builtin type 'list' instance (line 49)
                # Adding element type (line 49)
                # Getting the type of 'x' (line 49)
                x_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 38), 'x')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 37), list_210, x_211)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 49)
                row_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'row')
                # Getting the type of 'self' (line 49)
                self_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'self')
                # Obtaining the member 'squares' of a type (line 49)
                squares_214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), self_213, 'squares')
                # Obtaining the member '__getitem__' of a type (line 49)
                getitem___215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), squares_214, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 49)
                subscript_call_result_216 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), getitem___215, row_212)
                
                # Getting the type of 'col' (line 49)
                col_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 30), 'col')
                # Storing an element on a container (line 49)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 12), subscript_call_result_216, (col_217, list_210))
                
                # Call to update_neighbours(...): (line 50)
                # Processing the call arguments (line 50)
                # Getting the type of 'row' (line 50)
                row_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 35), 'row', False)
                # Getting the type of 'col' (line 50)
                col_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 40), 'col', False)
                # Getting the type of 'x' (line 50)
                x_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 45), 'x', False)
                # Processing the call keyword arguments (line 50)
                kwargs_223 = {}
                # Getting the type of 'self' (line 50)
                self_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'self', False)
                # Obtaining the member 'update_neighbours' of a type (line 50)
                update_neighbours_219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), self_218, 'update_neighbours')
                # Calling update_neighbours(args, kwargs) (line 50)
                update_neighbours_call_result_224 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), update_neighbours_219, *[row_220, col_221, x_222], **kwargs_223)
                
                
                # Assigning a Name to a Attribute (line 51):
                
                # Assigning a Name to a Attribute (line 51):
                # Getting the type of 'True' (line 51)
                True_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 28), 'True')
                # Getting the type of 'self' (line 51)
                self_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'self')
                # Setting the type of the member '_changed' of a type (line 51)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), self_226, '_changed', True_225)
                # SSA join for if statement (line 44)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 42)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'set_cell(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_cell' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_227)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_cell'
        return stypy_return_type_227


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
        x_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 15), 'x')
        
        # Call to range(...): (line 54)
        # Processing the call arguments (line 54)
        int_230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 26), 'int')
        int_231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 29), 'int')
        int_232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 33), 'int')
        # Applying the binary operator '+' (line 54)
        result_add_233 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 29), '+', int_231, int_232)
        
        # Processing the call keyword arguments (line 54)
        kwargs_234 = {}
        # Getting the type of 'range' (line 54)
        range_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'range', False)
        # Calling range(args, kwargs) (line 54)
        range_call_result_235 = invoke(stypy.reporting.localization.Localization(__file__, 54, 20), range_229, *[int_230, result_add_233], **kwargs_234)
        
        # Applying the binary operator 'in' (line 54)
        result_contains_236 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 15), 'in', x_228, range_call_result_235)
        
        
        # Getting the type of 'x' (line 55)
        x_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 11), 'x')
        
        # Obtaining the type of the subscript
        # Getting the type of 'col' (line 55)
        col_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 34), 'col')
        
        # Obtaining the type of the subscript
        # Getting the type of 'row' (line 55)
        row_239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 29), 'row')
        # Getting the type of 'self' (line 55)
        self_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'self')
        # Obtaining the member 'squares' of a type (line 55)
        squares_241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 16), self_240, 'squares')
        # Obtaining the member '__getitem__' of a type (line 55)
        getitem___242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 16), squares_241, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 55)
        subscript_call_result_243 = invoke(stypy.reporting.localization.Localization(__file__, 55, 16), getitem___242, row_239)
        
        # Obtaining the member '__getitem__' of a type (line 55)
        getitem___244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 16), subscript_call_result_243, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 55)
        subscript_call_result_245 = invoke(stypy.reporting.localization.Localization(__file__, 55, 16), getitem___244, col_238)
        
        # Applying the binary operator 'in' (line 55)
        result_contains_246 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 11), 'in', x_237, subscript_call_result_245)
        
        # Testing if the type of an if condition is none (line 55)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 55, 8), result_contains_246):
            pass
        else:
            
            # Testing the type of an if condition (line 55)
            if_condition_247 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 55, 8), result_contains_246)
            # Assigning a type to the variable 'if_condition_247' (line 55)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'if_condition_247', if_condition_247)
            # SSA begins for if statement (line 55)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to remove(...): (line 56)
            # Processing the call arguments (line 56)
            # Getting the type of 'x' (line 56)
            x_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 42), 'x', False)
            # Processing the call keyword arguments (line 56)
            kwargs_258 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'col' (line 56)
            col_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 30), 'col', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row' (line 56)
            row_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 25), 'row', False)
            # Getting the type of 'self' (line 56)
            self_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'self', False)
            # Obtaining the member 'squares' of a type (line 56)
            squares_251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 12), self_250, 'squares')
            # Obtaining the member '__getitem__' of a type (line 56)
            getitem___252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 12), squares_251, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 56)
            subscript_call_result_253 = invoke(stypy.reporting.localization.Localization(__file__, 56, 12), getitem___252, row_249)
            
            # Obtaining the member '__getitem__' of a type (line 56)
            getitem___254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 12), subscript_call_result_253, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 56)
            subscript_call_result_255 = invoke(stypy.reporting.localization.Localization(__file__, 56, 12), getitem___254, col_248)
            
            # Obtaining the member 'remove' of a type (line 56)
            remove_256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 12), subscript_call_result_255, 'remove')
            # Calling remove(args, kwargs) (line 56)
            remove_call_result_259 = invoke(stypy.reporting.localization.Localization(__file__, 56, 12), remove_256, *[x_257], **kwargs_258)
            
            # Evaluating assert statement condition
            
            
            # Call to len(...): (line 57)
            # Processing the call arguments (line 57)
            
            # Obtaining the type of the subscript
            # Getting the type of 'col' (line 57)
            col_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 41), 'col', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row' (line 57)
            row_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 36), 'row', False)
            # Getting the type of 'self' (line 57)
            self_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 23), 'self', False)
            # Obtaining the member 'squares' of a type (line 57)
            squares_264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 23), self_263, 'squares')
            # Obtaining the member '__getitem__' of a type (line 57)
            getitem___265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 23), squares_264, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 57)
            subscript_call_result_266 = invoke(stypy.reporting.localization.Localization(__file__, 57, 23), getitem___265, row_262)
            
            # Obtaining the member '__getitem__' of a type (line 57)
            getitem___267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 23), subscript_call_result_266, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 57)
            subscript_call_result_268 = invoke(stypy.reporting.localization.Localization(__file__, 57, 23), getitem___267, col_261)
            
            # Processing the call keyword arguments (line 57)
            kwargs_269 = {}
            # Getting the type of 'len' (line 57)
            len_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 19), 'len', False)
            # Calling len(args, kwargs) (line 57)
            len_call_result_270 = invoke(stypy.reporting.localization.Localization(__file__, 57, 19), len_260, *[subscript_call_result_268], **kwargs_269)
            
            int_271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 49), 'int')
            # Applying the binary operator '>' (line 57)
            result_gt_272 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 19), '>', len_call_result_270, int_271)
            
            
            
            # Call to len(...): (line 58)
            # Processing the call arguments (line 58)
            
            # Obtaining the type of the subscript
            # Getting the type of 'col' (line 58)
            col_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 37), 'col', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row' (line 58)
            row_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 32), 'row', False)
            # Getting the type of 'self' (line 58)
            self_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 19), 'self', False)
            # Obtaining the member 'squares' of a type (line 58)
            squares_277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 19), self_276, 'squares')
            # Obtaining the member '__getitem__' of a type (line 58)
            getitem___278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 19), squares_277, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 58)
            subscript_call_result_279 = invoke(stypy.reporting.localization.Localization(__file__, 58, 19), getitem___278, row_275)
            
            # Obtaining the member '__getitem__' of a type (line 58)
            getitem___280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 19), subscript_call_result_279, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 58)
            subscript_call_result_281 = invoke(stypy.reporting.localization.Localization(__file__, 58, 19), getitem___280, col_274)
            
            # Processing the call keyword arguments (line 58)
            kwargs_282 = {}
            # Getting the type of 'len' (line 58)
            len_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 15), 'len', False)
            # Calling len(args, kwargs) (line 58)
            len_call_result_283 = invoke(stypy.reporting.localization.Localization(__file__, 58, 15), len_273, *[subscript_call_result_281], **kwargs_282)
            
            int_284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 46), 'int')
            # Applying the binary operator '==' (line 58)
            result_eq_285 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 15), '==', len_call_result_283, int_284)
            
            # Testing if the type of an if condition is none (line 58)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 58, 12), result_eq_285):
                pass
            else:
                
                # Testing the type of an if condition (line 58)
                if_condition_286 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 12), result_eq_285)
                # Assigning a type to the variable 'if_condition_286' (line 58)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'if_condition_286', if_condition_286)
                # SSA begins for if statement (line 58)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Attribute (line 59):
                
                # Assigning a Name to a Attribute (line 59):
                # Getting the type of 'True' (line 59)
                True_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 32), 'True')
                # Getting the type of 'self' (line 59)
                self_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'self')
                # Setting the type of the member '_changed' of a type (line 59)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 16), self_288, '_changed', True_287)
                
                # Call to update_neighbours(...): (line 60)
                # Processing the call arguments (line 60)
                # Getting the type of 'row' (line 60)
                row_291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 39), 'row', False)
                # Getting the type of 'col' (line 60)
                col_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 44), 'col', False)
                
                # Obtaining the type of the subscript
                int_293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 72), 'int')
                
                # Obtaining the type of the subscript
                # Getting the type of 'col' (line 60)
                col_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 67), 'col', False)
                
                # Obtaining the type of the subscript
                # Getting the type of 'row' (line 60)
                row_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 62), 'row', False)
                # Getting the type of 'self' (line 60)
                self_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 49), 'self', False)
                # Obtaining the member 'squares' of a type (line 60)
                squares_297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 49), self_296, 'squares')
                # Obtaining the member '__getitem__' of a type (line 60)
                getitem___298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 49), squares_297, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 60)
                subscript_call_result_299 = invoke(stypy.reporting.localization.Localization(__file__, 60, 49), getitem___298, row_295)
                
                # Obtaining the member '__getitem__' of a type (line 60)
                getitem___300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 49), subscript_call_result_299, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 60)
                subscript_call_result_301 = invoke(stypy.reporting.localization.Localization(__file__, 60, 49), getitem___300, col_294)
                
                # Obtaining the member '__getitem__' of a type (line 60)
                getitem___302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 49), subscript_call_result_301, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 60)
                subscript_call_result_303 = invoke(stypy.reporting.localization.Localization(__file__, 60, 49), getitem___302, int_293)
                
                # Processing the call keyword arguments (line 60)
                kwargs_304 = {}
                # Getting the type of 'self' (line 60)
                self_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'self', False)
                # Obtaining the member 'update_neighbours' of a type (line 60)
                update_neighbours_290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 16), self_289, 'update_neighbours')
                # Calling update_neighbours(args, kwargs) (line 60)
                update_neighbours_call_result_305 = invoke(stypy.reporting.localization.Localization(__file__, 60, 16), update_neighbours_290, *[row_291, col_292, subscript_call_result_303], **kwargs_304)
                
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
        stypy_return_type_306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_306)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cell_exclude'
        return stypy_return_type_306


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
        int_308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 25), 'int')
        int_309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 28), 'int')
        # Processing the call keyword arguments (line 66)
        kwargs_310 = {}
        # Getting the type of 'range' (line 66)
        range_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 19), 'range', False)
        # Calling range(args, kwargs) (line 66)
        range_call_result_311 = invoke(stypy.reporting.localization.Localization(__file__, 66, 19), range_307, *[int_308, int_309], **kwargs_310)
        
        # Testing if the for loop is going to be iterated (line 66)
        # Testing the type of a for loop iterable (line 66)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 66, 8), range_call_result_311)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 66, 8), range_call_result_311):
            # Getting the type of the for loop variable (line 66)
            for_loop_var_312 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 66, 8), range_call_result_311)
            # Assigning a type to the variable 'row' (line 66)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'row', for_loop_var_312)
            # SSA begins for a for statement (line 66)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'row' (line 67)
            row_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 15), 'row')
            # Getting the type of 'set_row' (line 67)
            set_row_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 22), 'set_row')
            # Applying the binary operator '!=' (line 67)
            result_ne_315 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 15), '!=', row_313, set_row_314)
            
            # Testing if the type of an if condition is none (line 67)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 67, 12), result_ne_315):
                pass
            else:
                
                # Testing the type of an if condition (line 67)
                if_condition_316 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 12), result_ne_315)
                # Assigning a type to the variable 'if_condition_316' (line 67)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'if_condition_316', if_condition_316)
                # SSA begins for if statement (line 67)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to cell_exclude(...): (line 68)
                # Processing the call arguments (line 68)
                # Getting the type of 'row' (line 68)
                row_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 34), 'row', False)
                # Getting the type of 'set_col' (line 68)
                set_col_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 39), 'set_col', False)
                # Getting the type of 'x' (line 68)
                x_321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 48), 'x', False)
                # Processing the call keyword arguments (line 68)
                kwargs_322 = {}
                # Getting the type of 'self' (line 68)
                self_317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'self', False)
                # Obtaining the member 'cell_exclude' of a type (line 68)
                cell_exclude_318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 16), self_317, 'cell_exclude')
                # Calling cell_exclude(args, kwargs) (line 68)
                cell_exclude_call_result_323 = invoke(stypy.reporting.localization.Localization(__file__, 68, 16), cell_exclude_318, *[row_319, set_col_320, x_321], **kwargs_322)
                
                # SSA join for if statement (line 67)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to range(...): (line 69)
        # Processing the call arguments (line 69)
        int_325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 25), 'int')
        int_326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 28), 'int')
        # Processing the call keyword arguments (line 69)
        kwargs_327 = {}
        # Getting the type of 'range' (line 69)
        range_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 19), 'range', False)
        # Calling range(args, kwargs) (line 69)
        range_call_result_328 = invoke(stypy.reporting.localization.Localization(__file__, 69, 19), range_324, *[int_325, int_326], **kwargs_327)
        
        # Testing if the for loop is going to be iterated (line 69)
        # Testing the type of a for loop iterable (line 69)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 69, 8), range_call_result_328)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 69, 8), range_call_result_328):
            # Getting the type of the for loop variable (line 69)
            for_loop_var_329 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 69, 8), range_call_result_328)
            # Assigning a type to the variable 'col' (line 69)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'col', for_loop_var_329)
            # SSA begins for a for statement (line 69)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'col' (line 70)
            col_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 15), 'col')
            # Getting the type of 'set_col' (line 70)
            set_col_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 22), 'set_col')
            # Applying the binary operator '!=' (line 70)
            result_ne_332 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 15), '!=', col_330, set_col_331)
            
            # Testing if the type of an if condition is none (line 70)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 70, 12), result_ne_332):
                pass
            else:
                
                # Testing the type of an if condition (line 70)
                if_condition_333 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 12), result_ne_332)
                # Assigning a type to the variable 'if_condition_333' (line 70)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'if_condition_333', if_condition_333)
                # SSA begins for if statement (line 70)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to cell_exclude(...): (line 71)
                # Processing the call arguments (line 71)
                # Getting the type of 'set_row' (line 71)
                set_row_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 34), 'set_row', False)
                # Getting the type of 'col' (line 71)
                col_337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 43), 'col', False)
                # Getting the type of 'x' (line 71)
                x_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 48), 'x', False)
                # Processing the call keyword arguments (line 71)
                kwargs_339 = {}
                # Getting the type of 'self' (line 71)
                self_334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'self', False)
                # Obtaining the member 'cell_exclude' of a type (line 71)
                cell_exclude_335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 16), self_334, 'cell_exclude')
                # Calling cell_exclude(args, kwargs) (line 71)
                cell_exclude_call_result_340 = invoke(stypy.reporting.localization.Localization(__file__, 71, 16), cell_exclude_335, *[set_row_336, col_337, x_338], **kwargs_339)
                
                # SSA join for if statement (line 70)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 'TRIPLETS' (line 72)
        TRIPLETS_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 23), 'TRIPLETS')
        # Testing if the for loop is going to be iterated (line 72)
        # Testing the type of a for loop iterable (line 72)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 72, 8), TRIPLETS_341)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 72, 8), TRIPLETS_341):
            # Getting the type of the for loop variable (line 72)
            for_loop_var_342 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 72, 8), TRIPLETS_341)
            # Assigning a type to the variable 'triplet' (line 72)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'triplet', for_loop_var_342)
            # SSA begins for a for statement (line 72)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'set_row' (line 73)
            set_row_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 15), 'set_row')
            # Getting the type of 'triplet' (line 73)
            triplet_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 26), 'triplet')
            # Applying the binary operator 'in' (line 73)
            result_contains_345 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 15), 'in', set_row_343, triplet_344)
            
            # Testing if the type of an if condition is none (line 73)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 73, 12), result_contains_345):
                pass
            else:
                
                # Testing the type of an if condition (line 73)
                if_condition_346 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 12), result_contains_345)
                # Assigning a type to the variable 'if_condition_346' (line 73)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'if_condition_346', if_condition_346)
                # SSA begins for if statement (line 73)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Subscript to a Name (line 73):
                
                # Assigning a Subscript to a Name (line 73):
                
                # Obtaining the type of the subscript
                slice_347 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 73, 42), None, None, None)
                # Getting the type of 'triplet' (line 73)
                triplet_348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 42), 'triplet')
                # Obtaining the member '__getitem__' of a type (line 73)
                getitem___349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 42), triplet_348, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 73)
                subscript_call_result_350 = invoke(stypy.reporting.localization.Localization(__file__, 73, 42), getitem___349, slice_347)
                
                # Assigning a type to the variable 'rows' (line 73)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 35), 'rows', subscript_call_result_350)
                # SSA join for if statement (line 73)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'set_col' (line 74)
            set_col_351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 15), 'set_col')
            # Getting the type of 'triplet' (line 74)
            triplet_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 26), 'triplet')
            # Applying the binary operator 'in' (line 74)
            result_contains_353 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 15), 'in', set_col_351, triplet_352)
            
            # Testing if the type of an if condition is none (line 74)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 12), result_contains_353):
                pass
            else:
                
                # Testing the type of an if condition (line 74)
                if_condition_354 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 12), result_contains_353)
                # Assigning a type to the variable 'if_condition_354' (line 74)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'if_condition_354', if_condition_354)
                # SSA begins for if statement (line 74)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Subscript to a Name (line 74):
                
                # Assigning a Subscript to a Name (line 74):
                
                # Obtaining the type of the subscript
                slice_355 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 74, 42), None, None, None)
                # Getting the type of 'triplet' (line 74)
                triplet_356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 42), 'triplet')
                # Obtaining the member '__getitem__' of a type (line 74)
                getitem___357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 42), triplet_356, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 74)
                subscript_call_result_358 = invoke(stypy.reporting.localization.Localization(__file__, 74, 42), getitem___357, slice_355)
                
                # Assigning a type to the variable 'cols' (line 74)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 35), 'cols', subscript_call_result_358)
                # SSA join for if statement (line 74)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to remove(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'set_row' (line 75)
        set_row_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 20), 'set_row', False)
        # Processing the call keyword arguments (line 75)
        kwargs_362 = {}
        # Getting the type of 'rows' (line 75)
        rows_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'rows', False)
        # Obtaining the member 'remove' of a type (line 75)
        remove_360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), rows_359, 'remove')
        # Calling remove(args, kwargs) (line 75)
        remove_call_result_363 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), remove_360, *[set_row_361], **kwargs_362)
        
        
        # Call to remove(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'set_col' (line 76)
        set_col_366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 20), 'set_col', False)
        # Processing the call keyword arguments (line 76)
        kwargs_367 = {}
        # Getting the type of 'cols' (line 76)
        cols_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'cols', False)
        # Obtaining the member 'remove' of a type (line 76)
        remove_365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), cols_364, 'remove')
        # Calling remove(args, kwargs) (line 76)
        remove_call_result_368 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), remove_365, *[set_col_366], **kwargs_367)
        
        
        # Getting the type of 'rows' (line 77)
        rows_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 19), 'rows')
        # Testing if the for loop is going to be iterated (line 77)
        # Testing the type of a for loop iterable (line 77)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 77, 8), rows_369)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 77, 8), rows_369):
            # Getting the type of the for loop variable (line 77)
            for_loop_var_370 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 77, 8), rows_369)
            # Assigning a type to the variable 'row' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'row', for_loop_var_370)
            # SSA begins for a for statement (line 77)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'cols' (line 78)
            cols_371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'cols')
            # Testing if the for loop is going to be iterated (line 78)
            # Testing the type of a for loop iterable (line 78)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 78, 12), cols_371)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 78, 12), cols_371):
                # Getting the type of the for loop variable (line 78)
                for_loop_var_372 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 78, 12), cols_371)
                # Assigning a type to the variable 'col' (line 78)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'col', for_loop_var_372)
                # SSA begins for a for statement (line 78)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                # Evaluating assert statement condition
                
                # Evaluating a boolean operation
                
                # Getting the type of 'row' (line 79)
                row_373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 23), 'row')
                # Getting the type of 'set_row' (line 79)
                set_row_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 30), 'set_row')
                # Applying the binary operator '!=' (line 79)
                result_ne_375 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 23), '!=', row_373, set_row_374)
                
                
                # Getting the type of 'col' (line 79)
                col_376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 41), 'col')
                # Getting the type of 'set_col' (line 79)
                set_col_377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 48), 'set_col')
                # Applying the binary operator '!=' (line 79)
                result_ne_378 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 41), '!=', col_376, set_col_377)
                
                # Applying the binary operator 'or' (line 79)
                result_or_keyword_379 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 23), 'or', result_ne_375, result_ne_378)
                
                
                # Call to cell_exclude(...): (line 80)
                # Processing the call arguments (line 80)
                # Getting the type of 'row' (line 80)
                row_382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 34), 'row', False)
                # Getting the type of 'col' (line 80)
                col_383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 39), 'col', False)
                # Getting the type of 'x' (line 80)
                x_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 44), 'x', False)
                # Processing the call keyword arguments (line 80)
                kwargs_385 = {}
                # Getting the type of 'self' (line 80)
                self_380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'self', False)
                # Obtaining the member 'cell_exclude' of a type (line 80)
                cell_exclude_381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 16), self_380, 'cell_exclude')
                # Calling cell_exclude(args, kwargs) (line 80)
                cell_exclude_call_result_386 = invoke(stypy.reporting.localization.Localization(__file__, 80, 16), cell_exclude_381, *[row_382, col_383, x_384], **kwargs_385)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'update_neighbours(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update_neighbours' in the type store
        # Getting the type of 'stypy_return_type' (line 65)
        stypy_return_type_387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_387)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update_neighbours'
        return stypy_return_type_387


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
        col_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 33), 'col', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'row' (line 83)
        row_390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 28), 'row', False)
        # Getting the type of 'self' (line 83)
        self_391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 15), 'self', False)
        # Obtaining the member 'squares' of a type (line 83)
        squares_392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 15), self_391, 'squares')
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 15), squares_392, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_394 = invoke(stypy.reporting.localization.Localization(__file__, 83, 15), getitem___393, row_390)
        
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 15), subscript_call_result_394, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_396 = invoke(stypy.reporting.localization.Localization(__file__, 83, 15), getitem___395, col_389)
        
        # Processing the call keyword arguments (line 83)
        kwargs_397 = {}
        # Getting the type of 'len' (line 83)
        len_388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 11), 'len', False)
        # Calling len(args, kwargs) (line 83)
        len_call_result_398 = invoke(stypy.reporting.localization.Localization(__file__, 83, 11), len_388, *[subscript_call_result_396], **kwargs_397)
        
        int_399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 42), 'int')
        # Applying the binary operator '==' (line 83)
        result_eq_400 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 11), '==', len_call_result_398, int_399)
        
        # Testing if the type of an if condition is none (line 83)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 83, 8), result_eq_400):
            str_416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 19), 'str', '0')
            # Assigning a type to the variable 'stypy_return_type' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'stypy_return_type', str_416)
        else:
            
            # Testing the type of an if condition (line 83)
            if_condition_401 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 8), result_eq_400)
            # Assigning a type to the variable 'if_condition_401' (line 83)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'if_condition_401', if_condition_401)
            # SSA begins for if statement (line 83)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to str(...): (line 84)
            # Processing the call arguments (line 84)
            
            # Obtaining the type of the subscript
            int_403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 46), 'int')
            
            # Obtaining the type of the subscript
            # Getting the type of 'col' (line 84)
            col_404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 41), 'col', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row' (line 84)
            row_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 36), 'row', False)
            # Getting the type of 'self' (line 84)
            self_406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 23), 'self', False)
            # Obtaining the member 'squares' of a type (line 84)
            squares_407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 23), self_406, 'squares')
            # Obtaining the member '__getitem__' of a type (line 84)
            getitem___408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 23), squares_407, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 84)
            subscript_call_result_409 = invoke(stypy.reporting.localization.Localization(__file__, 84, 23), getitem___408, row_405)
            
            # Obtaining the member '__getitem__' of a type (line 84)
            getitem___410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 23), subscript_call_result_409, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 84)
            subscript_call_result_411 = invoke(stypy.reporting.localization.Localization(__file__, 84, 23), getitem___410, col_404)
            
            # Obtaining the member '__getitem__' of a type (line 84)
            getitem___412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 23), subscript_call_result_411, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 84)
            subscript_call_result_413 = invoke(stypy.reporting.localization.Localization(__file__, 84, 23), getitem___412, int_403)
            
            # Processing the call keyword arguments (line 84)
            kwargs_414 = {}
            # Getting the type of 'str' (line 84)
            str_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 19), 'str', False)
            # Calling str(args, kwargs) (line 84)
            str_call_result_415 = invoke(stypy.reporting.localization.Localization(__file__, 84, 19), str_402, *[subscript_call_result_413], **kwargs_414)
            
            # Assigning a type to the variable 'stypy_return_type' (line 84)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'stypy_return_type', str_call_result_415)
            # SSA branch for the else part of an if statement (line 83)
            module_type_store.open_ssa_branch('else')
            str_416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 19), 'str', '0')
            # Assigning a type to the variable 'stypy_return_type' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'stypy_return_type', str_416)
            # SSA join for if statement (line 83)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'get_cell_digit_str(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_cell_digit_str' in the type store
        # Getting the type of 'stypy_return_type' (line 82)
        stypy_return_type_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_417)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_cell_digit_str'
        return stypy_return_type_417


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
        str_418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 17), 'str', '   123   456   789\n')
        # Assigning a type to the variable 'answer' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'answer', str_418)
        
        
        # Call to range(...): (line 90)
        # Processing the call arguments (line 90)
        int_420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 25), 'int')
        int_421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 28), 'int')
        # Processing the call keyword arguments (line 90)
        kwargs_422 = {}
        # Getting the type of 'range' (line 90)
        range_419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 19), 'range', False)
        # Calling range(args, kwargs) (line 90)
        range_call_result_423 = invoke(stypy.reporting.localization.Localization(__file__, 90, 19), range_419, *[int_420, int_421], **kwargs_422)
        
        # Testing if the for loop is going to be iterated (line 90)
        # Testing the type of a for loop iterable (line 90)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 90, 8), range_call_result_423)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 90, 8), range_call_result_423):
            # Getting the type of the for loop variable (line 90)
            for_loop_var_424 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 90, 8), range_call_result_423)
            # Assigning a type to the variable 'row' (line 90)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'row', for_loop_var_424)
            # SSA begins for a for statement (line 90)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a BinOp to a Name (line 91):
            
            # Assigning a BinOp to a Name (line 91):
            # Getting the type of 'answer' (line 91)
            answer_425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 21), 'answer')
            
            # Call to str(...): (line 91)
            # Processing the call arguments (line 91)
            # Getting the type of 'row' (line 91)
            row_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 34), 'row', False)
            int_428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 40), 'int')
            # Applying the binary operator '+' (line 91)
            result_add_429 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 34), '+', row_427, int_428)
            
            # Processing the call keyword arguments (line 91)
            kwargs_430 = {}
            # Getting the type of 'str' (line 91)
            str_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 30), 'str', False)
            # Calling str(args, kwargs) (line 91)
            str_call_result_431 = invoke(stypy.reporting.localization.Localization(__file__, 91, 30), str_426, *[result_add_429], **kwargs_430)
            
            # Applying the binary operator '+' (line 91)
            result_add_432 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 21), '+', answer_425, str_call_result_431)
            
            str_433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 45), 'str', ' [')
            # Applying the binary operator '+' (line 91)
            result_add_434 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 43), '+', result_add_432, str_433)
            
            
            # Call to join(...): (line 91)
            # Processing the call arguments (line 91)
            # Calculating list comprehension
            # Calculating comprehension expression
            
            # Call to range(...): (line 92)
            # Processing the call arguments (line 92)
            int_449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 86), 'int')
            int_450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 89), 'int')
            # Processing the call keyword arguments (line 92)
            kwargs_451 = {}
            # Getting the type of 'range' (line 92)
            range_448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 80), 'range', False)
            # Calling range(args, kwargs) (line 92)
            range_call_result_452 = invoke(stypy.reporting.localization.Localization(__file__, 92, 80), range_448, *[int_449, int_450], **kwargs_451)
            
            comprehension_453 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 17), range_call_result_452)
            # Assigning a type to the variable 'col' (line 92)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 17), 'col', comprehension_453)
            
            # Call to replace(...): (line 92)
            # Processing the call arguments (line 92)
            str_444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 59), 'str', '0')
            str_445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 64), 'str', '?')
            # Processing the call keyword arguments (line 92)
            kwargs_446 = {}
            
            # Call to get_cell_digit_str(...): (line 92)
            # Processing the call arguments (line 92)
            # Getting the type of 'row' (line 92)
            row_439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 41), 'row', False)
            # Getting the type of 'col' (line 92)
            col_440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 46), 'col', False)
            # Processing the call keyword arguments (line 92)
            kwargs_441 = {}
            # Getting the type of 'self' (line 92)
            self_437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 17), 'self', False)
            # Obtaining the member 'get_cell_digit_str' of a type (line 92)
            get_cell_digit_str_438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 17), self_437, 'get_cell_digit_str')
            # Calling get_cell_digit_str(args, kwargs) (line 92)
            get_cell_digit_str_call_result_442 = invoke(stypy.reporting.localization.Localization(__file__, 92, 17), get_cell_digit_str_438, *[row_439, col_440], **kwargs_441)
            
            # Obtaining the member 'replace' of a type (line 92)
            replace_443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 17), get_cell_digit_str_call_result_442, 'replace')
            # Calling replace(args, kwargs) (line 92)
            replace_call_result_447 = invoke(stypy.reporting.localization.Localization(__file__, 92, 17), replace_443, *[str_444, str_445], **kwargs_446)
            
            list_454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 17), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 17), list_454, replace_call_result_447)
            # Processing the call keyword arguments (line 91)
            kwargs_455 = {}
            str_435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 52), 'str', '')
            # Obtaining the member 'join' of a type (line 91)
            join_436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 52), str_435, 'join')
            # Calling join(args, kwargs) (line 91)
            join_call_result_456 = invoke(stypy.reporting.localization.Localization(__file__, 91, 52), join_436, *[list_454], **kwargs_455)
            
            # Applying the binary operator '+' (line 91)
            result_add_457 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 50), '+', result_add_434, join_call_result_456)
            
            str_458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 96), 'str', '] [')
            # Applying the binary operator '+' (line 92)
            result_add_459 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 94), '+', result_add_457, str_458)
            
            
            # Call to join(...): (line 92)
            # Processing the call arguments (line 92)
            # Calculating list comprehension
            # Calculating comprehension expression
            
            # Call to range(...): (line 93)
            # Processing the call arguments (line 93)
            int_474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 86), 'int')
            int_475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 89), 'int')
            # Processing the call keyword arguments (line 93)
            kwargs_476 = {}
            # Getting the type of 'range' (line 93)
            range_473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 80), 'range', False)
            # Calling range(args, kwargs) (line 93)
            range_call_result_477 = invoke(stypy.reporting.localization.Localization(__file__, 93, 80), range_473, *[int_474, int_475], **kwargs_476)
            
            comprehension_478 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 17), range_call_result_477)
            # Assigning a type to the variable 'col' (line 93)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 17), 'col', comprehension_478)
            
            # Call to replace(...): (line 93)
            # Processing the call arguments (line 93)
            str_469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 59), 'str', '0')
            str_470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 64), 'str', '?')
            # Processing the call keyword arguments (line 93)
            kwargs_471 = {}
            
            # Call to get_cell_digit_str(...): (line 93)
            # Processing the call arguments (line 93)
            # Getting the type of 'row' (line 93)
            row_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 41), 'row', False)
            # Getting the type of 'col' (line 93)
            col_465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 46), 'col', False)
            # Processing the call keyword arguments (line 93)
            kwargs_466 = {}
            # Getting the type of 'self' (line 93)
            self_462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 17), 'self', False)
            # Obtaining the member 'get_cell_digit_str' of a type (line 93)
            get_cell_digit_str_463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 17), self_462, 'get_cell_digit_str')
            # Calling get_cell_digit_str(args, kwargs) (line 93)
            get_cell_digit_str_call_result_467 = invoke(stypy.reporting.localization.Localization(__file__, 93, 17), get_cell_digit_str_463, *[row_464, col_465], **kwargs_466)
            
            # Obtaining the member 'replace' of a type (line 93)
            replace_468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 17), get_cell_digit_str_call_result_467, 'replace')
            # Calling replace(args, kwargs) (line 93)
            replace_call_result_472 = invoke(stypy.reporting.localization.Localization(__file__, 93, 17), replace_468, *[str_469, str_470], **kwargs_471)
            
            list_479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 17), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 17), list_479, replace_call_result_472)
            # Processing the call keyword arguments (line 92)
            kwargs_480 = {}
            str_460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 104), 'str', '')
            # Obtaining the member 'join' of a type (line 92)
            join_461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 104), str_460, 'join')
            # Calling join(args, kwargs) (line 92)
            join_call_result_481 = invoke(stypy.reporting.localization.Localization(__file__, 92, 104), join_461, *[list_479], **kwargs_480)
            
            # Applying the binary operator '+' (line 92)
            result_add_482 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 102), '+', result_add_459, join_call_result_481)
            
            str_483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 96), 'str', '] [')
            # Applying the binary operator '+' (line 93)
            result_add_484 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 94), '+', result_add_482, str_483)
            
            
            # Call to join(...): (line 93)
            # Processing the call arguments (line 93)
            # Calculating list comprehension
            # Calculating comprehension expression
            
            # Call to range(...): (line 94)
            # Processing the call arguments (line 94)
            int_499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 86), 'int')
            int_500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 89), 'int')
            # Processing the call keyword arguments (line 94)
            kwargs_501 = {}
            # Getting the type of 'range' (line 94)
            range_498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 80), 'range', False)
            # Calling range(args, kwargs) (line 94)
            range_call_result_502 = invoke(stypy.reporting.localization.Localization(__file__, 94, 80), range_498, *[int_499, int_500], **kwargs_501)
            
            comprehension_503 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 17), range_call_result_502)
            # Assigning a type to the variable 'col' (line 94)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 17), 'col', comprehension_503)
            
            # Call to replace(...): (line 94)
            # Processing the call arguments (line 94)
            str_494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 59), 'str', '0')
            str_495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 64), 'str', '?')
            # Processing the call keyword arguments (line 94)
            kwargs_496 = {}
            
            # Call to get_cell_digit_str(...): (line 94)
            # Processing the call arguments (line 94)
            # Getting the type of 'row' (line 94)
            row_489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 41), 'row', False)
            # Getting the type of 'col' (line 94)
            col_490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 46), 'col', False)
            # Processing the call keyword arguments (line 94)
            kwargs_491 = {}
            # Getting the type of 'self' (line 94)
            self_487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 17), 'self', False)
            # Obtaining the member 'get_cell_digit_str' of a type (line 94)
            get_cell_digit_str_488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 17), self_487, 'get_cell_digit_str')
            # Calling get_cell_digit_str(args, kwargs) (line 94)
            get_cell_digit_str_call_result_492 = invoke(stypy.reporting.localization.Localization(__file__, 94, 17), get_cell_digit_str_488, *[row_489, col_490], **kwargs_491)
            
            # Obtaining the member 'replace' of a type (line 94)
            replace_493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 17), get_cell_digit_str_call_result_492, 'replace')
            # Calling replace(args, kwargs) (line 94)
            replace_call_result_497 = invoke(stypy.reporting.localization.Localization(__file__, 94, 17), replace_493, *[str_494, str_495], **kwargs_496)
            
            list_504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 17), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 17), list_504, replace_call_result_497)
            # Processing the call keyword arguments (line 93)
            kwargs_505 = {}
            str_485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 104), 'str', '')
            # Obtaining the member 'join' of a type (line 93)
            join_486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 104), str_485, 'join')
            # Calling join(args, kwargs) (line 93)
            join_call_result_506 = invoke(stypy.reporting.localization.Localization(__file__, 93, 104), join_486, *[list_504], **kwargs_505)
            
            # Applying the binary operator '+' (line 93)
            result_add_507 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 102), '+', result_add_484, join_call_result_506)
            
            str_508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 96), 'str', ']\n')
            # Applying the binary operator '+' (line 94)
            result_add_509 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 94), '+', result_add_507, str_508)
            
            # Assigning a type to the variable 'answer' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'answer', result_add_509)
            
            # Getting the type of 'row' (line 95)
            row_510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'row')
            int_511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 21), 'int')
            # Applying the binary operator '+' (line 95)
            result_add_512 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 15), '+', row_510, int_511)
            
            
            # Obtaining an instance of the builtin type 'list' (line 95)
            list_513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 26), 'list')
            # Adding type elements to the builtin type 'list' instance (line 95)
            # Adding element type (line 95)
            int_514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 27), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 26), list_513, int_514)
            # Adding element type (line 95)
            int_515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 30), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 26), list_513, int_515)
            
            # Applying the binary operator 'in' (line 95)
            result_contains_516 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 15), 'in', result_add_512, list_513)
            
            # Testing if the type of an if condition is none (line 95)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 95, 12), result_contains_516):
                pass
            else:
                
                # Testing the type of an if condition (line 95)
                if_condition_517 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 12), result_contains_516)
                # Assigning a type to the variable 'if_condition_517' (line 95)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'if_condition_517', if_condition_517)
                # SSA begins for if statement (line 95)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a BinOp to a Name (line 96):
                
                # Assigning a BinOp to a Name (line 96):
                # Getting the type of 'answer' (line 96)
                answer_518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 25), 'answer')
                str_519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 34), 'str', '   ---   ---   ---\n')
                # Applying the binary operator '+' (line 96)
                result_add_520 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 25), '+', answer_518, str_519)
                
                # Assigning a type to the variable 'answer' (line 96)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'answer', result_add_520)
                # SSA join for if statement (line 95)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'answer' (line 97)
        answer_521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 15), 'answer')
        # Assigning a type to the variable 'stypy_return_type' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'stypy_return_type', answer_521)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 88)
        stypy_return_type_522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_522)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_522


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
        True_523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 24), 'True')
        # Getting the type of 'self' (line 100)
        self_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'self')
        # Setting the type of the member '_changed' of a type (line 100)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), self_524, '_changed', True_523)
        
        # Getting the type of 'self' (line 101)
        self_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 14), 'self')
        # Obtaining the member '_changed' of a type (line 101)
        _changed_526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 14), self_525, '_changed')
        # Testing if the while is going to be iterated (line 101)
        # Testing the type of an if condition (line 101)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 8), _changed_526)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 101, 8), _changed_526):
            # SSA begins for while statement (line 101)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a Name to a Attribute (line 102):
            
            # Assigning a Name to a Attribute (line 102):
            # Getting the type of 'False' (line 102)
            False_527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 28), 'False')
            # Getting the type of 'self' (line 102)
            self_528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'self')
            # Setting the type of the member '_changed' of a type (line 102)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), self_528, '_changed', False_527)
            
            # Call to check_for_single_occurances(...): (line 103)
            # Processing the call keyword arguments (line 103)
            kwargs_531 = {}
            # Getting the type of 'self' (line 103)
            self_529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'self', False)
            # Obtaining the member 'check_for_single_occurances' of a type (line 103)
            check_for_single_occurances_530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), self_529, 'check_for_single_occurances')
            # Calling check_for_single_occurances(args, kwargs) (line 103)
            check_for_single_occurances_call_result_532 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), check_for_single_occurances_530, *[], **kwargs_531)
            
            
            # Call to check_for_last_in_row_col_3x3(...): (line 104)
            # Processing the call keyword arguments (line 104)
            kwargs_535 = {}
            # Getting the type of 'self' (line 104)
            self_533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'self', False)
            # Obtaining the member 'check_for_last_in_row_col_3x3' of a type (line 104)
            check_for_last_in_row_col_3x3_534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 12), self_533, 'check_for_last_in_row_col_3x3')
            # Calling check_for_last_in_row_col_3x3(args, kwargs) (line 104)
            check_for_last_in_row_col_3x3_call_result_536 = invoke(stypy.reporting.localization.Localization(__file__, 104, 12), check_for_last_in_row_col_3x3_534, *[], **kwargs_535)
            
            # SSA join for while statement (line 101)
            module_type_store = module_type_store.join_ssa_context()

        
        # Assigning a type to the variable 'stypy_return_type' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of 'check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check' in the type store
        # Getting the type of 'stypy_return_type' (line 99)
        stypy_return_type_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_537)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check'
        return stypy_return_type_537


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
        list_538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        # Adding element type (line 108)
        # Getting the type of 'ROW_ITER' (line 108)
        ROW_ITER_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 27), 'ROW_ITER')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 26), list_538, ROW_ITER_539)
        # Adding element type (line 108)
        # Getting the type of 'COL_ITER' (line 108)
        COL_ITER_540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 37), 'COL_ITER')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 26), list_538, COL_ITER_540)
        # Adding element type (line 108)
        # Getting the type of 'TxT_ITER' (line 108)
        TxT_ITER_541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 47), 'TxT_ITER')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 26), list_538, TxT_ITER_541)
        
        # Testing if the for loop is going to be iterated (line 108)
        # Testing the type of a for loop iterable (line 108)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 108, 8), list_538)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 108, 8), list_538):
            # Getting the type of the for loop variable (line 108)
            for_loop_var_542 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 108, 8), list_538)
            # Assigning a type to the variable 'check_type' (line 108)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'check_type', for_loop_var_542)
            # SSA begins for a for statement (line 108)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'check_type' (line 109)
            check_type_543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 30), 'check_type')
            # Testing if the for loop is going to be iterated (line 109)
            # Testing the type of a for loop iterable (line 109)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 109, 12), check_type_543)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 109, 12), check_type_543):
                # Getting the type of the for loop variable (line 109)
                for_loop_var_544 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 109, 12), check_type_543)
                # Assigning a type to the variable 'check_list' (line 109)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'check_list', for_loop_var_544)
                # SSA begins for a for statement (line 109)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Call to range(...): (line 110)
                # Processing the call arguments (line 110)
                int_546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 31), 'int')
                int_547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 34), 'int')
                int_548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 38), 'int')
                # Applying the binary operator '+' (line 110)
                result_add_549 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 34), '+', int_547, int_548)
                
                # Processing the call keyword arguments (line 110)
                kwargs_550 = {}
                # Getting the type of 'range' (line 110)
                range_545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 25), 'range', False)
                # Calling range(args, kwargs) (line 110)
                range_call_result_551 = invoke(stypy.reporting.localization.Localization(__file__, 110, 25), range_545, *[int_546, result_add_549], **kwargs_550)
                
                # Testing if the for loop is going to be iterated (line 110)
                # Testing the type of a for loop iterable (line 110)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 110, 16), range_call_result_551)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 110, 16), range_call_result_551):
                    # Getting the type of the for loop variable (line 110)
                    for_loop_var_552 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 110, 16), range_call_result_551)
                    # Assigning a type to the variable 'x' (line 110)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'x', for_loop_var_552)
                    # SSA begins for a for statement (line 110)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Assigning a List to a Name (line 111):
                    
                    # Assigning a List to a Name (line 111):
                    
                    # Obtaining an instance of the builtin type 'list' (line 111)
                    list_553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 32), 'list')
                    # Adding type elements to the builtin type 'list' instance (line 111)
                    
                    # Assigning a type to the variable 'x_in_list' (line 111)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 20), 'x_in_list', list_553)
                    
                    # Getting the type of 'check_list' (line 112)
                    check_list_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 38), 'check_list')
                    # Testing if the for loop is going to be iterated (line 112)
                    # Testing the type of a for loop iterable (line 112)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 112, 20), check_list_554)

                    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 112, 20), check_list_554):
                        # Getting the type of the for loop variable (line 112)
                        for_loop_var_555 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 112, 20), check_list_554)
                        # Assigning a type to the variable 'row' (line 112)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'row', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 20), for_loop_var_555))
                        # Assigning a type to the variable 'col' (line 112)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'col', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 20), for_loop_var_555))
                        # SSA begins for a for statement (line 112)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                        
                        # Getting the type of 'x' (line 113)
                        x_556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 27), 'x')
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'col' (line 113)
                        col_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 50), 'col')
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'row' (line 113)
                        row_558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 45), 'row')
                        # Getting the type of 'self' (line 113)
                        self_559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 32), 'self')
                        # Obtaining the member 'squares' of a type (line 113)
                        squares_560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 32), self_559, 'squares')
                        # Obtaining the member '__getitem__' of a type (line 113)
                        getitem___561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 32), squares_560, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 113)
                        subscript_call_result_562 = invoke(stypy.reporting.localization.Localization(__file__, 113, 32), getitem___561, row_558)
                        
                        # Obtaining the member '__getitem__' of a type (line 113)
                        getitem___563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 32), subscript_call_result_562, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 113)
                        subscript_call_result_564 = invoke(stypy.reporting.localization.Localization(__file__, 113, 32), getitem___563, col_557)
                        
                        # Applying the binary operator 'in' (line 113)
                        result_contains_565 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 27), 'in', x_556, subscript_call_result_564)
                        
                        # Testing if the type of an if condition is none (line 113)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 113, 24), result_contains_565):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 113)
                            if_condition_566 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 24), result_contains_565)
                            # Assigning a type to the variable 'if_condition_566' (line 113)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 24), 'if_condition_566', if_condition_566)
                            # SSA begins for if statement (line 113)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to append(...): (line 114)
                            # Processing the call arguments (line 114)
                            
                            # Obtaining an instance of the builtin type 'tuple' (line 114)
                            tuple_569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 46), 'tuple')
                            # Adding type elements to the builtin type 'tuple' instance (line 114)
                            # Adding element type (line 114)
                            # Getting the type of 'row' (line 114)
                            row_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 46), 'row', False)
                            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 46), tuple_569, row_570)
                            # Adding element type (line 114)
                            # Getting the type of 'col' (line 114)
                            col_571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 51), 'col', False)
                            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 46), tuple_569, col_571)
                            
                            # Processing the call keyword arguments (line 114)
                            kwargs_572 = {}
                            # Getting the type of 'x_in_list' (line 114)
                            x_in_list_567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 28), 'x_in_list', False)
                            # Obtaining the member 'append' of a type (line 114)
                            append_568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 28), x_in_list_567, 'append')
                            # Calling append(args, kwargs) (line 114)
                            append_call_result_573 = invoke(stypy.reporting.localization.Localization(__file__, 114, 28), append_568, *[tuple_569], **kwargs_572)
                            
                            # SSA join for if statement (line 113)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for a for statement
                        module_type_store = module_type_store.join_ssa_context()

                    
                    
                    
                    # Call to len(...): (line 115)
                    # Processing the call arguments (line 115)
                    # Getting the type of 'x_in_list' (line 115)
                    x_in_list_575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 27), 'x_in_list', False)
                    # Processing the call keyword arguments (line 115)
                    kwargs_576 = {}
                    # Getting the type of 'len' (line 115)
                    len_574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 23), 'len', False)
                    # Calling len(args, kwargs) (line 115)
                    len_call_result_577 = invoke(stypy.reporting.localization.Localization(__file__, 115, 23), len_574, *[x_in_list_575], **kwargs_576)
                    
                    int_578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 41), 'int')
                    # Applying the binary operator '==' (line 115)
                    result_eq_579 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 23), '==', len_call_result_577, int_578)
                    
                    # Testing if the type of an if condition is none (line 115)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 115, 20), result_eq_579):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 115)
                        if_condition_580 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 115, 20), result_eq_579)
                        # Assigning a type to the variable 'if_condition_580' (line 115)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 20), 'if_condition_580', if_condition_580)
                        # SSA begins for if statement (line 115)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Subscript to a Tuple (line 116):
                        
                        # Assigning a Subscript to a Name (line 116):
                        
                        # Obtaining the type of the subscript
                        int_581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 24), 'int')
                        
                        # Obtaining the type of the subscript
                        int_582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 47), 'int')
                        # Getting the type of 'x_in_list' (line 116)
                        x_in_list_583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 37), 'x_in_list')
                        # Obtaining the member '__getitem__' of a type (line 116)
                        getitem___584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 37), x_in_list_583, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 116)
                        subscript_call_result_585 = invoke(stypy.reporting.localization.Localization(__file__, 116, 37), getitem___584, int_582)
                        
                        # Obtaining the member '__getitem__' of a type (line 116)
                        getitem___586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 24), subscript_call_result_585, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 116)
                        subscript_call_result_587 = invoke(stypy.reporting.localization.Localization(__file__, 116, 24), getitem___586, int_581)
                        
                        # Assigning a type to the variable 'tuple_var_assignment_1' (line 116)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 24), 'tuple_var_assignment_1', subscript_call_result_587)
                        
                        # Assigning a Subscript to a Name (line 116):
                        
                        # Obtaining the type of the subscript
                        int_588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 24), 'int')
                        
                        # Obtaining the type of the subscript
                        int_589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 47), 'int')
                        # Getting the type of 'x_in_list' (line 116)
                        x_in_list_590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 37), 'x_in_list')
                        # Obtaining the member '__getitem__' of a type (line 116)
                        getitem___591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 37), x_in_list_590, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 116)
                        subscript_call_result_592 = invoke(stypy.reporting.localization.Localization(__file__, 116, 37), getitem___591, int_589)
                        
                        # Obtaining the member '__getitem__' of a type (line 116)
                        getitem___593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 24), subscript_call_result_592, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 116)
                        subscript_call_result_594 = invoke(stypy.reporting.localization.Localization(__file__, 116, 24), getitem___593, int_588)
                        
                        # Assigning a type to the variable 'tuple_var_assignment_2' (line 116)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 24), 'tuple_var_assignment_2', subscript_call_result_594)
                        
                        # Assigning a Name to a Name (line 116):
                        # Getting the type of 'tuple_var_assignment_1' (line 116)
                        tuple_var_assignment_1_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 24), 'tuple_var_assignment_1')
                        # Assigning a type to the variable 'row' (line 116)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 25), 'row', tuple_var_assignment_1_595)
                        
                        # Assigning a Name to a Name (line 116):
                        # Getting the type of 'tuple_var_assignment_2' (line 116)
                        tuple_var_assignment_2_596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 24), 'tuple_var_assignment_2')
                        # Assigning a type to the variable 'col' (line 116)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 30), 'col', tuple_var_assignment_2_596)
                        
                        
                        # Call to len(...): (line 117)
                        # Processing the call arguments (line 117)
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'col' (line 117)
                        col_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 49), 'col', False)
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'row' (line 117)
                        row_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 44), 'row', False)
                        # Getting the type of 'self' (line 117)
                        self_600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 31), 'self', False)
                        # Obtaining the member 'squares' of a type (line 117)
                        squares_601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 31), self_600, 'squares')
                        # Obtaining the member '__getitem__' of a type (line 117)
                        getitem___602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 31), squares_601, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 117)
                        subscript_call_result_603 = invoke(stypy.reporting.localization.Localization(__file__, 117, 31), getitem___602, row_599)
                        
                        # Obtaining the member '__getitem__' of a type (line 117)
                        getitem___604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 31), subscript_call_result_603, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 117)
                        subscript_call_result_605 = invoke(stypy.reporting.localization.Localization(__file__, 117, 31), getitem___604, col_598)
                        
                        # Processing the call keyword arguments (line 117)
                        kwargs_606 = {}
                        # Getting the type of 'len' (line 117)
                        len_597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 27), 'len', False)
                        # Calling len(args, kwargs) (line 117)
                        len_call_result_607 = invoke(stypy.reporting.localization.Localization(__file__, 117, 27), len_597, *[subscript_call_result_605], **kwargs_606)
                        
                        int_608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 57), 'int')
                        # Applying the binary operator '>' (line 117)
                        result_gt_609 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 27), '>', len_call_result_607, int_608)
                        
                        # Testing if the type of an if condition is none (line 117)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 117, 24), result_gt_609):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 117)
                            if_condition_610 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 117, 24), result_gt_609)
                            # Assigning a type to the variable 'if_condition_610' (line 117)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 24), 'if_condition_610', if_condition_610)
                            # SSA begins for if statement (line 117)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to set_cell(...): (line 118)
                            # Processing the call arguments (line 118)
                            # Getting the type of 'row' (line 118)
                            row_613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 42), 'row', False)
                            # Getting the type of 'col' (line 118)
                            col_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 47), 'col', False)
                            # Getting the type of 'x' (line 118)
                            x_615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 52), 'x', False)
                            # Processing the call keyword arguments (line 118)
                            kwargs_616 = {}
                            # Getting the type of 'self' (line 118)
                            self_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 28), 'self', False)
                            # Obtaining the member 'set_cell' of a type (line 118)
                            set_cell_612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 28), self_611, 'set_cell')
                            # Calling set_cell(args, kwargs) (line 118)
                            set_cell_call_result_617 = invoke(stypy.reporting.localization.Localization(__file__, 118, 28), set_cell_612, *[row_613, col_614, x_615], **kwargs_616)
                            
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
        stypy_return_type_618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_618)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_for_single_occurances'
        return stypy_return_type_618


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
        list_619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 121)
        # Adding element type (line 121)
        
        # Obtaining an instance of the builtin type 'tuple' (line 121)
        tuple_620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 121)
        # Adding element type (line 121)
        str_621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 41), 'str', 'Row')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 41), tuple_620, str_621)
        # Adding element type (line 121)
        # Getting the type of 'ROW_ITER' (line 121)
        ROW_ITER_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 48), 'ROW_ITER')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 41), tuple_620, ROW_ITER_622)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 39), list_619, tuple_620)
        # Adding element type (line 121)
        
        # Obtaining an instance of the builtin type 'tuple' (line 121)
        tuple_623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 60), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 121)
        # Adding element type (line 121)
        str_624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 60), 'str', 'Col')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 60), tuple_623, str_624)
        # Adding element type (line 121)
        # Getting the type of 'COL_ITER' (line 121)
        COL_ITER_625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 67), 'COL_ITER')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 60), tuple_623, COL_ITER_625)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 39), list_619, tuple_623)
        # Adding element type (line 121)
        
        # Obtaining an instance of the builtin type 'tuple' (line 121)
        tuple_626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 79), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 121)
        # Adding element type (line 121)
        str_627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 79), 'str', '3x3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 79), tuple_626, str_627)
        # Adding element type (line 121)
        # Getting the type of 'TxT_ITER' (line 121)
        TxT_ITER_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 86), 'TxT_ITER')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 79), tuple_626, TxT_ITER_628)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 39), list_619, tuple_626)
        
        # Testing if the for loop is going to be iterated (line 121)
        # Testing the type of a for loop iterable (line 121)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 121, 8), list_619)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 121, 8), list_619):
            # Getting the type of the for loop variable (line 121)
            for_loop_var_629 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 121, 8), list_619)
            # Assigning a type to the variable 'type_name' (line 121)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'type_name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 8), for_loop_var_629))
            # Assigning a type to the variable 'check_type' (line 121)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'check_type', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 8), for_loop_var_629))
            # SSA begins for a for statement (line 121)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'check_type' (line 122)
            check_type_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 30), 'check_type')
            # Testing if the for loop is going to be iterated (line 122)
            # Testing the type of a for loop iterable (line 122)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 122, 12), check_type_630)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 122, 12), check_type_630):
                # Getting the type of the for loop variable (line 122)
                for_loop_var_631 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 122, 12), check_type_630)
                # Assigning a type to the variable 'check_list' (line 122)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'check_list', for_loop_var_631)
                # SSA begins for a for statement (line 122)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a List to a Name (line 123):
                
                # Assigning a List to a Name (line 123):
                
                # Obtaining an instance of the builtin type 'list' (line 123)
                list_632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 34), 'list')
                # Adding type elements to the builtin type 'list' instance (line 123)
                
                # Assigning a type to the variable 'unknown_entries' (line 123)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'unknown_entries', list_632)
                
                # Assigning a Call to a Name (line 124):
                
                # Assigning a Call to a Name (line 124):
                
                # Call to range(...): (line 124)
                # Processing the call arguments (line 124)
                int_634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 42), 'int')
                int_635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 45), 'int')
                int_636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 49), 'int')
                # Applying the binary operator '+' (line 124)
                result_add_637 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 45), '+', int_635, int_636)
                
                # Processing the call keyword arguments (line 124)
                kwargs_638 = {}
                # Getting the type of 'range' (line 124)
                range_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 36), 'range', False)
                # Calling range(args, kwargs) (line 124)
                range_call_result_639 = invoke(stypy.reporting.localization.Localization(__file__, 124, 36), range_633, *[int_634, result_add_637], **kwargs_638)
                
                # Assigning a type to the variable 'unassigned_values' (line 124)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'unassigned_values', range_call_result_639)
                
                # Assigning a List to a Name (line 125):
                
                # Assigning a List to a Name (line 125):
                
                # Obtaining an instance of the builtin type 'list' (line 125)
                list_640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 31), 'list')
                # Adding type elements to the builtin type 'list' instance (line 125)
                
                # Assigning a type to the variable 'known_values' (line 125)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'known_values', list_640)
                
                # Getting the type of 'check_list' (line 126)
                check_list_641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 34), 'check_list')
                # Testing if the for loop is going to be iterated (line 126)
                # Testing the type of a for loop iterable (line 126)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 126, 16), check_list_641)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 126, 16), check_list_641):
                    # Getting the type of the for loop variable (line 126)
                    for_loop_var_642 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 126, 16), check_list_641)
                    # Assigning a type to the variable 'row' (line 126)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'row', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 16), for_loop_var_642))
                    # Assigning a type to the variable 'col' (line 126)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'col', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 16), for_loop_var_642))
                    # SSA begins for a for statement (line 126)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    
                    # Call to len(...): (line 127)
                    # Processing the call arguments (line 127)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'col' (line 127)
                    col_644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 45), 'col', False)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'row' (line 127)
                    row_645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 40), 'row', False)
                    # Getting the type of 'self' (line 127)
                    self_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 27), 'self', False)
                    # Obtaining the member 'squares' of a type (line 127)
                    squares_647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 27), self_646, 'squares')
                    # Obtaining the member '__getitem__' of a type (line 127)
                    getitem___648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 27), squares_647, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                    subscript_call_result_649 = invoke(stypy.reporting.localization.Localization(__file__, 127, 27), getitem___648, row_645)
                    
                    # Obtaining the member '__getitem__' of a type (line 127)
                    getitem___650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 27), subscript_call_result_649, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                    subscript_call_result_651 = invoke(stypy.reporting.localization.Localization(__file__, 127, 27), getitem___650, col_644)
                    
                    # Processing the call keyword arguments (line 127)
                    kwargs_652 = {}
                    # Getting the type of 'len' (line 127)
                    len_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'len', False)
                    # Calling len(args, kwargs) (line 127)
                    len_call_result_653 = invoke(stypy.reporting.localization.Localization(__file__, 127, 23), len_643, *[subscript_call_result_651], **kwargs_652)
                    
                    int_654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 54), 'int')
                    # Applying the binary operator '==' (line 127)
                    result_eq_655 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 23), '==', len_call_result_653, int_654)
                    
                    # Testing if the type of an if condition is none (line 127)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 127, 20), result_eq_655):
                        
                        # Call to append(...): (line 136)
                        # Processing the call arguments (line 136)
                        
                        # Obtaining an instance of the builtin type 'tuple' (line 136)
                        tuple_715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 48), 'tuple')
                        # Adding type elements to the builtin type 'tuple' instance (line 136)
                        # Adding element type (line 136)
                        # Getting the type of 'row' (line 136)
                        row_716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 48), 'row', False)
                        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 48), tuple_715, row_716)
                        # Adding element type (line 136)
                        # Getting the type of 'col' (line 136)
                        col_717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 53), 'col', False)
                        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 48), tuple_715, col_717)
                        
                        # Processing the call keyword arguments (line 136)
                        kwargs_718 = {}
                        # Getting the type of 'unknown_entries' (line 136)
                        unknown_entries_713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 24), 'unknown_entries', False)
                        # Obtaining the member 'append' of a type (line 136)
                        append_714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 24), unknown_entries_713, 'append')
                        # Calling append(args, kwargs) (line 136)
                        append_call_result_719 = invoke(stypy.reporting.localization.Localization(__file__, 136, 24), append_714, *[tuple_715], **kwargs_718)
                        
                    else:
                        
                        # Testing the type of an if condition (line 127)
                        if_condition_656 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 127, 20), result_eq_655)
                        # Assigning a type to the variable 'if_condition_656' (line 127)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 20), 'if_condition_656', if_condition_656)
                        # SSA begins for if statement (line 127)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Evaluating assert statement condition
                        
                        
                        # Obtaining the type of the subscript
                        int_657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 54), 'int')
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'col' (line 128)
                        col_658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 49), 'col')
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'row' (line 128)
                        row_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 44), 'row')
                        # Getting the type of 'self' (line 128)
                        self_660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 31), 'self')
                        # Obtaining the member 'squares' of a type (line 128)
                        squares_661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 31), self_660, 'squares')
                        # Obtaining the member '__getitem__' of a type (line 128)
                        getitem___662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 31), squares_661, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
                        subscript_call_result_663 = invoke(stypy.reporting.localization.Localization(__file__, 128, 31), getitem___662, row_659)
                        
                        # Obtaining the member '__getitem__' of a type (line 128)
                        getitem___664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 31), subscript_call_result_663, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
                        subscript_call_result_665 = invoke(stypy.reporting.localization.Localization(__file__, 128, 31), getitem___664, col_658)
                        
                        # Obtaining the member '__getitem__' of a type (line 128)
                        getitem___666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 31), subscript_call_result_665, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
                        subscript_call_result_667 = invoke(stypy.reporting.localization.Localization(__file__, 128, 31), getitem___666, int_657)
                        
                        # Getting the type of 'known_values' (line 128)
                        known_values_668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 64), 'known_values')
                        # Applying the binary operator 'notin' (line 128)
                        result_contains_669 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 31), 'notin', subscript_call_result_667, known_values_668)
                        
                        
                        # Call to append(...): (line 130)
                        # Processing the call arguments (line 130)
                        
                        # Obtaining the type of the subscript
                        int_672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 67), 'int')
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'col' (line 130)
                        col_673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 62), 'col', False)
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'row' (line 130)
                        row_674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 57), 'row', False)
                        # Getting the type of 'self' (line 130)
                        self_675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 44), 'self', False)
                        # Obtaining the member 'squares' of a type (line 130)
                        squares_676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 44), self_675, 'squares')
                        # Obtaining the member '__getitem__' of a type (line 130)
                        getitem___677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 44), squares_676, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                        subscript_call_result_678 = invoke(stypy.reporting.localization.Localization(__file__, 130, 44), getitem___677, row_674)
                        
                        # Obtaining the member '__getitem__' of a type (line 130)
                        getitem___679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 44), subscript_call_result_678, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                        subscript_call_result_680 = invoke(stypy.reporting.localization.Localization(__file__, 130, 44), getitem___679, col_673)
                        
                        # Obtaining the member '__getitem__' of a type (line 130)
                        getitem___681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 44), subscript_call_result_680, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                        subscript_call_result_682 = invoke(stypy.reporting.localization.Localization(__file__, 130, 44), getitem___681, int_672)
                        
                        # Processing the call keyword arguments (line 130)
                        kwargs_683 = {}
                        # Getting the type of 'known_values' (line 130)
                        known_values_670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 24), 'known_values', False)
                        # Obtaining the member 'append' of a type (line 130)
                        append_671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 24), known_values_670, 'append')
                        # Calling append(args, kwargs) (line 130)
                        append_call_result_684 = invoke(stypy.reporting.localization.Localization(__file__, 130, 24), append_671, *[subscript_call_result_682], **kwargs_683)
                        
                        # Evaluating assert statement condition
                        
                        
                        # Obtaining the type of the subscript
                        int_685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 54), 'int')
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'col' (line 132)
                        col_686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 49), 'col')
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'row' (line 132)
                        row_687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 44), 'row')
                        # Getting the type of 'self' (line 132)
                        self_688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 31), 'self')
                        # Obtaining the member 'squares' of a type (line 132)
                        squares_689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 31), self_688, 'squares')
                        # Obtaining the member '__getitem__' of a type (line 132)
                        getitem___690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 31), squares_689, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 132)
                        subscript_call_result_691 = invoke(stypy.reporting.localization.Localization(__file__, 132, 31), getitem___690, row_687)
                        
                        # Obtaining the member '__getitem__' of a type (line 132)
                        getitem___692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 31), subscript_call_result_691, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 132)
                        subscript_call_result_693 = invoke(stypy.reporting.localization.Localization(__file__, 132, 31), getitem___692, col_686)
                        
                        # Obtaining the member '__getitem__' of a type (line 132)
                        getitem___694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 31), subscript_call_result_693, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 132)
                        subscript_call_result_695 = invoke(stypy.reporting.localization.Localization(__file__, 132, 31), getitem___694, int_685)
                        
                        # Getting the type of 'unassigned_values' (line 132)
                        unassigned_values_696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 60), 'unassigned_values')
                        # Applying the binary operator 'in' (line 132)
                        result_contains_697 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 31), 'in', subscript_call_result_695, unassigned_values_696)
                        
                        
                        # Call to remove(...): (line 134)
                        # Processing the call arguments (line 134)
                        
                        # Obtaining the type of the subscript
                        int_700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 72), 'int')
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'col' (line 134)
                        col_701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 67), 'col', False)
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'row' (line 134)
                        row_702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 62), 'row', False)
                        # Getting the type of 'self' (line 134)
                        self_703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 49), 'self', False)
                        # Obtaining the member 'squares' of a type (line 134)
                        squares_704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 49), self_703, 'squares')
                        # Obtaining the member '__getitem__' of a type (line 134)
                        getitem___705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 49), squares_704, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 134)
                        subscript_call_result_706 = invoke(stypy.reporting.localization.Localization(__file__, 134, 49), getitem___705, row_702)
                        
                        # Obtaining the member '__getitem__' of a type (line 134)
                        getitem___707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 49), subscript_call_result_706, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 134)
                        subscript_call_result_708 = invoke(stypy.reporting.localization.Localization(__file__, 134, 49), getitem___707, col_701)
                        
                        # Obtaining the member '__getitem__' of a type (line 134)
                        getitem___709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 49), subscript_call_result_708, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 134)
                        subscript_call_result_710 = invoke(stypy.reporting.localization.Localization(__file__, 134, 49), getitem___709, int_700)
                        
                        # Processing the call keyword arguments (line 134)
                        kwargs_711 = {}
                        # Getting the type of 'unassigned_values' (line 134)
                        unassigned_values_698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 24), 'unassigned_values', False)
                        # Obtaining the member 'remove' of a type (line 134)
                        remove_699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 24), unassigned_values_698, 'remove')
                        # Calling remove(args, kwargs) (line 134)
                        remove_call_result_712 = invoke(stypy.reporting.localization.Localization(__file__, 134, 24), remove_699, *[subscript_call_result_710], **kwargs_711)
                        
                        # SSA branch for the else part of an if statement (line 127)
                        module_type_store.open_ssa_branch('else')
                        
                        # Call to append(...): (line 136)
                        # Processing the call arguments (line 136)
                        
                        # Obtaining an instance of the builtin type 'tuple' (line 136)
                        tuple_715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 48), 'tuple')
                        # Adding type elements to the builtin type 'tuple' instance (line 136)
                        # Adding element type (line 136)
                        # Getting the type of 'row' (line 136)
                        row_716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 48), 'row', False)
                        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 48), tuple_715, row_716)
                        # Adding element type (line 136)
                        # Getting the type of 'col' (line 136)
                        col_717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 53), 'col', False)
                        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 48), tuple_715, col_717)
                        
                        # Processing the call keyword arguments (line 136)
                        kwargs_718 = {}
                        # Getting the type of 'unknown_entries' (line 136)
                        unknown_entries_713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 24), 'unknown_entries', False)
                        # Obtaining the member 'append' of a type (line 136)
                        append_714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 24), unknown_entries_713, 'append')
                        # Calling append(args, kwargs) (line 136)
                        append_call_result_719 = invoke(stypy.reporting.localization.Localization(__file__, 136, 24), append_714, *[tuple_715], **kwargs_718)
                        
                        # SSA join for if statement (line 127)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # Evaluating assert statement condition
                
                
                # Call to len(...): (line 137)
                # Processing the call arguments (line 137)
                # Getting the type of 'unknown_entries' (line 137)
                unknown_entries_721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 27), 'unknown_entries', False)
                # Processing the call keyword arguments (line 137)
                kwargs_722 = {}
                # Getting the type of 'len' (line 137)
                len_720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 23), 'len', False)
                # Calling len(args, kwargs) (line 137)
                len_call_result_723 = invoke(stypy.reporting.localization.Localization(__file__, 137, 23), len_720, *[unknown_entries_721], **kwargs_722)
                
                
                # Call to len(...): (line 137)
                # Processing the call arguments (line 137)
                # Getting the type of 'known_values' (line 137)
                known_values_725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 50), 'known_values', False)
                # Processing the call keyword arguments (line 137)
                kwargs_726 = {}
                # Getting the type of 'len' (line 137)
                len_724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 46), 'len', False)
                # Calling len(args, kwargs) (line 137)
                len_call_result_727 = invoke(stypy.reporting.localization.Localization(__file__, 137, 46), len_724, *[known_values_725], **kwargs_726)
                
                # Applying the binary operator '+' (line 137)
                result_add_728 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 23), '+', len_call_result_723, len_call_result_727)
                
                int_729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 67), 'int')
                # Applying the binary operator '==' (line 137)
                result_eq_730 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 23), '==', result_add_728, int_729)
                
                # Evaluating assert statement condition
                
                
                # Call to len(...): (line 138)
                # Processing the call arguments (line 138)
                # Getting the type of 'unknown_entries' (line 138)
                unknown_entries_732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 27), 'unknown_entries', False)
                # Processing the call keyword arguments (line 138)
                kwargs_733 = {}
                # Getting the type of 'len' (line 138)
                len_731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 23), 'len', False)
                # Calling len(args, kwargs) (line 138)
                len_call_result_734 = invoke(stypy.reporting.localization.Localization(__file__, 138, 23), len_731, *[unknown_entries_732], **kwargs_733)
                
                
                # Call to len(...): (line 138)
                # Processing the call arguments (line 138)
                # Getting the type of 'unassigned_values' (line 138)
                unassigned_values_736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 51), 'unassigned_values', False)
                # Processing the call keyword arguments (line 138)
                kwargs_737 = {}
                # Getting the type of 'len' (line 138)
                len_735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 47), 'len', False)
                # Calling len(args, kwargs) (line 138)
                len_call_result_738 = invoke(stypy.reporting.localization.Localization(__file__, 138, 47), len_735, *[unassigned_values_736], **kwargs_737)
                
                # Applying the binary operator '==' (line 138)
                result_eq_739 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 23), '==', len_call_result_734, len_call_result_738)
                
                
                
                # Call to len(...): (line 139)
                # Processing the call arguments (line 139)
                # Getting the type of 'unknown_entries' (line 139)
                unknown_entries_741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 23), 'unknown_entries', False)
                # Processing the call keyword arguments (line 139)
                kwargs_742 = {}
                # Getting the type of 'len' (line 139)
                len_740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 19), 'len', False)
                # Calling len(args, kwargs) (line 139)
                len_call_result_743 = invoke(stypy.reporting.localization.Localization(__file__, 139, 19), len_740, *[unknown_entries_741], **kwargs_742)
                
                int_744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 43), 'int')
                # Applying the binary operator '==' (line 139)
                result_eq_745 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 19), '==', len_call_result_743, int_744)
                
                # Testing if the type of an if condition is none (line 139)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 139, 16), result_eq_745):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 139)
                    if_condition_746 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 16), result_eq_745)
                    # Assigning a type to the variable 'if_condition_746' (line 139)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 'if_condition_746', if_condition_746)
                    # SSA begins for if statement (line 139)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Subscript to a Name (line 140):
                    
                    # Assigning a Subscript to a Name (line 140):
                    
                    # Obtaining the type of the subscript
                    int_747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 42), 'int')
                    # Getting the type of 'unassigned_values' (line 140)
                    unassigned_values_748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 24), 'unassigned_values')
                    # Obtaining the member '__getitem__' of a type (line 140)
                    getitem___749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 24), unassigned_values_748, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 140)
                    subscript_call_result_750 = invoke(stypy.reporting.localization.Localization(__file__, 140, 24), getitem___749, int_747)
                    
                    # Assigning a type to the variable 'x' (line 140)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 20), 'x', subscript_call_result_750)
                    
                    # Assigning a Subscript to a Tuple (line 141):
                    
                    # Assigning a Subscript to a Name (line 141):
                    
                    # Obtaining the type of the subscript
                    int_751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 20), 'int')
                    
                    # Obtaining the type of the subscript
                    int_752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 49), 'int')
                    # Getting the type of 'unknown_entries' (line 141)
                    unknown_entries_753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 33), 'unknown_entries')
                    # Obtaining the member '__getitem__' of a type (line 141)
                    getitem___754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 33), unknown_entries_753, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 141)
                    subscript_call_result_755 = invoke(stypy.reporting.localization.Localization(__file__, 141, 33), getitem___754, int_752)
                    
                    # Obtaining the member '__getitem__' of a type (line 141)
                    getitem___756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 20), subscript_call_result_755, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 141)
                    subscript_call_result_757 = invoke(stypy.reporting.localization.Localization(__file__, 141, 20), getitem___756, int_751)
                    
                    # Assigning a type to the variable 'tuple_var_assignment_3' (line 141)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 20), 'tuple_var_assignment_3', subscript_call_result_757)
                    
                    # Assigning a Subscript to a Name (line 141):
                    
                    # Obtaining the type of the subscript
                    int_758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 20), 'int')
                    
                    # Obtaining the type of the subscript
                    int_759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 49), 'int')
                    # Getting the type of 'unknown_entries' (line 141)
                    unknown_entries_760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 33), 'unknown_entries')
                    # Obtaining the member '__getitem__' of a type (line 141)
                    getitem___761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 33), unknown_entries_760, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 141)
                    subscript_call_result_762 = invoke(stypy.reporting.localization.Localization(__file__, 141, 33), getitem___761, int_759)
                    
                    # Obtaining the member '__getitem__' of a type (line 141)
                    getitem___763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 20), subscript_call_result_762, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 141)
                    subscript_call_result_764 = invoke(stypy.reporting.localization.Localization(__file__, 141, 20), getitem___763, int_758)
                    
                    # Assigning a type to the variable 'tuple_var_assignment_4' (line 141)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 20), 'tuple_var_assignment_4', subscript_call_result_764)
                    
                    # Assigning a Name to a Name (line 141):
                    # Getting the type of 'tuple_var_assignment_3' (line 141)
                    tuple_var_assignment_3_765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 20), 'tuple_var_assignment_3')
                    # Assigning a type to the variable 'row' (line 141)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 21), 'row', tuple_var_assignment_3_765)
                    
                    # Assigning a Name to a Name (line 141):
                    # Getting the type of 'tuple_var_assignment_4' (line 141)
                    tuple_var_assignment_4_766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 20), 'tuple_var_assignment_4')
                    # Assigning a type to the variable 'col' (line 141)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 26), 'col', tuple_var_assignment_4_766)
                    
                    # Call to set_cell(...): (line 142)
                    # Processing the call arguments (line 142)
                    # Getting the type of 'row' (line 142)
                    row_769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 34), 'row', False)
                    # Getting the type of 'col' (line 142)
                    col_770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 39), 'col', False)
                    # Getting the type of 'x' (line 142)
                    x_771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 44), 'x', False)
                    # Processing the call keyword arguments (line 142)
                    kwargs_772 = {}
                    # Getting the type of 'self' (line 142)
                    self_767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 20), 'self', False)
                    # Obtaining the member 'set_cell' of a type (line 142)
                    set_cell_768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 20), self_767, 'set_cell')
                    # Calling set_cell(args, kwargs) (line 142)
                    set_cell_call_result_773 = invoke(stypy.reporting.localization.Localization(__file__, 142, 20), set_cell_768, *[row_769, col_770, x_771], **kwargs_772)
                    
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
        stypy_return_type_774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_774)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_for_last_in_row_col_3x3'
        return stypy_return_type_774


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
        True_775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 19), 'True')
        # Assigning a type to the variable 'progress' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'progress', True_775)
        
        # Getting the type of 'progress' (line 147)
        progress_776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 14), 'progress')
        # Testing if the while is going to be iterated (line 147)
        # Testing the type of an if condition (line 147)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 8), progress_776)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 147, 8), progress_776):
            # SSA begins for while statement (line 147)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a Name to a Name (line 148):
            
            # Assigning a Name to a Name (line 148):
            # Getting the type of 'False' (line 148)
            False_777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 23), 'False')
            # Assigning a type to the variable 'progress' (line 148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'progress', False_777)
            
            
            # Call to range(...): (line 149)
            # Processing the call arguments (line 149)
            int_779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 29), 'int')
            int_780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 32), 'int')
            # Processing the call keyword arguments (line 149)
            kwargs_781 = {}
            # Getting the type of 'range' (line 149)
            range_778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 23), 'range', False)
            # Calling range(args, kwargs) (line 149)
            range_call_result_782 = invoke(stypy.reporting.localization.Localization(__file__, 149, 23), range_778, *[int_779, int_780], **kwargs_781)
            
            # Testing if the for loop is going to be iterated (line 149)
            # Testing the type of a for loop iterable (line 149)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 149, 12), range_call_result_782)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 149, 12), range_call_result_782):
                # Getting the type of the for loop variable (line 149)
                for_loop_var_783 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 149, 12), range_call_result_782)
                # Assigning a type to the variable 'row' (line 149)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'row', for_loop_var_783)
                # SSA begins for a for statement (line 149)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Call to range(...): (line 150)
                # Processing the call arguments (line 150)
                int_785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 33), 'int')
                int_786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 36), 'int')
                # Processing the call keyword arguments (line 150)
                kwargs_787 = {}
                # Getting the type of 'range' (line 150)
                range_784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 27), 'range', False)
                # Calling range(args, kwargs) (line 150)
                range_call_result_788 = invoke(stypy.reporting.localization.Localization(__file__, 150, 27), range_784, *[int_785, int_786], **kwargs_787)
                
                # Testing if the for loop is going to be iterated (line 150)
                # Testing the type of a for loop iterable (line 150)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 150, 16), range_call_result_788)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 150, 16), range_call_result_788):
                    # Getting the type of the for loop variable (line 150)
                    for_loop_var_789 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 150, 16), range_call_result_788)
                    # Assigning a type to the variable 'col' (line 150)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'col', for_loop_var_789)
                    # SSA begins for a for statement (line 150)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    
                    # Call to len(...): (line 151)
                    # Processing the call arguments (line 151)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'col' (line 151)
                    col_791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 45), 'col', False)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'row' (line 151)
                    row_792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 40), 'row', False)
                    # Getting the type of 'self' (line 151)
                    self_793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 27), 'self', False)
                    # Obtaining the member 'squares' of a type (line 151)
                    squares_794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 27), self_793, 'squares')
                    # Obtaining the member '__getitem__' of a type (line 151)
                    getitem___795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 27), squares_794, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 151)
                    subscript_call_result_796 = invoke(stypy.reporting.localization.Localization(__file__, 151, 27), getitem___795, row_792)
                    
                    # Obtaining the member '__getitem__' of a type (line 151)
                    getitem___797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 27), subscript_call_result_796, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 151)
                    subscript_call_result_798 = invoke(stypy.reporting.localization.Localization(__file__, 151, 27), getitem___797, col_791)
                    
                    # Processing the call keyword arguments (line 151)
                    kwargs_799 = {}
                    # Getting the type of 'len' (line 151)
                    len_790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 23), 'len', False)
                    # Calling len(args, kwargs) (line 151)
                    len_call_result_800 = invoke(stypy.reporting.localization.Localization(__file__, 151, 23), len_790, *[subscript_call_result_798], **kwargs_799)
                    
                    int_801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 53), 'int')
                    # Applying the binary operator '>' (line 151)
                    result_gt_802 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 23), '>', len_call_result_800, int_801)
                    
                    # Testing if the type of an if condition is none (line 151)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 151, 20), result_gt_802):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 151)
                        if_condition_803 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 20), result_gt_802)
                        # Assigning a type to the variable 'if_condition_803' (line 151)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 20), 'if_condition_803', if_condition_803)
                        # SSA begins for if statement (line 151)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a List to a Name (line 152):
                        
                        # Assigning a List to a Name (line 152):
                        
                        # Obtaining an instance of the builtin type 'list' (line 152)
                        list_804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 32), 'list')
                        # Adding type elements to the builtin type 'list' instance (line 152)
                        
                        # Assigning a type to the variable 'bad_x' (line 152)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 24), 'bad_x', list_804)
                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'col' (line 153)
                        col_805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 51), 'col')
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'row' (line 153)
                        row_806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 46), 'row')
                        # Getting the type of 'self' (line 153)
                        self_807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 33), 'self')
                        # Obtaining the member 'squares' of a type (line 153)
                        squares_808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 33), self_807, 'squares')
                        # Obtaining the member '__getitem__' of a type (line 153)
                        getitem___809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 33), squares_808, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 153)
                        subscript_call_result_810 = invoke(stypy.reporting.localization.Localization(__file__, 153, 33), getitem___809, row_806)
                        
                        # Obtaining the member '__getitem__' of a type (line 153)
                        getitem___811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 33), subscript_call_result_810, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 153)
                        subscript_call_result_812 = invoke(stypy.reporting.localization.Localization(__file__, 153, 33), getitem___811, col_805)
                        
                        # Testing if the for loop is going to be iterated (line 153)
                        # Testing the type of a for loop iterable (line 153)
                        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 153, 24), subscript_call_result_812)

                        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 153, 24), subscript_call_result_812):
                            # Getting the type of the for loop variable (line 153)
                            for_loop_var_813 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 153, 24), subscript_call_result_812)
                            # Assigning a type to the variable 'x' (line 153)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 24), 'x', for_loop_var_813)
                            # SSA begins for a for statement (line 153)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                            
                            # Assigning a Call to a Name (line 154):
                            
                            # Assigning a Call to a Name (line 154):
                            
                            # Call to copy(...): (line 154)
                            # Processing the call keyword arguments (line 154)
                            kwargs_816 = {}
                            # Getting the type of 'self' (line 154)
                            self_814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 42), 'self', False)
                            # Obtaining the member 'copy' of a type (line 154)
                            copy_815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 42), self_814, 'copy')
                            # Calling copy(args, kwargs) (line 154)
                            copy_call_result_817 = invoke(stypy.reporting.localization.Localization(__file__, 154, 42), copy_815, *[], **kwargs_816)
                            
                            # Assigning a type to the variable 'soduko_copy' (line 154)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 28), 'soduko_copy', copy_call_result_817)
                            
                            
                            # SSA begins for try-except statement (line 155)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
                            
                            # Call to set_cell(...): (line 156)
                            # Processing the call arguments (line 156)
                            # Getting the type of 'row' (line 156)
                            row_820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 53), 'row', False)
                            # Getting the type of 'col' (line 156)
                            col_821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 58), 'col', False)
                            # Getting the type of 'x' (line 156)
                            x_822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 63), 'x', False)
                            # Processing the call keyword arguments (line 156)
                            kwargs_823 = {}
                            # Getting the type of 'soduko_copy' (line 156)
                            soduko_copy_818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 32), 'soduko_copy', False)
                            # Obtaining the member 'set_cell' of a type (line 156)
                            set_cell_819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 32), soduko_copy_818, 'set_cell')
                            # Calling set_cell(args, kwargs) (line 156)
                            set_cell_call_result_824 = invoke(stypy.reporting.localization.Localization(__file__, 156, 32), set_cell_819, *[row_820, col_821, x_822], **kwargs_823)
                            
                            
                            # Call to check(...): (line 157)
                            # Processing the call keyword arguments (line 157)
                            kwargs_827 = {}
                            # Getting the type of 'soduko_copy' (line 157)
                            soduko_copy_825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 32), 'soduko_copy', False)
                            # Obtaining the member 'check' of a type (line 157)
                            check_826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 32), soduko_copy_825, 'check')
                            # Calling check(args, kwargs) (line 157)
                            check_call_result_828 = invoke(stypy.reporting.localization.Localization(__file__, 157, 32), check_826, *[], **kwargs_827)
                            
                            # SSA branch for the except part of a try statement (line 155)
                            # SSA branch for the except 'AssertionError' branch of a try statement (line 155)
                            # Storing handler type
                            module_type_store.open_ssa_branch('except')
                            # Getting the type of 'AssertionError' (line 158)
                            AssertionError_829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 35), 'AssertionError')
                            # Assigning a type to the variable 'e' (line 158)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 28), 'e', AssertionError_829)
                            
                            # Call to append(...): (line 159)
                            # Processing the call arguments (line 159)
                            # Getting the type of 'x' (line 159)
                            x_832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 45), 'x', False)
                            # Processing the call keyword arguments (line 159)
                            kwargs_833 = {}
                            # Getting the type of 'bad_x' (line 159)
                            bad_x_830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 32), 'bad_x', False)
                            # Obtaining the member 'append' of a type (line 159)
                            append_831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 32), bad_x_830, 'append')
                            # Calling append(args, kwargs) (line 159)
                            append_call_result_834 = invoke(stypy.reporting.localization.Localization(__file__, 159, 32), append_831, *[x_832], **kwargs_833)
                            
                            # SSA join for try-except statement (line 155)
                            module_type_store = module_type_store.join_ssa_context()
                            
                            # Deleting a member
                            module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 160, 28), module_type_store, 'soduko_copy')
                            # SSA join for a for statement
                            module_type_store = module_type_store.join_ssa_context()

                        
                        
                        
                        # Call to len(...): (line 161)
                        # Processing the call arguments (line 161)
                        # Getting the type of 'bad_x' (line 161)
                        bad_x_836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 31), 'bad_x', False)
                        # Processing the call keyword arguments (line 161)
                        kwargs_837 = {}
                        # Getting the type of 'len' (line 161)
                        len_835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 27), 'len', False)
                        # Calling len(args, kwargs) (line 161)
                        len_call_result_838 = invoke(stypy.reporting.localization.Localization(__file__, 161, 27), len_835, *[bad_x_836], **kwargs_837)
                        
                        int_839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 41), 'int')
                        # Applying the binary operator '==' (line 161)
                        result_eq_840 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 27), '==', len_call_result_838, int_839)
                        
                        # Testing if the type of an if condition is none (line 161)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 161, 24), result_eq_840):
                            
                            
                            # Call to len(...): (line 163)
                            # Processing the call arguments (line 163)
                            # Getting the type of 'bad_x' (line 163)
                            bad_x_843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 33), 'bad_x', False)
                            # Processing the call keyword arguments (line 163)
                            kwargs_844 = {}
                            # Getting the type of 'len' (line 163)
                            len_842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 29), 'len', False)
                            # Calling len(args, kwargs) (line 163)
                            len_call_result_845 = invoke(stypy.reporting.localization.Localization(__file__, 163, 29), len_842, *[bad_x_843], **kwargs_844)
                            
                            
                            # Call to len(...): (line 163)
                            # Processing the call arguments (line 163)
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'col' (line 163)
                            col_847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 64), 'col', False)
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'row' (line 163)
                            row_848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 59), 'row', False)
                            # Getting the type of 'self' (line 163)
                            self_849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 46), 'self', False)
                            # Obtaining the member 'squares' of a type (line 163)
                            squares_850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 46), self_849, 'squares')
                            # Obtaining the member '__getitem__' of a type (line 163)
                            getitem___851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 46), squares_850, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 163)
                            subscript_call_result_852 = invoke(stypy.reporting.localization.Localization(__file__, 163, 46), getitem___851, row_848)
                            
                            # Obtaining the member '__getitem__' of a type (line 163)
                            getitem___853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 46), subscript_call_result_852, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 163)
                            subscript_call_result_854 = invoke(stypy.reporting.localization.Localization(__file__, 163, 46), getitem___853, col_847)
                            
                            # Processing the call keyword arguments (line 163)
                            kwargs_855 = {}
                            # Getting the type of 'len' (line 163)
                            len_846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 42), 'len', False)
                            # Calling len(args, kwargs) (line 163)
                            len_call_result_856 = invoke(stypy.reporting.localization.Localization(__file__, 163, 42), len_846, *[subscript_call_result_854], **kwargs_855)
                            
                            # Applying the binary operator '<' (line 163)
                            result_lt_857 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 29), '<', len_call_result_845, len_call_result_856)
                            
                            # Testing if the type of an if condition is none (line 163)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 163, 29), result_lt_857):
                                # Evaluating assert statement condition
                                # Getting the type of 'False' (line 169)
                                False_873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 35), 'False')
                            else:
                                
                                # Testing the type of an if condition (line 163)
                                if_condition_858 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 29), result_lt_857)
                                # Assigning a type to the variable 'if_condition_858' (line 163)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 29), 'if_condition_858', if_condition_858)
                                # SSA begins for if statement (line 163)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Getting the type of 'bad_x' (line 164)
                                bad_x_859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 37), 'bad_x')
                                # Testing if the for loop is going to be iterated (line 164)
                                # Testing the type of a for loop iterable (line 164)
                                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 164, 28), bad_x_859)

                                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 164, 28), bad_x_859):
                                    # Getting the type of the for loop variable (line 164)
                                    for_loop_var_860 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 164, 28), bad_x_859)
                                    # Assigning a type to the variable 'x' (line 164)
                                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'x', for_loop_var_860)
                                    # SSA begins for a for statement (line 164)
                                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                                    
                                    # Call to cell_exclude(...): (line 165)
                                    # Processing the call arguments (line 165)
                                    # Getting the type of 'row' (line 165)
                                    row_863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 50), 'row', False)
                                    # Getting the type of 'col' (line 165)
                                    col_864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 55), 'col', False)
                                    # Getting the type of 'x' (line 165)
                                    x_865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 60), 'x', False)
                                    # Processing the call keyword arguments (line 165)
                                    kwargs_866 = {}
                                    # Getting the type of 'self' (line 165)
                                    self_861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 32), 'self', False)
                                    # Obtaining the member 'cell_exclude' of a type (line 165)
                                    cell_exclude_862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 32), self_861, 'cell_exclude')
                                    # Calling cell_exclude(args, kwargs) (line 165)
                                    cell_exclude_call_result_867 = invoke(stypy.reporting.localization.Localization(__file__, 165, 32), cell_exclude_862, *[row_863, col_864, x_865], **kwargs_866)
                                    
                                    
                                    # Call to check(...): (line 166)
                                    # Processing the call keyword arguments (line 166)
                                    kwargs_870 = {}
                                    # Getting the type of 'self' (line 166)
                                    self_868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 32), 'self', False)
                                    # Obtaining the member 'check' of a type (line 166)
                                    check_869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 32), self_868, 'check')
                                    # Calling check(args, kwargs) (line 166)
                                    check_call_result_871 = invoke(stypy.reporting.localization.Localization(__file__, 166, 32), check_869, *[], **kwargs_870)
                                    
                                    # SSA join for a for statement
                                    module_type_store = module_type_store.join_ssa_context()

                                
                                
                                # Assigning a Name to a Name (line 167):
                                
                                # Assigning a Name to a Name (line 167):
                                # Getting the type of 'True' (line 167)
                                True_872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 39), 'True')
                                # Assigning a type to the variable 'progress' (line 167)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 28), 'progress', True_872)
                                # SSA branch for the else part of an if statement (line 163)
                                module_type_store.open_ssa_branch('else')
                                # Evaluating assert statement condition
                                # Getting the type of 'False' (line 169)
                                False_873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 35), 'False')
                                # SSA join for if statement (line 163)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 161)
                            if_condition_841 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 24), result_eq_840)
                            # Assigning a type to the variable 'if_condition_841' (line 161)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 24), 'if_condition_841', if_condition_841)
                            # SSA begins for if statement (line 161)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            pass
                            # SSA branch for the else part of an if statement (line 161)
                            module_type_store.open_ssa_branch('else')
                            
                            
                            # Call to len(...): (line 163)
                            # Processing the call arguments (line 163)
                            # Getting the type of 'bad_x' (line 163)
                            bad_x_843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 33), 'bad_x', False)
                            # Processing the call keyword arguments (line 163)
                            kwargs_844 = {}
                            # Getting the type of 'len' (line 163)
                            len_842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 29), 'len', False)
                            # Calling len(args, kwargs) (line 163)
                            len_call_result_845 = invoke(stypy.reporting.localization.Localization(__file__, 163, 29), len_842, *[bad_x_843], **kwargs_844)
                            
                            
                            # Call to len(...): (line 163)
                            # Processing the call arguments (line 163)
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'col' (line 163)
                            col_847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 64), 'col', False)
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'row' (line 163)
                            row_848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 59), 'row', False)
                            # Getting the type of 'self' (line 163)
                            self_849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 46), 'self', False)
                            # Obtaining the member 'squares' of a type (line 163)
                            squares_850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 46), self_849, 'squares')
                            # Obtaining the member '__getitem__' of a type (line 163)
                            getitem___851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 46), squares_850, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 163)
                            subscript_call_result_852 = invoke(stypy.reporting.localization.Localization(__file__, 163, 46), getitem___851, row_848)
                            
                            # Obtaining the member '__getitem__' of a type (line 163)
                            getitem___853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 46), subscript_call_result_852, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 163)
                            subscript_call_result_854 = invoke(stypy.reporting.localization.Localization(__file__, 163, 46), getitem___853, col_847)
                            
                            # Processing the call keyword arguments (line 163)
                            kwargs_855 = {}
                            # Getting the type of 'len' (line 163)
                            len_846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 42), 'len', False)
                            # Calling len(args, kwargs) (line 163)
                            len_call_result_856 = invoke(stypy.reporting.localization.Localization(__file__, 163, 42), len_846, *[subscript_call_result_854], **kwargs_855)
                            
                            # Applying the binary operator '<' (line 163)
                            result_lt_857 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 29), '<', len_call_result_845, len_call_result_856)
                            
                            # Testing if the type of an if condition is none (line 163)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 163, 29), result_lt_857):
                                # Evaluating assert statement condition
                                # Getting the type of 'False' (line 169)
                                False_873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 35), 'False')
                            else:
                                
                                # Testing the type of an if condition (line 163)
                                if_condition_858 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 29), result_lt_857)
                                # Assigning a type to the variable 'if_condition_858' (line 163)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 29), 'if_condition_858', if_condition_858)
                                # SSA begins for if statement (line 163)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Getting the type of 'bad_x' (line 164)
                                bad_x_859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 37), 'bad_x')
                                # Testing if the for loop is going to be iterated (line 164)
                                # Testing the type of a for loop iterable (line 164)
                                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 164, 28), bad_x_859)

                                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 164, 28), bad_x_859):
                                    # Getting the type of the for loop variable (line 164)
                                    for_loop_var_860 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 164, 28), bad_x_859)
                                    # Assigning a type to the variable 'x' (line 164)
                                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'x', for_loop_var_860)
                                    # SSA begins for a for statement (line 164)
                                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                                    
                                    # Call to cell_exclude(...): (line 165)
                                    # Processing the call arguments (line 165)
                                    # Getting the type of 'row' (line 165)
                                    row_863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 50), 'row', False)
                                    # Getting the type of 'col' (line 165)
                                    col_864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 55), 'col', False)
                                    # Getting the type of 'x' (line 165)
                                    x_865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 60), 'x', False)
                                    # Processing the call keyword arguments (line 165)
                                    kwargs_866 = {}
                                    # Getting the type of 'self' (line 165)
                                    self_861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 32), 'self', False)
                                    # Obtaining the member 'cell_exclude' of a type (line 165)
                                    cell_exclude_862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 32), self_861, 'cell_exclude')
                                    # Calling cell_exclude(args, kwargs) (line 165)
                                    cell_exclude_call_result_867 = invoke(stypy.reporting.localization.Localization(__file__, 165, 32), cell_exclude_862, *[row_863, col_864, x_865], **kwargs_866)
                                    
                                    
                                    # Call to check(...): (line 166)
                                    # Processing the call keyword arguments (line 166)
                                    kwargs_870 = {}
                                    # Getting the type of 'self' (line 166)
                                    self_868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 32), 'self', False)
                                    # Obtaining the member 'check' of a type (line 166)
                                    check_869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 32), self_868, 'check')
                                    # Calling check(args, kwargs) (line 166)
                                    check_call_result_871 = invoke(stypy.reporting.localization.Localization(__file__, 166, 32), check_869, *[], **kwargs_870)
                                    
                                    # SSA join for a for statement
                                    module_type_store = module_type_store.join_ssa_context()

                                
                                
                                # Assigning a Name to a Name (line 167):
                                
                                # Assigning a Name to a Name (line 167):
                                # Getting the type of 'True' (line 167)
                                True_872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 39), 'True')
                                # Assigning a type to the variable 'progress' (line 167)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 28), 'progress', True_872)
                                # SSA branch for the else part of an if statement (line 163)
                                module_type_store.open_ssa_branch('else')
                                # Evaluating assert statement condition
                                # Getting the type of 'False' (line 169)
                                False_873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 35), 'False')
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
        stypy_return_type_874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_874)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'one_level_supposition'
        return stypy_return_type_874


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
    int_876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 19), 'int')
    # Processing the call keyword arguments (line 173)
    kwargs_877 = {}
    # Getting the type of 'range' (line 173)
    range_875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 13), 'range', False)
    # Calling range(args, kwargs) (line 173)
    range_call_result_878 = invoke(stypy.reporting.localization.Localization(__file__, 173, 13), range_875, *[int_876], **kwargs_877)
    
    # Testing if the for loop is going to be iterated (line 173)
    # Testing the type of a for loop iterable (line 173)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 173, 4), range_call_result_878)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 173, 4), range_call_result_878):
        # Getting the type of the for loop variable (line 173)
        for_loop_var_879 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 173, 4), range_call_result_878)
        # Assigning a type to the variable 'x' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'x', for_loop_var_879)
        # SSA begins for a for statement (line 173)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 174):
        
        # Assigning a Call to a Name (line 174):
        
        # Call to soduko(...): (line 174)
        # Processing the call arguments (line 174)
        
        # Obtaining an instance of the builtin type 'list' (line 174)
        list_881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 174)
        # Adding element type (line 174)
        str_882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 20), 'str', '800000600')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 19), list_881, str_882)
        # Adding element type (line 174)
        str_883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 20), 'str', '040500100')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 19), list_881, str_883)
        # Adding element type (line 174)
        str_884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 20), 'str', '070090000')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 19), list_881, str_884)
        # Adding element type (line 174)
        str_885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 20), 'str', '030020007')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 19), list_881, str_885)
        # Adding element type (line 174)
        str_886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 20), 'str', '600008004')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 19), list_881, str_886)
        # Adding element type (line 174)
        str_887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 20), 'str', '500000090')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 19), list_881, str_887)
        # Adding element type (line 174)
        str_888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 20), 'str', '000030020')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 19), list_881, str_888)
        # Adding element type (line 174)
        str_889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 20), 'str', '001006050')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 19), list_881, str_889)
        # Adding element type (line 174)
        str_890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 20), 'str', '004000003')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 19), list_881, str_890)
        
        # Processing the call keyword arguments (line 174)
        kwargs_891 = {}
        # Getting the type of 'soduko' (line 174)
        soduko_880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'soduko', False)
        # Calling soduko(args, kwargs) (line 174)
        soduko_call_result_892 = invoke(stypy.reporting.localization.Localization(__file__, 174, 12), soduko_880, *[list_881], **kwargs_891)
        
        # Assigning a type to the variable 't' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 't', soduko_call_result_892)
        
        # Call to check(...): (line 184)
        # Processing the call keyword arguments (line 184)
        kwargs_895 = {}
        # Getting the type of 't' (line 184)
        t_893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 't', False)
        # Obtaining the member 'check' of a type (line 184)
        check_894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), t_893, 'check')
        # Calling check(args, kwargs) (line 184)
        check_call_result_896 = invoke(stypy.reporting.localization.Localization(__file__, 184, 8), check_894, *[], **kwargs_895)
        
        
        # Call to one_level_supposition(...): (line 185)
        # Processing the call keyword arguments (line 185)
        kwargs_899 = {}
        # Getting the type of 't' (line 185)
        t_897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 't', False)
        # Obtaining the member 'one_level_supposition' of a type (line 185)
        one_level_supposition_898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 8), t_897, 'one_level_supposition')
        # Calling one_level_supposition(args, kwargs) (line 185)
        one_level_supposition_call_result_900 = invoke(stypy.reporting.localization.Localization(__file__, 185, 8), one_level_supposition_898, *[], **kwargs_899)
        
        
        # Call to check(...): (line 186)
        # Processing the call keyword arguments (line 186)
        kwargs_903 = {}
        # Getting the type of 't' (line 186)
        t_901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 't', False)
        # Obtaining the member 'check' of a type (line 186)
        check_902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), t_901, 'check')
        # Calling check(args, kwargs) (line 186)
        check_call_result_904 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), check_902, *[], **kwargs_903)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 172)
    stypy_return_type_905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_905)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_905

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
    kwargs_907 = {}
    # Getting the type of 'main' (line 191)
    main_906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'main', False)
    # Calling main(args, kwargs) (line 191)
    main_call_result_908 = invoke(stypy.reporting.localization.Localization(__file__, 191, 4), main_906, *[], **kwargs_907)
    
    # Getting the type of 'True' (line 192)
    True_909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'stypy_return_type', True_909)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 190)
    stypy_return_type_910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_910)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_910

# Assigning a type to the variable 'run' (line 190)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'run', run)

# Call to run(...): (line 195)
# Processing the call keyword arguments (line 195)
kwargs_912 = {}
# Getting the type of 'run' (line 195)
run_911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 0), 'run', False)
# Calling run(args, kwargs) (line 195)
run_call_result_913 = invoke(stypy.reporting.localization.Localization(__file__, 195, 0), run_911, *[], **kwargs_912)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

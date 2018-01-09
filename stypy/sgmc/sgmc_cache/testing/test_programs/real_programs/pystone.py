
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # #! /usr/bin/env python
2: 
3: 
4: LOOPS = 50000
5: 
6: from time import clock
7: 
8: __version__ = "1.1"
9: 
10: [Ident1, Ident2, Ident3, Ident4, Ident5] = range(1, 6)
11: 
12: 
13: class Record:
14:     def __init__(self, PtrComp=None, Discr=0, EnumComp=0,
15:                  IntComp=0, StringComp=0):
16:         self.PtrComp = PtrComp
17:         self.Discr = Discr
18:         self.EnumComp = EnumComp
19:         self.IntComp = IntComp
20:         self.StringComp = StringComp
21: 
22:     def copy(self):
23:         return Record(self.PtrComp, self.Discr, self.EnumComp,
24:                       self.IntComp, self.StringComp)
25: 
26: 
27: TRUE = 1
28: FALSE = 0
29: 
30: 
31: def main(loops=LOOPS):
32:     benchtime, stones = pystones(loops)
33:     print "Pystone(%s) time for %d passes = %g" % \
34:           (__version__, loops, benchtime)
35:     print "This machine benchmarks at %g pystones/second" % stones
36: 
37: 
38: def pystones(loops=LOOPS):
39:     return Proc0(loops)
40: 
41: 
42: IntGlob = 0
43: BoolGlob = FALSE
44: Char1Glob = '\0'
45: Char2Glob = '\0'
46: Array1Glob = [0] * 51
47: Array2Glob = map(lambda x: x[:], [Array1Glob] * 51)
48: PtrGlb = None
49: PtrGlbNext = None
50: 
51: 
52: def Proc0(loops=LOOPS):
53:     global IntGlob
54:     global BoolGlob
55:     global Char1Glob
56:     global Char2Glob
57:     global Array1Glob
58:     global Array2Glob
59:     global PtrGlb
60:     global PtrGlbNext
61: 
62:     starttime = clock()
63:     for i in range(loops):
64:         pass
65:     nulltime = clock() - starttime
66: 
67:     PtrGlbNext = Record()
68:     PtrGlb = Record()
69:     PtrGlb.PtrComp = PtrGlbNext
70:     PtrGlb.Discr = Ident1
71:     PtrGlb.EnumComp = Ident3
72:     PtrGlb.IntComp = 40
73:     PtrGlb.StringComp = "DHRYSTONE PROGRAM, SOME STRING"
74:     String1Loc = "DHRYSTONE PROGRAM, 1'ST STRING"
75:     Array2Glob[8][7] = 10
76: 
77:     starttime = clock()
78: 
79:     for i in range(loops):
80:         Proc5()
81:         Proc4()
82:         IntLoc1 = 2
83:         IntLoc2 = 3
84:         String2Loc = "DHRYSTONE PROGRAM, 2'ND STRING"
85:         EnumLoc = Ident2
86:         BoolGlob = not Func2(String1Loc, String2Loc)
87:         while IntLoc1 < IntLoc2:
88:             IntLoc3 = 5 * IntLoc1 - IntLoc2
89:             IntLoc3 = Proc7(IntLoc1, IntLoc2)
90:             IntLoc1 = IntLoc1 + 1
91:         Proc8(Array1Glob, Array2Glob, IntLoc1, IntLoc3)
92:         PtrGlb = Proc1(PtrGlb)
93:         CharIndex = 'A'
94:         while CharIndex <= Char2Glob:
95:             if EnumLoc == Func1(CharIndex, 'C'):
96:                 EnumLoc = Proc6(Ident1)
97:             CharIndex = chr(ord(CharIndex) + 1)
98:         IntLoc3 = IntLoc2 * IntLoc1
99:         IntLoc2 = IntLoc3 / IntLoc1
100:         IntLoc2 = 7 * (IntLoc3 - IntLoc2) - IntLoc1
101:         IntLoc1 = Proc2(IntLoc1)
102: 
103:     benchtime = clock() - starttime - nulltime
104:     if benchtime == 0.0:
105:         loopsPerBenchtime = 0.0
106:         # my_local = 1
107: 
108:     else:
109:         # my_local = "aa"
110:         loopsPerBenchtime = (loops / benchtime)
111: 
112:     # my_local = my_local / 4
113:     return benchtime, loopsPerBenchtime
114: 
115: 
116: def Proc1(PtrParIn):
117:     PtrParIn.PtrComp = NextRecord = PtrGlb.copy()
118:     PtrParIn.IntComp = 5
119:     NextRecord.IntComp = PtrParIn.IntComp
120:     NextRecord.PtrComp = PtrParIn.PtrComp
121:     NextRecord.PtrComp = Proc3(NextRecord.PtrComp)
122:     if NextRecord.Discr == Ident1:
123:         NextRecord.IntComp = 6
124:         NextRecord.EnumComp = Proc6(PtrParIn.EnumComp)
125:         NextRecord.PtrComp = PtrGlb.PtrComp
126:         NextRecord.IntComp = Proc7(NextRecord.IntComp, 10)
127:     else:
128:         PtrParIn = NextRecord.copy()
129:     NextRecord.PtrComp = None
130:     return PtrParIn
131: 
132: 
133: def Proc2(IntParIO):
134:     IntLoc = IntParIO + 10
135:     while 1:
136:         if Char1Glob == 'A':
137:             IntLoc = IntLoc - 1
138:             IntParIO = IntLoc - IntGlob
139:             EnumLoc = Ident1
140:         if EnumLoc == Ident1:
141:             break
142:     return IntParIO
143: 
144: 
145: def Proc3(PtrParOut):
146:     global IntGlob
147: 
148:     if PtrGlb is not None:
149:         PtrParOut = PtrGlb.PtrComp
150:     else:
151:         IntGlob = 100
152:     PtrGlb.IntComp = Proc7(10, IntGlob)
153:     return PtrParOut
154: 
155: 
156: def Proc4():
157:     global Char2Glob
158: 
159:     BoolLoc = Char1Glob == 'A'
160:     BoolLoc = BoolLoc or BoolGlob
161:     Char2Glob = 'B'
162: 
163: 
164: def Proc5():
165:     global Char1Glob
166:     global BoolGlob
167: 
168:     Char1Glob = 'A'
169:     BoolGlob = FALSE
170: 
171: 
172: def Proc6(EnumParIn):
173:     EnumParOut = EnumParIn
174:     if not Func3(EnumParIn):
175:         EnumParOut = Ident4
176:     if EnumParIn == Ident1:
177:         EnumParOut = Ident1
178:     elif EnumParIn == Ident2:
179:         if IntGlob > 100:
180:             EnumParOut = Ident1
181:         else:
182:             EnumParOut = Ident4
183:     elif EnumParIn == Ident3:
184:         EnumParOut = Ident2
185:     elif EnumParIn == Ident4:
186:         pass
187:     elif EnumParIn == Ident5:
188:         EnumParOut = Ident3
189:     return EnumParOut
190: 
191: 
192: def Proc7(IntParI1, IntParI2):
193:     IntLoc = IntParI1 + 2
194:     IntParOut = IntParI2 + IntLoc
195:     return IntParOut
196: 
197: 
198: def Proc8(Array1Par, Array2Par, IntParI1, IntParI2):
199:     global IntGlob
200: 
201:     IntLoc = IntParI1 + 5
202:     Array1Par[IntLoc] = IntParI2
203:     Array1Par[IntLoc + 1] = Array1Par[IntLoc]
204:     Array1Par[IntLoc + 30] = IntLoc
205:     for IntIndex in range(IntLoc, IntLoc + 2):
206:         Array2Par[IntLoc][IntIndex] = IntLoc
207:     Array2Par[IntLoc][IntLoc - 1] = Array2Par[IntLoc][IntLoc - 1] + 1
208:     Array2Par[IntLoc + 20][IntLoc] = Array1Par[IntLoc]
209:     IntGlob = 5
210: 
211: 
212: def Func1(CharPar1, CharPar2):
213:     CharLoc1 = CharPar1
214:     CharLoc2 = CharLoc1
215:     if CharLoc2 != CharPar2:
216:         return Ident1
217:     else:
218:         return Ident2
219: 
220: 
221: def Func2(StrParI1, StrParI2):
222:     IntLoc = 1
223:     while IntLoc <= 1:
224:         if Func1(StrParI1[IntLoc], StrParI2[IntLoc + 1]) == Ident1:
225:             CharLoc = 'A'
226:             IntLoc = IntLoc + 1
227:     if CharLoc >= 'W' and CharLoc <= 'Z':
228:         IntLoc = 7
229:     if CharLoc == 'X':
230:         return TRUE
231:     else:
232:         if StrParI1 > StrParI2:
233:             IntLoc = IntLoc + 7
234:             return TRUE
235:         else:
236:             return FALSE
237: 
238: 
239: def Func3(EnumParIn):
240:     EnumLoc = EnumParIn
241:     if EnumLoc == Ident3: return TRUE
242:     return FALSE
243: 
244: 
245: if __name__ == '__main__':
246:     #    import sys
247:     #
248:     #     def error(msg):
249:     #         print >> sys.stderr, msg,
250:     #         print >> sys.stderr, "usage: %s [number_of_loops]" % sys.argv[0]
251:     #         sys.exit(100)
252:     #
253:     #     nargs = len(sys.argv) - 1
254:     #     if nargs > 1:
255:     #         error("%d arguments are too many;" % nargs)
256:     #     elif nargs == 1:
257:     #         try:
258:     #             loops = int(sys.argv[1])
259:     #         except ValueError:
260:     #             error("Invalid argument %r;" % sys.argv[1])
261:     #     else:
262:     #         loops = LOOPS
263:     #     main(loops)
264: 
265:     main(LOOPS)
266: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 4):

# Assigning a Num to a Name (line 4):
int_5647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 8), 'int')
# Assigning a type to the variable 'LOOPS' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'LOOPS', int_5647)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from time import clock' statement (line 6)
from time import clock

import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'time', None, module_type_store, ['clock'], [clock])


# Assigning a Str to a Name (line 8):

# Assigning a Str to a Name (line 8):
str_5648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 14), 'str', '1.1')
# Assigning a type to the variable '__version__' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '__version__', str_5648)

# Assigning a Call to a List (line 10):

# Assigning a Subscript to a Name (line 10):

# Obtaining the type of the subscript
int_5649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 0), 'int')

# Call to range(...): (line 10)
# Processing the call arguments (line 10)
int_5651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 49), 'int')
int_5652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 52), 'int')
# Processing the call keyword arguments (line 10)
kwargs_5653 = {}
# Getting the type of 'range' (line 10)
range_5650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 43), 'range', False)
# Calling range(args, kwargs) (line 10)
range_call_result_5654 = invoke(stypy.reporting.localization.Localization(__file__, 10, 43), range_5650, *[int_5651, int_5652], **kwargs_5653)

# Obtaining the member '__getitem__' of a type (line 10)
getitem___5655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 0), range_call_result_5654, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 10)
subscript_call_result_5656 = invoke(stypy.reporting.localization.Localization(__file__, 10, 0), getitem___5655, int_5649)

# Assigning a type to the variable 'list_var_assignment_5640' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'list_var_assignment_5640', subscript_call_result_5656)

# Assigning a Subscript to a Name (line 10):

# Obtaining the type of the subscript
int_5657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 0), 'int')

# Call to range(...): (line 10)
# Processing the call arguments (line 10)
int_5659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 49), 'int')
int_5660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 52), 'int')
# Processing the call keyword arguments (line 10)
kwargs_5661 = {}
# Getting the type of 'range' (line 10)
range_5658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 43), 'range', False)
# Calling range(args, kwargs) (line 10)
range_call_result_5662 = invoke(stypy.reporting.localization.Localization(__file__, 10, 43), range_5658, *[int_5659, int_5660], **kwargs_5661)

# Obtaining the member '__getitem__' of a type (line 10)
getitem___5663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 0), range_call_result_5662, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 10)
subscript_call_result_5664 = invoke(stypy.reporting.localization.Localization(__file__, 10, 0), getitem___5663, int_5657)

# Assigning a type to the variable 'list_var_assignment_5641' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'list_var_assignment_5641', subscript_call_result_5664)

# Assigning a Subscript to a Name (line 10):

# Obtaining the type of the subscript
int_5665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 0), 'int')

# Call to range(...): (line 10)
# Processing the call arguments (line 10)
int_5667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 49), 'int')
int_5668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 52), 'int')
# Processing the call keyword arguments (line 10)
kwargs_5669 = {}
# Getting the type of 'range' (line 10)
range_5666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 43), 'range', False)
# Calling range(args, kwargs) (line 10)
range_call_result_5670 = invoke(stypy.reporting.localization.Localization(__file__, 10, 43), range_5666, *[int_5667, int_5668], **kwargs_5669)

# Obtaining the member '__getitem__' of a type (line 10)
getitem___5671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 0), range_call_result_5670, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 10)
subscript_call_result_5672 = invoke(stypy.reporting.localization.Localization(__file__, 10, 0), getitem___5671, int_5665)

# Assigning a type to the variable 'list_var_assignment_5642' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'list_var_assignment_5642', subscript_call_result_5672)

# Assigning a Subscript to a Name (line 10):

# Obtaining the type of the subscript
int_5673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 0), 'int')

# Call to range(...): (line 10)
# Processing the call arguments (line 10)
int_5675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 49), 'int')
int_5676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 52), 'int')
# Processing the call keyword arguments (line 10)
kwargs_5677 = {}
# Getting the type of 'range' (line 10)
range_5674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 43), 'range', False)
# Calling range(args, kwargs) (line 10)
range_call_result_5678 = invoke(stypy.reporting.localization.Localization(__file__, 10, 43), range_5674, *[int_5675, int_5676], **kwargs_5677)

# Obtaining the member '__getitem__' of a type (line 10)
getitem___5679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 0), range_call_result_5678, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 10)
subscript_call_result_5680 = invoke(stypy.reporting.localization.Localization(__file__, 10, 0), getitem___5679, int_5673)

# Assigning a type to the variable 'list_var_assignment_5643' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'list_var_assignment_5643', subscript_call_result_5680)

# Assigning a Subscript to a Name (line 10):

# Obtaining the type of the subscript
int_5681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 0), 'int')

# Call to range(...): (line 10)
# Processing the call arguments (line 10)
int_5683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 49), 'int')
int_5684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 52), 'int')
# Processing the call keyword arguments (line 10)
kwargs_5685 = {}
# Getting the type of 'range' (line 10)
range_5682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 43), 'range', False)
# Calling range(args, kwargs) (line 10)
range_call_result_5686 = invoke(stypy.reporting.localization.Localization(__file__, 10, 43), range_5682, *[int_5683, int_5684], **kwargs_5685)

# Obtaining the member '__getitem__' of a type (line 10)
getitem___5687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 0), range_call_result_5686, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 10)
subscript_call_result_5688 = invoke(stypy.reporting.localization.Localization(__file__, 10, 0), getitem___5687, int_5681)

# Assigning a type to the variable 'list_var_assignment_5644' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'list_var_assignment_5644', subscript_call_result_5688)

# Assigning a Name to a Name (line 10):
# Getting the type of 'list_var_assignment_5640' (line 10)
list_var_assignment_5640_5689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'list_var_assignment_5640')
# Assigning a type to the variable 'Ident1' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 1), 'Ident1', list_var_assignment_5640_5689)

# Assigning a Name to a Name (line 10):
# Getting the type of 'list_var_assignment_5641' (line 10)
list_var_assignment_5641_5690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'list_var_assignment_5641')
# Assigning a type to the variable 'Ident2' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'Ident2', list_var_assignment_5641_5690)

# Assigning a Name to a Name (line 10):
# Getting the type of 'list_var_assignment_5642' (line 10)
list_var_assignment_5642_5691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'list_var_assignment_5642')
# Assigning a type to the variable 'Ident3' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 17), 'Ident3', list_var_assignment_5642_5691)

# Assigning a Name to a Name (line 10):
# Getting the type of 'list_var_assignment_5643' (line 10)
list_var_assignment_5643_5692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'list_var_assignment_5643')
# Assigning a type to the variable 'Ident4' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 25), 'Ident4', list_var_assignment_5643_5692)

# Assigning a Name to a Name (line 10):
# Getting the type of 'list_var_assignment_5644' (line 10)
list_var_assignment_5644_5693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'list_var_assignment_5644')
# Assigning a type to the variable 'Ident5' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 33), 'Ident5', list_var_assignment_5644_5693)
# Declaration of the 'Record' class

class Record:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 14)
        None_5694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 31), 'None')
        int_5695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 43), 'int')
        int_5696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 55), 'int')
        int_5697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 25), 'int')
        int_5698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 39), 'int')
        defaults = [None_5694, int_5695, int_5696, int_5697, int_5698]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 14, 4, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Record.__init__', ['PtrComp', 'Discr', 'EnumComp', 'IntComp', 'StringComp'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['PtrComp', 'Discr', 'EnumComp', 'IntComp', 'StringComp'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 16):
        
        # Assigning a Name to a Attribute (line 16):
        # Getting the type of 'PtrComp' (line 16)
        PtrComp_5699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 23), 'PtrComp')
        # Getting the type of 'self' (line 16)
        self_5700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'self')
        # Setting the type of the member 'PtrComp' of a type (line 16)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), self_5700, 'PtrComp', PtrComp_5699)
        
        # Assigning a Name to a Attribute (line 17):
        
        # Assigning a Name to a Attribute (line 17):
        # Getting the type of 'Discr' (line 17)
        Discr_5701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 21), 'Discr')
        # Getting the type of 'self' (line 17)
        self_5702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'self')
        # Setting the type of the member 'Discr' of a type (line 17)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), self_5702, 'Discr', Discr_5701)
        
        # Assigning a Name to a Attribute (line 18):
        
        # Assigning a Name to a Attribute (line 18):
        # Getting the type of 'EnumComp' (line 18)
        EnumComp_5703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 24), 'EnumComp')
        # Getting the type of 'self' (line 18)
        self_5704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'self')
        # Setting the type of the member 'EnumComp' of a type (line 18)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), self_5704, 'EnumComp', EnumComp_5703)
        
        # Assigning a Name to a Attribute (line 19):
        
        # Assigning a Name to a Attribute (line 19):
        # Getting the type of 'IntComp' (line 19)
        IntComp_5705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 23), 'IntComp')
        # Getting the type of 'self' (line 19)
        self_5706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'self')
        # Setting the type of the member 'IntComp' of a type (line 19)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 8), self_5706, 'IntComp', IntComp_5705)
        
        # Assigning a Name to a Attribute (line 20):
        
        # Assigning a Name to a Attribute (line 20):
        # Getting the type of 'StringComp' (line 20)
        StringComp_5707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 26), 'StringComp')
        # Getting the type of 'self' (line 20)
        self_5708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'self')
        # Setting the type of the member 'StringComp' of a type (line 20)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), self_5708, 'StringComp', StringComp_5707)
        
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
        module_type_store = module_type_store.open_function_context('copy', 22, 4, False)
        # Assigning a type to the variable 'self' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Record.copy.__dict__.__setitem__('stypy_localization', localization)
        Record.copy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Record.copy.__dict__.__setitem__('stypy_type_store', module_type_store)
        Record.copy.__dict__.__setitem__('stypy_function_name', 'Record.copy')
        Record.copy.__dict__.__setitem__('stypy_param_names_list', [])
        Record.copy.__dict__.__setitem__('stypy_varargs_param_name', None)
        Record.copy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Record.copy.__dict__.__setitem__('stypy_call_defaults', defaults)
        Record.copy.__dict__.__setitem__('stypy_call_varargs', varargs)
        Record.copy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Record.copy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Record.copy', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to Record(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'self' (line 23)
        self_5710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 22), 'self', False)
        # Obtaining the member 'PtrComp' of a type (line 23)
        PtrComp_5711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 22), self_5710, 'PtrComp')
        # Getting the type of 'self' (line 23)
        self_5712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 36), 'self', False)
        # Obtaining the member 'Discr' of a type (line 23)
        Discr_5713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 36), self_5712, 'Discr')
        # Getting the type of 'self' (line 23)
        self_5714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 48), 'self', False)
        # Obtaining the member 'EnumComp' of a type (line 23)
        EnumComp_5715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 48), self_5714, 'EnumComp')
        # Getting the type of 'self' (line 24)
        self_5716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 22), 'self', False)
        # Obtaining the member 'IntComp' of a type (line 24)
        IntComp_5717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 22), self_5716, 'IntComp')
        # Getting the type of 'self' (line 24)
        self_5718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 36), 'self', False)
        # Obtaining the member 'StringComp' of a type (line 24)
        StringComp_5719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 36), self_5718, 'StringComp')
        # Processing the call keyword arguments (line 23)
        kwargs_5720 = {}
        # Getting the type of 'Record' (line 23)
        Record_5709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 15), 'Record', False)
        # Calling Record(args, kwargs) (line 23)
        Record_call_result_5721 = invoke(stypy.reporting.localization.Localization(__file__, 23, 15), Record_5709, *[PtrComp_5711, Discr_5713, EnumComp_5715, IntComp_5717, StringComp_5719], **kwargs_5720)
        
        # Assigning a type to the variable 'stypy_return_type' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'stypy_return_type', Record_call_result_5721)
        
        # ################# End of 'copy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'copy' in the type store
        # Getting the type of 'stypy_return_type' (line 22)
        stypy_return_type_5722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5722)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'copy'
        return stypy_return_type_5722


# Assigning a type to the variable 'Record' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'Record', Record)

# Assigning a Num to a Name (line 27):

# Assigning a Num to a Name (line 27):
int_5723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 7), 'int')
# Assigning a type to the variable 'TRUE' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'TRUE', int_5723)

# Assigning a Num to a Name (line 28):

# Assigning a Num to a Name (line 28):
int_5724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 8), 'int')
# Assigning a type to the variable 'FALSE' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'FALSE', int_5724)

@norecursion
def main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'LOOPS' (line 31)
    LOOPS_5725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'LOOPS')
    defaults = [LOOPS_5725]
    # Create a new context for function 'main'
    module_type_store = module_type_store.open_function_context('main', 31, 0, False)
    
    # Passed parameters checking function
    main.stypy_localization = localization
    main.stypy_type_of_self = None
    main.stypy_type_store = module_type_store
    main.stypy_function_name = 'main'
    main.stypy_param_names_list = ['loops']
    main.stypy_varargs_param_name = None
    main.stypy_kwargs_param_name = None
    main.stypy_call_defaults = defaults
    main.stypy_call_varargs = varargs
    main.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'main', ['loops'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'main', localization, ['loops'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'main(...)' code ##################

    
    # Assigning a Call to a Tuple (line 32):
    
    # Assigning a Subscript to a Name (line 32):
    
    # Obtaining the type of the subscript
    int_5726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 4), 'int')
    
    # Call to pystones(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'loops' (line 32)
    loops_5728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 33), 'loops', False)
    # Processing the call keyword arguments (line 32)
    kwargs_5729 = {}
    # Getting the type of 'pystones' (line 32)
    pystones_5727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 24), 'pystones', False)
    # Calling pystones(args, kwargs) (line 32)
    pystones_call_result_5730 = invoke(stypy.reporting.localization.Localization(__file__, 32, 24), pystones_5727, *[loops_5728], **kwargs_5729)
    
    # Obtaining the member '__getitem__' of a type (line 32)
    getitem___5731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 4), pystones_call_result_5730, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 32)
    subscript_call_result_5732 = invoke(stypy.reporting.localization.Localization(__file__, 32, 4), getitem___5731, int_5726)
    
    # Assigning a type to the variable 'tuple_var_assignment_5645' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'tuple_var_assignment_5645', subscript_call_result_5732)
    
    # Assigning a Subscript to a Name (line 32):
    
    # Obtaining the type of the subscript
    int_5733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 4), 'int')
    
    # Call to pystones(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'loops' (line 32)
    loops_5735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 33), 'loops', False)
    # Processing the call keyword arguments (line 32)
    kwargs_5736 = {}
    # Getting the type of 'pystones' (line 32)
    pystones_5734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 24), 'pystones', False)
    # Calling pystones(args, kwargs) (line 32)
    pystones_call_result_5737 = invoke(stypy.reporting.localization.Localization(__file__, 32, 24), pystones_5734, *[loops_5735], **kwargs_5736)
    
    # Obtaining the member '__getitem__' of a type (line 32)
    getitem___5738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 4), pystones_call_result_5737, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 32)
    subscript_call_result_5739 = invoke(stypy.reporting.localization.Localization(__file__, 32, 4), getitem___5738, int_5733)
    
    # Assigning a type to the variable 'tuple_var_assignment_5646' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'tuple_var_assignment_5646', subscript_call_result_5739)
    
    # Assigning a Name to a Name (line 32):
    # Getting the type of 'tuple_var_assignment_5645' (line 32)
    tuple_var_assignment_5645_5740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'tuple_var_assignment_5645')
    # Assigning a type to the variable 'benchtime' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'benchtime', tuple_var_assignment_5645_5740)
    
    # Assigning a Name to a Name (line 32):
    # Getting the type of 'tuple_var_assignment_5646' (line 32)
    tuple_var_assignment_5646_5741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'tuple_var_assignment_5646')
    # Assigning a type to the variable 'stones' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'stones', tuple_var_assignment_5646_5741)
    str_5742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 10), 'str', 'Pystone(%s) time for %d passes = %g')
    
    # Obtaining an instance of the builtin type 'tuple' (line 34)
    tuple_5743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 34)
    # Adding element type (line 34)
    # Getting the type of '__version__' (line 34)
    version___5744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 11), '__version__')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 11), tuple_5743, version___5744)
    # Adding element type (line 34)
    # Getting the type of 'loops' (line 34)
    loops_5745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 24), 'loops')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 11), tuple_5743, loops_5745)
    # Adding element type (line 34)
    # Getting the type of 'benchtime' (line 34)
    benchtime_5746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 31), 'benchtime')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 11), tuple_5743, benchtime_5746)
    
    # Applying the binary operator '%' (line 33)
    result_mod_5747 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 10), '%', str_5742, tuple_5743)
    
    str_5748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 10), 'str', 'This machine benchmarks at %g pystones/second')
    # Getting the type of 'stones' (line 35)
    stones_5749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 60), 'stones')
    # Applying the binary operator '%' (line 35)
    result_mod_5750 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 10), '%', str_5748, stones_5749)
    
    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 31)
    stypy_return_type_5751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5751)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_5751

# Assigning a type to the variable 'main' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'main', main)

@norecursion
def pystones(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'LOOPS' (line 38)
    LOOPS_5752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 19), 'LOOPS')
    defaults = [LOOPS_5752]
    # Create a new context for function 'pystones'
    module_type_store = module_type_store.open_function_context('pystones', 38, 0, False)
    
    # Passed parameters checking function
    pystones.stypy_localization = localization
    pystones.stypy_type_of_self = None
    pystones.stypy_type_store = module_type_store
    pystones.stypy_function_name = 'pystones'
    pystones.stypy_param_names_list = ['loops']
    pystones.stypy_varargs_param_name = None
    pystones.stypy_kwargs_param_name = None
    pystones.stypy_call_defaults = defaults
    pystones.stypy_call_varargs = varargs
    pystones.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pystones', ['loops'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pystones', localization, ['loops'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pystones(...)' code ##################

    
    # Call to Proc0(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'loops' (line 39)
    loops_5754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 17), 'loops', False)
    # Processing the call keyword arguments (line 39)
    kwargs_5755 = {}
    # Getting the type of 'Proc0' (line 39)
    Proc0_5753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 11), 'Proc0', False)
    # Calling Proc0(args, kwargs) (line 39)
    Proc0_call_result_5756 = invoke(stypy.reporting.localization.Localization(__file__, 39, 11), Proc0_5753, *[loops_5754], **kwargs_5755)
    
    # Assigning a type to the variable 'stypy_return_type' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type', Proc0_call_result_5756)
    
    # ################# End of 'pystones(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pystones' in the type store
    # Getting the type of 'stypy_return_type' (line 38)
    stypy_return_type_5757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5757)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pystones'
    return stypy_return_type_5757

# Assigning a type to the variable 'pystones' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'pystones', pystones)

# Assigning a Num to a Name (line 42):

# Assigning a Num to a Name (line 42):
int_5758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 10), 'int')
# Assigning a type to the variable 'IntGlob' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'IntGlob', int_5758)

# Assigning a Name to a Name (line 43):

# Assigning a Name to a Name (line 43):
# Getting the type of 'FALSE' (line 43)
FALSE_5759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 11), 'FALSE')
# Assigning a type to the variable 'BoolGlob' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'BoolGlob', FALSE_5759)

# Assigning a Str to a Name (line 44):

# Assigning a Str to a Name (line 44):
str_5760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 12), 'str', '\x00')
# Assigning a type to the variable 'Char1Glob' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'Char1Glob', str_5760)

# Assigning a Str to a Name (line 45):

# Assigning a Str to a Name (line 45):
str_5761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 12), 'str', '\x00')
# Assigning a type to the variable 'Char2Glob' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'Char2Glob', str_5761)

# Assigning a BinOp to a Name (line 46):

# Assigning a BinOp to a Name (line 46):

# Obtaining an instance of the builtin type 'list' (line 46)
list_5762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 46)
# Adding element type (line 46)
int_5763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 13), list_5762, int_5763)

int_5764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 19), 'int')
# Applying the binary operator '*' (line 46)
result_mul_5765 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 13), '*', list_5762, int_5764)

# Assigning a type to the variable 'Array1Glob' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'Array1Glob', result_mul_5765)

# Assigning a Call to a Name (line 47):

# Assigning a Call to a Name (line 47):

# Call to map(...): (line 47)
# Processing the call arguments (line 47)

@norecursion
def _stypy_temp_lambda_8(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_8'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_8', 47, 17, True)
    # Passed parameters checking function
    _stypy_temp_lambda_8.stypy_localization = localization
    _stypy_temp_lambda_8.stypy_type_of_self = None
    _stypy_temp_lambda_8.stypy_type_store = module_type_store
    _stypy_temp_lambda_8.stypy_function_name = '_stypy_temp_lambda_8'
    _stypy_temp_lambda_8.stypy_param_names_list = ['x']
    _stypy_temp_lambda_8.stypy_varargs_param_name = None
    _stypy_temp_lambda_8.stypy_kwargs_param_name = None
    _stypy_temp_lambda_8.stypy_call_defaults = defaults
    _stypy_temp_lambda_8.stypy_call_varargs = varargs
    _stypy_temp_lambda_8.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_8', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_8', ['x'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Obtaining the type of the subscript
    slice_5767 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 47, 27), None, None, None)
    # Getting the type of 'x' (line 47)
    x_5768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 27), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 47)
    getitem___5769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 27), x_5768, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 47)
    subscript_call_result_5770 = invoke(stypy.reporting.localization.Localization(__file__, 47, 27), getitem___5769, slice_5767)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 17), 'stypy_return_type', subscript_call_result_5770)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_8' in the type store
    # Getting the type of 'stypy_return_type' (line 47)
    stypy_return_type_5771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 17), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5771)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_8'
    return stypy_return_type_5771

# Assigning a type to the variable '_stypy_temp_lambda_8' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 17), '_stypy_temp_lambda_8', _stypy_temp_lambda_8)
# Getting the type of '_stypy_temp_lambda_8' (line 47)
_stypy_temp_lambda_8_5772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 17), '_stypy_temp_lambda_8')

# Obtaining an instance of the builtin type 'list' (line 47)
list_5773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 33), 'list')
# Adding type elements to the builtin type 'list' instance (line 47)
# Adding element type (line 47)
# Getting the type of 'Array1Glob' (line 47)
Array1Glob_5774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 34), 'Array1Glob', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 33), list_5773, Array1Glob_5774)

int_5775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 48), 'int')
# Applying the binary operator '*' (line 47)
result_mul_5776 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 33), '*', list_5773, int_5775)

# Processing the call keyword arguments (line 47)
kwargs_5777 = {}
# Getting the type of 'map' (line 47)
map_5766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 13), 'map', False)
# Calling map(args, kwargs) (line 47)
map_call_result_5778 = invoke(stypy.reporting.localization.Localization(__file__, 47, 13), map_5766, *[_stypy_temp_lambda_8_5772, result_mul_5776], **kwargs_5777)

# Assigning a type to the variable 'Array2Glob' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'Array2Glob', map_call_result_5778)

# Assigning a Name to a Name (line 48):

# Assigning a Name to a Name (line 48):
# Getting the type of 'None' (line 48)
None_5779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 9), 'None')
# Assigning a type to the variable 'PtrGlb' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'PtrGlb', None_5779)

# Assigning a Name to a Name (line 49):

# Assigning a Name to a Name (line 49):
# Getting the type of 'None' (line 49)
None_5780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 13), 'None')
# Assigning a type to the variable 'PtrGlbNext' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'PtrGlbNext', None_5780)

@norecursion
def Proc0(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'LOOPS' (line 52)
    LOOPS_5781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'LOOPS')
    defaults = [LOOPS_5781]
    # Create a new context for function 'Proc0'
    module_type_store = module_type_store.open_function_context('Proc0', 52, 0, False)
    
    # Passed parameters checking function
    Proc0.stypy_localization = localization
    Proc0.stypy_type_of_self = None
    Proc0.stypy_type_store = module_type_store
    Proc0.stypy_function_name = 'Proc0'
    Proc0.stypy_param_names_list = ['loops']
    Proc0.stypy_varargs_param_name = None
    Proc0.stypy_kwargs_param_name = None
    Proc0.stypy_call_defaults = defaults
    Proc0.stypy_call_varargs = varargs
    Proc0.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'Proc0', ['loops'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'Proc0', localization, ['loops'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'Proc0(...)' code ##################

    # Marking variables as global (line 53)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 53, 4), 'IntGlob')
    # Marking variables as global (line 54)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 54, 4), 'BoolGlob')
    # Marking variables as global (line 55)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 55, 4), 'Char1Glob')
    # Marking variables as global (line 56)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 56, 4), 'Char2Glob')
    # Marking variables as global (line 57)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 57, 4), 'Array1Glob')
    # Marking variables as global (line 58)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 58, 4), 'Array2Glob')
    # Marking variables as global (line 59)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 59, 4), 'PtrGlb')
    # Marking variables as global (line 60)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 60, 4), 'PtrGlbNext')
    
    # Assigning a Call to a Name (line 62):
    
    # Assigning a Call to a Name (line 62):
    
    # Call to clock(...): (line 62)
    # Processing the call keyword arguments (line 62)
    kwargs_5783 = {}
    # Getting the type of 'clock' (line 62)
    clock_5782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'clock', False)
    # Calling clock(args, kwargs) (line 62)
    clock_call_result_5784 = invoke(stypy.reporting.localization.Localization(__file__, 62, 16), clock_5782, *[], **kwargs_5783)
    
    # Assigning a type to the variable 'starttime' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'starttime', clock_call_result_5784)
    
    
    # Call to range(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'loops' (line 63)
    loops_5786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 19), 'loops', False)
    # Processing the call keyword arguments (line 63)
    kwargs_5787 = {}
    # Getting the type of 'range' (line 63)
    range_5785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 13), 'range', False)
    # Calling range(args, kwargs) (line 63)
    range_call_result_5788 = invoke(stypy.reporting.localization.Localization(__file__, 63, 13), range_5785, *[loops_5786], **kwargs_5787)
    
    # Testing the type of a for loop iterable (line 63)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 63, 4), range_call_result_5788)
    # Getting the type of the for loop variable (line 63)
    for_loop_var_5789 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 63, 4), range_call_result_5788)
    # Assigning a type to the variable 'i' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'i', for_loop_var_5789)
    # SSA begins for a for statement (line 63)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    pass
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 65):
    
    # Assigning a BinOp to a Name (line 65):
    
    # Call to clock(...): (line 65)
    # Processing the call keyword arguments (line 65)
    kwargs_5791 = {}
    # Getting the type of 'clock' (line 65)
    clock_5790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 15), 'clock', False)
    # Calling clock(args, kwargs) (line 65)
    clock_call_result_5792 = invoke(stypy.reporting.localization.Localization(__file__, 65, 15), clock_5790, *[], **kwargs_5791)
    
    # Getting the type of 'starttime' (line 65)
    starttime_5793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 25), 'starttime')
    # Applying the binary operator '-' (line 65)
    result_sub_5794 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 15), '-', clock_call_result_5792, starttime_5793)
    
    # Assigning a type to the variable 'nulltime' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'nulltime', result_sub_5794)
    
    # Assigning a Call to a Name (line 67):
    
    # Assigning a Call to a Name (line 67):
    
    # Call to Record(...): (line 67)
    # Processing the call keyword arguments (line 67)
    kwargs_5796 = {}
    # Getting the type of 'Record' (line 67)
    Record_5795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 17), 'Record', False)
    # Calling Record(args, kwargs) (line 67)
    Record_call_result_5797 = invoke(stypy.reporting.localization.Localization(__file__, 67, 17), Record_5795, *[], **kwargs_5796)
    
    # Assigning a type to the variable 'PtrGlbNext' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'PtrGlbNext', Record_call_result_5797)
    
    # Assigning a Call to a Name (line 68):
    
    # Assigning a Call to a Name (line 68):
    
    # Call to Record(...): (line 68)
    # Processing the call keyword arguments (line 68)
    kwargs_5799 = {}
    # Getting the type of 'Record' (line 68)
    Record_5798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 13), 'Record', False)
    # Calling Record(args, kwargs) (line 68)
    Record_call_result_5800 = invoke(stypy.reporting.localization.Localization(__file__, 68, 13), Record_5798, *[], **kwargs_5799)
    
    # Assigning a type to the variable 'PtrGlb' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'PtrGlb', Record_call_result_5800)
    
    # Assigning a Name to a Attribute (line 69):
    
    # Assigning a Name to a Attribute (line 69):
    # Getting the type of 'PtrGlbNext' (line 69)
    PtrGlbNext_5801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 21), 'PtrGlbNext')
    # Getting the type of 'PtrGlb' (line 69)
    PtrGlb_5802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'PtrGlb')
    # Setting the type of the member 'PtrComp' of a type (line 69)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 4), PtrGlb_5802, 'PtrComp', PtrGlbNext_5801)
    
    # Assigning a Name to a Attribute (line 70):
    
    # Assigning a Name to a Attribute (line 70):
    # Getting the type of 'Ident1' (line 70)
    Ident1_5803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 19), 'Ident1')
    # Getting the type of 'PtrGlb' (line 70)
    PtrGlb_5804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'PtrGlb')
    # Setting the type of the member 'Discr' of a type (line 70)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 4), PtrGlb_5804, 'Discr', Ident1_5803)
    
    # Assigning a Name to a Attribute (line 71):
    
    # Assigning a Name to a Attribute (line 71):
    # Getting the type of 'Ident3' (line 71)
    Ident3_5805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 22), 'Ident3')
    # Getting the type of 'PtrGlb' (line 71)
    PtrGlb_5806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'PtrGlb')
    # Setting the type of the member 'EnumComp' of a type (line 71)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 4), PtrGlb_5806, 'EnumComp', Ident3_5805)
    
    # Assigning a Num to a Attribute (line 72):
    
    # Assigning a Num to a Attribute (line 72):
    int_5807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 21), 'int')
    # Getting the type of 'PtrGlb' (line 72)
    PtrGlb_5808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'PtrGlb')
    # Setting the type of the member 'IntComp' of a type (line 72)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 4), PtrGlb_5808, 'IntComp', int_5807)
    
    # Assigning a Str to a Attribute (line 73):
    
    # Assigning a Str to a Attribute (line 73):
    str_5809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 24), 'str', 'DHRYSTONE PROGRAM, SOME STRING')
    # Getting the type of 'PtrGlb' (line 73)
    PtrGlb_5810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'PtrGlb')
    # Setting the type of the member 'StringComp' of a type (line 73)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 4), PtrGlb_5810, 'StringComp', str_5809)
    
    # Assigning a Str to a Name (line 74):
    
    # Assigning a Str to a Name (line 74):
    str_5811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 17), 'str', "DHRYSTONE PROGRAM, 1'ST STRING")
    # Assigning a type to the variable 'String1Loc' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'String1Loc', str_5811)
    
    # Assigning a Num to a Subscript (line 75):
    
    # Assigning a Num to a Subscript (line 75):
    int_5812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 23), 'int')
    
    # Obtaining the type of the subscript
    int_5813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 15), 'int')
    # Getting the type of 'Array2Glob' (line 75)
    Array2Glob_5814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'Array2Glob')
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___5815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 4), Array2Glob_5814, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_5816 = invoke(stypy.reporting.localization.Localization(__file__, 75, 4), getitem___5815, int_5813)
    
    int_5817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 18), 'int')
    # Storing an element on a container (line 75)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 4), subscript_call_result_5816, (int_5817, int_5812))
    
    # Assigning a Call to a Name (line 77):
    
    # Assigning a Call to a Name (line 77):
    
    # Call to clock(...): (line 77)
    # Processing the call keyword arguments (line 77)
    kwargs_5819 = {}
    # Getting the type of 'clock' (line 77)
    clock_5818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 16), 'clock', False)
    # Calling clock(args, kwargs) (line 77)
    clock_call_result_5820 = invoke(stypy.reporting.localization.Localization(__file__, 77, 16), clock_5818, *[], **kwargs_5819)
    
    # Assigning a type to the variable 'starttime' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'starttime', clock_call_result_5820)
    
    
    # Call to range(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'loops' (line 79)
    loops_5822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 19), 'loops', False)
    # Processing the call keyword arguments (line 79)
    kwargs_5823 = {}
    # Getting the type of 'range' (line 79)
    range_5821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 13), 'range', False)
    # Calling range(args, kwargs) (line 79)
    range_call_result_5824 = invoke(stypy.reporting.localization.Localization(__file__, 79, 13), range_5821, *[loops_5822], **kwargs_5823)
    
    # Testing the type of a for loop iterable (line 79)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 79, 4), range_call_result_5824)
    # Getting the type of the for loop variable (line 79)
    for_loop_var_5825 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 79, 4), range_call_result_5824)
    # Assigning a type to the variable 'i' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'i', for_loop_var_5825)
    # SSA begins for a for statement (line 79)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to Proc5(...): (line 80)
    # Processing the call keyword arguments (line 80)
    kwargs_5827 = {}
    # Getting the type of 'Proc5' (line 80)
    Proc5_5826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'Proc5', False)
    # Calling Proc5(args, kwargs) (line 80)
    Proc5_call_result_5828 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), Proc5_5826, *[], **kwargs_5827)
    
    
    # Call to Proc4(...): (line 81)
    # Processing the call keyword arguments (line 81)
    kwargs_5830 = {}
    # Getting the type of 'Proc4' (line 81)
    Proc4_5829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'Proc4', False)
    # Calling Proc4(args, kwargs) (line 81)
    Proc4_call_result_5831 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), Proc4_5829, *[], **kwargs_5830)
    
    
    # Assigning a Num to a Name (line 82):
    
    # Assigning a Num to a Name (line 82):
    int_5832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 18), 'int')
    # Assigning a type to the variable 'IntLoc1' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'IntLoc1', int_5832)
    
    # Assigning a Num to a Name (line 83):
    
    # Assigning a Num to a Name (line 83):
    int_5833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 18), 'int')
    # Assigning a type to the variable 'IntLoc2' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'IntLoc2', int_5833)
    
    # Assigning a Str to a Name (line 84):
    
    # Assigning a Str to a Name (line 84):
    str_5834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 21), 'str', "DHRYSTONE PROGRAM, 2'ND STRING")
    # Assigning a type to the variable 'String2Loc' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'String2Loc', str_5834)
    
    # Assigning a Name to a Name (line 85):
    
    # Assigning a Name to a Name (line 85):
    # Getting the type of 'Ident2' (line 85)
    Ident2_5835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 18), 'Ident2')
    # Assigning a type to the variable 'EnumLoc' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'EnumLoc', Ident2_5835)
    
    # Assigning a UnaryOp to a Name (line 86):
    
    # Assigning a UnaryOp to a Name (line 86):
    
    
    # Call to Func2(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'String1Loc' (line 86)
    String1Loc_5837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 29), 'String1Loc', False)
    # Getting the type of 'String2Loc' (line 86)
    String2Loc_5838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 41), 'String2Loc', False)
    # Processing the call keyword arguments (line 86)
    kwargs_5839 = {}
    # Getting the type of 'Func2' (line 86)
    Func2_5836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 23), 'Func2', False)
    # Calling Func2(args, kwargs) (line 86)
    Func2_call_result_5840 = invoke(stypy.reporting.localization.Localization(__file__, 86, 23), Func2_5836, *[String1Loc_5837, String2Loc_5838], **kwargs_5839)
    
    # Applying the 'not' unary operator (line 86)
    result_not__5841 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 19), 'not', Func2_call_result_5840)
    
    # Assigning a type to the variable 'BoolGlob' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'BoolGlob', result_not__5841)
    
    
    # Getting the type of 'IntLoc1' (line 87)
    IntLoc1_5842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 14), 'IntLoc1')
    # Getting the type of 'IntLoc2' (line 87)
    IntLoc2_5843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 24), 'IntLoc2')
    # Applying the binary operator '<' (line 87)
    result_lt_5844 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 14), '<', IntLoc1_5842, IntLoc2_5843)
    
    # Testing the type of an if condition (line 87)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 8), result_lt_5844)
    # SSA begins for while statement (line 87)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a BinOp to a Name (line 88):
    
    # Assigning a BinOp to a Name (line 88):
    int_5845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 22), 'int')
    # Getting the type of 'IntLoc1' (line 88)
    IntLoc1_5846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 26), 'IntLoc1')
    # Applying the binary operator '*' (line 88)
    result_mul_5847 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 22), '*', int_5845, IntLoc1_5846)
    
    # Getting the type of 'IntLoc2' (line 88)
    IntLoc2_5848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 36), 'IntLoc2')
    # Applying the binary operator '-' (line 88)
    result_sub_5849 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 22), '-', result_mul_5847, IntLoc2_5848)
    
    # Assigning a type to the variable 'IntLoc3' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'IntLoc3', result_sub_5849)
    
    # Assigning a Call to a Name (line 89):
    
    # Assigning a Call to a Name (line 89):
    
    # Call to Proc7(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'IntLoc1' (line 89)
    IntLoc1_5851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 28), 'IntLoc1', False)
    # Getting the type of 'IntLoc2' (line 89)
    IntLoc2_5852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 37), 'IntLoc2', False)
    # Processing the call keyword arguments (line 89)
    kwargs_5853 = {}
    # Getting the type of 'Proc7' (line 89)
    Proc7_5850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 22), 'Proc7', False)
    # Calling Proc7(args, kwargs) (line 89)
    Proc7_call_result_5854 = invoke(stypy.reporting.localization.Localization(__file__, 89, 22), Proc7_5850, *[IntLoc1_5851, IntLoc2_5852], **kwargs_5853)
    
    # Assigning a type to the variable 'IntLoc3' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'IntLoc3', Proc7_call_result_5854)
    
    # Assigning a BinOp to a Name (line 90):
    
    # Assigning a BinOp to a Name (line 90):
    # Getting the type of 'IntLoc1' (line 90)
    IntLoc1_5855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 22), 'IntLoc1')
    int_5856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 32), 'int')
    # Applying the binary operator '+' (line 90)
    result_add_5857 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 22), '+', IntLoc1_5855, int_5856)
    
    # Assigning a type to the variable 'IntLoc1' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'IntLoc1', result_add_5857)
    # SSA join for while statement (line 87)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to Proc8(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'Array1Glob' (line 91)
    Array1Glob_5859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 14), 'Array1Glob', False)
    # Getting the type of 'Array2Glob' (line 91)
    Array2Glob_5860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 26), 'Array2Glob', False)
    # Getting the type of 'IntLoc1' (line 91)
    IntLoc1_5861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 38), 'IntLoc1', False)
    # Getting the type of 'IntLoc3' (line 91)
    IntLoc3_5862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 47), 'IntLoc3', False)
    # Processing the call keyword arguments (line 91)
    kwargs_5863 = {}
    # Getting the type of 'Proc8' (line 91)
    Proc8_5858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'Proc8', False)
    # Calling Proc8(args, kwargs) (line 91)
    Proc8_call_result_5864 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), Proc8_5858, *[Array1Glob_5859, Array2Glob_5860, IntLoc1_5861, IntLoc3_5862], **kwargs_5863)
    
    
    # Assigning a Call to a Name (line 92):
    
    # Assigning a Call to a Name (line 92):
    
    # Call to Proc1(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'PtrGlb' (line 92)
    PtrGlb_5866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 23), 'PtrGlb', False)
    # Processing the call keyword arguments (line 92)
    kwargs_5867 = {}
    # Getting the type of 'Proc1' (line 92)
    Proc1_5865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 17), 'Proc1', False)
    # Calling Proc1(args, kwargs) (line 92)
    Proc1_call_result_5868 = invoke(stypy.reporting.localization.Localization(__file__, 92, 17), Proc1_5865, *[PtrGlb_5866], **kwargs_5867)
    
    # Assigning a type to the variable 'PtrGlb' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'PtrGlb', Proc1_call_result_5868)
    
    # Assigning a Str to a Name (line 93):
    
    # Assigning a Str to a Name (line 93):
    str_5869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 20), 'str', 'A')
    # Assigning a type to the variable 'CharIndex' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'CharIndex', str_5869)
    
    
    # Getting the type of 'CharIndex' (line 94)
    CharIndex_5870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 14), 'CharIndex')
    # Getting the type of 'Char2Glob' (line 94)
    Char2Glob_5871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 27), 'Char2Glob')
    # Applying the binary operator '<=' (line 94)
    result_le_5872 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 14), '<=', CharIndex_5870, Char2Glob_5871)
    
    # Testing the type of an if condition (line 94)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 8), result_le_5872)
    # SSA begins for while statement (line 94)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    
    # Getting the type of 'EnumLoc' (line 95)
    EnumLoc_5873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'EnumLoc')
    
    # Call to Func1(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'CharIndex' (line 95)
    CharIndex_5875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 32), 'CharIndex', False)
    str_5876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 43), 'str', 'C')
    # Processing the call keyword arguments (line 95)
    kwargs_5877 = {}
    # Getting the type of 'Func1' (line 95)
    Func1_5874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 26), 'Func1', False)
    # Calling Func1(args, kwargs) (line 95)
    Func1_call_result_5878 = invoke(stypy.reporting.localization.Localization(__file__, 95, 26), Func1_5874, *[CharIndex_5875, str_5876], **kwargs_5877)
    
    # Applying the binary operator '==' (line 95)
    result_eq_5879 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 15), '==', EnumLoc_5873, Func1_call_result_5878)
    
    # Testing the type of an if condition (line 95)
    if_condition_5880 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 12), result_eq_5879)
    # Assigning a type to the variable 'if_condition_5880' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'if_condition_5880', if_condition_5880)
    # SSA begins for if statement (line 95)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 96):
    
    # Assigning a Call to a Name (line 96):
    
    # Call to Proc6(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'Ident1' (line 96)
    Ident1_5882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 32), 'Ident1', False)
    # Processing the call keyword arguments (line 96)
    kwargs_5883 = {}
    # Getting the type of 'Proc6' (line 96)
    Proc6_5881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 26), 'Proc6', False)
    # Calling Proc6(args, kwargs) (line 96)
    Proc6_call_result_5884 = invoke(stypy.reporting.localization.Localization(__file__, 96, 26), Proc6_5881, *[Ident1_5882], **kwargs_5883)
    
    # Assigning a type to the variable 'EnumLoc' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'EnumLoc', Proc6_call_result_5884)
    # SSA join for if statement (line 95)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 97):
    
    # Assigning a Call to a Name (line 97):
    
    # Call to chr(...): (line 97)
    # Processing the call arguments (line 97)
    
    # Call to ord(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'CharIndex' (line 97)
    CharIndex_5887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 32), 'CharIndex', False)
    # Processing the call keyword arguments (line 97)
    kwargs_5888 = {}
    # Getting the type of 'ord' (line 97)
    ord_5886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 28), 'ord', False)
    # Calling ord(args, kwargs) (line 97)
    ord_call_result_5889 = invoke(stypy.reporting.localization.Localization(__file__, 97, 28), ord_5886, *[CharIndex_5887], **kwargs_5888)
    
    int_5890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 45), 'int')
    # Applying the binary operator '+' (line 97)
    result_add_5891 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 28), '+', ord_call_result_5889, int_5890)
    
    # Processing the call keyword arguments (line 97)
    kwargs_5892 = {}
    # Getting the type of 'chr' (line 97)
    chr_5885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 24), 'chr', False)
    # Calling chr(args, kwargs) (line 97)
    chr_call_result_5893 = invoke(stypy.reporting.localization.Localization(__file__, 97, 24), chr_5885, *[result_add_5891], **kwargs_5892)
    
    # Assigning a type to the variable 'CharIndex' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'CharIndex', chr_call_result_5893)
    # SSA join for while statement (line 94)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 98):
    
    # Assigning a BinOp to a Name (line 98):
    # Getting the type of 'IntLoc2' (line 98)
    IntLoc2_5894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 18), 'IntLoc2')
    # Getting the type of 'IntLoc1' (line 98)
    IntLoc1_5895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 28), 'IntLoc1')
    # Applying the binary operator '*' (line 98)
    result_mul_5896 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 18), '*', IntLoc2_5894, IntLoc1_5895)
    
    # Assigning a type to the variable 'IntLoc3' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'IntLoc3', result_mul_5896)
    
    # Assigning a BinOp to a Name (line 99):
    
    # Assigning a BinOp to a Name (line 99):
    # Getting the type of 'IntLoc3' (line 99)
    IntLoc3_5897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 18), 'IntLoc3')
    # Getting the type of 'IntLoc1' (line 99)
    IntLoc1_5898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 28), 'IntLoc1')
    # Applying the binary operator 'div' (line 99)
    result_div_5899 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 18), 'div', IntLoc3_5897, IntLoc1_5898)
    
    # Assigning a type to the variable 'IntLoc2' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'IntLoc2', result_div_5899)
    
    # Assigning a BinOp to a Name (line 100):
    
    # Assigning a BinOp to a Name (line 100):
    int_5900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 18), 'int')
    # Getting the type of 'IntLoc3' (line 100)
    IntLoc3_5901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 23), 'IntLoc3')
    # Getting the type of 'IntLoc2' (line 100)
    IntLoc2_5902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 33), 'IntLoc2')
    # Applying the binary operator '-' (line 100)
    result_sub_5903 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 23), '-', IntLoc3_5901, IntLoc2_5902)
    
    # Applying the binary operator '*' (line 100)
    result_mul_5904 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 18), '*', int_5900, result_sub_5903)
    
    # Getting the type of 'IntLoc1' (line 100)
    IntLoc1_5905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 44), 'IntLoc1')
    # Applying the binary operator '-' (line 100)
    result_sub_5906 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 18), '-', result_mul_5904, IntLoc1_5905)
    
    # Assigning a type to the variable 'IntLoc2' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'IntLoc2', result_sub_5906)
    
    # Assigning a Call to a Name (line 101):
    
    # Assigning a Call to a Name (line 101):
    
    # Call to Proc2(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'IntLoc1' (line 101)
    IntLoc1_5908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'IntLoc1', False)
    # Processing the call keyword arguments (line 101)
    kwargs_5909 = {}
    # Getting the type of 'Proc2' (line 101)
    Proc2_5907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 18), 'Proc2', False)
    # Calling Proc2(args, kwargs) (line 101)
    Proc2_call_result_5910 = invoke(stypy.reporting.localization.Localization(__file__, 101, 18), Proc2_5907, *[IntLoc1_5908], **kwargs_5909)
    
    # Assigning a type to the variable 'IntLoc1' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'IntLoc1', Proc2_call_result_5910)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 103):
    
    # Assigning a BinOp to a Name (line 103):
    
    # Call to clock(...): (line 103)
    # Processing the call keyword arguments (line 103)
    kwargs_5912 = {}
    # Getting the type of 'clock' (line 103)
    clock_5911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'clock', False)
    # Calling clock(args, kwargs) (line 103)
    clock_call_result_5913 = invoke(stypy.reporting.localization.Localization(__file__, 103, 16), clock_5911, *[], **kwargs_5912)
    
    # Getting the type of 'starttime' (line 103)
    starttime_5914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 26), 'starttime')
    # Applying the binary operator '-' (line 103)
    result_sub_5915 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 16), '-', clock_call_result_5913, starttime_5914)
    
    # Getting the type of 'nulltime' (line 103)
    nulltime_5916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 38), 'nulltime')
    # Applying the binary operator '-' (line 103)
    result_sub_5917 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 36), '-', result_sub_5915, nulltime_5916)
    
    # Assigning a type to the variable 'benchtime' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'benchtime', result_sub_5917)
    
    
    # Getting the type of 'benchtime' (line 104)
    benchtime_5918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 7), 'benchtime')
    float_5919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 20), 'float')
    # Applying the binary operator '==' (line 104)
    result_eq_5920 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 7), '==', benchtime_5918, float_5919)
    
    # Testing the type of an if condition (line 104)
    if_condition_5921 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 4), result_eq_5920)
    # Assigning a type to the variable 'if_condition_5921' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'if_condition_5921', if_condition_5921)
    # SSA begins for if statement (line 104)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 105):
    
    # Assigning a Num to a Name (line 105):
    float_5922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 28), 'float')
    # Assigning a type to the variable 'loopsPerBenchtime' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'loopsPerBenchtime', float_5922)
    # SSA branch for the else part of an if statement (line 104)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 110):
    
    # Assigning a BinOp to a Name (line 110):
    # Getting the type of 'loops' (line 110)
    loops_5923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 29), 'loops')
    # Getting the type of 'benchtime' (line 110)
    benchtime_5924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 37), 'benchtime')
    # Applying the binary operator 'div' (line 110)
    result_div_5925 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 29), 'div', loops_5923, benchtime_5924)
    
    # Assigning a type to the variable 'loopsPerBenchtime' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'loopsPerBenchtime', result_div_5925)
    # SSA join for if statement (line 104)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 113)
    tuple_5926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 113)
    # Adding element type (line 113)
    # Getting the type of 'benchtime' (line 113)
    benchtime_5927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 11), 'benchtime')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 11), tuple_5926, benchtime_5927)
    # Adding element type (line 113)
    # Getting the type of 'loopsPerBenchtime' (line 113)
    loopsPerBenchtime_5928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 22), 'loopsPerBenchtime')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 11), tuple_5926, loopsPerBenchtime_5928)
    
    # Assigning a type to the variable 'stypy_return_type' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'stypy_return_type', tuple_5926)
    
    # ################# End of 'Proc0(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Proc0' in the type store
    # Getting the type of 'stypy_return_type' (line 52)
    stypy_return_type_5929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5929)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Proc0'
    return stypy_return_type_5929

# Assigning a type to the variable 'Proc0' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'Proc0', Proc0)

@norecursion
def Proc1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Proc1'
    module_type_store = module_type_store.open_function_context('Proc1', 116, 0, False)
    
    # Passed parameters checking function
    Proc1.stypy_localization = localization
    Proc1.stypy_type_of_self = None
    Proc1.stypy_type_store = module_type_store
    Proc1.stypy_function_name = 'Proc1'
    Proc1.stypy_param_names_list = ['PtrParIn']
    Proc1.stypy_varargs_param_name = None
    Proc1.stypy_kwargs_param_name = None
    Proc1.stypy_call_defaults = defaults
    Proc1.stypy_call_varargs = varargs
    Proc1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'Proc1', ['PtrParIn'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'Proc1', localization, ['PtrParIn'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'Proc1(...)' code ##################

    
    # Multiple assignment of 2 elements.
    
    # Assigning a Call to a Name (line 117):
    
    # Call to copy(...): (line 117)
    # Processing the call keyword arguments (line 117)
    kwargs_5932 = {}
    # Getting the type of 'PtrGlb' (line 117)
    PtrGlb_5930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 36), 'PtrGlb', False)
    # Obtaining the member 'copy' of a type (line 117)
    copy_5931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 36), PtrGlb_5930, 'copy')
    # Calling copy(args, kwargs) (line 117)
    copy_call_result_5933 = invoke(stypy.reporting.localization.Localization(__file__, 117, 36), copy_5931, *[], **kwargs_5932)
    
    # Assigning a type to the variable 'NextRecord' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 23), 'NextRecord', copy_call_result_5933)
    
    # Assigning a Name to a Attribute (line 117):
    # Getting the type of 'NextRecord' (line 117)
    NextRecord_5934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 23), 'NextRecord')
    # Getting the type of 'PtrParIn' (line 117)
    PtrParIn_5935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'PtrParIn')
    # Setting the type of the member 'PtrComp' of a type (line 117)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 4), PtrParIn_5935, 'PtrComp', NextRecord_5934)
    
    # Assigning a Num to a Attribute (line 118):
    
    # Assigning a Num to a Attribute (line 118):
    int_5936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 23), 'int')
    # Getting the type of 'PtrParIn' (line 118)
    PtrParIn_5937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'PtrParIn')
    # Setting the type of the member 'IntComp' of a type (line 118)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 4), PtrParIn_5937, 'IntComp', int_5936)
    
    # Assigning a Attribute to a Attribute (line 119):
    
    # Assigning a Attribute to a Attribute (line 119):
    # Getting the type of 'PtrParIn' (line 119)
    PtrParIn_5938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 25), 'PtrParIn')
    # Obtaining the member 'IntComp' of a type (line 119)
    IntComp_5939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 25), PtrParIn_5938, 'IntComp')
    # Getting the type of 'NextRecord' (line 119)
    NextRecord_5940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'NextRecord')
    # Setting the type of the member 'IntComp' of a type (line 119)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 4), NextRecord_5940, 'IntComp', IntComp_5939)
    
    # Assigning a Attribute to a Attribute (line 120):
    
    # Assigning a Attribute to a Attribute (line 120):
    # Getting the type of 'PtrParIn' (line 120)
    PtrParIn_5941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 25), 'PtrParIn')
    # Obtaining the member 'PtrComp' of a type (line 120)
    PtrComp_5942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 25), PtrParIn_5941, 'PtrComp')
    # Getting the type of 'NextRecord' (line 120)
    NextRecord_5943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'NextRecord')
    # Setting the type of the member 'PtrComp' of a type (line 120)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 4), NextRecord_5943, 'PtrComp', PtrComp_5942)
    
    # Assigning a Call to a Attribute (line 121):
    
    # Assigning a Call to a Attribute (line 121):
    
    # Call to Proc3(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'NextRecord' (line 121)
    NextRecord_5945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 31), 'NextRecord', False)
    # Obtaining the member 'PtrComp' of a type (line 121)
    PtrComp_5946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 31), NextRecord_5945, 'PtrComp')
    # Processing the call keyword arguments (line 121)
    kwargs_5947 = {}
    # Getting the type of 'Proc3' (line 121)
    Proc3_5944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 25), 'Proc3', False)
    # Calling Proc3(args, kwargs) (line 121)
    Proc3_call_result_5948 = invoke(stypy.reporting.localization.Localization(__file__, 121, 25), Proc3_5944, *[PtrComp_5946], **kwargs_5947)
    
    # Getting the type of 'NextRecord' (line 121)
    NextRecord_5949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'NextRecord')
    # Setting the type of the member 'PtrComp' of a type (line 121)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 4), NextRecord_5949, 'PtrComp', Proc3_call_result_5948)
    
    
    # Getting the type of 'NextRecord' (line 122)
    NextRecord_5950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 7), 'NextRecord')
    # Obtaining the member 'Discr' of a type (line 122)
    Discr_5951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 7), NextRecord_5950, 'Discr')
    # Getting the type of 'Ident1' (line 122)
    Ident1_5952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 27), 'Ident1')
    # Applying the binary operator '==' (line 122)
    result_eq_5953 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 7), '==', Discr_5951, Ident1_5952)
    
    # Testing the type of an if condition (line 122)
    if_condition_5954 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 4), result_eq_5953)
    # Assigning a type to the variable 'if_condition_5954' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'if_condition_5954', if_condition_5954)
    # SSA begins for if statement (line 122)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Attribute (line 123):
    
    # Assigning a Num to a Attribute (line 123):
    int_5955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 29), 'int')
    # Getting the type of 'NextRecord' (line 123)
    NextRecord_5956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'NextRecord')
    # Setting the type of the member 'IntComp' of a type (line 123)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), NextRecord_5956, 'IntComp', int_5955)
    
    # Assigning a Call to a Attribute (line 124):
    
    # Assigning a Call to a Attribute (line 124):
    
    # Call to Proc6(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'PtrParIn' (line 124)
    PtrParIn_5958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 36), 'PtrParIn', False)
    # Obtaining the member 'EnumComp' of a type (line 124)
    EnumComp_5959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 36), PtrParIn_5958, 'EnumComp')
    # Processing the call keyword arguments (line 124)
    kwargs_5960 = {}
    # Getting the type of 'Proc6' (line 124)
    Proc6_5957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 30), 'Proc6', False)
    # Calling Proc6(args, kwargs) (line 124)
    Proc6_call_result_5961 = invoke(stypy.reporting.localization.Localization(__file__, 124, 30), Proc6_5957, *[EnumComp_5959], **kwargs_5960)
    
    # Getting the type of 'NextRecord' (line 124)
    NextRecord_5962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'NextRecord')
    # Setting the type of the member 'EnumComp' of a type (line 124)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 8), NextRecord_5962, 'EnumComp', Proc6_call_result_5961)
    
    # Assigning a Attribute to a Attribute (line 125):
    
    # Assigning a Attribute to a Attribute (line 125):
    # Getting the type of 'PtrGlb' (line 125)
    PtrGlb_5963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 29), 'PtrGlb')
    # Obtaining the member 'PtrComp' of a type (line 125)
    PtrComp_5964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 29), PtrGlb_5963, 'PtrComp')
    # Getting the type of 'NextRecord' (line 125)
    NextRecord_5965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'NextRecord')
    # Setting the type of the member 'PtrComp' of a type (line 125)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), NextRecord_5965, 'PtrComp', PtrComp_5964)
    
    # Assigning a Call to a Attribute (line 126):
    
    # Assigning a Call to a Attribute (line 126):
    
    # Call to Proc7(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'NextRecord' (line 126)
    NextRecord_5967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 35), 'NextRecord', False)
    # Obtaining the member 'IntComp' of a type (line 126)
    IntComp_5968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 35), NextRecord_5967, 'IntComp')
    int_5969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 55), 'int')
    # Processing the call keyword arguments (line 126)
    kwargs_5970 = {}
    # Getting the type of 'Proc7' (line 126)
    Proc7_5966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 29), 'Proc7', False)
    # Calling Proc7(args, kwargs) (line 126)
    Proc7_call_result_5971 = invoke(stypy.reporting.localization.Localization(__file__, 126, 29), Proc7_5966, *[IntComp_5968, int_5969], **kwargs_5970)
    
    # Getting the type of 'NextRecord' (line 126)
    NextRecord_5972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'NextRecord')
    # Setting the type of the member 'IntComp' of a type (line 126)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), NextRecord_5972, 'IntComp', Proc7_call_result_5971)
    # SSA branch for the else part of an if statement (line 122)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 128):
    
    # Assigning a Call to a Name (line 128):
    
    # Call to copy(...): (line 128)
    # Processing the call keyword arguments (line 128)
    kwargs_5975 = {}
    # Getting the type of 'NextRecord' (line 128)
    NextRecord_5973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 19), 'NextRecord', False)
    # Obtaining the member 'copy' of a type (line 128)
    copy_5974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 19), NextRecord_5973, 'copy')
    # Calling copy(args, kwargs) (line 128)
    copy_call_result_5976 = invoke(stypy.reporting.localization.Localization(__file__, 128, 19), copy_5974, *[], **kwargs_5975)
    
    # Assigning a type to the variable 'PtrParIn' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'PtrParIn', copy_call_result_5976)
    # SSA join for if statement (line 122)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Attribute (line 129):
    
    # Assigning a Name to a Attribute (line 129):
    # Getting the type of 'None' (line 129)
    None_5977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 25), 'None')
    # Getting the type of 'NextRecord' (line 129)
    NextRecord_5978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'NextRecord')
    # Setting the type of the member 'PtrComp' of a type (line 129)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 4), NextRecord_5978, 'PtrComp', None_5977)
    # Getting the type of 'PtrParIn' (line 130)
    PtrParIn_5979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 11), 'PtrParIn')
    # Assigning a type to the variable 'stypy_return_type' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'stypy_return_type', PtrParIn_5979)
    
    # ################# End of 'Proc1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Proc1' in the type store
    # Getting the type of 'stypy_return_type' (line 116)
    stypy_return_type_5980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5980)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Proc1'
    return stypy_return_type_5980

# Assigning a type to the variable 'Proc1' (line 116)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), 'Proc1', Proc1)

@norecursion
def Proc2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Proc2'
    module_type_store = module_type_store.open_function_context('Proc2', 133, 0, False)
    
    # Passed parameters checking function
    Proc2.stypy_localization = localization
    Proc2.stypy_type_of_self = None
    Proc2.stypy_type_store = module_type_store
    Proc2.stypy_function_name = 'Proc2'
    Proc2.stypy_param_names_list = ['IntParIO']
    Proc2.stypy_varargs_param_name = None
    Proc2.stypy_kwargs_param_name = None
    Proc2.stypy_call_defaults = defaults
    Proc2.stypy_call_varargs = varargs
    Proc2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'Proc2', ['IntParIO'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'Proc2', localization, ['IntParIO'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'Proc2(...)' code ##################

    
    # Assigning a BinOp to a Name (line 134):
    
    # Assigning a BinOp to a Name (line 134):
    # Getting the type of 'IntParIO' (line 134)
    IntParIO_5981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 13), 'IntParIO')
    int_5982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 24), 'int')
    # Applying the binary operator '+' (line 134)
    result_add_5983 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 13), '+', IntParIO_5981, int_5982)
    
    # Assigning a type to the variable 'IntLoc' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'IntLoc', result_add_5983)
    
    int_5984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 10), 'int')
    # Testing the type of an if condition (line 135)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 4), int_5984)
    # SSA begins for while statement (line 135)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    
    # Getting the type of 'Char1Glob' (line 136)
    Char1Glob_5985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 11), 'Char1Glob')
    str_5986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 24), 'str', 'A')
    # Applying the binary operator '==' (line 136)
    result_eq_5987 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 11), '==', Char1Glob_5985, str_5986)
    
    # Testing the type of an if condition (line 136)
    if_condition_5988 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 136, 8), result_eq_5987)
    # Assigning a type to the variable 'if_condition_5988' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'if_condition_5988', if_condition_5988)
    # SSA begins for if statement (line 136)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 137):
    
    # Assigning a BinOp to a Name (line 137):
    # Getting the type of 'IntLoc' (line 137)
    IntLoc_5989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 21), 'IntLoc')
    int_5990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 30), 'int')
    # Applying the binary operator '-' (line 137)
    result_sub_5991 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 21), '-', IntLoc_5989, int_5990)
    
    # Assigning a type to the variable 'IntLoc' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'IntLoc', result_sub_5991)
    
    # Assigning a BinOp to a Name (line 138):
    
    # Assigning a BinOp to a Name (line 138):
    # Getting the type of 'IntLoc' (line 138)
    IntLoc_5992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 23), 'IntLoc')
    # Getting the type of 'IntGlob' (line 138)
    IntGlob_5993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 32), 'IntGlob')
    # Applying the binary operator '-' (line 138)
    result_sub_5994 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 23), '-', IntLoc_5992, IntGlob_5993)
    
    # Assigning a type to the variable 'IntParIO' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'IntParIO', result_sub_5994)
    
    # Assigning a Name to a Name (line 139):
    
    # Assigning a Name to a Name (line 139):
    # Getting the type of 'Ident1' (line 139)
    Ident1_5995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 22), 'Ident1')
    # Assigning a type to the variable 'EnumLoc' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'EnumLoc', Ident1_5995)
    # SSA join for if statement (line 136)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'EnumLoc' (line 140)
    EnumLoc_5996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 11), 'EnumLoc')
    # Getting the type of 'Ident1' (line 140)
    Ident1_5997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 22), 'Ident1')
    # Applying the binary operator '==' (line 140)
    result_eq_5998 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 11), '==', EnumLoc_5996, Ident1_5997)
    
    # Testing the type of an if condition (line 140)
    if_condition_5999 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 140, 8), result_eq_5998)
    # Assigning a type to the variable 'if_condition_5999' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'if_condition_5999', if_condition_5999)
    # SSA begins for if statement (line 140)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 140)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 135)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'IntParIO' (line 142)
    IntParIO_6000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 11), 'IntParIO')
    # Assigning a type to the variable 'stypy_return_type' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'stypy_return_type', IntParIO_6000)
    
    # ################# End of 'Proc2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Proc2' in the type store
    # Getting the type of 'stypy_return_type' (line 133)
    stypy_return_type_6001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6001)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Proc2'
    return stypy_return_type_6001

# Assigning a type to the variable 'Proc2' (line 133)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 0), 'Proc2', Proc2)

@norecursion
def Proc3(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Proc3'
    module_type_store = module_type_store.open_function_context('Proc3', 145, 0, False)
    
    # Passed parameters checking function
    Proc3.stypy_localization = localization
    Proc3.stypy_type_of_self = None
    Proc3.stypy_type_store = module_type_store
    Proc3.stypy_function_name = 'Proc3'
    Proc3.stypy_param_names_list = ['PtrParOut']
    Proc3.stypy_varargs_param_name = None
    Proc3.stypy_kwargs_param_name = None
    Proc3.stypy_call_defaults = defaults
    Proc3.stypy_call_varargs = varargs
    Proc3.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'Proc3', ['PtrParOut'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'Proc3', localization, ['PtrParOut'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'Proc3(...)' code ##################

    # Marking variables as global (line 146)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 146, 4), 'IntGlob')
    
    # Type idiom detected: calculating its left and rigth part (line 148)
    # Getting the type of 'PtrGlb' (line 148)
    PtrGlb_6002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'PtrGlb')
    # Getting the type of 'None' (line 148)
    None_6003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 21), 'None')
    
    (may_be_6004, more_types_in_union_6005) = may_not_be_none(PtrGlb_6002, None_6003)

    if may_be_6004:

        if more_types_in_union_6005:
            # Runtime conditional SSA (line 148)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 149):
        
        # Assigning a Attribute to a Name (line 149):
        # Getting the type of 'PtrGlb' (line 149)
        PtrGlb_6006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 20), 'PtrGlb')
        # Obtaining the member 'PtrComp' of a type (line 149)
        PtrComp_6007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 20), PtrGlb_6006, 'PtrComp')
        # Assigning a type to the variable 'PtrParOut' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'PtrParOut', PtrComp_6007)

        if more_types_in_union_6005:
            # Runtime conditional SSA for else branch (line 148)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_6004) or more_types_in_union_6005):
        
        # Assigning a Num to a Name (line 151):
        
        # Assigning a Num to a Name (line 151):
        int_6008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 18), 'int')
        # Assigning a type to the variable 'IntGlob' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'IntGlob', int_6008)

        if (may_be_6004 and more_types_in_union_6005):
            # SSA join for if statement (line 148)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Attribute (line 152):
    
    # Assigning a Call to a Attribute (line 152):
    
    # Call to Proc7(...): (line 152)
    # Processing the call arguments (line 152)
    int_6010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 27), 'int')
    # Getting the type of 'IntGlob' (line 152)
    IntGlob_6011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 31), 'IntGlob', False)
    # Processing the call keyword arguments (line 152)
    kwargs_6012 = {}
    # Getting the type of 'Proc7' (line 152)
    Proc7_6009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 21), 'Proc7', False)
    # Calling Proc7(args, kwargs) (line 152)
    Proc7_call_result_6013 = invoke(stypy.reporting.localization.Localization(__file__, 152, 21), Proc7_6009, *[int_6010, IntGlob_6011], **kwargs_6012)
    
    # Getting the type of 'PtrGlb' (line 152)
    PtrGlb_6014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'PtrGlb')
    # Setting the type of the member 'IntComp' of a type (line 152)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 4), PtrGlb_6014, 'IntComp', Proc7_call_result_6013)
    # Getting the type of 'PtrParOut' (line 153)
    PtrParOut_6015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 11), 'PtrParOut')
    # Assigning a type to the variable 'stypy_return_type' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'stypy_return_type', PtrParOut_6015)
    
    # ################# End of 'Proc3(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Proc3' in the type store
    # Getting the type of 'stypy_return_type' (line 145)
    stypy_return_type_6016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6016)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Proc3'
    return stypy_return_type_6016

# Assigning a type to the variable 'Proc3' (line 145)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 0), 'Proc3', Proc3)

@norecursion
def Proc4(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Proc4'
    module_type_store = module_type_store.open_function_context('Proc4', 156, 0, False)
    
    # Passed parameters checking function
    Proc4.stypy_localization = localization
    Proc4.stypy_type_of_self = None
    Proc4.stypy_type_store = module_type_store
    Proc4.stypy_function_name = 'Proc4'
    Proc4.stypy_param_names_list = []
    Proc4.stypy_varargs_param_name = None
    Proc4.stypy_kwargs_param_name = None
    Proc4.stypy_call_defaults = defaults
    Proc4.stypy_call_varargs = varargs
    Proc4.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'Proc4', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'Proc4', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'Proc4(...)' code ##################

    # Marking variables as global (line 157)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 157, 4), 'Char2Glob')
    
    # Assigning a Compare to a Name (line 159):
    
    # Assigning a Compare to a Name (line 159):
    
    # Getting the type of 'Char1Glob' (line 159)
    Char1Glob_6017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 14), 'Char1Glob')
    str_6018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 27), 'str', 'A')
    # Applying the binary operator '==' (line 159)
    result_eq_6019 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 14), '==', Char1Glob_6017, str_6018)
    
    # Assigning a type to the variable 'BoolLoc' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'BoolLoc', result_eq_6019)
    
    # Assigning a BoolOp to a Name (line 160):
    
    # Assigning a BoolOp to a Name (line 160):
    
    # Evaluating a boolean operation
    # Getting the type of 'BoolLoc' (line 160)
    BoolLoc_6020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 14), 'BoolLoc')
    # Getting the type of 'BoolGlob' (line 160)
    BoolGlob_6021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 25), 'BoolGlob')
    # Applying the binary operator 'or' (line 160)
    result_or_keyword_6022 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 14), 'or', BoolLoc_6020, BoolGlob_6021)
    
    # Assigning a type to the variable 'BoolLoc' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'BoolLoc', result_or_keyword_6022)
    
    # Assigning a Str to a Name (line 161):
    
    # Assigning a Str to a Name (line 161):
    str_6023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 16), 'str', 'B')
    # Assigning a type to the variable 'Char2Glob' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'Char2Glob', str_6023)
    
    # ################# End of 'Proc4(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Proc4' in the type store
    # Getting the type of 'stypy_return_type' (line 156)
    stypy_return_type_6024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6024)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Proc4'
    return stypy_return_type_6024

# Assigning a type to the variable 'Proc4' (line 156)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 0), 'Proc4', Proc4)

@norecursion
def Proc5(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Proc5'
    module_type_store = module_type_store.open_function_context('Proc5', 164, 0, False)
    
    # Passed parameters checking function
    Proc5.stypy_localization = localization
    Proc5.stypy_type_of_self = None
    Proc5.stypy_type_store = module_type_store
    Proc5.stypy_function_name = 'Proc5'
    Proc5.stypy_param_names_list = []
    Proc5.stypy_varargs_param_name = None
    Proc5.stypy_kwargs_param_name = None
    Proc5.stypy_call_defaults = defaults
    Proc5.stypy_call_varargs = varargs
    Proc5.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'Proc5', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'Proc5', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'Proc5(...)' code ##################

    # Marking variables as global (line 165)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 165, 4), 'Char1Glob')
    # Marking variables as global (line 166)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 166, 4), 'BoolGlob')
    
    # Assigning a Str to a Name (line 168):
    
    # Assigning a Str to a Name (line 168):
    str_6025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 16), 'str', 'A')
    # Assigning a type to the variable 'Char1Glob' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'Char1Glob', str_6025)
    
    # Assigning a Name to a Name (line 169):
    
    # Assigning a Name to a Name (line 169):
    # Getting the type of 'FALSE' (line 169)
    FALSE_6026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'FALSE')
    # Assigning a type to the variable 'BoolGlob' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'BoolGlob', FALSE_6026)
    
    # ################# End of 'Proc5(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Proc5' in the type store
    # Getting the type of 'stypy_return_type' (line 164)
    stypy_return_type_6027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6027)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Proc5'
    return stypy_return_type_6027

# Assigning a type to the variable 'Proc5' (line 164)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 0), 'Proc5', Proc5)

@norecursion
def Proc6(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Proc6'
    module_type_store = module_type_store.open_function_context('Proc6', 172, 0, False)
    
    # Passed parameters checking function
    Proc6.stypy_localization = localization
    Proc6.stypy_type_of_self = None
    Proc6.stypy_type_store = module_type_store
    Proc6.stypy_function_name = 'Proc6'
    Proc6.stypy_param_names_list = ['EnumParIn']
    Proc6.stypy_varargs_param_name = None
    Proc6.stypy_kwargs_param_name = None
    Proc6.stypy_call_defaults = defaults
    Proc6.stypy_call_varargs = varargs
    Proc6.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'Proc6', ['EnumParIn'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'Proc6', localization, ['EnumParIn'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'Proc6(...)' code ##################

    
    # Assigning a Name to a Name (line 173):
    
    # Assigning a Name to a Name (line 173):
    # Getting the type of 'EnumParIn' (line 173)
    EnumParIn_6028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 17), 'EnumParIn')
    # Assigning a type to the variable 'EnumParOut' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'EnumParOut', EnumParIn_6028)
    
    
    
    # Call to Func3(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'EnumParIn' (line 174)
    EnumParIn_6030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 17), 'EnumParIn', False)
    # Processing the call keyword arguments (line 174)
    kwargs_6031 = {}
    # Getting the type of 'Func3' (line 174)
    Func3_6029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 11), 'Func3', False)
    # Calling Func3(args, kwargs) (line 174)
    Func3_call_result_6032 = invoke(stypy.reporting.localization.Localization(__file__, 174, 11), Func3_6029, *[EnumParIn_6030], **kwargs_6031)
    
    # Applying the 'not' unary operator (line 174)
    result_not__6033 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 7), 'not', Func3_call_result_6032)
    
    # Testing the type of an if condition (line 174)
    if_condition_6034 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 174, 4), result_not__6033)
    # Assigning a type to the variable 'if_condition_6034' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'if_condition_6034', if_condition_6034)
    # SSA begins for if statement (line 174)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 175):
    
    # Assigning a Name to a Name (line 175):
    # Getting the type of 'Ident4' (line 175)
    Ident4_6035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 21), 'Ident4')
    # Assigning a type to the variable 'EnumParOut' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'EnumParOut', Ident4_6035)
    # SSA join for if statement (line 174)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'EnumParIn' (line 176)
    EnumParIn_6036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 7), 'EnumParIn')
    # Getting the type of 'Ident1' (line 176)
    Ident1_6037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 20), 'Ident1')
    # Applying the binary operator '==' (line 176)
    result_eq_6038 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 7), '==', EnumParIn_6036, Ident1_6037)
    
    # Testing the type of an if condition (line 176)
    if_condition_6039 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 176, 4), result_eq_6038)
    # Assigning a type to the variable 'if_condition_6039' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'if_condition_6039', if_condition_6039)
    # SSA begins for if statement (line 176)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 177):
    
    # Assigning a Name to a Name (line 177):
    # Getting the type of 'Ident1' (line 177)
    Ident1_6040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 21), 'Ident1')
    # Assigning a type to the variable 'EnumParOut' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'EnumParOut', Ident1_6040)
    # SSA branch for the else part of an if statement (line 176)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'EnumParIn' (line 178)
    EnumParIn_6041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 9), 'EnumParIn')
    # Getting the type of 'Ident2' (line 178)
    Ident2_6042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 22), 'Ident2')
    # Applying the binary operator '==' (line 178)
    result_eq_6043 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 9), '==', EnumParIn_6041, Ident2_6042)
    
    # Testing the type of an if condition (line 178)
    if_condition_6044 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 178, 9), result_eq_6043)
    # Assigning a type to the variable 'if_condition_6044' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 9), 'if_condition_6044', if_condition_6044)
    # SSA begins for if statement (line 178)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'IntGlob' (line 179)
    IntGlob_6045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 11), 'IntGlob')
    int_6046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 21), 'int')
    # Applying the binary operator '>' (line 179)
    result_gt_6047 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 11), '>', IntGlob_6045, int_6046)
    
    # Testing the type of an if condition (line 179)
    if_condition_6048 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 179, 8), result_gt_6047)
    # Assigning a type to the variable 'if_condition_6048' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'if_condition_6048', if_condition_6048)
    # SSA begins for if statement (line 179)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 180):
    
    # Assigning a Name to a Name (line 180):
    # Getting the type of 'Ident1' (line 180)
    Ident1_6049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 25), 'Ident1')
    # Assigning a type to the variable 'EnumParOut' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'EnumParOut', Ident1_6049)
    # SSA branch for the else part of an if statement (line 179)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 182):
    
    # Assigning a Name to a Name (line 182):
    # Getting the type of 'Ident4' (line 182)
    Ident4_6050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 25), 'Ident4')
    # Assigning a type to the variable 'EnumParOut' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'EnumParOut', Ident4_6050)
    # SSA join for if statement (line 179)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 178)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'EnumParIn' (line 183)
    EnumParIn_6051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 9), 'EnumParIn')
    # Getting the type of 'Ident3' (line 183)
    Ident3_6052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 22), 'Ident3')
    # Applying the binary operator '==' (line 183)
    result_eq_6053 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 9), '==', EnumParIn_6051, Ident3_6052)
    
    # Testing the type of an if condition (line 183)
    if_condition_6054 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 183, 9), result_eq_6053)
    # Assigning a type to the variable 'if_condition_6054' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 9), 'if_condition_6054', if_condition_6054)
    # SSA begins for if statement (line 183)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 184):
    
    # Assigning a Name to a Name (line 184):
    # Getting the type of 'Ident2' (line 184)
    Ident2_6055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 21), 'Ident2')
    # Assigning a type to the variable 'EnumParOut' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'EnumParOut', Ident2_6055)
    # SSA branch for the else part of an if statement (line 183)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'EnumParIn' (line 185)
    EnumParIn_6056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 9), 'EnumParIn')
    # Getting the type of 'Ident4' (line 185)
    Ident4_6057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 22), 'Ident4')
    # Applying the binary operator '==' (line 185)
    result_eq_6058 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 9), '==', EnumParIn_6056, Ident4_6057)
    
    # Testing the type of an if condition (line 185)
    if_condition_6059 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 185, 9), result_eq_6058)
    # Assigning a type to the variable 'if_condition_6059' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 9), 'if_condition_6059', if_condition_6059)
    # SSA begins for if statement (line 185)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    pass
    # SSA branch for the else part of an if statement (line 185)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'EnumParIn' (line 187)
    EnumParIn_6060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 9), 'EnumParIn')
    # Getting the type of 'Ident5' (line 187)
    Ident5_6061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 22), 'Ident5')
    # Applying the binary operator '==' (line 187)
    result_eq_6062 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 9), '==', EnumParIn_6060, Ident5_6061)
    
    # Testing the type of an if condition (line 187)
    if_condition_6063 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 187, 9), result_eq_6062)
    # Assigning a type to the variable 'if_condition_6063' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 9), 'if_condition_6063', if_condition_6063)
    # SSA begins for if statement (line 187)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 188):
    
    # Assigning a Name to a Name (line 188):
    # Getting the type of 'Ident3' (line 188)
    Ident3_6064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 21), 'Ident3')
    # Assigning a type to the variable 'EnumParOut' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'EnumParOut', Ident3_6064)
    # SSA join for if statement (line 187)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 185)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 183)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 178)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 176)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'EnumParOut' (line 189)
    EnumParOut_6065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 11), 'EnumParOut')
    # Assigning a type to the variable 'stypy_return_type' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'stypy_return_type', EnumParOut_6065)
    
    # ################# End of 'Proc6(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Proc6' in the type store
    # Getting the type of 'stypy_return_type' (line 172)
    stypy_return_type_6066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6066)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Proc6'
    return stypy_return_type_6066

# Assigning a type to the variable 'Proc6' (line 172)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 0), 'Proc6', Proc6)

@norecursion
def Proc7(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Proc7'
    module_type_store = module_type_store.open_function_context('Proc7', 192, 0, False)
    
    # Passed parameters checking function
    Proc7.stypy_localization = localization
    Proc7.stypy_type_of_self = None
    Proc7.stypy_type_store = module_type_store
    Proc7.stypy_function_name = 'Proc7'
    Proc7.stypy_param_names_list = ['IntParI1', 'IntParI2']
    Proc7.stypy_varargs_param_name = None
    Proc7.stypy_kwargs_param_name = None
    Proc7.stypy_call_defaults = defaults
    Proc7.stypy_call_varargs = varargs
    Proc7.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'Proc7', ['IntParI1', 'IntParI2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'Proc7', localization, ['IntParI1', 'IntParI2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'Proc7(...)' code ##################

    
    # Assigning a BinOp to a Name (line 193):
    
    # Assigning a BinOp to a Name (line 193):
    # Getting the type of 'IntParI1' (line 193)
    IntParI1_6067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 13), 'IntParI1')
    int_6068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 24), 'int')
    # Applying the binary operator '+' (line 193)
    result_add_6069 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 13), '+', IntParI1_6067, int_6068)
    
    # Assigning a type to the variable 'IntLoc' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'IntLoc', result_add_6069)
    
    # Assigning a BinOp to a Name (line 194):
    
    # Assigning a BinOp to a Name (line 194):
    # Getting the type of 'IntParI2' (line 194)
    IntParI2_6070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 16), 'IntParI2')
    # Getting the type of 'IntLoc' (line 194)
    IntLoc_6071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 27), 'IntLoc')
    # Applying the binary operator '+' (line 194)
    result_add_6072 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 16), '+', IntParI2_6070, IntLoc_6071)
    
    # Assigning a type to the variable 'IntParOut' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'IntParOut', result_add_6072)
    # Getting the type of 'IntParOut' (line 195)
    IntParOut_6073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 11), 'IntParOut')
    # Assigning a type to the variable 'stypy_return_type' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'stypy_return_type', IntParOut_6073)
    
    # ################# End of 'Proc7(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Proc7' in the type store
    # Getting the type of 'stypy_return_type' (line 192)
    stypy_return_type_6074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6074)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Proc7'
    return stypy_return_type_6074

# Assigning a type to the variable 'Proc7' (line 192)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 0), 'Proc7', Proc7)

@norecursion
def Proc8(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Proc8'
    module_type_store = module_type_store.open_function_context('Proc8', 198, 0, False)
    
    # Passed parameters checking function
    Proc8.stypy_localization = localization
    Proc8.stypy_type_of_self = None
    Proc8.stypy_type_store = module_type_store
    Proc8.stypy_function_name = 'Proc8'
    Proc8.stypy_param_names_list = ['Array1Par', 'Array2Par', 'IntParI1', 'IntParI2']
    Proc8.stypy_varargs_param_name = None
    Proc8.stypy_kwargs_param_name = None
    Proc8.stypy_call_defaults = defaults
    Proc8.stypy_call_varargs = varargs
    Proc8.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'Proc8', ['Array1Par', 'Array2Par', 'IntParI1', 'IntParI2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'Proc8', localization, ['Array1Par', 'Array2Par', 'IntParI1', 'IntParI2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'Proc8(...)' code ##################

    # Marking variables as global (line 199)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 199, 4), 'IntGlob')
    
    # Assigning a BinOp to a Name (line 201):
    
    # Assigning a BinOp to a Name (line 201):
    # Getting the type of 'IntParI1' (line 201)
    IntParI1_6075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 13), 'IntParI1')
    int_6076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 24), 'int')
    # Applying the binary operator '+' (line 201)
    result_add_6077 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 13), '+', IntParI1_6075, int_6076)
    
    # Assigning a type to the variable 'IntLoc' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'IntLoc', result_add_6077)
    
    # Assigning a Name to a Subscript (line 202):
    
    # Assigning a Name to a Subscript (line 202):
    # Getting the type of 'IntParI2' (line 202)
    IntParI2_6078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 24), 'IntParI2')
    # Getting the type of 'Array1Par' (line 202)
    Array1Par_6079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'Array1Par')
    # Getting the type of 'IntLoc' (line 202)
    IntLoc_6080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 14), 'IntLoc')
    # Storing an element on a container (line 202)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 4), Array1Par_6079, (IntLoc_6080, IntParI2_6078))
    
    # Assigning a Subscript to a Subscript (line 203):
    
    # Assigning a Subscript to a Subscript (line 203):
    
    # Obtaining the type of the subscript
    # Getting the type of 'IntLoc' (line 203)
    IntLoc_6081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 38), 'IntLoc')
    # Getting the type of 'Array1Par' (line 203)
    Array1Par_6082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 28), 'Array1Par')
    # Obtaining the member '__getitem__' of a type (line 203)
    getitem___6083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 28), Array1Par_6082, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 203)
    subscript_call_result_6084 = invoke(stypy.reporting.localization.Localization(__file__, 203, 28), getitem___6083, IntLoc_6081)
    
    # Getting the type of 'Array1Par' (line 203)
    Array1Par_6085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'Array1Par')
    # Getting the type of 'IntLoc' (line 203)
    IntLoc_6086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 14), 'IntLoc')
    int_6087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 23), 'int')
    # Applying the binary operator '+' (line 203)
    result_add_6088 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 14), '+', IntLoc_6086, int_6087)
    
    # Storing an element on a container (line 203)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 4), Array1Par_6085, (result_add_6088, subscript_call_result_6084))
    
    # Assigning a Name to a Subscript (line 204):
    
    # Assigning a Name to a Subscript (line 204):
    # Getting the type of 'IntLoc' (line 204)
    IntLoc_6089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 29), 'IntLoc')
    # Getting the type of 'Array1Par' (line 204)
    Array1Par_6090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'Array1Par')
    # Getting the type of 'IntLoc' (line 204)
    IntLoc_6091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 14), 'IntLoc')
    int_6092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 23), 'int')
    # Applying the binary operator '+' (line 204)
    result_add_6093 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 14), '+', IntLoc_6091, int_6092)
    
    # Storing an element on a container (line 204)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 4), Array1Par_6090, (result_add_6093, IntLoc_6089))
    
    
    # Call to range(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'IntLoc' (line 205)
    IntLoc_6095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 26), 'IntLoc', False)
    # Getting the type of 'IntLoc' (line 205)
    IntLoc_6096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 34), 'IntLoc', False)
    int_6097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 43), 'int')
    # Applying the binary operator '+' (line 205)
    result_add_6098 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 34), '+', IntLoc_6096, int_6097)
    
    # Processing the call keyword arguments (line 205)
    kwargs_6099 = {}
    # Getting the type of 'range' (line 205)
    range_6094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'range', False)
    # Calling range(args, kwargs) (line 205)
    range_call_result_6100 = invoke(stypy.reporting.localization.Localization(__file__, 205, 20), range_6094, *[IntLoc_6095, result_add_6098], **kwargs_6099)
    
    # Testing the type of a for loop iterable (line 205)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 205, 4), range_call_result_6100)
    # Getting the type of the for loop variable (line 205)
    for_loop_var_6101 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 205, 4), range_call_result_6100)
    # Assigning a type to the variable 'IntIndex' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'IntIndex', for_loop_var_6101)
    # SSA begins for a for statement (line 205)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Subscript (line 206):
    
    # Assigning a Name to a Subscript (line 206):
    # Getting the type of 'IntLoc' (line 206)
    IntLoc_6102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 38), 'IntLoc')
    
    # Obtaining the type of the subscript
    # Getting the type of 'IntLoc' (line 206)
    IntLoc_6103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 18), 'IntLoc')
    # Getting the type of 'Array2Par' (line 206)
    Array2Par_6104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'Array2Par')
    # Obtaining the member '__getitem__' of a type (line 206)
    getitem___6105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), Array2Par_6104, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 206)
    subscript_call_result_6106 = invoke(stypy.reporting.localization.Localization(__file__, 206, 8), getitem___6105, IntLoc_6103)
    
    # Getting the type of 'IntIndex' (line 206)
    IntIndex_6107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 26), 'IntIndex')
    # Storing an element on a container (line 206)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 8), subscript_call_result_6106, (IntIndex_6107, IntLoc_6102))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Subscript (line 207):
    
    # Assigning a BinOp to a Subscript (line 207):
    
    # Obtaining the type of the subscript
    # Getting the type of 'IntLoc' (line 207)
    IntLoc_6108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 54), 'IntLoc')
    int_6109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 63), 'int')
    # Applying the binary operator '-' (line 207)
    result_sub_6110 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 54), '-', IntLoc_6108, int_6109)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'IntLoc' (line 207)
    IntLoc_6111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 46), 'IntLoc')
    # Getting the type of 'Array2Par' (line 207)
    Array2Par_6112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 36), 'Array2Par')
    # Obtaining the member '__getitem__' of a type (line 207)
    getitem___6113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 36), Array2Par_6112, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 207)
    subscript_call_result_6114 = invoke(stypy.reporting.localization.Localization(__file__, 207, 36), getitem___6113, IntLoc_6111)
    
    # Obtaining the member '__getitem__' of a type (line 207)
    getitem___6115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 36), subscript_call_result_6114, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 207)
    subscript_call_result_6116 = invoke(stypy.reporting.localization.Localization(__file__, 207, 36), getitem___6115, result_sub_6110)
    
    int_6117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 68), 'int')
    # Applying the binary operator '+' (line 207)
    result_add_6118 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 36), '+', subscript_call_result_6116, int_6117)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'IntLoc' (line 207)
    IntLoc_6119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 14), 'IntLoc')
    # Getting the type of 'Array2Par' (line 207)
    Array2Par_6120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'Array2Par')
    # Obtaining the member '__getitem__' of a type (line 207)
    getitem___6121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 4), Array2Par_6120, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 207)
    subscript_call_result_6122 = invoke(stypy.reporting.localization.Localization(__file__, 207, 4), getitem___6121, IntLoc_6119)
    
    # Getting the type of 'IntLoc' (line 207)
    IntLoc_6123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 22), 'IntLoc')
    int_6124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 31), 'int')
    # Applying the binary operator '-' (line 207)
    result_sub_6125 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 22), '-', IntLoc_6123, int_6124)
    
    # Storing an element on a container (line 207)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 4), subscript_call_result_6122, (result_sub_6125, result_add_6118))
    
    # Assigning a Subscript to a Subscript (line 208):
    
    # Assigning a Subscript to a Subscript (line 208):
    
    # Obtaining the type of the subscript
    # Getting the type of 'IntLoc' (line 208)
    IntLoc_6126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 47), 'IntLoc')
    # Getting the type of 'Array1Par' (line 208)
    Array1Par_6127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 37), 'Array1Par')
    # Obtaining the member '__getitem__' of a type (line 208)
    getitem___6128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 37), Array1Par_6127, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 208)
    subscript_call_result_6129 = invoke(stypy.reporting.localization.Localization(__file__, 208, 37), getitem___6128, IntLoc_6126)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'IntLoc' (line 208)
    IntLoc_6130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 14), 'IntLoc')
    int_6131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 23), 'int')
    # Applying the binary operator '+' (line 208)
    result_add_6132 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 14), '+', IntLoc_6130, int_6131)
    
    # Getting the type of 'Array2Par' (line 208)
    Array2Par_6133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'Array2Par')
    # Obtaining the member '__getitem__' of a type (line 208)
    getitem___6134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 4), Array2Par_6133, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 208)
    subscript_call_result_6135 = invoke(stypy.reporting.localization.Localization(__file__, 208, 4), getitem___6134, result_add_6132)
    
    # Getting the type of 'IntLoc' (line 208)
    IntLoc_6136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 27), 'IntLoc')
    # Storing an element on a container (line 208)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 4), subscript_call_result_6135, (IntLoc_6136, subscript_call_result_6129))
    
    # Assigning a Num to a Name (line 209):
    
    # Assigning a Num to a Name (line 209):
    int_6137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 14), 'int')
    # Assigning a type to the variable 'IntGlob' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'IntGlob', int_6137)
    
    # ################# End of 'Proc8(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Proc8' in the type store
    # Getting the type of 'stypy_return_type' (line 198)
    stypy_return_type_6138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6138)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Proc8'
    return stypy_return_type_6138

# Assigning a type to the variable 'Proc8' (line 198)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 0), 'Proc8', Proc8)

@norecursion
def Func1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Func1'
    module_type_store = module_type_store.open_function_context('Func1', 212, 0, False)
    
    # Passed parameters checking function
    Func1.stypy_localization = localization
    Func1.stypy_type_of_self = None
    Func1.stypy_type_store = module_type_store
    Func1.stypy_function_name = 'Func1'
    Func1.stypy_param_names_list = ['CharPar1', 'CharPar2']
    Func1.stypy_varargs_param_name = None
    Func1.stypy_kwargs_param_name = None
    Func1.stypy_call_defaults = defaults
    Func1.stypy_call_varargs = varargs
    Func1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'Func1', ['CharPar1', 'CharPar2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'Func1', localization, ['CharPar1', 'CharPar2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'Func1(...)' code ##################

    
    # Assigning a Name to a Name (line 213):
    
    # Assigning a Name to a Name (line 213):
    # Getting the type of 'CharPar1' (line 213)
    CharPar1_6139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 15), 'CharPar1')
    # Assigning a type to the variable 'CharLoc1' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'CharLoc1', CharPar1_6139)
    
    # Assigning a Name to a Name (line 214):
    
    # Assigning a Name to a Name (line 214):
    # Getting the type of 'CharLoc1' (line 214)
    CharLoc1_6140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 15), 'CharLoc1')
    # Assigning a type to the variable 'CharLoc2' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'CharLoc2', CharLoc1_6140)
    
    
    # Getting the type of 'CharLoc2' (line 215)
    CharLoc2_6141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 7), 'CharLoc2')
    # Getting the type of 'CharPar2' (line 215)
    CharPar2_6142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 19), 'CharPar2')
    # Applying the binary operator '!=' (line 215)
    result_ne_6143 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 7), '!=', CharLoc2_6141, CharPar2_6142)
    
    # Testing the type of an if condition (line 215)
    if_condition_6144 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 4), result_ne_6143)
    # Assigning a type to the variable 'if_condition_6144' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'if_condition_6144', if_condition_6144)
    # SSA begins for if statement (line 215)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'Ident1' (line 216)
    Ident1_6145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 15), 'Ident1')
    # Assigning a type to the variable 'stypy_return_type' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'stypy_return_type', Ident1_6145)
    # SSA branch for the else part of an if statement (line 215)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'Ident2' (line 218)
    Ident2_6146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 15), 'Ident2')
    # Assigning a type to the variable 'stypy_return_type' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'stypy_return_type', Ident2_6146)
    # SSA join for if statement (line 215)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'Func1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Func1' in the type store
    # Getting the type of 'stypy_return_type' (line 212)
    stypy_return_type_6147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6147)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Func1'
    return stypy_return_type_6147

# Assigning a type to the variable 'Func1' (line 212)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 0), 'Func1', Func1)

@norecursion
def Func2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Func2'
    module_type_store = module_type_store.open_function_context('Func2', 221, 0, False)
    
    # Passed parameters checking function
    Func2.stypy_localization = localization
    Func2.stypy_type_of_self = None
    Func2.stypy_type_store = module_type_store
    Func2.stypy_function_name = 'Func2'
    Func2.stypy_param_names_list = ['StrParI1', 'StrParI2']
    Func2.stypy_varargs_param_name = None
    Func2.stypy_kwargs_param_name = None
    Func2.stypy_call_defaults = defaults
    Func2.stypy_call_varargs = varargs
    Func2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'Func2', ['StrParI1', 'StrParI2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'Func2', localization, ['StrParI1', 'StrParI2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'Func2(...)' code ##################

    
    # Assigning a Num to a Name (line 222):
    
    # Assigning a Num to a Name (line 222):
    int_6148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 13), 'int')
    # Assigning a type to the variable 'IntLoc' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'IntLoc', int_6148)
    
    
    # Getting the type of 'IntLoc' (line 223)
    IntLoc_6149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 10), 'IntLoc')
    int_6150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 20), 'int')
    # Applying the binary operator '<=' (line 223)
    result_le_6151 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 10), '<=', IntLoc_6149, int_6150)
    
    # Testing the type of an if condition (line 223)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 4), result_le_6151)
    # SSA begins for while statement (line 223)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    
    
    # Call to Func1(...): (line 224)
    # Processing the call arguments (line 224)
    
    # Obtaining the type of the subscript
    # Getting the type of 'IntLoc' (line 224)
    IntLoc_6153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 26), 'IntLoc', False)
    # Getting the type of 'StrParI1' (line 224)
    StrParI1_6154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 17), 'StrParI1', False)
    # Obtaining the member '__getitem__' of a type (line 224)
    getitem___6155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 17), StrParI1_6154, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 224)
    subscript_call_result_6156 = invoke(stypy.reporting.localization.Localization(__file__, 224, 17), getitem___6155, IntLoc_6153)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'IntLoc' (line 224)
    IntLoc_6157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 44), 'IntLoc', False)
    int_6158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 53), 'int')
    # Applying the binary operator '+' (line 224)
    result_add_6159 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 44), '+', IntLoc_6157, int_6158)
    
    # Getting the type of 'StrParI2' (line 224)
    StrParI2_6160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 35), 'StrParI2', False)
    # Obtaining the member '__getitem__' of a type (line 224)
    getitem___6161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 35), StrParI2_6160, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 224)
    subscript_call_result_6162 = invoke(stypy.reporting.localization.Localization(__file__, 224, 35), getitem___6161, result_add_6159)
    
    # Processing the call keyword arguments (line 224)
    kwargs_6163 = {}
    # Getting the type of 'Func1' (line 224)
    Func1_6152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 11), 'Func1', False)
    # Calling Func1(args, kwargs) (line 224)
    Func1_call_result_6164 = invoke(stypy.reporting.localization.Localization(__file__, 224, 11), Func1_6152, *[subscript_call_result_6156, subscript_call_result_6162], **kwargs_6163)
    
    # Getting the type of 'Ident1' (line 224)
    Ident1_6165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 60), 'Ident1')
    # Applying the binary operator '==' (line 224)
    result_eq_6166 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 11), '==', Func1_call_result_6164, Ident1_6165)
    
    # Testing the type of an if condition (line 224)
    if_condition_6167 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 224, 8), result_eq_6166)
    # Assigning a type to the variable 'if_condition_6167' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'if_condition_6167', if_condition_6167)
    # SSA begins for if statement (line 224)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 225):
    
    # Assigning a Str to a Name (line 225):
    str_6168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 22), 'str', 'A')
    # Assigning a type to the variable 'CharLoc' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'CharLoc', str_6168)
    
    # Assigning a BinOp to a Name (line 226):
    
    # Assigning a BinOp to a Name (line 226):
    # Getting the type of 'IntLoc' (line 226)
    IntLoc_6169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 21), 'IntLoc')
    int_6170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 30), 'int')
    # Applying the binary operator '+' (line 226)
    result_add_6171 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 21), '+', IntLoc_6169, int_6170)
    
    # Assigning a type to the variable 'IntLoc' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'IntLoc', result_add_6171)
    # SSA join for if statement (line 224)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 223)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'CharLoc' (line 227)
    CharLoc_6172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 7), 'CharLoc')
    str_6173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 18), 'str', 'W')
    # Applying the binary operator '>=' (line 227)
    result_ge_6174 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 7), '>=', CharLoc_6172, str_6173)
    
    
    # Getting the type of 'CharLoc' (line 227)
    CharLoc_6175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 26), 'CharLoc')
    str_6176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 37), 'str', 'Z')
    # Applying the binary operator '<=' (line 227)
    result_le_6177 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 26), '<=', CharLoc_6175, str_6176)
    
    # Applying the binary operator 'and' (line 227)
    result_and_keyword_6178 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 7), 'and', result_ge_6174, result_le_6177)
    
    # Testing the type of an if condition (line 227)
    if_condition_6179 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 4), result_and_keyword_6178)
    # Assigning a type to the variable 'if_condition_6179' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'if_condition_6179', if_condition_6179)
    # SSA begins for if statement (line 227)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 228):
    
    # Assigning a Num to a Name (line 228):
    int_6180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 17), 'int')
    # Assigning a type to the variable 'IntLoc' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'IntLoc', int_6180)
    # SSA join for if statement (line 227)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'CharLoc' (line 229)
    CharLoc_6181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 7), 'CharLoc')
    str_6182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 18), 'str', 'X')
    # Applying the binary operator '==' (line 229)
    result_eq_6183 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 7), '==', CharLoc_6181, str_6182)
    
    # Testing the type of an if condition (line 229)
    if_condition_6184 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 229, 4), result_eq_6183)
    # Assigning a type to the variable 'if_condition_6184' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'if_condition_6184', if_condition_6184)
    # SSA begins for if statement (line 229)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'TRUE' (line 230)
    TRUE_6185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 15), 'TRUE')
    # Assigning a type to the variable 'stypy_return_type' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'stypy_return_type', TRUE_6185)
    # SSA branch for the else part of an if statement (line 229)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'StrParI1' (line 232)
    StrParI1_6186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 11), 'StrParI1')
    # Getting the type of 'StrParI2' (line 232)
    StrParI2_6187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 22), 'StrParI2')
    # Applying the binary operator '>' (line 232)
    result_gt_6188 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 11), '>', StrParI1_6186, StrParI2_6187)
    
    # Testing the type of an if condition (line 232)
    if_condition_6189 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 232, 8), result_gt_6188)
    # Assigning a type to the variable 'if_condition_6189' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'if_condition_6189', if_condition_6189)
    # SSA begins for if statement (line 232)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 233):
    
    # Assigning a BinOp to a Name (line 233):
    # Getting the type of 'IntLoc' (line 233)
    IntLoc_6190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 21), 'IntLoc')
    int_6191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 30), 'int')
    # Applying the binary operator '+' (line 233)
    result_add_6192 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 21), '+', IntLoc_6190, int_6191)
    
    # Assigning a type to the variable 'IntLoc' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'IntLoc', result_add_6192)
    # Getting the type of 'TRUE' (line 234)
    TRUE_6193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 19), 'TRUE')
    # Assigning a type to the variable 'stypy_return_type' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'stypy_return_type', TRUE_6193)
    # SSA branch for the else part of an if statement (line 232)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'FALSE' (line 236)
    FALSE_6194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 19), 'FALSE')
    # Assigning a type to the variable 'stypy_return_type' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'stypy_return_type', FALSE_6194)
    # SSA join for if statement (line 232)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 229)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'Func2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Func2' in the type store
    # Getting the type of 'stypy_return_type' (line 221)
    stypy_return_type_6195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6195)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Func2'
    return stypy_return_type_6195

# Assigning a type to the variable 'Func2' (line 221)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 0), 'Func2', Func2)

@norecursion
def Func3(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Func3'
    module_type_store = module_type_store.open_function_context('Func3', 239, 0, False)
    
    # Passed parameters checking function
    Func3.stypy_localization = localization
    Func3.stypy_type_of_self = None
    Func3.stypy_type_store = module_type_store
    Func3.stypy_function_name = 'Func3'
    Func3.stypy_param_names_list = ['EnumParIn']
    Func3.stypy_varargs_param_name = None
    Func3.stypy_kwargs_param_name = None
    Func3.stypy_call_defaults = defaults
    Func3.stypy_call_varargs = varargs
    Func3.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'Func3', ['EnumParIn'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'Func3', localization, ['EnumParIn'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'Func3(...)' code ##################

    
    # Assigning a Name to a Name (line 240):
    
    # Assigning a Name to a Name (line 240):
    # Getting the type of 'EnumParIn' (line 240)
    EnumParIn_6196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 14), 'EnumParIn')
    # Assigning a type to the variable 'EnumLoc' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'EnumLoc', EnumParIn_6196)
    
    
    # Getting the type of 'EnumLoc' (line 241)
    EnumLoc_6197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 7), 'EnumLoc')
    # Getting the type of 'Ident3' (line 241)
    Ident3_6198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 18), 'Ident3')
    # Applying the binary operator '==' (line 241)
    result_eq_6199 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 7), '==', EnumLoc_6197, Ident3_6198)
    
    # Testing the type of an if condition (line 241)
    if_condition_6200 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 4), result_eq_6199)
    # Assigning a type to the variable 'if_condition_6200' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'if_condition_6200', if_condition_6200)
    # SSA begins for if statement (line 241)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'TRUE' (line 241)
    TRUE_6201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 33), 'TRUE')
    # Assigning a type to the variable 'stypy_return_type' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 26), 'stypy_return_type', TRUE_6201)
    # SSA join for if statement (line 241)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'FALSE' (line 242)
    FALSE_6202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 11), 'FALSE')
    # Assigning a type to the variable 'stypy_return_type' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'stypy_return_type', FALSE_6202)
    
    # ################# End of 'Func3(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Func3' in the type store
    # Getting the type of 'stypy_return_type' (line 239)
    stypy_return_type_6203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6203)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Func3'
    return stypy_return_type_6203

# Assigning a type to the variable 'Func3' (line 239)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 0), 'Func3', Func3)

if (__name__ == '__main__'):
    
    # Call to main(...): (line 265)
    # Processing the call arguments (line 265)
    # Getting the type of 'LOOPS' (line 265)
    LOOPS_6205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 9), 'LOOPS', False)
    # Processing the call keyword arguments (line 265)
    kwargs_6206 = {}
    # Getting the type of 'main' (line 265)
    main_6204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'main', False)
    # Calling main(args, kwargs) (line 265)
    main_call_result_6207 = invoke(stypy.reporting.localization.Localization(__file__, 265, 4), main_6204, *[LOOPS_6205], **kwargs_6206)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

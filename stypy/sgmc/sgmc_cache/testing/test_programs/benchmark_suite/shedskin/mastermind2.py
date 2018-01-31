
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Recipe 496907: Mastermind-style code-breaking, by Raymond Hettinger
2: # http://code.activestate.com/recipes/496907/
3: # Version speed up and adapted to Psyco D by leonardo maffi, V.1.0, Apr 4 2009
4: 
5: import random
6: from math import log
7: from collections import defaultdict
8: from time import clock
9: 
10: DIGITS = 4
11: TRIALS = 1
12: fmt = '%0' + str(DIGITS) + 'd'
13: searchspace = [[int(f) for f in fmt % i] for i in xrange(0, 10 ** DIGITS)]
14: count1 = [0] * 10
15: count2 = [0] * 10
16: 
17: 
18: def compare(a, b):
19:     N = 10
20:     for i in xrange(N):
21:         count1[i] = 0
22:         count2[i] = 0
23: 
24:     strikes = 0
25:     i = 0
26:     for dig1 in a:
27:         dig2 = b[i]
28:         i += 1
29:         if dig1 == dig2:
30:             strikes += 1
31:         count1[dig1] += 1
32:         count2[dig2] += 1
33: 
34:     balls = (count1[0] if count1[0] < count2[0] else count2[0])
35:     balls += (count1[1] if count1[1] < count2[1] else count2[1])
36:     balls += (count1[2] if count1[2] < count2[2] else count2[2])
37:     balls += (count1[3] if count1[3] < count2[3] else count2[3])
38:     balls += (count1[4] if count1[4] < count2[4] else count2[4])
39:     balls += (count1[5] if count1[5] < count2[5] else count2[5])
40:     balls += (count1[6] if count1[6] < count2[6] else count2[6])
41:     balls += (count1[7] if count1[7] < count2[7] else count2[7])
42:     balls += (count1[8] if count1[8] < count2[8] else count2[8])
43:     balls += (count1[9] if count1[9] < count2[9] else count2[9])
44: 
45:     return (strikes << 16) | (balls - strikes)
46: 
47: 
48: def rungame(target, strategy, verbose=True, maxtries=15):
49:     possibles = searchspace
50:     for i in xrange(maxtries):
51:         g = strategy(i, possibles)
52:         if verbose:
53:             pass
54:         ##            print "Out of %7d possibilities.  I'll guess %r" % (len(possibles), g),
55:         score = compare(g, target)
56:         if verbose:
57:             pass  # print ' ---> ', score
58:         if (score >> 16) == DIGITS:  # score >> 16 is strikes
59:             if verbose:
60:                 pass  # print "That's it.  After %d tries, I won." % (i+1)
61:             break
62:         possibles = [n for n in possibles if compare(g, n) == score]
63:     return i + 1
64: 
65: 
66: # Strategy support =============================================
67: 
68: def utility(play, possibles):
69:     b = defaultdict(int)
70:     for poss in possibles:
71:         b[compare(play, poss)] += 1
72: 
73:     # info
74:     bits = 0
75:     s = float(len(possibles))
76:     for i in b.itervalues():
77:         p = i / s
78:         bits -= p * log(p, 2)
79:     return bits
80: 
81: 
82: def nodup(play):
83:     return len(set(play)) == DIGITS
84: 
85: 
86: # Strategies =============================================
87: 
88: def s_allrand(i, possibles):
89:     return random.choice(possibles)
90: 
91: 
92: def s_trynodup(i, possibles):
93:     for j in xrange(20):
94:         g = random.choice(possibles)
95:         if nodup(g):
96:             break
97:     return g
98: 
99: 
100: def s_bestinfo(i, possibles):
101:     if i == 0:
102:         return s_trynodup(i, possibles)
103:     plays = random.sample(possibles, min(20, len(possibles)))
104:     _, play = max([(utility(play, possibles), play) for play in plays])
105:     return play
106: 
107: 
108: def s_worstinfo(i, possibles):
109:     if i == 0:
110:         return s_trynodup(i, possibles)
111:     plays = random.sample(possibles, min(20, len(possibles)))
112:     _, play = min([(utility(play, possibles), play) for play in plays])
113:     return play
114: 
115: 
116: def s_samplebest(i, possibles):
117:     if i == 0:
118:         return s_trynodup(i, possibles)
119:     if len(possibles) > 150:
120:         possibles = random.sample(possibles, 150)
121:         plays = possibles[:20]
122:     elif len(possibles) > 20:
123:         plays = random.sample(possibles, 20)
124:     else:
125:         plays = possibles
126:     _, play = max([(utility(play, possibles), play) for play in plays])
127:     return play
128: 
129: 
130: # Evaluate Strategies =============================================
131: 
132: def average(seqn):
133:     return sum(seqn) / float(len(seqn))
134: 
135: 
136: def counts(seqn):
137:     limit = max(10, max(seqn)) + 1
138:     tally = [0] * limit
139:     for i in seqn:
140:         tally[i] += 1
141:     return tally[1:]
142: 
143: 
144: def eval_strategy(name, strategy):
145:     start = clock()
146:     data = [rungame(random.choice(searchspace), strategy, verbose=False) for i in xrange(TRIALS)]
147: 
148: 
149: ##    print 'mean=%.2f %r  %s n=%d dig=%d' % (average(data), counts(data), name, len(data), DIGITS)
150: ##    print 'Time elapsed %.2f' % (clock() - start,)
151: 
152: def main():
153:     random.seed(1)
154:     ##    print '-' * 60
155: 
156:     names = "s_bestinfo s_samplebest s_worstinfo s_allrand s_trynodup s_bestinfo".split()
157:     eval_strategy('s_bestinfo', s_bestinfo)
158:     eval_strategy('s_samplebest', s_samplebest)
159:     eval_strategy('s_worstinfo', s_worstinfo)
160:     eval_strategy('s_trynodup', s_trynodup)
161:     eval_strategy('s_allrand', s_allrand)
162: 
163: 
164: def run():
165:     main()
166:     return True
167: 
168: 
169: run()
170: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import random' statement (line 5)
import random

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'random', random, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from math import log' statement (line 6)
try:
    from math import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'math', None, module_type_store, ['log'], [log])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from collections import defaultdict' statement (line 7)
try:
    from collections import defaultdict

except:
    defaultdict = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'collections', None, module_type_store, ['defaultdict'], [defaultdict])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from time import clock' statement (line 8)
try:
    from time import clock

except:
    clock = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'time', None, module_type_store, ['clock'], [clock])


# Assigning a Num to a Name (line 10):

# Assigning a Num to a Name (line 10):
int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 9), 'int')
# Assigning a type to the variable 'DIGITS' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'DIGITS', int_10)

# Assigning a Num to a Name (line 11):

# Assigning a Num to a Name (line 11):
int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 9), 'int')
# Assigning a type to the variable 'TRIALS' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'TRIALS', int_11)

# Assigning a BinOp to a Name (line 12):

# Assigning a BinOp to a Name (line 12):
str_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 6), 'str', '%0')

# Call to str(...): (line 12)
# Processing the call arguments (line 12)
# Getting the type of 'DIGITS' (line 12)
DIGITS_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 17), 'DIGITS', False)
# Processing the call keyword arguments (line 12)
kwargs_15 = {}
# Getting the type of 'str' (line 12)
str_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 13), 'str', False)
# Calling str(args, kwargs) (line 12)
str_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 12, 13), str_13, *[DIGITS_14], **kwargs_15)

# Applying the binary operator '+' (line 12)
result_add_17 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 6), '+', str_12, str_call_result_16)

str_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 27), 'str', 'd')
# Applying the binary operator '+' (line 12)
result_add_19 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 25), '+', result_add_17, str_18)

# Assigning a type to the variable 'fmt' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'fmt', result_add_19)

# Assigning a ListComp to a Name (line 13):

# Assigning a ListComp to a Name (line 13):
# Calculating list comprehension
# Calculating comprehension expression

# Call to xrange(...): (line 13)
# Processing the call arguments (line 13)
int_30 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 57), 'int')
int_31 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 60), 'int')
# Getting the type of 'DIGITS' (line 13)
DIGITS_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 66), 'DIGITS', False)
# Applying the binary operator '**' (line 13)
result_pow_33 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 60), '**', int_31, DIGITS_32)

# Processing the call keyword arguments (line 13)
kwargs_34 = {}
# Getting the type of 'xrange' (line 13)
xrange_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 50), 'xrange', False)
# Calling xrange(args, kwargs) (line 13)
xrange_call_result_35 = invoke(stypy.reporting.localization.Localization(__file__, 13, 50), xrange_29, *[int_30, result_pow_33], **kwargs_34)

comprehension_36 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 15), xrange_call_result_35)
# Assigning a type to the variable 'i' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 15), 'i', comprehension_36)
# Calculating list comprehension
# Calculating comprehension expression
# Getting the type of 'fmt' (line 13)
fmt_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 32), 'fmt')
# Getting the type of 'i' (line 13)
i_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 38), 'i')
# Applying the binary operator '%' (line 13)
result_mod_26 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 32), '%', fmt_24, i_25)

comprehension_27 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 16), result_mod_26)
# Assigning a type to the variable 'f' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 16), 'f', comprehension_27)

# Call to int(...): (line 13)
# Processing the call arguments (line 13)
# Getting the type of 'f' (line 13)
f_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 20), 'f', False)
# Processing the call keyword arguments (line 13)
kwargs_22 = {}
# Getting the type of 'int' (line 13)
int_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 16), 'int', False)
# Calling int(args, kwargs) (line 13)
int_call_result_23 = invoke(stypy.reporting.localization.Localization(__file__, 13, 16), int_20, *[f_21], **kwargs_22)

list_28 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 16), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 16), list_28, int_call_result_23)
list_37 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 15), list_37, list_28)
# Assigning a type to the variable 'searchspace' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'searchspace', list_37)

# Assigning a BinOp to a Name (line 14):

# Assigning a BinOp to a Name (line 14):

# Obtaining an instance of the builtin type 'list' (line 14)
list_38 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 9), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
int_39 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 10), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 9), list_38, int_39)

int_40 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 15), 'int')
# Applying the binary operator '*' (line 14)
result_mul_41 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '*', list_38, int_40)

# Assigning a type to the variable 'count1' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'count1', result_mul_41)

# Assigning a BinOp to a Name (line 15):

# Assigning a BinOp to a Name (line 15):

# Obtaining an instance of the builtin type 'list' (line 15)
list_42 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 9), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)
int_43 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 10), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 9), list_42, int_43)

int_44 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'int')
# Applying the binary operator '*' (line 15)
result_mul_45 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 9), '*', list_42, int_44)

# Assigning a type to the variable 'count2' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'count2', result_mul_45)

@norecursion
def compare(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'compare'
    module_type_store = module_type_store.open_function_context('compare', 18, 0, False)
    
    # Passed parameters checking function
    compare.stypy_localization = localization
    compare.stypy_type_of_self = None
    compare.stypy_type_store = module_type_store
    compare.stypy_function_name = 'compare'
    compare.stypy_param_names_list = ['a', 'b']
    compare.stypy_varargs_param_name = None
    compare.stypy_kwargs_param_name = None
    compare.stypy_call_defaults = defaults
    compare.stypy_call_varargs = varargs
    compare.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'compare', ['a', 'b'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'compare', localization, ['a', 'b'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'compare(...)' code ##################

    
    # Assigning a Num to a Name (line 19):
    
    # Assigning a Num to a Name (line 19):
    int_46 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 8), 'int')
    # Assigning a type to the variable 'N' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'N', int_46)
    
    
    # Call to xrange(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'N' (line 20)
    N_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 20), 'N', False)
    # Processing the call keyword arguments (line 20)
    kwargs_49 = {}
    # Getting the type of 'xrange' (line 20)
    xrange_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 20)
    xrange_call_result_50 = invoke(stypy.reporting.localization.Localization(__file__, 20, 13), xrange_47, *[N_48], **kwargs_49)
    
    # Testing if the for loop is going to be iterated (line 20)
    # Testing the type of a for loop iterable (line 20)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 20, 4), xrange_call_result_50)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 20, 4), xrange_call_result_50):
        # Getting the type of the for loop variable (line 20)
        for_loop_var_51 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 20, 4), xrange_call_result_50)
        # Assigning a type to the variable 'i' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'i', for_loop_var_51)
        # SSA begins for a for statement (line 20)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Num to a Subscript (line 21):
        
        # Assigning a Num to a Subscript (line 21):
        int_52 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 20), 'int')
        # Getting the type of 'count1' (line 21)
        count1_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'count1')
        # Getting the type of 'i' (line 21)
        i_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'i')
        # Storing an element on a container (line 21)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 8), count1_53, (i_54, int_52))
        
        # Assigning a Num to a Subscript (line 22):
        
        # Assigning a Num to a Subscript (line 22):
        int_55 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 20), 'int')
        # Getting the type of 'count2' (line 22)
        count2_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'count2')
        # Getting the type of 'i' (line 22)
        i_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 15), 'i')
        # Storing an element on a container (line 22)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 8), count2_56, (i_57, int_55))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a Num to a Name (line 24):
    
    # Assigning a Num to a Name (line 24):
    int_58 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 14), 'int')
    # Assigning a type to the variable 'strikes' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'strikes', int_58)
    
    # Assigning a Num to a Name (line 25):
    
    # Assigning a Num to a Name (line 25):
    int_59 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 8), 'int')
    # Assigning a type to the variable 'i' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'i', int_59)
    
    # Getting the type of 'a' (line 26)
    a_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 16), 'a')
    # Testing if the for loop is going to be iterated (line 26)
    # Testing the type of a for loop iterable (line 26)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 26, 4), a_60)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 26, 4), a_60):
        # Getting the type of the for loop variable (line 26)
        for_loop_var_61 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 26, 4), a_60)
        # Assigning a type to the variable 'dig1' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'dig1', for_loop_var_61)
        # SSA begins for a for statement (line 26)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 27):
        
        # Assigning a Subscript to a Name (line 27):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 27)
        i_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 17), 'i')
        # Getting the type of 'b' (line 27)
        b_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 15), 'b')
        # Obtaining the member '__getitem__' of a type (line 27)
        getitem___64 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 15), b_63, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 27)
        subscript_call_result_65 = invoke(stypy.reporting.localization.Localization(__file__, 27, 15), getitem___64, i_62)
        
        # Assigning a type to the variable 'dig2' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'dig2', subscript_call_result_65)
        
        # Getting the type of 'i' (line 28)
        i_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'i')
        int_67 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 13), 'int')
        # Applying the binary operator '+=' (line 28)
        result_iadd_68 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 8), '+=', i_66, int_67)
        # Assigning a type to the variable 'i' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'i', result_iadd_68)
        
        
        # Getting the type of 'dig1' (line 29)
        dig1_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 11), 'dig1')
        # Getting the type of 'dig2' (line 29)
        dig2_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 19), 'dig2')
        # Applying the binary operator '==' (line 29)
        result_eq_71 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 11), '==', dig1_69, dig2_70)
        
        # Testing if the type of an if condition is none (line 29)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 29, 8), result_eq_71):
            pass
        else:
            
            # Testing the type of an if condition (line 29)
            if_condition_72 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 8), result_eq_71)
            # Assigning a type to the variable 'if_condition_72' (line 29)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'if_condition_72', if_condition_72)
            # SSA begins for if statement (line 29)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'strikes' (line 30)
            strikes_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'strikes')
            int_74 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 23), 'int')
            # Applying the binary operator '+=' (line 30)
            result_iadd_75 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 12), '+=', strikes_73, int_74)
            # Assigning a type to the variable 'strikes' (line 30)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'strikes', result_iadd_75)
            
            # SSA join for if statement (line 29)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'count1' (line 31)
        count1_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'count1')
        
        # Obtaining the type of the subscript
        # Getting the type of 'dig1' (line 31)
        dig1_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'dig1')
        # Getting the type of 'count1' (line 31)
        count1_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'count1')
        # Obtaining the member '__getitem__' of a type (line 31)
        getitem___79 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), count1_78, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 31)
        subscript_call_result_80 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), getitem___79, dig1_77)
        
        int_81 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 24), 'int')
        # Applying the binary operator '+=' (line 31)
        result_iadd_82 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 8), '+=', subscript_call_result_80, int_81)
        # Getting the type of 'count1' (line 31)
        count1_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'count1')
        # Getting the type of 'dig1' (line 31)
        dig1_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'dig1')
        # Storing an element on a container (line 31)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 8), count1_83, (dig1_84, result_iadd_82))
        
        
        # Getting the type of 'count2' (line 32)
        count2_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'count2')
        
        # Obtaining the type of the subscript
        # Getting the type of 'dig2' (line 32)
        dig2_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'dig2')
        # Getting the type of 'count2' (line 32)
        count2_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'count2')
        # Obtaining the member '__getitem__' of a type (line 32)
        getitem___88 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), count2_87, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 32)
        subscript_call_result_89 = invoke(stypy.reporting.localization.Localization(__file__, 32, 8), getitem___88, dig2_86)
        
        int_90 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 24), 'int')
        # Applying the binary operator '+=' (line 32)
        result_iadd_91 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 8), '+=', subscript_call_result_89, int_90)
        # Getting the type of 'count2' (line 32)
        count2_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'count2')
        # Getting the type of 'dig2' (line 32)
        dig2_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'dig2')
        # Storing an element on a container (line 32)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 8), count2_92, (dig2_93, result_iadd_91))
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a IfExp to a Name (line 34):
    
    # Assigning a IfExp to a Name (line 34):
    
    
    
    # Obtaining the type of the subscript
    int_94 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 33), 'int')
    # Getting the type of 'count1' (line 34)
    count1_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 26), 'count1')
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___96 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 26), count1_95, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_97 = invoke(stypy.reporting.localization.Localization(__file__, 34, 26), getitem___96, int_94)
    
    
    # Obtaining the type of the subscript
    int_98 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 45), 'int')
    # Getting the type of 'count2' (line 34)
    count2_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 38), 'count2')
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 38), count2_99, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_101 = invoke(stypy.reporting.localization.Localization(__file__, 34, 38), getitem___100, int_98)
    
    # Applying the binary operator '<' (line 34)
    result_lt_102 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 26), '<', subscript_call_result_97, subscript_call_result_101)
    
    # Testing the type of an if expression (line 34)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 34, 13), result_lt_102)
    # SSA begins for if expression (line 34)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Obtaining the type of the subscript
    int_103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 20), 'int')
    # Getting the type of 'count1' (line 34)
    count1_104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 13), 'count1')
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 13), count1_104, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_106 = invoke(stypy.reporting.localization.Localization(__file__, 34, 13), getitem___105, int_103)
    
    # SSA branch for the else part of an if expression (line 34)
    module_type_store.open_ssa_branch('if expression else')
    
    # Obtaining the type of the subscript
    int_107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 60), 'int')
    # Getting the type of 'count2' (line 34)
    count2_108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 53), 'count2')
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 53), count2_108, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_110 = invoke(stypy.reporting.localization.Localization(__file__, 34, 53), getitem___109, int_107)
    
    # SSA join for if expression (line 34)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_111 = union_type.UnionType.add(subscript_call_result_106, subscript_call_result_110)
    
    # Assigning a type to the variable 'balls' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'balls', if_exp_111)
    
    # Getting the type of 'balls' (line 35)
    balls_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'balls')
    
    
    
    # Obtaining the type of the subscript
    int_113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 34), 'int')
    # Getting the type of 'count1' (line 35)
    count1_114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 27), 'count1')
    # Obtaining the member '__getitem__' of a type (line 35)
    getitem___115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 27), count1_114, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 35)
    subscript_call_result_116 = invoke(stypy.reporting.localization.Localization(__file__, 35, 27), getitem___115, int_113)
    
    
    # Obtaining the type of the subscript
    int_117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 46), 'int')
    # Getting the type of 'count2' (line 35)
    count2_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 39), 'count2')
    # Obtaining the member '__getitem__' of a type (line 35)
    getitem___119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 39), count2_118, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 35)
    subscript_call_result_120 = invoke(stypy.reporting.localization.Localization(__file__, 35, 39), getitem___119, int_117)
    
    # Applying the binary operator '<' (line 35)
    result_lt_121 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 27), '<', subscript_call_result_116, subscript_call_result_120)
    
    # Testing the type of an if expression (line 35)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 14), result_lt_121)
    # SSA begins for if expression (line 35)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Obtaining the type of the subscript
    int_122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 21), 'int')
    # Getting the type of 'count1' (line 35)
    count1_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 14), 'count1')
    # Obtaining the member '__getitem__' of a type (line 35)
    getitem___124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 14), count1_123, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 35)
    subscript_call_result_125 = invoke(stypy.reporting.localization.Localization(__file__, 35, 14), getitem___124, int_122)
    
    # SSA branch for the else part of an if expression (line 35)
    module_type_store.open_ssa_branch('if expression else')
    
    # Obtaining the type of the subscript
    int_126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 61), 'int')
    # Getting the type of 'count2' (line 35)
    count2_127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 54), 'count2')
    # Obtaining the member '__getitem__' of a type (line 35)
    getitem___128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 54), count2_127, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 35)
    subscript_call_result_129 = invoke(stypy.reporting.localization.Localization(__file__, 35, 54), getitem___128, int_126)
    
    # SSA join for if expression (line 35)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_130 = union_type.UnionType.add(subscript_call_result_125, subscript_call_result_129)
    
    # Applying the binary operator '+=' (line 35)
    result_iadd_131 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 4), '+=', balls_112, if_exp_130)
    # Assigning a type to the variable 'balls' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'balls', result_iadd_131)
    
    
    # Getting the type of 'balls' (line 36)
    balls_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'balls')
    
    
    
    # Obtaining the type of the subscript
    int_133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 34), 'int')
    # Getting the type of 'count1' (line 36)
    count1_134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 27), 'count1')
    # Obtaining the member '__getitem__' of a type (line 36)
    getitem___135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 27), count1_134, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 36)
    subscript_call_result_136 = invoke(stypy.reporting.localization.Localization(__file__, 36, 27), getitem___135, int_133)
    
    
    # Obtaining the type of the subscript
    int_137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 46), 'int')
    # Getting the type of 'count2' (line 36)
    count2_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 39), 'count2')
    # Obtaining the member '__getitem__' of a type (line 36)
    getitem___139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 39), count2_138, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 36)
    subscript_call_result_140 = invoke(stypy.reporting.localization.Localization(__file__, 36, 39), getitem___139, int_137)
    
    # Applying the binary operator '<' (line 36)
    result_lt_141 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 27), '<', subscript_call_result_136, subscript_call_result_140)
    
    # Testing the type of an if expression (line 36)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 14), result_lt_141)
    # SSA begins for if expression (line 36)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Obtaining the type of the subscript
    int_142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 21), 'int')
    # Getting the type of 'count1' (line 36)
    count1_143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 14), 'count1')
    # Obtaining the member '__getitem__' of a type (line 36)
    getitem___144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 14), count1_143, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 36)
    subscript_call_result_145 = invoke(stypy.reporting.localization.Localization(__file__, 36, 14), getitem___144, int_142)
    
    # SSA branch for the else part of an if expression (line 36)
    module_type_store.open_ssa_branch('if expression else')
    
    # Obtaining the type of the subscript
    int_146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 61), 'int')
    # Getting the type of 'count2' (line 36)
    count2_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 54), 'count2')
    # Obtaining the member '__getitem__' of a type (line 36)
    getitem___148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 54), count2_147, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 36)
    subscript_call_result_149 = invoke(stypy.reporting.localization.Localization(__file__, 36, 54), getitem___148, int_146)
    
    # SSA join for if expression (line 36)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_150 = union_type.UnionType.add(subscript_call_result_145, subscript_call_result_149)
    
    # Applying the binary operator '+=' (line 36)
    result_iadd_151 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 4), '+=', balls_132, if_exp_150)
    # Assigning a type to the variable 'balls' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'balls', result_iadd_151)
    
    
    # Getting the type of 'balls' (line 37)
    balls_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'balls')
    
    
    
    # Obtaining the type of the subscript
    int_153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 34), 'int')
    # Getting the type of 'count1' (line 37)
    count1_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 27), 'count1')
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 27), count1_154, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 37)
    subscript_call_result_156 = invoke(stypy.reporting.localization.Localization(__file__, 37, 27), getitem___155, int_153)
    
    
    # Obtaining the type of the subscript
    int_157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 46), 'int')
    # Getting the type of 'count2' (line 37)
    count2_158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 39), 'count2')
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 39), count2_158, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 37)
    subscript_call_result_160 = invoke(stypy.reporting.localization.Localization(__file__, 37, 39), getitem___159, int_157)
    
    # Applying the binary operator '<' (line 37)
    result_lt_161 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 27), '<', subscript_call_result_156, subscript_call_result_160)
    
    # Testing the type of an if expression (line 37)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 37, 14), result_lt_161)
    # SSA begins for if expression (line 37)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Obtaining the type of the subscript
    int_162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 21), 'int')
    # Getting the type of 'count1' (line 37)
    count1_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 14), 'count1')
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 14), count1_163, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 37)
    subscript_call_result_165 = invoke(stypy.reporting.localization.Localization(__file__, 37, 14), getitem___164, int_162)
    
    # SSA branch for the else part of an if expression (line 37)
    module_type_store.open_ssa_branch('if expression else')
    
    # Obtaining the type of the subscript
    int_166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 61), 'int')
    # Getting the type of 'count2' (line 37)
    count2_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 54), 'count2')
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 54), count2_167, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 37)
    subscript_call_result_169 = invoke(stypy.reporting.localization.Localization(__file__, 37, 54), getitem___168, int_166)
    
    # SSA join for if expression (line 37)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_170 = union_type.UnionType.add(subscript_call_result_165, subscript_call_result_169)
    
    # Applying the binary operator '+=' (line 37)
    result_iadd_171 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 4), '+=', balls_152, if_exp_170)
    # Assigning a type to the variable 'balls' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'balls', result_iadd_171)
    
    
    # Getting the type of 'balls' (line 38)
    balls_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'balls')
    
    
    
    # Obtaining the type of the subscript
    int_173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 34), 'int')
    # Getting the type of 'count1' (line 38)
    count1_174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 27), 'count1')
    # Obtaining the member '__getitem__' of a type (line 38)
    getitem___175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 27), count1_174, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 38)
    subscript_call_result_176 = invoke(stypy.reporting.localization.Localization(__file__, 38, 27), getitem___175, int_173)
    
    
    # Obtaining the type of the subscript
    int_177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 46), 'int')
    # Getting the type of 'count2' (line 38)
    count2_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 39), 'count2')
    # Obtaining the member '__getitem__' of a type (line 38)
    getitem___179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 39), count2_178, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 38)
    subscript_call_result_180 = invoke(stypy.reporting.localization.Localization(__file__, 38, 39), getitem___179, int_177)
    
    # Applying the binary operator '<' (line 38)
    result_lt_181 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 27), '<', subscript_call_result_176, subscript_call_result_180)
    
    # Testing the type of an if expression (line 38)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 14), result_lt_181)
    # SSA begins for if expression (line 38)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Obtaining the type of the subscript
    int_182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 21), 'int')
    # Getting the type of 'count1' (line 38)
    count1_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 14), 'count1')
    # Obtaining the member '__getitem__' of a type (line 38)
    getitem___184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 14), count1_183, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 38)
    subscript_call_result_185 = invoke(stypy.reporting.localization.Localization(__file__, 38, 14), getitem___184, int_182)
    
    # SSA branch for the else part of an if expression (line 38)
    module_type_store.open_ssa_branch('if expression else')
    
    # Obtaining the type of the subscript
    int_186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 61), 'int')
    # Getting the type of 'count2' (line 38)
    count2_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 54), 'count2')
    # Obtaining the member '__getitem__' of a type (line 38)
    getitem___188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 54), count2_187, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 38)
    subscript_call_result_189 = invoke(stypy.reporting.localization.Localization(__file__, 38, 54), getitem___188, int_186)
    
    # SSA join for if expression (line 38)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_190 = union_type.UnionType.add(subscript_call_result_185, subscript_call_result_189)
    
    # Applying the binary operator '+=' (line 38)
    result_iadd_191 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 4), '+=', balls_172, if_exp_190)
    # Assigning a type to the variable 'balls' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'balls', result_iadd_191)
    
    
    # Getting the type of 'balls' (line 39)
    balls_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'balls')
    
    
    
    # Obtaining the type of the subscript
    int_193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 34), 'int')
    # Getting the type of 'count1' (line 39)
    count1_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 27), 'count1')
    # Obtaining the member '__getitem__' of a type (line 39)
    getitem___195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 27), count1_194, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 39)
    subscript_call_result_196 = invoke(stypy.reporting.localization.Localization(__file__, 39, 27), getitem___195, int_193)
    
    
    # Obtaining the type of the subscript
    int_197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 46), 'int')
    # Getting the type of 'count2' (line 39)
    count2_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 39), 'count2')
    # Obtaining the member '__getitem__' of a type (line 39)
    getitem___199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 39), count2_198, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 39)
    subscript_call_result_200 = invoke(stypy.reporting.localization.Localization(__file__, 39, 39), getitem___199, int_197)
    
    # Applying the binary operator '<' (line 39)
    result_lt_201 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 27), '<', subscript_call_result_196, subscript_call_result_200)
    
    # Testing the type of an if expression (line 39)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 14), result_lt_201)
    # SSA begins for if expression (line 39)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Obtaining the type of the subscript
    int_202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 21), 'int')
    # Getting the type of 'count1' (line 39)
    count1_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 14), 'count1')
    # Obtaining the member '__getitem__' of a type (line 39)
    getitem___204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 14), count1_203, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 39)
    subscript_call_result_205 = invoke(stypy.reporting.localization.Localization(__file__, 39, 14), getitem___204, int_202)
    
    # SSA branch for the else part of an if expression (line 39)
    module_type_store.open_ssa_branch('if expression else')
    
    # Obtaining the type of the subscript
    int_206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 61), 'int')
    # Getting the type of 'count2' (line 39)
    count2_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 54), 'count2')
    # Obtaining the member '__getitem__' of a type (line 39)
    getitem___208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 54), count2_207, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 39)
    subscript_call_result_209 = invoke(stypy.reporting.localization.Localization(__file__, 39, 54), getitem___208, int_206)
    
    # SSA join for if expression (line 39)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_210 = union_type.UnionType.add(subscript_call_result_205, subscript_call_result_209)
    
    # Applying the binary operator '+=' (line 39)
    result_iadd_211 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 4), '+=', balls_192, if_exp_210)
    # Assigning a type to the variable 'balls' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'balls', result_iadd_211)
    
    
    # Getting the type of 'balls' (line 40)
    balls_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'balls')
    
    
    
    # Obtaining the type of the subscript
    int_213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 34), 'int')
    # Getting the type of 'count1' (line 40)
    count1_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 27), 'count1')
    # Obtaining the member '__getitem__' of a type (line 40)
    getitem___215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 27), count1_214, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
    subscript_call_result_216 = invoke(stypy.reporting.localization.Localization(__file__, 40, 27), getitem___215, int_213)
    
    
    # Obtaining the type of the subscript
    int_217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 46), 'int')
    # Getting the type of 'count2' (line 40)
    count2_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 39), 'count2')
    # Obtaining the member '__getitem__' of a type (line 40)
    getitem___219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 39), count2_218, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
    subscript_call_result_220 = invoke(stypy.reporting.localization.Localization(__file__, 40, 39), getitem___219, int_217)
    
    # Applying the binary operator '<' (line 40)
    result_lt_221 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 27), '<', subscript_call_result_216, subscript_call_result_220)
    
    # Testing the type of an if expression (line 40)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 14), result_lt_221)
    # SSA begins for if expression (line 40)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Obtaining the type of the subscript
    int_222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 21), 'int')
    # Getting the type of 'count1' (line 40)
    count1_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 14), 'count1')
    # Obtaining the member '__getitem__' of a type (line 40)
    getitem___224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 14), count1_223, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
    subscript_call_result_225 = invoke(stypy.reporting.localization.Localization(__file__, 40, 14), getitem___224, int_222)
    
    # SSA branch for the else part of an if expression (line 40)
    module_type_store.open_ssa_branch('if expression else')
    
    # Obtaining the type of the subscript
    int_226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 61), 'int')
    # Getting the type of 'count2' (line 40)
    count2_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 54), 'count2')
    # Obtaining the member '__getitem__' of a type (line 40)
    getitem___228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 54), count2_227, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
    subscript_call_result_229 = invoke(stypy.reporting.localization.Localization(__file__, 40, 54), getitem___228, int_226)
    
    # SSA join for if expression (line 40)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_230 = union_type.UnionType.add(subscript_call_result_225, subscript_call_result_229)
    
    # Applying the binary operator '+=' (line 40)
    result_iadd_231 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 4), '+=', balls_212, if_exp_230)
    # Assigning a type to the variable 'balls' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'balls', result_iadd_231)
    
    
    # Getting the type of 'balls' (line 41)
    balls_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'balls')
    
    
    
    # Obtaining the type of the subscript
    int_233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 34), 'int')
    # Getting the type of 'count1' (line 41)
    count1_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 27), 'count1')
    # Obtaining the member '__getitem__' of a type (line 41)
    getitem___235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 27), count1_234, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 41)
    subscript_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 41, 27), getitem___235, int_233)
    
    
    # Obtaining the type of the subscript
    int_237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 46), 'int')
    # Getting the type of 'count2' (line 41)
    count2_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 39), 'count2')
    # Obtaining the member '__getitem__' of a type (line 41)
    getitem___239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 39), count2_238, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 41)
    subscript_call_result_240 = invoke(stypy.reporting.localization.Localization(__file__, 41, 39), getitem___239, int_237)
    
    # Applying the binary operator '<' (line 41)
    result_lt_241 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 27), '<', subscript_call_result_236, subscript_call_result_240)
    
    # Testing the type of an if expression (line 41)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 14), result_lt_241)
    # SSA begins for if expression (line 41)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Obtaining the type of the subscript
    int_242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 21), 'int')
    # Getting the type of 'count1' (line 41)
    count1_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 14), 'count1')
    # Obtaining the member '__getitem__' of a type (line 41)
    getitem___244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 14), count1_243, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 41)
    subscript_call_result_245 = invoke(stypy.reporting.localization.Localization(__file__, 41, 14), getitem___244, int_242)
    
    # SSA branch for the else part of an if expression (line 41)
    module_type_store.open_ssa_branch('if expression else')
    
    # Obtaining the type of the subscript
    int_246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 61), 'int')
    # Getting the type of 'count2' (line 41)
    count2_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 54), 'count2')
    # Obtaining the member '__getitem__' of a type (line 41)
    getitem___248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 54), count2_247, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 41)
    subscript_call_result_249 = invoke(stypy.reporting.localization.Localization(__file__, 41, 54), getitem___248, int_246)
    
    # SSA join for if expression (line 41)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_250 = union_type.UnionType.add(subscript_call_result_245, subscript_call_result_249)
    
    # Applying the binary operator '+=' (line 41)
    result_iadd_251 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 4), '+=', balls_232, if_exp_250)
    # Assigning a type to the variable 'balls' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'balls', result_iadd_251)
    
    
    # Getting the type of 'balls' (line 42)
    balls_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'balls')
    
    
    
    # Obtaining the type of the subscript
    int_253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 34), 'int')
    # Getting the type of 'count1' (line 42)
    count1_254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 27), 'count1')
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 27), count1_254, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_256 = invoke(stypy.reporting.localization.Localization(__file__, 42, 27), getitem___255, int_253)
    
    
    # Obtaining the type of the subscript
    int_257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 46), 'int')
    # Getting the type of 'count2' (line 42)
    count2_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 39), 'count2')
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 39), count2_258, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_260 = invoke(stypy.reporting.localization.Localization(__file__, 42, 39), getitem___259, int_257)
    
    # Applying the binary operator '<' (line 42)
    result_lt_261 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 27), '<', subscript_call_result_256, subscript_call_result_260)
    
    # Testing the type of an if expression (line 42)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 14), result_lt_261)
    # SSA begins for if expression (line 42)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Obtaining the type of the subscript
    int_262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 21), 'int')
    # Getting the type of 'count1' (line 42)
    count1_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 14), 'count1')
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 14), count1_263, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_265 = invoke(stypy.reporting.localization.Localization(__file__, 42, 14), getitem___264, int_262)
    
    # SSA branch for the else part of an if expression (line 42)
    module_type_store.open_ssa_branch('if expression else')
    
    # Obtaining the type of the subscript
    int_266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 61), 'int')
    # Getting the type of 'count2' (line 42)
    count2_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 54), 'count2')
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 54), count2_267, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_269 = invoke(stypy.reporting.localization.Localization(__file__, 42, 54), getitem___268, int_266)
    
    # SSA join for if expression (line 42)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_270 = union_type.UnionType.add(subscript_call_result_265, subscript_call_result_269)
    
    # Applying the binary operator '+=' (line 42)
    result_iadd_271 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 4), '+=', balls_252, if_exp_270)
    # Assigning a type to the variable 'balls' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'balls', result_iadd_271)
    
    
    # Getting the type of 'balls' (line 43)
    balls_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'balls')
    
    
    
    # Obtaining the type of the subscript
    int_273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 34), 'int')
    # Getting the type of 'count1' (line 43)
    count1_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 27), 'count1')
    # Obtaining the member '__getitem__' of a type (line 43)
    getitem___275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 27), count1_274, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 43)
    subscript_call_result_276 = invoke(stypy.reporting.localization.Localization(__file__, 43, 27), getitem___275, int_273)
    
    
    # Obtaining the type of the subscript
    int_277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 46), 'int')
    # Getting the type of 'count2' (line 43)
    count2_278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 39), 'count2')
    # Obtaining the member '__getitem__' of a type (line 43)
    getitem___279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 39), count2_278, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 43)
    subscript_call_result_280 = invoke(stypy.reporting.localization.Localization(__file__, 43, 39), getitem___279, int_277)
    
    # Applying the binary operator '<' (line 43)
    result_lt_281 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 27), '<', subscript_call_result_276, subscript_call_result_280)
    
    # Testing the type of an if expression (line 43)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 14), result_lt_281)
    # SSA begins for if expression (line 43)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Obtaining the type of the subscript
    int_282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 21), 'int')
    # Getting the type of 'count1' (line 43)
    count1_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 14), 'count1')
    # Obtaining the member '__getitem__' of a type (line 43)
    getitem___284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 14), count1_283, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 43)
    subscript_call_result_285 = invoke(stypy.reporting.localization.Localization(__file__, 43, 14), getitem___284, int_282)
    
    # SSA branch for the else part of an if expression (line 43)
    module_type_store.open_ssa_branch('if expression else')
    
    # Obtaining the type of the subscript
    int_286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 61), 'int')
    # Getting the type of 'count2' (line 43)
    count2_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 54), 'count2')
    # Obtaining the member '__getitem__' of a type (line 43)
    getitem___288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 54), count2_287, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 43)
    subscript_call_result_289 = invoke(stypy.reporting.localization.Localization(__file__, 43, 54), getitem___288, int_286)
    
    # SSA join for if expression (line 43)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_290 = union_type.UnionType.add(subscript_call_result_285, subscript_call_result_289)
    
    # Applying the binary operator '+=' (line 43)
    result_iadd_291 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 4), '+=', balls_272, if_exp_290)
    # Assigning a type to the variable 'balls' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'balls', result_iadd_291)
    
    # Getting the type of 'strikes' (line 45)
    strikes_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'strikes')
    int_293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 23), 'int')
    # Applying the binary operator '<<' (line 45)
    result_lshift_294 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 12), '<<', strikes_292, int_293)
    
    # Getting the type of 'balls' (line 45)
    balls_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 30), 'balls')
    # Getting the type of 'strikes' (line 45)
    strikes_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 38), 'strikes')
    # Applying the binary operator '-' (line 45)
    result_sub_297 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 30), '-', balls_295, strikes_296)
    
    # Applying the binary operator '|' (line 45)
    result_or__298 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 11), '|', result_lshift_294, result_sub_297)
    
    # Assigning a type to the variable 'stypy_return_type' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type', result_or__298)
    
    # ################# End of 'compare(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'compare' in the type store
    # Getting the type of 'stypy_return_type' (line 18)
    stypy_return_type_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_299)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'compare'
    return stypy_return_type_299

# Assigning a type to the variable 'compare' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'compare', compare)

@norecursion
def rungame(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 48)
    True_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 38), 'True')
    int_301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 53), 'int')
    defaults = [True_300, int_301]
    # Create a new context for function 'rungame'
    module_type_store = module_type_store.open_function_context('rungame', 48, 0, False)
    
    # Passed parameters checking function
    rungame.stypy_localization = localization
    rungame.stypy_type_of_self = None
    rungame.stypy_type_store = module_type_store
    rungame.stypy_function_name = 'rungame'
    rungame.stypy_param_names_list = ['target', 'strategy', 'verbose', 'maxtries']
    rungame.stypy_varargs_param_name = None
    rungame.stypy_kwargs_param_name = None
    rungame.stypy_call_defaults = defaults
    rungame.stypy_call_varargs = varargs
    rungame.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rungame', ['target', 'strategy', 'verbose', 'maxtries'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rungame', localization, ['target', 'strategy', 'verbose', 'maxtries'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rungame(...)' code ##################

    
    # Assigning a Name to a Name (line 49):
    
    # Assigning a Name to a Name (line 49):
    # Getting the type of 'searchspace' (line 49)
    searchspace_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'searchspace')
    # Assigning a type to the variable 'possibles' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'possibles', searchspace_302)
    
    
    # Call to xrange(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'maxtries' (line 50)
    maxtries_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 20), 'maxtries', False)
    # Processing the call keyword arguments (line 50)
    kwargs_305 = {}
    # Getting the type of 'xrange' (line 50)
    xrange_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 50)
    xrange_call_result_306 = invoke(stypy.reporting.localization.Localization(__file__, 50, 13), xrange_303, *[maxtries_304], **kwargs_305)
    
    # Testing if the for loop is going to be iterated (line 50)
    # Testing the type of a for loop iterable (line 50)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 50, 4), xrange_call_result_306)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 50, 4), xrange_call_result_306):
        # Getting the type of the for loop variable (line 50)
        for_loop_var_307 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 50, 4), xrange_call_result_306)
        # Assigning a type to the variable 'i' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'i', for_loop_var_307)
        # SSA begins for a for statement (line 50)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 51):
        
        # Assigning a Call to a Name (line 51):
        
        # Call to strategy(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'i' (line 51)
        i_309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 21), 'i', False)
        # Getting the type of 'possibles' (line 51)
        possibles_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 24), 'possibles', False)
        # Processing the call keyword arguments (line 51)
        kwargs_311 = {}
        # Getting the type of 'strategy' (line 51)
        strategy_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'strategy', False)
        # Calling strategy(args, kwargs) (line 51)
        strategy_call_result_312 = invoke(stypy.reporting.localization.Localization(__file__, 51, 12), strategy_308, *[i_309, possibles_310], **kwargs_311)
        
        # Assigning a type to the variable 'g' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'g', strategy_call_result_312)
        # Getting the type of 'verbose' (line 52)
        verbose_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'verbose')
        # Testing if the type of an if condition is none (line 52)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 52, 8), verbose_313):
            pass
        else:
            
            # Testing the type of an if condition (line 52)
            if_condition_314 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 52, 8), verbose_313)
            # Assigning a type to the variable 'if_condition_314' (line 52)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'if_condition_314', if_condition_314)
            # SSA begins for if statement (line 52)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            pass
            # SSA join for if statement (line 52)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 55):
        
        # Assigning a Call to a Name (line 55):
        
        # Call to compare(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'g' (line 55)
        g_316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 24), 'g', False)
        # Getting the type of 'target' (line 55)
        target_317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 27), 'target', False)
        # Processing the call keyword arguments (line 55)
        kwargs_318 = {}
        # Getting the type of 'compare' (line 55)
        compare_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'compare', False)
        # Calling compare(args, kwargs) (line 55)
        compare_call_result_319 = invoke(stypy.reporting.localization.Localization(__file__, 55, 16), compare_315, *[g_316, target_317], **kwargs_318)
        
        # Assigning a type to the variable 'score' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'score', compare_call_result_319)
        # Getting the type of 'verbose' (line 56)
        verbose_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 11), 'verbose')
        # Testing if the type of an if condition is none (line 56)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 56, 8), verbose_320):
            pass
        else:
            
            # Testing the type of an if condition (line 56)
            if_condition_321 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 56, 8), verbose_320)
            # Assigning a type to the variable 'if_condition_321' (line 56)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'if_condition_321', if_condition_321)
            # SSA begins for if statement (line 56)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            pass
            # SSA join for if statement (line 56)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'score' (line 58)
        score_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'score')
        int_323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 21), 'int')
        # Applying the binary operator '>>' (line 58)
        result_rshift_324 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 12), '>>', score_322, int_323)
        
        # Getting the type of 'DIGITS' (line 58)
        DIGITS_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 28), 'DIGITS')
        # Applying the binary operator '==' (line 58)
        result_eq_326 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 11), '==', result_rshift_324, DIGITS_325)
        
        # Testing if the type of an if condition is none (line 58)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 58, 8), result_eq_326):
            pass
        else:
            
            # Testing the type of an if condition (line 58)
            if_condition_327 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 8), result_eq_326)
            # Assigning a type to the variable 'if_condition_327' (line 58)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'if_condition_327', if_condition_327)
            # SSA begins for if statement (line 58)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'verbose' (line 59)
            verbose_328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'verbose')
            # Testing if the type of an if condition is none (line 59)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 59, 12), verbose_328):
                pass
            else:
                
                # Testing the type of an if condition (line 59)
                if_condition_329 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 12), verbose_328)
                # Assigning a type to the variable 'if_condition_329' (line 59)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'if_condition_329', if_condition_329)
                # SSA begins for if statement (line 59)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                pass
                # SSA join for if statement (line 59)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 58)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a ListComp to a Name (line 62):
        
        # Assigning a ListComp to a Name (line 62):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'possibles' (line 62)
        possibles_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 32), 'possibles')
        comprehension_339 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 21), possibles_338)
        # Assigning a type to the variable 'n' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 21), 'n', comprehension_339)
        
        
        # Call to compare(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'g' (line 62)
        g_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 53), 'g', False)
        # Getting the type of 'n' (line 62)
        n_333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 56), 'n', False)
        # Processing the call keyword arguments (line 62)
        kwargs_334 = {}
        # Getting the type of 'compare' (line 62)
        compare_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 45), 'compare', False)
        # Calling compare(args, kwargs) (line 62)
        compare_call_result_335 = invoke(stypy.reporting.localization.Localization(__file__, 62, 45), compare_331, *[g_332, n_333], **kwargs_334)
        
        # Getting the type of 'score' (line 62)
        score_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 62), 'score')
        # Applying the binary operator '==' (line 62)
        result_eq_337 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 45), '==', compare_call_result_335, score_336)
        
        # Getting the type of 'n' (line 62)
        n_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 21), 'n')
        list_340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 21), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 21), list_340, n_330)
        # Assigning a type to the variable 'possibles' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'possibles', list_340)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'i' (line 63)
    i_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 11), 'i')
    int_342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 15), 'int')
    # Applying the binary operator '+' (line 63)
    result_add_343 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 11), '+', i_341, int_342)
    
    # Assigning a type to the variable 'stypy_return_type' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'stypy_return_type', result_add_343)
    
    # ################# End of 'rungame(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rungame' in the type store
    # Getting the type of 'stypy_return_type' (line 48)
    stypy_return_type_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_344)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rungame'
    return stypy_return_type_344

# Assigning a type to the variable 'rungame' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'rungame', rungame)

@norecursion
def utility(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'utility'
    module_type_store = module_type_store.open_function_context('utility', 68, 0, False)
    
    # Passed parameters checking function
    utility.stypy_localization = localization
    utility.stypy_type_of_self = None
    utility.stypy_type_store = module_type_store
    utility.stypy_function_name = 'utility'
    utility.stypy_param_names_list = ['play', 'possibles']
    utility.stypy_varargs_param_name = None
    utility.stypy_kwargs_param_name = None
    utility.stypy_call_defaults = defaults
    utility.stypy_call_varargs = varargs
    utility.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'utility', ['play', 'possibles'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'utility', localization, ['play', 'possibles'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'utility(...)' code ##################

    
    # Assigning a Call to a Name (line 69):
    
    # Assigning a Call to a Name (line 69):
    
    # Call to defaultdict(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'int' (line 69)
    int_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 20), 'int', False)
    # Processing the call keyword arguments (line 69)
    kwargs_347 = {}
    # Getting the type of 'defaultdict' (line 69)
    defaultdict_345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'defaultdict', False)
    # Calling defaultdict(args, kwargs) (line 69)
    defaultdict_call_result_348 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), defaultdict_345, *[int_346], **kwargs_347)
    
    # Assigning a type to the variable 'b' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'b', defaultdict_call_result_348)
    
    # Getting the type of 'possibles' (line 70)
    possibles_349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'possibles')
    # Testing if the for loop is going to be iterated (line 70)
    # Testing the type of a for loop iterable (line 70)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 70, 4), possibles_349)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 70, 4), possibles_349):
        # Getting the type of the for loop variable (line 70)
        for_loop_var_350 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 70, 4), possibles_349)
        # Assigning a type to the variable 'poss' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'poss', for_loop_var_350)
        # SSA begins for a for statement (line 70)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'b' (line 71)
        b_351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'b')
        
        # Obtaining the type of the subscript
        
        # Call to compare(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'play' (line 71)
        play_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 18), 'play', False)
        # Getting the type of 'poss' (line 71)
        poss_354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 24), 'poss', False)
        # Processing the call keyword arguments (line 71)
        kwargs_355 = {}
        # Getting the type of 'compare' (line 71)
        compare_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 10), 'compare', False)
        # Calling compare(args, kwargs) (line 71)
        compare_call_result_356 = invoke(stypy.reporting.localization.Localization(__file__, 71, 10), compare_352, *[play_353, poss_354], **kwargs_355)
        
        # Getting the type of 'b' (line 71)
        b_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'b')
        # Obtaining the member '__getitem__' of a type (line 71)
        getitem___358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), b_357, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 71)
        subscript_call_result_359 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), getitem___358, compare_call_result_356)
        
        int_360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 34), 'int')
        # Applying the binary operator '+=' (line 71)
        result_iadd_361 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 8), '+=', subscript_call_result_359, int_360)
        # Getting the type of 'b' (line 71)
        b_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'b')
        
        # Call to compare(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'play' (line 71)
        play_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 18), 'play', False)
        # Getting the type of 'poss' (line 71)
        poss_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 24), 'poss', False)
        # Processing the call keyword arguments (line 71)
        kwargs_366 = {}
        # Getting the type of 'compare' (line 71)
        compare_363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 10), 'compare', False)
        # Calling compare(args, kwargs) (line 71)
        compare_call_result_367 = invoke(stypy.reporting.localization.Localization(__file__, 71, 10), compare_363, *[play_364, poss_365], **kwargs_366)
        
        # Storing an element on a container (line 71)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 8), b_362, (compare_call_result_367, result_iadd_361))
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a Num to a Name (line 74):
    
    # Assigning a Num to a Name (line 74):
    int_368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 11), 'int')
    # Assigning a type to the variable 'bits' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'bits', int_368)
    
    # Assigning a Call to a Name (line 75):
    
    # Assigning a Call to a Name (line 75):
    
    # Call to float(...): (line 75)
    # Processing the call arguments (line 75)
    
    # Call to len(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'possibles' (line 75)
    possibles_371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 18), 'possibles', False)
    # Processing the call keyword arguments (line 75)
    kwargs_372 = {}
    # Getting the type of 'len' (line 75)
    len_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 14), 'len', False)
    # Calling len(args, kwargs) (line 75)
    len_call_result_373 = invoke(stypy.reporting.localization.Localization(__file__, 75, 14), len_370, *[possibles_371], **kwargs_372)
    
    # Processing the call keyword arguments (line 75)
    kwargs_374 = {}
    # Getting the type of 'float' (line 75)
    float_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'float', False)
    # Calling float(args, kwargs) (line 75)
    float_call_result_375 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), float_369, *[len_call_result_373], **kwargs_374)
    
    # Assigning a type to the variable 's' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 's', float_call_result_375)
    
    
    # Call to itervalues(...): (line 76)
    # Processing the call keyword arguments (line 76)
    kwargs_378 = {}
    # Getting the type of 'b' (line 76)
    b_376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 13), 'b', False)
    # Obtaining the member 'itervalues' of a type (line 76)
    itervalues_377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 13), b_376, 'itervalues')
    # Calling itervalues(args, kwargs) (line 76)
    itervalues_call_result_379 = invoke(stypy.reporting.localization.Localization(__file__, 76, 13), itervalues_377, *[], **kwargs_378)
    
    # Testing if the for loop is going to be iterated (line 76)
    # Testing the type of a for loop iterable (line 76)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 76, 4), itervalues_call_result_379)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 76, 4), itervalues_call_result_379):
        # Getting the type of the for loop variable (line 76)
        for_loop_var_380 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 76, 4), itervalues_call_result_379)
        # Assigning a type to the variable 'i' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'i', for_loop_var_380)
        # SSA begins for a for statement (line 76)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 77):
        
        # Assigning a BinOp to a Name (line 77):
        # Getting the type of 'i' (line 77)
        i_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'i')
        # Getting the type of 's' (line 77)
        s_382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 16), 's')
        # Applying the binary operator 'div' (line 77)
        result_div_383 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 12), 'div', i_381, s_382)
        
        # Assigning a type to the variable 'p' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'p', result_div_383)
        
        # Getting the type of 'bits' (line 78)
        bits_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'bits')
        # Getting the type of 'p' (line 78)
        p_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'p')
        
        # Call to log(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'p' (line 78)
        p_387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 24), 'p', False)
        int_388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 27), 'int')
        # Processing the call keyword arguments (line 78)
        kwargs_389 = {}
        # Getting the type of 'log' (line 78)
        log_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 20), 'log', False)
        # Calling log(args, kwargs) (line 78)
        log_call_result_390 = invoke(stypy.reporting.localization.Localization(__file__, 78, 20), log_386, *[p_387, int_388], **kwargs_389)
        
        # Applying the binary operator '*' (line 78)
        result_mul_391 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 16), '*', p_385, log_call_result_390)
        
        # Applying the binary operator '-=' (line 78)
        result_isub_392 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 8), '-=', bits_384, result_mul_391)
        # Assigning a type to the variable 'bits' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'bits', result_isub_392)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'bits' (line 79)
    bits_393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), 'bits')
    # Assigning a type to the variable 'stypy_return_type' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'stypy_return_type', bits_393)
    
    # ################# End of 'utility(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'utility' in the type store
    # Getting the type of 'stypy_return_type' (line 68)
    stypy_return_type_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_394)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'utility'
    return stypy_return_type_394

# Assigning a type to the variable 'utility' (line 68)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'utility', utility)

@norecursion
def nodup(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'nodup'
    module_type_store = module_type_store.open_function_context('nodup', 82, 0, False)
    
    # Passed parameters checking function
    nodup.stypy_localization = localization
    nodup.stypy_type_of_self = None
    nodup.stypy_type_store = module_type_store
    nodup.stypy_function_name = 'nodup'
    nodup.stypy_param_names_list = ['play']
    nodup.stypy_varargs_param_name = None
    nodup.stypy_kwargs_param_name = None
    nodup.stypy_call_defaults = defaults
    nodup.stypy_call_varargs = varargs
    nodup.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'nodup', ['play'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'nodup', localization, ['play'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'nodup(...)' code ##################

    
    
    # Call to len(...): (line 83)
    # Processing the call arguments (line 83)
    
    # Call to set(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'play' (line 83)
    play_397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 19), 'play', False)
    # Processing the call keyword arguments (line 83)
    kwargs_398 = {}
    # Getting the type of 'set' (line 83)
    set_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 15), 'set', False)
    # Calling set(args, kwargs) (line 83)
    set_call_result_399 = invoke(stypy.reporting.localization.Localization(__file__, 83, 15), set_396, *[play_397], **kwargs_398)
    
    # Processing the call keyword arguments (line 83)
    kwargs_400 = {}
    # Getting the type of 'len' (line 83)
    len_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 11), 'len', False)
    # Calling len(args, kwargs) (line 83)
    len_call_result_401 = invoke(stypy.reporting.localization.Localization(__file__, 83, 11), len_395, *[set_call_result_399], **kwargs_400)
    
    # Getting the type of 'DIGITS' (line 83)
    DIGITS_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 29), 'DIGITS')
    # Applying the binary operator '==' (line 83)
    result_eq_403 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 11), '==', len_call_result_401, DIGITS_402)
    
    # Assigning a type to the variable 'stypy_return_type' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'stypy_return_type', result_eq_403)
    
    # ################# End of 'nodup(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'nodup' in the type store
    # Getting the type of 'stypy_return_type' (line 82)
    stypy_return_type_404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_404)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'nodup'
    return stypy_return_type_404

# Assigning a type to the variable 'nodup' (line 82)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'nodup', nodup)

@norecursion
def s_allrand(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 's_allrand'
    module_type_store = module_type_store.open_function_context('s_allrand', 88, 0, False)
    
    # Passed parameters checking function
    s_allrand.stypy_localization = localization
    s_allrand.stypy_type_of_self = None
    s_allrand.stypy_type_store = module_type_store
    s_allrand.stypy_function_name = 's_allrand'
    s_allrand.stypy_param_names_list = ['i', 'possibles']
    s_allrand.stypy_varargs_param_name = None
    s_allrand.stypy_kwargs_param_name = None
    s_allrand.stypy_call_defaults = defaults
    s_allrand.stypy_call_varargs = varargs
    s_allrand.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 's_allrand', ['i', 'possibles'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 's_allrand', localization, ['i', 'possibles'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 's_allrand(...)' code ##################

    
    # Call to choice(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'possibles' (line 89)
    possibles_407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 25), 'possibles', False)
    # Processing the call keyword arguments (line 89)
    kwargs_408 = {}
    # Getting the type of 'random' (line 89)
    random_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 11), 'random', False)
    # Obtaining the member 'choice' of a type (line 89)
    choice_406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 11), random_405, 'choice')
    # Calling choice(args, kwargs) (line 89)
    choice_call_result_409 = invoke(stypy.reporting.localization.Localization(__file__, 89, 11), choice_406, *[possibles_407], **kwargs_408)
    
    # Assigning a type to the variable 'stypy_return_type' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'stypy_return_type', choice_call_result_409)
    
    # ################# End of 's_allrand(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 's_allrand' in the type store
    # Getting the type of 'stypy_return_type' (line 88)
    stypy_return_type_410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_410)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 's_allrand'
    return stypy_return_type_410

# Assigning a type to the variable 's_allrand' (line 88)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 's_allrand', s_allrand)

@norecursion
def s_trynodup(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 's_trynodup'
    module_type_store = module_type_store.open_function_context('s_trynodup', 92, 0, False)
    
    # Passed parameters checking function
    s_trynodup.stypy_localization = localization
    s_trynodup.stypy_type_of_self = None
    s_trynodup.stypy_type_store = module_type_store
    s_trynodup.stypy_function_name = 's_trynodup'
    s_trynodup.stypy_param_names_list = ['i', 'possibles']
    s_trynodup.stypy_varargs_param_name = None
    s_trynodup.stypy_kwargs_param_name = None
    s_trynodup.stypy_call_defaults = defaults
    s_trynodup.stypy_call_varargs = varargs
    s_trynodup.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 's_trynodup', ['i', 'possibles'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 's_trynodup', localization, ['i', 'possibles'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 's_trynodup(...)' code ##################

    
    
    # Call to xrange(...): (line 93)
    # Processing the call arguments (line 93)
    int_412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 20), 'int')
    # Processing the call keyword arguments (line 93)
    kwargs_413 = {}
    # Getting the type of 'xrange' (line 93)
    xrange_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 93)
    xrange_call_result_414 = invoke(stypy.reporting.localization.Localization(__file__, 93, 13), xrange_411, *[int_412], **kwargs_413)
    
    # Testing if the for loop is going to be iterated (line 93)
    # Testing the type of a for loop iterable (line 93)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 93, 4), xrange_call_result_414)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 93, 4), xrange_call_result_414):
        # Getting the type of the for loop variable (line 93)
        for_loop_var_415 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 93, 4), xrange_call_result_414)
        # Assigning a type to the variable 'j' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'j', for_loop_var_415)
        # SSA begins for a for statement (line 93)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 94):
        
        # Assigning a Call to a Name (line 94):
        
        # Call to choice(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'possibles' (line 94)
        possibles_418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 26), 'possibles', False)
        # Processing the call keyword arguments (line 94)
        kwargs_419 = {}
        # Getting the type of 'random' (line 94)
        random_416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'random', False)
        # Obtaining the member 'choice' of a type (line 94)
        choice_417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), random_416, 'choice')
        # Calling choice(args, kwargs) (line 94)
        choice_call_result_420 = invoke(stypy.reporting.localization.Localization(__file__, 94, 12), choice_417, *[possibles_418], **kwargs_419)
        
        # Assigning a type to the variable 'g' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'g', choice_call_result_420)
        
        # Call to nodup(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'g' (line 95)
        g_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 17), 'g', False)
        # Processing the call keyword arguments (line 95)
        kwargs_423 = {}
        # Getting the type of 'nodup' (line 95)
        nodup_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 11), 'nodup', False)
        # Calling nodup(args, kwargs) (line 95)
        nodup_call_result_424 = invoke(stypy.reporting.localization.Localization(__file__, 95, 11), nodup_421, *[g_422], **kwargs_423)
        
        # Testing if the type of an if condition is none (line 95)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 95, 8), nodup_call_result_424):
            pass
        else:
            
            # Testing the type of an if condition (line 95)
            if_condition_425 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 8), nodup_call_result_424)
            # Assigning a type to the variable 'if_condition_425' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'if_condition_425', if_condition_425)
            # SSA begins for if statement (line 95)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 95)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'g' (line 97)
    g_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 'g')
    # Assigning a type to the variable 'stypy_return_type' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type', g_426)
    
    # ################# End of 's_trynodup(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 's_trynodup' in the type store
    # Getting the type of 'stypy_return_type' (line 92)
    stypy_return_type_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_427)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 's_trynodup'
    return stypy_return_type_427

# Assigning a type to the variable 's_trynodup' (line 92)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), 's_trynodup', s_trynodup)

@norecursion
def s_bestinfo(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 's_bestinfo'
    module_type_store = module_type_store.open_function_context('s_bestinfo', 100, 0, False)
    
    # Passed parameters checking function
    s_bestinfo.stypy_localization = localization
    s_bestinfo.stypy_type_of_self = None
    s_bestinfo.stypy_type_store = module_type_store
    s_bestinfo.stypy_function_name = 's_bestinfo'
    s_bestinfo.stypy_param_names_list = ['i', 'possibles']
    s_bestinfo.stypy_varargs_param_name = None
    s_bestinfo.stypy_kwargs_param_name = None
    s_bestinfo.stypy_call_defaults = defaults
    s_bestinfo.stypy_call_varargs = varargs
    s_bestinfo.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 's_bestinfo', ['i', 'possibles'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 's_bestinfo', localization, ['i', 'possibles'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 's_bestinfo(...)' code ##################

    
    # Getting the type of 'i' (line 101)
    i_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 7), 'i')
    int_429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 12), 'int')
    # Applying the binary operator '==' (line 101)
    result_eq_430 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 7), '==', i_428, int_429)
    
    # Testing if the type of an if condition is none (line 101)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 101, 4), result_eq_430):
        pass
    else:
        
        # Testing the type of an if condition (line 101)
        if_condition_431 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 4), result_eq_430)
        # Assigning a type to the variable 'if_condition_431' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'if_condition_431', if_condition_431)
        # SSA begins for if statement (line 101)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to s_trynodup(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'i' (line 102)
        i_433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 26), 'i', False)
        # Getting the type of 'possibles' (line 102)
        possibles_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 29), 'possibles', False)
        # Processing the call keyword arguments (line 102)
        kwargs_435 = {}
        # Getting the type of 's_trynodup' (line 102)
        s_trynodup_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 15), 's_trynodup', False)
        # Calling s_trynodup(args, kwargs) (line 102)
        s_trynodup_call_result_436 = invoke(stypy.reporting.localization.Localization(__file__, 102, 15), s_trynodup_432, *[i_433, possibles_434], **kwargs_435)
        
        # Assigning a type to the variable 'stypy_return_type' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'stypy_return_type', s_trynodup_call_result_436)
        # SSA join for if statement (line 101)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 103):
    
    # Assigning a Call to a Name (line 103):
    
    # Call to sample(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'possibles' (line 103)
    possibles_439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 26), 'possibles', False)
    
    # Call to min(...): (line 103)
    # Processing the call arguments (line 103)
    int_441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 41), 'int')
    
    # Call to len(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'possibles' (line 103)
    possibles_443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 49), 'possibles', False)
    # Processing the call keyword arguments (line 103)
    kwargs_444 = {}
    # Getting the type of 'len' (line 103)
    len_442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 45), 'len', False)
    # Calling len(args, kwargs) (line 103)
    len_call_result_445 = invoke(stypy.reporting.localization.Localization(__file__, 103, 45), len_442, *[possibles_443], **kwargs_444)
    
    # Processing the call keyword arguments (line 103)
    kwargs_446 = {}
    # Getting the type of 'min' (line 103)
    min_440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 37), 'min', False)
    # Calling min(args, kwargs) (line 103)
    min_call_result_447 = invoke(stypy.reporting.localization.Localization(__file__, 103, 37), min_440, *[int_441, len_call_result_445], **kwargs_446)
    
    # Processing the call keyword arguments (line 103)
    kwargs_448 = {}
    # Getting the type of 'random' (line 103)
    random_437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'random', False)
    # Obtaining the member 'sample' of a type (line 103)
    sample_438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), random_437, 'sample')
    # Calling sample(args, kwargs) (line 103)
    sample_call_result_449 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), sample_438, *[possibles_439, min_call_result_447], **kwargs_448)
    
    # Assigning a type to the variable 'plays' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'plays', sample_call_result_449)
    
    # Assigning a Call to a Tuple (line 104):
    
    # Assigning a Call to a Name:
    
    # Call to max(...): (line 104)
    # Processing the call arguments (line 104)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'plays' (line 104)
    plays_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 64), 'plays', False)
    comprehension_459 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 19), plays_458)
    # Assigning a type to the variable 'play' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 19), 'play', comprehension_459)
    
    # Obtaining an instance of the builtin type 'tuple' (line 104)
    tuple_451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 104)
    # Adding element type (line 104)
    
    # Call to utility(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'play' (line 104)
    play_453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 28), 'play', False)
    # Getting the type of 'possibles' (line 104)
    possibles_454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 34), 'possibles', False)
    # Processing the call keyword arguments (line 104)
    kwargs_455 = {}
    # Getting the type of 'utility' (line 104)
    utility_452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 20), 'utility', False)
    # Calling utility(args, kwargs) (line 104)
    utility_call_result_456 = invoke(stypy.reporting.localization.Localization(__file__, 104, 20), utility_452, *[play_453, possibles_454], **kwargs_455)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 20), tuple_451, utility_call_result_456)
    # Adding element type (line 104)
    # Getting the type of 'play' (line 104)
    play_457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 46), 'play', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 20), tuple_451, play_457)
    
    list_460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 19), list_460, tuple_451)
    # Processing the call keyword arguments (line 104)
    kwargs_461 = {}
    # Getting the type of 'max' (line 104)
    max_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 14), 'max', False)
    # Calling max(args, kwargs) (line 104)
    max_call_result_462 = invoke(stypy.reporting.localization.Localization(__file__, 104, 14), max_450, *[list_460], **kwargs_461)
    
    # Assigning a type to the variable 'call_assignment_1' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'call_assignment_1', max_call_result_462)
    
    # Assigning a Call to a Name (line 104):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 4), 'int')
    # Processing the call keyword arguments
    kwargs_466 = {}
    # Getting the type of 'call_assignment_1' (line 104)
    call_assignment_1_463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'call_assignment_1', False)
    # Obtaining the member '__getitem__' of a type (line 104)
    getitem___464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 4), call_assignment_1_463, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_467 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___464, *[int_465], **kwargs_466)
    
    # Assigning a type to the variable 'call_assignment_2' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'call_assignment_2', getitem___call_result_467)
    
    # Assigning a Name to a Name (line 104):
    # Getting the type of 'call_assignment_2' (line 104)
    call_assignment_2_468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'call_assignment_2')
    # Assigning a type to the variable '_' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), '_', call_assignment_2_468)
    
    # Assigning a Call to a Name (line 104):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 4), 'int')
    # Processing the call keyword arguments
    kwargs_472 = {}
    # Getting the type of 'call_assignment_1' (line 104)
    call_assignment_1_469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'call_assignment_1', False)
    # Obtaining the member '__getitem__' of a type (line 104)
    getitem___470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 4), call_assignment_1_469, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_473 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___470, *[int_471], **kwargs_472)
    
    # Assigning a type to the variable 'call_assignment_3' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'call_assignment_3', getitem___call_result_473)
    
    # Assigning a Name to a Name (line 104):
    # Getting the type of 'call_assignment_3' (line 104)
    call_assignment_3_474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'call_assignment_3')
    # Assigning a type to the variable 'play' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 7), 'play', call_assignment_3_474)
    # Getting the type of 'play' (line 105)
    play_475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 11), 'play')
    # Assigning a type to the variable 'stypy_return_type' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type', play_475)
    
    # ################# End of 's_bestinfo(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 's_bestinfo' in the type store
    # Getting the type of 'stypy_return_type' (line 100)
    stypy_return_type_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_476)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 's_bestinfo'
    return stypy_return_type_476

# Assigning a type to the variable 's_bestinfo' (line 100)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 's_bestinfo', s_bestinfo)

@norecursion
def s_worstinfo(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 's_worstinfo'
    module_type_store = module_type_store.open_function_context('s_worstinfo', 108, 0, False)
    
    # Passed parameters checking function
    s_worstinfo.stypy_localization = localization
    s_worstinfo.stypy_type_of_self = None
    s_worstinfo.stypy_type_store = module_type_store
    s_worstinfo.stypy_function_name = 's_worstinfo'
    s_worstinfo.stypy_param_names_list = ['i', 'possibles']
    s_worstinfo.stypy_varargs_param_name = None
    s_worstinfo.stypy_kwargs_param_name = None
    s_worstinfo.stypy_call_defaults = defaults
    s_worstinfo.stypy_call_varargs = varargs
    s_worstinfo.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 's_worstinfo', ['i', 'possibles'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 's_worstinfo', localization, ['i', 'possibles'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 's_worstinfo(...)' code ##################

    
    # Getting the type of 'i' (line 109)
    i_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 7), 'i')
    int_478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 12), 'int')
    # Applying the binary operator '==' (line 109)
    result_eq_479 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 7), '==', i_477, int_478)
    
    # Testing if the type of an if condition is none (line 109)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 109, 4), result_eq_479):
        pass
    else:
        
        # Testing the type of an if condition (line 109)
        if_condition_480 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 4), result_eq_479)
        # Assigning a type to the variable 'if_condition_480' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'if_condition_480', if_condition_480)
        # SSA begins for if statement (line 109)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to s_trynodup(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'i' (line 110)
        i_482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 26), 'i', False)
        # Getting the type of 'possibles' (line 110)
        possibles_483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 29), 'possibles', False)
        # Processing the call keyword arguments (line 110)
        kwargs_484 = {}
        # Getting the type of 's_trynodup' (line 110)
        s_trynodup_481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 15), 's_trynodup', False)
        # Calling s_trynodup(args, kwargs) (line 110)
        s_trynodup_call_result_485 = invoke(stypy.reporting.localization.Localization(__file__, 110, 15), s_trynodup_481, *[i_482, possibles_483], **kwargs_484)
        
        # Assigning a type to the variable 'stypy_return_type' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'stypy_return_type', s_trynodup_call_result_485)
        # SSA join for if statement (line 109)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 111):
    
    # Assigning a Call to a Name (line 111):
    
    # Call to sample(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'possibles' (line 111)
    possibles_488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 26), 'possibles', False)
    
    # Call to min(...): (line 111)
    # Processing the call arguments (line 111)
    int_490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 41), 'int')
    
    # Call to len(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'possibles' (line 111)
    possibles_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 49), 'possibles', False)
    # Processing the call keyword arguments (line 111)
    kwargs_493 = {}
    # Getting the type of 'len' (line 111)
    len_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 45), 'len', False)
    # Calling len(args, kwargs) (line 111)
    len_call_result_494 = invoke(stypy.reporting.localization.Localization(__file__, 111, 45), len_491, *[possibles_492], **kwargs_493)
    
    # Processing the call keyword arguments (line 111)
    kwargs_495 = {}
    # Getting the type of 'min' (line 111)
    min_489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 37), 'min', False)
    # Calling min(args, kwargs) (line 111)
    min_call_result_496 = invoke(stypy.reporting.localization.Localization(__file__, 111, 37), min_489, *[int_490, len_call_result_494], **kwargs_495)
    
    # Processing the call keyword arguments (line 111)
    kwargs_497 = {}
    # Getting the type of 'random' (line 111)
    random_486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'random', False)
    # Obtaining the member 'sample' of a type (line 111)
    sample_487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 12), random_486, 'sample')
    # Calling sample(args, kwargs) (line 111)
    sample_call_result_498 = invoke(stypy.reporting.localization.Localization(__file__, 111, 12), sample_487, *[possibles_488, min_call_result_496], **kwargs_497)
    
    # Assigning a type to the variable 'plays' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'plays', sample_call_result_498)
    
    # Assigning a Call to a Tuple (line 112):
    
    # Assigning a Call to a Name:
    
    # Call to min(...): (line 112)
    # Processing the call arguments (line 112)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'plays' (line 112)
    plays_507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 64), 'plays', False)
    comprehension_508 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 19), plays_507)
    # Assigning a type to the variable 'play' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 19), 'play', comprehension_508)
    
    # Obtaining an instance of the builtin type 'tuple' (line 112)
    tuple_500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 112)
    # Adding element type (line 112)
    
    # Call to utility(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'play' (line 112)
    play_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 28), 'play', False)
    # Getting the type of 'possibles' (line 112)
    possibles_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 34), 'possibles', False)
    # Processing the call keyword arguments (line 112)
    kwargs_504 = {}
    # Getting the type of 'utility' (line 112)
    utility_501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'utility', False)
    # Calling utility(args, kwargs) (line 112)
    utility_call_result_505 = invoke(stypy.reporting.localization.Localization(__file__, 112, 20), utility_501, *[play_502, possibles_503], **kwargs_504)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 20), tuple_500, utility_call_result_505)
    # Adding element type (line 112)
    # Getting the type of 'play' (line 112)
    play_506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 46), 'play', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 20), tuple_500, play_506)
    
    list_509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 19), list_509, tuple_500)
    # Processing the call keyword arguments (line 112)
    kwargs_510 = {}
    # Getting the type of 'min' (line 112)
    min_499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 14), 'min', False)
    # Calling min(args, kwargs) (line 112)
    min_call_result_511 = invoke(stypy.reporting.localization.Localization(__file__, 112, 14), min_499, *[list_509], **kwargs_510)
    
    # Assigning a type to the variable 'call_assignment_4' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'call_assignment_4', min_call_result_511)
    
    # Assigning a Call to a Name (line 112):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 4), 'int')
    # Processing the call keyword arguments
    kwargs_515 = {}
    # Getting the type of 'call_assignment_4' (line 112)
    call_assignment_4_512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'call_assignment_4', False)
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 4), call_assignment_4_512, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_516 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___513, *[int_514], **kwargs_515)
    
    # Assigning a type to the variable 'call_assignment_5' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'call_assignment_5', getitem___call_result_516)
    
    # Assigning a Name to a Name (line 112):
    # Getting the type of 'call_assignment_5' (line 112)
    call_assignment_5_517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'call_assignment_5')
    # Assigning a type to the variable '_' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), '_', call_assignment_5_517)
    
    # Assigning a Call to a Name (line 112):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 4), 'int')
    # Processing the call keyword arguments
    kwargs_521 = {}
    # Getting the type of 'call_assignment_4' (line 112)
    call_assignment_4_518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'call_assignment_4', False)
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 4), call_assignment_4_518, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_522 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___519, *[int_520], **kwargs_521)
    
    # Assigning a type to the variable 'call_assignment_6' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'call_assignment_6', getitem___call_result_522)
    
    # Assigning a Name to a Name (line 112):
    # Getting the type of 'call_assignment_6' (line 112)
    call_assignment_6_523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'call_assignment_6')
    # Assigning a type to the variable 'play' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 7), 'play', call_assignment_6_523)
    # Getting the type of 'play' (line 113)
    play_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 11), 'play')
    # Assigning a type to the variable 'stypy_return_type' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'stypy_return_type', play_524)
    
    # ################# End of 's_worstinfo(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 's_worstinfo' in the type store
    # Getting the type of 'stypy_return_type' (line 108)
    stypy_return_type_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_525)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 's_worstinfo'
    return stypy_return_type_525

# Assigning a type to the variable 's_worstinfo' (line 108)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 's_worstinfo', s_worstinfo)

@norecursion
def s_samplebest(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 's_samplebest'
    module_type_store = module_type_store.open_function_context('s_samplebest', 116, 0, False)
    
    # Passed parameters checking function
    s_samplebest.stypy_localization = localization
    s_samplebest.stypy_type_of_self = None
    s_samplebest.stypy_type_store = module_type_store
    s_samplebest.stypy_function_name = 's_samplebest'
    s_samplebest.stypy_param_names_list = ['i', 'possibles']
    s_samplebest.stypy_varargs_param_name = None
    s_samplebest.stypy_kwargs_param_name = None
    s_samplebest.stypy_call_defaults = defaults
    s_samplebest.stypy_call_varargs = varargs
    s_samplebest.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 's_samplebest', ['i', 'possibles'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 's_samplebest', localization, ['i', 'possibles'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 's_samplebest(...)' code ##################

    
    # Getting the type of 'i' (line 117)
    i_526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 7), 'i')
    int_527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 12), 'int')
    # Applying the binary operator '==' (line 117)
    result_eq_528 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 7), '==', i_526, int_527)
    
    # Testing if the type of an if condition is none (line 117)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 117, 4), result_eq_528):
        pass
    else:
        
        # Testing the type of an if condition (line 117)
        if_condition_529 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 117, 4), result_eq_528)
        # Assigning a type to the variable 'if_condition_529' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'if_condition_529', if_condition_529)
        # SSA begins for if statement (line 117)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to s_trynodup(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'i' (line 118)
        i_531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 26), 'i', False)
        # Getting the type of 'possibles' (line 118)
        possibles_532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 29), 'possibles', False)
        # Processing the call keyword arguments (line 118)
        kwargs_533 = {}
        # Getting the type of 's_trynodup' (line 118)
        s_trynodup_530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 's_trynodup', False)
        # Calling s_trynodup(args, kwargs) (line 118)
        s_trynodup_call_result_534 = invoke(stypy.reporting.localization.Localization(__file__, 118, 15), s_trynodup_530, *[i_531, possibles_532], **kwargs_533)
        
        # Assigning a type to the variable 'stypy_return_type' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'stypy_return_type', s_trynodup_call_result_534)
        # SSA join for if statement (line 117)
        module_type_store = module_type_store.join_ssa_context()
        

    
    
    # Call to len(...): (line 119)
    # Processing the call arguments (line 119)
    # Getting the type of 'possibles' (line 119)
    possibles_536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 11), 'possibles', False)
    # Processing the call keyword arguments (line 119)
    kwargs_537 = {}
    # Getting the type of 'len' (line 119)
    len_535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 7), 'len', False)
    # Calling len(args, kwargs) (line 119)
    len_call_result_538 = invoke(stypy.reporting.localization.Localization(__file__, 119, 7), len_535, *[possibles_536], **kwargs_537)
    
    int_539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 24), 'int')
    # Applying the binary operator '>' (line 119)
    result_gt_540 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 7), '>', len_call_result_538, int_539)
    
    # Testing if the type of an if condition is none (line 119)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 119, 4), result_gt_540):
        
        
        # Call to len(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'possibles' (line 122)
        possibles_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 13), 'possibles', False)
        # Processing the call keyword arguments (line 122)
        kwargs_555 = {}
        # Getting the type of 'len' (line 122)
        len_553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 9), 'len', False)
        # Calling len(args, kwargs) (line 122)
        len_call_result_556 = invoke(stypy.reporting.localization.Localization(__file__, 122, 9), len_553, *[possibles_554], **kwargs_555)
        
        int_557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 26), 'int')
        # Applying the binary operator '>' (line 122)
        result_gt_558 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 9), '>', len_call_result_556, int_557)
        
        # Testing if the type of an if condition is none (line 122)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 122, 9), result_gt_558):
            
            # Assigning a Name to a Name (line 125):
            
            # Assigning a Name to a Name (line 125):
            # Getting the type of 'possibles' (line 125)
            possibles_566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'possibles')
            # Assigning a type to the variable 'plays' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'plays', possibles_566)
        else:
            
            # Testing the type of an if condition (line 122)
            if_condition_559 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 9), result_gt_558)
            # Assigning a type to the variable 'if_condition_559' (line 122)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 9), 'if_condition_559', if_condition_559)
            # SSA begins for if statement (line 122)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 123):
            
            # Assigning a Call to a Name (line 123):
            
            # Call to sample(...): (line 123)
            # Processing the call arguments (line 123)
            # Getting the type of 'possibles' (line 123)
            possibles_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 30), 'possibles', False)
            int_563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 41), 'int')
            # Processing the call keyword arguments (line 123)
            kwargs_564 = {}
            # Getting the type of 'random' (line 123)
            random_560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'random', False)
            # Obtaining the member 'sample' of a type (line 123)
            sample_561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 16), random_560, 'sample')
            # Calling sample(args, kwargs) (line 123)
            sample_call_result_565 = invoke(stypy.reporting.localization.Localization(__file__, 123, 16), sample_561, *[possibles_562, int_563], **kwargs_564)
            
            # Assigning a type to the variable 'plays' (line 123)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'plays', sample_call_result_565)
            # SSA branch for the else part of an if statement (line 122)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Name (line 125):
            
            # Assigning a Name to a Name (line 125):
            # Getting the type of 'possibles' (line 125)
            possibles_566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'possibles')
            # Assigning a type to the variable 'plays' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'plays', possibles_566)
            # SSA join for if statement (line 122)
            module_type_store = module_type_store.join_ssa_context()
            

    else:
        
        # Testing the type of an if condition (line 119)
        if_condition_541 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 4), result_gt_540)
        # Assigning a type to the variable 'if_condition_541' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'if_condition_541', if_condition_541)
        # SSA begins for if statement (line 119)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 120):
        
        # Assigning a Call to a Name (line 120):
        
        # Call to sample(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'possibles' (line 120)
        possibles_544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 34), 'possibles', False)
        int_545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 45), 'int')
        # Processing the call keyword arguments (line 120)
        kwargs_546 = {}
        # Getting the type of 'random' (line 120)
        random_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 20), 'random', False)
        # Obtaining the member 'sample' of a type (line 120)
        sample_543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 20), random_542, 'sample')
        # Calling sample(args, kwargs) (line 120)
        sample_call_result_547 = invoke(stypy.reporting.localization.Localization(__file__, 120, 20), sample_543, *[possibles_544, int_545], **kwargs_546)
        
        # Assigning a type to the variable 'possibles' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'possibles', sample_call_result_547)
        
        # Assigning a Subscript to a Name (line 121):
        
        # Assigning a Subscript to a Name (line 121):
        
        # Obtaining the type of the subscript
        int_548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 27), 'int')
        slice_549 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 121, 16), None, int_548, None)
        # Getting the type of 'possibles' (line 121)
        possibles_550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'possibles')
        # Obtaining the member '__getitem__' of a type (line 121)
        getitem___551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 16), possibles_550, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 121)
        subscript_call_result_552 = invoke(stypy.reporting.localization.Localization(__file__, 121, 16), getitem___551, slice_549)
        
        # Assigning a type to the variable 'plays' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'plays', subscript_call_result_552)
        # SSA branch for the else part of an if statement (line 119)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to len(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'possibles' (line 122)
        possibles_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 13), 'possibles', False)
        # Processing the call keyword arguments (line 122)
        kwargs_555 = {}
        # Getting the type of 'len' (line 122)
        len_553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 9), 'len', False)
        # Calling len(args, kwargs) (line 122)
        len_call_result_556 = invoke(stypy.reporting.localization.Localization(__file__, 122, 9), len_553, *[possibles_554], **kwargs_555)
        
        int_557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 26), 'int')
        # Applying the binary operator '>' (line 122)
        result_gt_558 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 9), '>', len_call_result_556, int_557)
        
        # Testing if the type of an if condition is none (line 122)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 122, 9), result_gt_558):
            
            # Assigning a Name to a Name (line 125):
            
            # Assigning a Name to a Name (line 125):
            # Getting the type of 'possibles' (line 125)
            possibles_566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'possibles')
            # Assigning a type to the variable 'plays' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'plays', possibles_566)
        else:
            
            # Testing the type of an if condition (line 122)
            if_condition_559 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 9), result_gt_558)
            # Assigning a type to the variable 'if_condition_559' (line 122)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 9), 'if_condition_559', if_condition_559)
            # SSA begins for if statement (line 122)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 123):
            
            # Assigning a Call to a Name (line 123):
            
            # Call to sample(...): (line 123)
            # Processing the call arguments (line 123)
            # Getting the type of 'possibles' (line 123)
            possibles_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 30), 'possibles', False)
            int_563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 41), 'int')
            # Processing the call keyword arguments (line 123)
            kwargs_564 = {}
            # Getting the type of 'random' (line 123)
            random_560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'random', False)
            # Obtaining the member 'sample' of a type (line 123)
            sample_561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 16), random_560, 'sample')
            # Calling sample(args, kwargs) (line 123)
            sample_call_result_565 = invoke(stypy.reporting.localization.Localization(__file__, 123, 16), sample_561, *[possibles_562, int_563], **kwargs_564)
            
            # Assigning a type to the variable 'plays' (line 123)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'plays', sample_call_result_565)
            # SSA branch for the else part of an if statement (line 122)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Name (line 125):
            
            # Assigning a Name to a Name (line 125):
            # Getting the type of 'possibles' (line 125)
            possibles_566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'possibles')
            # Assigning a type to the variable 'plays' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'plays', possibles_566)
            # SSA join for if statement (line 122)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 119)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Tuple (line 126):
    
    # Assigning a Call to a Name:
    
    # Call to max(...): (line 126)
    # Processing the call arguments (line 126)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'plays' (line 126)
    plays_575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 64), 'plays', False)
    comprehension_576 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 19), plays_575)
    # Assigning a type to the variable 'play' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 19), 'play', comprehension_576)
    
    # Obtaining an instance of the builtin type 'tuple' (line 126)
    tuple_568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 126)
    # Adding element type (line 126)
    
    # Call to utility(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'play' (line 126)
    play_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 28), 'play', False)
    # Getting the type of 'possibles' (line 126)
    possibles_571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 34), 'possibles', False)
    # Processing the call keyword arguments (line 126)
    kwargs_572 = {}
    # Getting the type of 'utility' (line 126)
    utility_569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 20), 'utility', False)
    # Calling utility(args, kwargs) (line 126)
    utility_call_result_573 = invoke(stypy.reporting.localization.Localization(__file__, 126, 20), utility_569, *[play_570, possibles_571], **kwargs_572)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 20), tuple_568, utility_call_result_573)
    # Adding element type (line 126)
    # Getting the type of 'play' (line 126)
    play_574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 46), 'play', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 20), tuple_568, play_574)
    
    list_577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 19), list_577, tuple_568)
    # Processing the call keyword arguments (line 126)
    kwargs_578 = {}
    # Getting the type of 'max' (line 126)
    max_567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 14), 'max', False)
    # Calling max(args, kwargs) (line 126)
    max_call_result_579 = invoke(stypy.reporting.localization.Localization(__file__, 126, 14), max_567, *[list_577], **kwargs_578)
    
    # Assigning a type to the variable 'call_assignment_7' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'call_assignment_7', max_call_result_579)
    
    # Assigning a Call to a Name (line 126):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 4), 'int')
    # Processing the call keyword arguments
    kwargs_583 = {}
    # Getting the type of 'call_assignment_7' (line 126)
    call_assignment_7_580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'call_assignment_7', False)
    # Obtaining the member '__getitem__' of a type (line 126)
    getitem___581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 4), call_assignment_7_580, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_584 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___581, *[int_582], **kwargs_583)
    
    # Assigning a type to the variable 'call_assignment_8' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'call_assignment_8', getitem___call_result_584)
    
    # Assigning a Name to a Name (line 126):
    # Getting the type of 'call_assignment_8' (line 126)
    call_assignment_8_585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'call_assignment_8')
    # Assigning a type to the variable '_' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), '_', call_assignment_8_585)
    
    # Assigning a Call to a Name (line 126):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 4), 'int')
    # Processing the call keyword arguments
    kwargs_589 = {}
    # Getting the type of 'call_assignment_7' (line 126)
    call_assignment_7_586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'call_assignment_7', False)
    # Obtaining the member '__getitem__' of a type (line 126)
    getitem___587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 4), call_assignment_7_586, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_590 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___587, *[int_588], **kwargs_589)
    
    # Assigning a type to the variable 'call_assignment_9' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'call_assignment_9', getitem___call_result_590)
    
    # Assigning a Name to a Name (line 126):
    # Getting the type of 'call_assignment_9' (line 126)
    call_assignment_9_591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'call_assignment_9')
    # Assigning a type to the variable 'play' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 7), 'play', call_assignment_9_591)
    # Getting the type of 'play' (line 127)
    play_592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 11), 'play')
    # Assigning a type to the variable 'stypy_return_type' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'stypy_return_type', play_592)
    
    # ################# End of 's_samplebest(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 's_samplebest' in the type store
    # Getting the type of 'stypy_return_type' (line 116)
    stypy_return_type_593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_593)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 's_samplebest'
    return stypy_return_type_593

# Assigning a type to the variable 's_samplebest' (line 116)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), 's_samplebest', s_samplebest)

@norecursion
def average(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'average'
    module_type_store = module_type_store.open_function_context('average', 132, 0, False)
    
    # Passed parameters checking function
    average.stypy_localization = localization
    average.stypy_type_of_self = None
    average.stypy_type_store = module_type_store
    average.stypy_function_name = 'average'
    average.stypy_param_names_list = ['seqn']
    average.stypy_varargs_param_name = None
    average.stypy_kwargs_param_name = None
    average.stypy_call_defaults = defaults
    average.stypy_call_varargs = varargs
    average.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'average', ['seqn'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'average', localization, ['seqn'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'average(...)' code ##################

    
    # Call to sum(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'seqn' (line 133)
    seqn_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), 'seqn', False)
    # Processing the call keyword arguments (line 133)
    kwargs_596 = {}
    # Getting the type of 'sum' (line 133)
    sum_594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 11), 'sum', False)
    # Calling sum(args, kwargs) (line 133)
    sum_call_result_597 = invoke(stypy.reporting.localization.Localization(__file__, 133, 11), sum_594, *[seqn_595], **kwargs_596)
    
    
    # Call to float(...): (line 133)
    # Processing the call arguments (line 133)
    
    # Call to len(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'seqn' (line 133)
    seqn_600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 33), 'seqn', False)
    # Processing the call keyword arguments (line 133)
    kwargs_601 = {}
    # Getting the type of 'len' (line 133)
    len_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 29), 'len', False)
    # Calling len(args, kwargs) (line 133)
    len_call_result_602 = invoke(stypy.reporting.localization.Localization(__file__, 133, 29), len_599, *[seqn_600], **kwargs_601)
    
    # Processing the call keyword arguments (line 133)
    kwargs_603 = {}
    # Getting the type of 'float' (line 133)
    float_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 23), 'float', False)
    # Calling float(args, kwargs) (line 133)
    float_call_result_604 = invoke(stypy.reporting.localization.Localization(__file__, 133, 23), float_598, *[len_call_result_602], **kwargs_603)
    
    # Applying the binary operator 'div' (line 133)
    result_div_605 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 11), 'div', sum_call_result_597, float_call_result_604)
    
    # Assigning a type to the variable 'stypy_return_type' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type', result_div_605)
    
    # ################# End of 'average(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'average' in the type store
    # Getting the type of 'stypy_return_type' (line 132)
    stypy_return_type_606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_606)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'average'
    return stypy_return_type_606

# Assigning a type to the variable 'average' (line 132)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 0), 'average', average)

@norecursion
def counts(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'counts'
    module_type_store = module_type_store.open_function_context('counts', 136, 0, False)
    
    # Passed parameters checking function
    counts.stypy_localization = localization
    counts.stypy_type_of_self = None
    counts.stypy_type_store = module_type_store
    counts.stypy_function_name = 'counts'
    counts.stypy_param_names_list = ['seqn']
    counts.stypy_varargs_param_name = None
    counts.stypy_kwargs_param_name = None
    counts.stypy_call_defaults = defaults
    counts.stypy_call_varargs = varargs
    counts.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'counts', ['seqn'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'counts', localization, ['seqn'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'counts(...)' code ##################

    
    # Assigning a BinOp to a Name (line 137):
    
    # Assigning a BinOp to a Name (line 137):
    
    # Call to max(...): (line 137)
    # Processing the call arguments (line 137)
    int_608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 16), 'int')
    
    # Call to max(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'seqn' (line 137)
    seqn_610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 24), 'seqn', False)
    # Processing the call keyword arguments (line 137)
    kwargs_611 = {}
    # Getting the type of 'max' (line 137)
    max_609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 20), 'max', False)
    # Calling max(args, kwargs) (line 137)
    max_call_result_612 = invoke(stypy.reporting.localization.Localization(__file__, 137, 20), max_609, *[seqn_610], **kwargs_611)
    
    # Processing the call keyword arguments (line 137)
    kwargs_613 = {}
    # Getting the type of 'max' (line 137)
    max_607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'max', False)
    # Calling max(args, kwargs) (line 137)
    max_call_result_614 = invoke(stypy.reporting.localization.Localization(__file__, 137, 12), max_607, *[int_608, max_call_result_612], **kwargs_613)
    
    int_615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 33), 'int')
    # Applying the binary operator '+' (line 137)
    result_add_616 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 12), '+', max_call_result_614, int_615)
    
    # Assigning a type to the variable 'limit' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'limit', result_add_616)
    
    # Assigning a BinOp to a Name (line 138):
    
    # Assigning a BinOp to a Name (line 138):
    
    # Obtaining an instance of the builtin type 'list' (line 138)
    list_617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 138)
    # Adding element type (line 138)
    int_618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 12), list_617, int_618)
    
    # Getting the type of 'limit' (line 138)
    limit_619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 18), 'limit')
    # Applying the binary operator '*' (line 138)
    result_mul_620 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 12), '*', list_617, limit_619)
    
    # Assigning a type to the variable 'tally' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'tally', result_mul_620)
    
    # Getting the type of 'seqn' (line 139)
    seqn_621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 13), 'seqn')
    # Testing if the for loop is going to be iterated (line 139)
    # Testing the type of a for loop iterable (line 139)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 139, 4), seqn_621)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 139, 4), seqn_621):
        # Getting the type of the for loop variable (line 139)
        for_loop_var_622 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 139, 4), seqn_621)
        # Assigning a type to the variable 'i' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'i', for_loop_var_622)
        # SSA begins for a for statement (line 139)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'tally' (line 140)
        tally_623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'tally')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 140)
        i_624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 14), 'i')
        # Getting the type of 'tally' (line 140)
        tally_625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'tally')
        # Obtaining the member '__getitem__' of a type (line 140)
        getitem___626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), tally_625, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 140)
        subscript_call_result_627 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), getitem___626, i_624)
        
        int_628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 20), 'int')
        # Applying the binary operator '+=' (line 140)
        result_iadd_629 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 8), '+=', subscript_call_result_627, int_628)
        # Getting the type of 'tally' (line 140)
        tally_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'tally')
        # Getting the type of 'i' (line 140)
        i_631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 14), 'i')
        # Storing an element on a container (line 140)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 8), tally_630, (i_631, result_iadd_629))
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Obtaining the type of the subscript
    int_632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 17), 'int')
    slice_633 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 141, 11), int_632, None, None)
    # Getting the type of 'tally' (line 141)
    tally_634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 11), 'tally')
    # Obtaining the member '__getitem__' of a type (line 141)
    getitem___635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 11), tally_634, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 141)
    subscript_call_result_636 = invoke(stypy.reporting.localization.Localization(__file__, 141, 11), getitem___635, slice_633)
    
    # Assigning a type to the variable 'stypy_return_type' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'stypy_return_type', subscript_call_result_636)
    
    # ################# End of 'counts(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'counts' in the type store
    # Getting the type of 'stypy_return_type' (line 136)
    stypy_return_type_637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_637)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'counts'
    return stypy_return_type_637

# Assigning a type to the variable 'counts' (line 136)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), 'counts', counts)

@norecursion
def eval_strategy(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'eval_strategy'
    module_type_store = module_type_store.open_function_context('eval_strategy', 144, 0, False)
    
    # Passed parameters checking function
    eval_strategy.stypy_localization = localization
    eval_strategy.stypy_type_of_self = None
    eval_strategy.stypy_type_store = module_type_store
    eval_strategy.stypy_function_name = 'eval_strategy'
    eval_strategy.stypy_param_names_list = ['name', 'strategy']
    eval_strategy.stypy_varargs_param_name = None
    eval_strategy.stypy_kwargs_param_name = None
    eval_strategy.stypy_call_defaults = defaults
    eval_strategy.stypy_call_varargs = varargs
    eval_strategy.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'eval_strategy', ['name', 'strategy'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'eval_strategy', localization, ['name', 'strategy'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'eval_strategy(...)' code ##################

    
    # Assigning a Call to a Name (line 145):
    
    # Assigning a Call to a Name (line 145):
    
    # Call to clock(...): (line 145)
    # Processing the call keyword arguments (line 145)
    kwargs_639 = {}
    # Getting the type of 'clock' (line 145)
    clock_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'clock', False)
    # Calling clock(args, kwargs) (line 145)
    clock_call_result_640 = invoke(stypy.reporting.localization.Localization(__file__, 145, 12), clock_638, *[], **kwargs_639)
    
    # Assigning a type to the variable 'start' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'start', clock_call_result_640)
    
    # Assigning a ListComp to a Name (line 146):
    
    # Assigning a ListComp to a Name (line 146):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to xrange(...): (line 146)
    # Processing the call arguments (line 146)
    # Getting the type of 'TRIALS' (line 146)
    TRIALS_653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 89), 'TRIALS', False)
    # Processing the call keyword arguments (line 146)
    kwargs_654 = {}
    # Getting the type of 'xrange' (line 146)
    xrange_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 82), 'xrange', False)
    # Calling xrange(args, kwargs) (line 146)
    xrange_call_result_655 = invoke(stypy.reporting.localization.Localization(__file__, 146, 82), xrange_652, *[TRIALS_653], **kwargs_654)
    
    comprehension_656 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 12), xrange_call_result_655)
    # Assigning a type to the variable 'i' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'i', comprehension_656)
    
    # Call to rungame(...): (line 146)
    # Processing the call arguments (line 146)
    
    # Call to choice(...): (line 146)
    # Processing the call arguments (line 146)
    # Getting the type of 'searchspace' (line 146)
    searchspace_644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 34), 'searchspace', False)
    # Processing the call keyword arguments (line 146)
    kwargs_645 = {}
    # Getting the type of 'random' (line 146)
    random_642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 20), 'random', False)
    # Obtaining the member 'choice' of a type (line 146)
    choice_643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 20), random_642, 'choice')
    # Calling choice(args, kwargs) (line 146)
    choice_call_result_646 = invoke(stypy.reporting.localization.Localization(__file__, 146, 20), choice_643, *[searchspace_644], **kwargs_645)
    
    # Getting the type of 'strategy' (line 146)
    strategy_647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 48), 'strategy', False)
    # Processing the call keyword arguments (line 146)
    # Getting the type of 'False' (line 146)
    False_648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 66), 'False', False)
    keyword_649 = False_648
    kwargs_650 = {'verbose': keyword_649}
    # Getting the type of 'rungame' (line 146)
    rungame_641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'rungame', False)
    # Calling rungame(args, kwargs) (line 146)
    rungame_call_result_651 = invoke(stypy.reporting.localization.Localization(__file__, 146, 12), rungame_641, *[choice_call_result_646, strategy_647], **kwargs_650)
    
    list_657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 12), list_657, rungame_call_result_651)
    # Assigning a type to the variable 'data' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'data', list_657)
    
    # ################# End of 'eval_strategy(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'eval_strategy' in the type store
    # Getting the type of 'stypy_return_type' (line 144)
    stypy_return_type_658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_658)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'eval_strategy'
    return stypy_return_type_658

# Assigning a type to the variable 'eval_strategy' (line 144)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 0), 'eval_strategy', eval_strategy)

@norecursion
def main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'main'
    module_type_store = module_type_store.open_function_context('main', 152, 0, False)
    
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

    
    # Call to seed(...): (line 153)
    # Processing the call arguments (line 153)
    int_661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 16), 'int')
    # Processing the call keyword arguments (line 153)
    kwargs_662 = {}
    # Getting the type of 'random' (line 153)
    random_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'random', False)
    # Obtaining the member 'seed' of a type (line 153)
    seed_660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 4), random_659, 'seed')
    # Calling seed(args, kwargs) (line 153)
    seed_call_result_663 = invoke(stypy.reporting.localization.Localization(__file__, 153, 4), seed_660, *[int_661], **kwargs_662)
    
    
    # Assigning a Call to a Name (line 156):
    
    # Assigning a Call to a Name (line 156):
    
    # Call to split(...): (line 156)
    # Processing the call keyword arguments (line 156)
    kwargs_666 = {}
    str_664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 12), 'str', 's_bestinfo s_samplebest s_worstinfo s_allrand s_trynodup s_bestinfo')
    # Obtaining the member 'split' of a type (line 156)
    split_665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 12), str_664, 'split')
    # Calling split(args, kwargs) (line 156)
    split_call_result_667 = invoke(stypy.reporting.localization.Localization(__file__, 156, 12), split_665, *[], **kwargs_666)
    
    # Assigning a type to the variable 'names' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'names', split_call_result_667)
    
    # Call to eval_strategy(...): (line 157)
    # Processing the call arguments (line 157)
    str_669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 18), 'str', 's_bestinfo')
    # Getting the type of 's_bestinfo' (line 157)
    s_bestinfo_670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 32), 's_bestinfo', False)
    # Processing the call keyword arguments (line 157)
    kwargs_671 = {}
    # Getting the type of 'eval_strategy' (line 157)
    eval_strategy_668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'eval_strategy', False)
    # Calling eval_strategy(args, kwargs) (line 157)
    eval_strategy_call_result_672 = invoke(stypy.reporting.localization.Localization(__file__, 157, 4), eval_strategy_668, *[str_669, s_bestinfo_670], **kwargs_671)
    
    
    # Call to eval_strategy(...): (line 158)
    # Processing the call arguments (line 158)
    str_674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 18), 'str', 's_samplebest')
    # Getting the type of 's_samplebest' (line 158)
    s_samplebest_675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 34), 's_samplebest', False)
    # Processing the call keyword arguments (line 158)
    kwargs_676 = {}
    # Getting the type of 'eval_strategy' (line 158)
    eval_strategy_673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'eval_strategy', False)
    # Calling eval_strategy(args, kwargs) (line 158)
    eval_strategy_call_result_677 = invoke(stypy.reporting.localization.Localization(__file__, 158, 4), eval_strategy_673, *[str_674, s_samplebest_675], **kwargs_676)
    
    
    # Call to eval_strategy(...): (line 159)
    # Processing the call arguments (line 159)
    str_679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 18), 'str', 's_worstinfo')
    # Getting the type of 's_worstinfo' (line 159)
    s_worstinfo_680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 33), 's_worstinfo', False)
    # Processing the call keyword arguments (line 159)
    kwargs_681 = {}
    # Getting the type of 'eval_strategy' (line 159)
    eval_strategy_678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'eval_strategy', False)
    # Calling eval_strategy(args, kwargs) (line 159)
    eval_strategy_call_result_682 = invoke(stypy.reporting.localization.Localization(__file__, 159, 4), eval_strategy_678, *[str_679, s_worstinfo_680], **kwargs_681)
    
    
    # Call to eval_strategy(...): (line 160)
    # Processing the call arguments (line 160)
    str_684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 18), 'str', 's_trynodup')
    # Getting the type of 's_trynodup' (line 160)
    s_trynodup_685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 32), 's_trynodup', False)
    # Processing the call keyword arguments (line 160)
    kwargs_686 = {}
    # Getting the type of 'eval_strategy' (line 160)
    eval_strategy_683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'eval_strategy', False)
    # Calling eval_strategy(args, kwargs) (line 160)
    eval_strategy_call_result_687 = invoke(stypy.reporting.localization.Localization(__file__, 160, 4), eval_strategy_683, *[str_684, s_trynodup_685], **kwargs_686)
    
    
    # Call to eval_strategy(...): (line 161)
    # Processing the call arguments (line 161)
    str_689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 18), 'str', 's_allrand')
    # Getting the type of 's_allrand' (line 161)
    s_allrand_690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 31), 's_allrand', False)
    # Processing the call keyword arguments (line 161)
    kwargs_691 = {}
    # Getting the type of 'eval_strategy' (line 161)
    eval_strategy_688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'eval_strategy', False)
    # Calling eval_strategy(args, kwargs) (line 161)
    eval_strategy_call_result_692 = invoke(stypy.reporting.localization.Localization(__file__, 161, 4), eval_strategy_688, *[str_689, s_allrand_690], **kwargs_691)
    
    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 152)
    stypy_return_type_693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_693)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_693

# Assigning a type to the variable 'main' (line 152)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'main', main)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 164, 0, False)
    
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

    
    # Call to main(...): (line 165)
    # Processing the call keyword arguments (line 165)
    kwargs_695 = {}
    # Getting the type of 'main' (line 165)
    main_694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'main', False)
    # Calling main(args, kwargs) (line 165)
    main_call_result_696 = invoke(stypy.reporting.localization.Localization(__file__, 165, 4), main_694, *[], **kwargs_695)
    
    # Getting the type of 'True' (line 166)
    True_697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'stypy_return_type', True_697)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 164)
    stypy_return_type_698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_698)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_698

# Assigning a type to the variable 'run' (line 164)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 0), 'run', run)

# Call to run(...): (line 169)
# Processing the call keyword arguments (line 169)
kwargs_700 = {}
# Getting the type of 'run' (line 169)
run_699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 0), 'run', False)
# Calling run(args, kwargs) (line 169)
run_call_result_701 = invoke(stypy.reporting.localization.Localization(__file__, 169, 0), run_699, *[], **kwargs_700)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

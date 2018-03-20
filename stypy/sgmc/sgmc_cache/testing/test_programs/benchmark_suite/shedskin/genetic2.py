
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: 
3: # placed in the public domain by Stavros Korokithakis
4: 
5: import time
6: import copy
7: from random import randrange, randint, seed, random, choice, triangular
8: seed(42)
9: 
10: # The size of the data bus of the multiplexer.
11: DATA_SIZE = 4
12: # The size of the selector, in bits (this is log2   (DATA_SIZE)).
13: MUX_SIZE = 2
14: MAX_DEPTH = 5
15: 
16: OPCODE_NONE = 0
17: OPCODE_AND = 1
18: OPCODE_OR = 2
19: OPCODE_NOT = 3
20: OPCODE_IF = 4
21: 
22: ARG_NUM = (0, 2, 2, 1, 3)
23: 
24: STATE_OPCODE = 0
25: STATE_ARGUMENT = 1
26: STATE_SUBTREE = 2
27: 
28: def fitness(individual):
29:     return individual.fitness
30: 
31: def make_random_genome(node, depth=0):
32:     if depth >= MAX_DEPTH or random() > 0.7:
33:         node.opcode = OPCODE_NONE
34:         node.args = None
35:         node.value = randrange(MUX_SIZE + DATA_SIZE)
36:     else:
37:         node.opcode = randint(OPCODE_AND, OPCODE_IF)
38:         node.value = 0
39:         node.args = tuple([TreeNode() for _ in range(ARG_NUM[node.opcode])])
40:         for arg in node.args:
41:             make_random_genome(arg, depth+1)
42: 
43: class TreeNode:
44:     def __init__(self, opcode=OPCODE_NONE, value=-1, args=None):
45:         self.opcode = opcode
46:         self.value = value
47:         self.args = args
48: 
49:     def mutate(self):
50:         '''Mutate this node.'''
51:         # If we're a terminal node, stop so we don't exceed our depth.
52:         if self.opcode == OPCODE_NONE:
53:             return
54: 
55:         if random() > 0.5:
56:             # Turn this node into a terminal node.
57:             make_random_genome(self, MAX_DEPTH)
58:         else:
59:             # Turn this into a different node.
60:             make_random_genome(self, MAX_DEPTH-1)
61: 
62: 
63:     def execute(self, input):
64:         if self.opcode == OPCODE_NONE:
65:             return (input & (1 << self.value)) >> self.value
66:         elif self.opcode == OPCODE_AND:
67:             return self.args[0].execute(input) & \
68:                    self.args[1].execute(input)
69:         elif self.opcode == OPCODE_OR:
70:             return self.args[0].execute(input) | \
71:                    self.args[1].execute(input)
72:         elif self.opcode == OPCODE_NOT:
73:             return 1 ^ self.args[0].execute(input)
74:         elif self.opcode == OPCODE_IF:
75:             if self.args[0].execute(input):
76:                 return self.args[1].execute(input)
77:             else:
78:                 return self.args[2].execute(input)
79: 
80:     def __str__(self):
81:         if self.opcode == OPCODE_NONE:
82:             output = "(bit %s)" % self.value
83:         elif self.opcode == OPCODE_AND:
84:             output = "(and %s %s)" % self.args
85:         elif self.opcode == OPCODE_OR:
86:             output = "(or %s %s)" % self.args
87:         elif self.opcode == OPCODE_NOT:
88:             output = "(not %s)" % self.args
89:         elif self.opcode == OPCODE_IF:
90:             output = "(if %s then %s else %s)" % self.args
91: 
92:         return output
93: 
94: class Individual:
95:     def __init__(self, genome=None):
96:         '''
97:         Initialise the multiplexer with a genome and data size.
98:         '''
99:         if genome is None:
100:             self.genome = TreeNode()
101:             make_random_genome(self.genome, 0)
102:         else:
103:             self.genome = genome
104:         # Memoize fitness for sorting.
105:         self.fitness = 0.0
106: 
107:     def __str__(self):
108:         '''Represent this individual.'''
109:         return "Genome: %s, fitness %s." % (self.genome, self.fitness)
110: 
111:     def copy(self):
112:         return Individual(copy.deepcopy(self.genome))
113: 
114:     def mutate(self):
115:         '''Mutate this individual.'''
116:         if self.genome.args:
117:             node, choice = self.get_random_node()
118:             node.args[choice].mutate()
119: 
120:     def get_random_node(self, max_depth=MAX_DEPTH):
121:         '''Get a random node from the tree.'''
122:         root = self.genome
123:         previous_root = root
124:         choice = 0
125:         for counter in range(max_depth):
126:             if root.args and random() > 1 / MAX_DEPTH:
127:                 previous_root = root
128:                 choice = randrange(len(root.args))
129:                 root = root.args[choice]
130:             else:
131:                 break
132:         return (previous_root, choice)
133: 
134:     def update_fitness(self, full_test=False):
135:         '''Calculate the individual's fitness and update it.'''
136:         correct = 0
137:         if full_test:
138:             data = (1 << DATA_SIZE) - 1
139:             for mux in range(DATA_SIZE):
140:                 for _ in range(2):
141:                     # Flip the bit in question.
142:                     data ^= (1 << mux)
143:                     input = (data << 2) | mux
144:                     output = self.genome.execute(input)
145: 
146:                     # Do some bit twiddling...
147:                     correct_output = (data & (1 << mux)) >> mux
148:                     if output == correct_output:
149:                         correct += 1
150:             total = DATA_SIZE * 2
151:         else:
152:             for mux in range(DATA_SIZE):
153:                 for data in range(1 << DATA_SIZE):
154:                     input = (data << 2) | mux
155:                     output = self.genome.execute(input)
156: 
157:                     # Do some bit twiddling...
158:                     correct_output = (data & (1 << mux)) >> mux
159:                     if output == correct_output:
160:                         correct += 1
161:             total = (1 << DATA_SIZE) * DATA_SIZE
162: 
163:         self.fitness = (1.0 * correct) / total
164:         return self.fitness
165: 
166: class Pool:
167:     population_size = 300
168: 
169:     def __init__(self):
170:         '''Initialise the pool.'''
171:         self.population = [Individual() for _ in range(Pool.population_size)]
172:         self.epoch = 0
173: 
174:     def crossover(self, father, mother):
175:         son = father.copy()
176:         daughter = mother.copy()
177:         son_node, son_choice = son.get_random_node()
178:         daughter_node, daughter_choice = daughter.get_random_node()
179:         if son_node.args and daughter_node.args:
180:             temp_node = son_node.args[son_choice]
181:             son_node.args = son_node.args[:son_choice] + (daughter_node.args[daughter_choice], ) + son_node.args[son_choice+1:]
182:             daughter_node.args = daughter_node.args[:daughter_choice] + (temp_node, ) + daughter_node.args[daughter_choice+1:]
183:         return son, daughter
184: 
185:     def advance_epoch(self):
186:         '''Pass the time.'''
187:         # Sort ascending because this is cost rather than fitness.
188:         self.population.sort(key=fitness, reverse=True)
189:         new_population = []
190: 
191:         # Clone our best people.
192:         iters = int(Pool.population_size * 0.4)
193:         for counter in range(iters):
194:             new_individual = self.population[counter].copy()
195:             new_population.append(new_individual)
196: 
197:         # Breed our best people, producing four offspring for each couple.
198:         iters = int(Pool.population_size * 0.6)
199:         for counter in range(0, iters, 2):
200:             # Perform rank roulette selection.
201:             father = self.population[int(triangular(0, iters, 0))]
202:             mother = self.population[int(triangular(0, iters, 0))]
203:             children = self.crossover(father, mother)
204:             children[0].mutate()
205:             new_population += children
206: 
207:         self.population = new_population
208:         for person in self.population:
209:             person.update_fitness()
210:         self.epoch += 1
211: 
212:     def get_best_individual(self):
213:         '''Get the best individual of this pool.'''
214:         return max(self.population, key=fitness)
215: 
216: 
217: def main():
218:     pool = Pool()
219:     start_time = time.time()
220:     for epoch in range(100):
221:         pool.advance_epoch()
222:         best_individual = pool.get_best_individual()
223:         if not epoch % 10:
224:             pass#print "Epoch: %s, best fitness: %s" % (epoch, best_individual.fitness)
225: 
226:     #print "Epoch: %s, best fitness: %s" % (epoch, best_individual.fitness)
227:     #print "Finished in %0.3f sec, best individual: %s" % (time.time() - start_time, best_individual)
228: 
229: def run():
230:     main()
231:     return True
232: 
233: run()
234: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import time' statement (line 5)
import time

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'time', time, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import copy' statement (line 6)
import copy

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'copy', copy, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from random import randrange, randint, seed, random, choice, triangular' statement (line 7)
try:
    from random import randrange, randint, seed, random, choice, triangular

except:
    randrange = UndefinedType
    randint = UndefinedType
    seed = UndefinedType
    random = UndefinedType
    choice = UndefinedType
    triangular = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'random', None, module_type_store, ['randrange', 'randint', 'seed', 'random', 'choice', 'triangular'], [randrange, randint, seed, random, choice, triangular])


# Call to seed(...): (line 8)
# Processing the call arguments (line 8)
int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 5), 'int')
# Processing the call keyword arguments (line 8)
kwargs_12 = {}
# Getting the type of 'seed' (line 8)
seed_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'seed', False)
# Calling seed(args, kwargs) (line 8)
seed_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 8, 0), seed_10, *[int_11], **kwargs_12)


# Assigning a Num to a Name (line 11):

# Assigning a Num to a Name (line 11):
int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 12), 'int')
# Assigning a type to the variable 'DATA_SIZE' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'DATA_SIZE', int_14)

# Assigning a Num to a Name (line 13):

# Assigning a Num to a Name (line 13):
int_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 11), 'int')
# Assigning a type to the variable 'MUX_SIZE' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'MUX_SIZE', int_15)

# Assigning a Num to a Name (line 14):

# Assigning a Num to a Name (line 14):
int_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 12), 'int')
# Assigning a type to the variable 'MAX_DEPTH' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'MAX_DEPTH', int_16)

# Assigning a Num to a Name (line 16):

# Assigning a Num to a Name (line 16):
int_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 14), 'int')
# Assigning a type to the variable 'OPCODE_NONE' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'OPCODE_NONE', int_17)

# Assigning a Num to a Name (line 17):

# Assigning a Num to a Name (line 17):
int_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 13), 'int')
# Assigning a type to the variable 'OPCODE_AND' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'OPCODE_AND', int_18)

# Assigning a Num to a Name (line 18):

# Assigning a Num to a Name (line 18):
int_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 12), 'int')
# Assigning a type to the variable 'OPCODE_OR' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'OPCODE_OR', int_19)

# Assigning a Num to a Name (line 19):

# Assigning a Num to a Name (line 19):
int_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 13), 'int')
# Assigning a type to the variable 'OPCODE_NOT' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'OPCODE_NOT', int_20)

# Assigning a Num to a Name (line 20):

# Assigning a Num to a Name (line 20):
int_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 12), 'int')
# Assigning a type to the variable 'OPCODE_IF' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'OPCODE_IF', int_21)

# Assigning a Tuple to a Name (line 22):

# Assigning a Tuple to a Name (line 22):

# Obtaining an instance of the builtin type 'tuple' (line 22)
tuple_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 11), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 22)
# Adding element type (line 22)
int_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 11), tuple_22, int_23)
# Adding element type (line 22)
int_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 11), tuple_22, int_24)
# Adding element type (line 22)
int_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 11), tuple_22, int_25)
# Adding element type (line 22)
int_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 11), tuple_22, int_26)
# Adding element type (line 22)
int_27 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 11), tuple_22, int_27)

# Assigning a type to the variable 'ARG_NUM' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'ARG_NUM', tuple_22)

# Assigning a Num to a Name (line 24):

# Assigning a Num to a Name (line 24):
int_28 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 15), 'int')
# Assigning a type to the variable 'STATE_OPCODE' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'STATE_OPCODE', int_28)

# Assigning a Num to a Name (line 25):

# Assigning a Num to a Name (line 25):
int_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 17), 'int')
# Assigning a type to the variable 'STATE_ARGUMENT' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'STATE_ARGUMENT', int_29)

# Assigning a Num to a Name (line 26):

# Assigning a Num to a Name (line 26):
int_30 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 16), 'int')
# Assigning a type to the variable 'STATE_SUBTREE' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'STATE_SUBTREE', int_30)

@norecursion
def fitness(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'fitness'
    module_type_store = module_type_store.open_function_context('fitness', 28, 0, False)
    
    # Passed parameters checking function
    fitness.stypy_localization = localization
    fitness.stypy_type_of_self = None
    fitness.stypy_type_store = module_type_store
    fitness.stypy_function_name = 'fitness'
    fitness.stypy_param_names_list = ['individual']
    fitness.stypy_varargs_param_name = None
    fitness.stypy_kwargs_param_name = None
    fitness.stypy_call_defaults = defaults
    fitness.stypy_call_varargs = varargs
    fitness.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fitness', ['individual'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fitness', localization, ['individual'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fitness(...)' code ##################

    # Getting the type of 'individual' (line 29)
    individual_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 11), 'individual')
    # Obtaining the member 'fitness' of a type (line 29)
    fitness_32 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 11), individual_31, 'fitness')
    # Assigning a type to the variable 'stypy_return_type' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type', fitness_32)
    
    # ################# End of 'fitness(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fitness' in the type store
    # Getting the type of 'stypy_return_type' (line 28)
    stypy_return_type_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_33)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fitness'
    return stypy_return_type_33

# Assigning a type to the variable 'fitness' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'fitness', fitness)

@norecursion
def make_random_genome(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_34 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 35), 'int')
    defaults = [int_34]
    # Create a new context for function 'make_random_genome'
    module_type_store = module_type_store.open_function_context('make_random_genome', 31, 0, False)
    
    # Passed parameters checking function
    make_random_genome.stypy_localization = localization
    make_random_genome.stypy_type_of_self = None
    make_random_genome.stypy_type_store = module_type_store
    make_random_genome.stypy_function_name = 'make_random_genome'
    make_random_genome.stypy_param_names_list = ['node', 'depth']
    make_random_genome.stypy_varargs_param_name = None
    make_random_genome.stypy_kwargs_param_name = None
    make_random_genome.stypy_call_defaults = defaults
    make_random_genome.stypy_call_varargs = varargs
    make_random_genome.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'make_random_genome', ['node', 'depth'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'make_random_genome', localization, ['node', 'depth'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'make_random_genome(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Getting the type of 'depth' (line 32)
    depth_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 7), 'depth')
    # Getting the type of 'MAX_DEPTH' (line 32)
    MAX_DEPTH_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 16), 'MAX_DEPTH')
    # Applying the binary operator '>=' (line 32)
    result_ge_37 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 7), '>=', depth_35, MAX_DEPTH_36)
    
    
    
    # Call to random(...): (line 32)
    # Processing the call keyword arguments (line 32)
    kwargs_39 = {}
    # Getting the type of 'random' (line 32)
    random_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 29), 'random', False)
    # Calling random(args, kwargs) (line 32)
    random_call_result_40 = invoke(stypy.reporting.localization.Localization(__file__, 32, 29), random_38, *[], **kwargs_39)
    
    float_41 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 40), 'float')
    # Applying the binary operator '>' (line 32)
    result_gt_42 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 29), '>', random_call_result_40, float_41)
    
    # Applying the binary operator 'or' (line 32)
    result_or_keyword_43 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 7), 'or', result_ge_37, result_gt_42)
    
    # Testing if the type of an if condition is none (line 32)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 32, 4), result_or_keyword_43):
        
        # Assigning a Call to a Attribute (line 37):
        
        # Assigning a Call to a Attribute (line 37):
        
        # Call to randint(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'OPCODE_AND' (line 37)
        OPCODE_AND_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 30), 'OPCODE_AND', False)
        # Getting the type of 'OPCODE_IF' (line 37)
        OPCODE_IF_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 42), 'OPCODE_IF', False)
        # Processing the call keyword arguments (line 37)
        kwargs_59 = {}
        # Getting the type of 'randint' (line 37)
        randint_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 22), 'randint', False)
        # Calling randint(args, kwargs) (line 37)
        randint_call_result_60 = invoke(stypy.reporting.localization.Localization(__file__, 37, 22), randint_56, *[OPCODE_AND_57, OPCODE_IF_58], **kwargs_59)
        
        # Getting the type of 'node' (line 37)
        node_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'node')
        # Setting the type of the member 'opcode' of a type (line 37)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), node_61, 'opcode', randint_call_result_60)
        
        # Assigning a Num to a Attribute (line 38):
        
        # Assigning a Num to a Attribute (line 38):
        int_62 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 21), 'int')
        # Getting the type of 'node' (line 38)
        node_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'node')
        # Setting the type of the member 'value' of a type (line 38)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), node_63, 'value', int_62)
        
        # Assigning a Call to a Attribute (line 39):
        
        # Assigning a Call to a Attribute (line 39):
        
        # Call to tuple(...): (line 39)
        # Processing the call arguments (line 39)
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 39)
        # Processing the call arguments (line 39)
        
        # Obtaining the type of the subscript
        # Getting the type of 'node' (line 39)
        node_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 61), 'node', False)
        # Obtaining the member 'opcode' of a type (line 39)
        opcode_70 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 61), node_69, 'opcode')
        # Getting the type of 'ARG_NUM' (line 39)
        ARG_NUM_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 53), 'ARG_NUM', False)
        # Obtaining the member '__getitem__' of a type (line 39)
        getitem___72 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 53), ARG_NUM_71, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 39)
        subscript_call_result_73 = invoke(stypy.reporting.localization.Localization(__file__, 39, 53), getitem___72, opcode_70)
        
        # Processing the call keyword arguments (line 39)
        kwargs_74 = {}
        # Getting the type of 'range' (line 39)
        range_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 47), 'range', False)
        # Calling range(args, kwargs) (line 39)
        range_call_result_75 = invoke(stypy.reporting.localization.Localization(__file__, 39, 47), range_68, *[subscript_call_result_73], **kwargs_74)
        
        comprehension_76 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 27), range_call_result_75)
        # Assigning a type to the variable '_' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 27), '_', comprehension_76)
        
        # Call to TreeNode(...): (line 39)
        # Processing the call keyword arguments (line 39)
        kwargs_66 = {}
        # Getting the type of 'TreeNode' (line 39)
        TreeNode_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 27), 'TreeNode', False)
        # Calling TreeNode(args, kwargs) (line 39)
        TreeNode_call_result_67 = invoke(stypy.reporting.localization.Localization(__file__, 39, 27), TreeNode_65, *[], **kwargs_66)
        
        list_77 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 27), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 27), list_77, TreeNode_call_result_67)
        # Processing the call keyword arguments (line 39)
        kwargs_78 = {}
        # Getting the type of 'tuple' (line 39)
        tuple_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 'tuple', False)
        # Calling tuple(args, kwargs) (line 39)
        tuple_call_result_79 = invoke(stypy.reporting.localization.Localization(__file__, 39, 20), tuple_64, *[list_77], **kwargs_78)
        
        # Getting the type of 'node' (line 39)
        node_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'node')
        # Setting the type of the member 'args' of a type (line 39)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), node_80, 'args', tuple_call_result_79)
        
        # Getting the type of 'node' (line 40)
        node_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'node')
        # Obtaining the member 'args' of a type (line 40)
        args_82 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 19), node_81, 'args')
        # Assigning a type to the variable 'args_82' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'args_82', args_82)
        # Testing if the for loop is going to be iterated (line 40)
        # Testing the type of a for loop iterable (line 40)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 40, 8), args_82)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 40, 8), args_82):
            # Getting the type of the for loop variable (line 40)
            for_loop_var_83 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 40, 8), args_82)
            # Assigning a type to the variable 'arg' (line 40)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'arg', for_loop_var_83)
            # SSA begins for a for statement (line 40)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to make_random_genome(...): (line 41)
            # Processing the call arguments (line 41)
            # Getting the type of 'arg' (line 41)
            arg_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 31), 'arg', False)
            # Getting the type of 'depth' (line 41)
            depth_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 36), 'depth', False)
            int_87 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 42), 'int')
            # Applying the binary operator '+' (line 41)
            result_add_88 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 36), '+', depth_86, int_87)
            
            # Processing the call keyword arguments (line 41)
            kwargs_89 = {}
            # Getting the type of 'make_random_genome' (line 41)
            make_random_genome_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'make_random_genome', False)
            # Calling make_random_genome(args, kwargs) (line 41)
            make_random_genome_call_result_90 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), make_random_genome_84, *[arg_85, result_add_88], **kwargs_89)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
    else:
        
        # Testing the type of an if condition (line 32)
        if_condition_44 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 32, 4), result_or_keyword_43)
        # Assigning a type to the variable 'if_condition_44' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'if_condition_44', if_condition_44)
        # SSA begins for if statement (line 32)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 33):
        
        # Assigning a Name to a Attribute (line 33):
        # Getting the type of 'OPCODE_NONE' (line 33)
        OPCODE_NONE_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 22), 'OPCODE_NONE')
        # Getting the type of 'node' (line 33)
        node_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'node')
        # Setting the type of the member 'opcode' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), node_46, 'opcode', OPCODE_NONE_45)
        
        # Assigning a Name to a Attribute (line 34):
        
        # Assigning a Name to a Attribute (line 34):
        # Getting the type of 'None' (line 34)
        None_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 20), 'None')
        # Getting the type of 'node' (line 34)
        node_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'node')
        # Setting the type of the member 'args' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), node_48, 'args', None_47)
        
        # Assigning a Call to a Attribute (line 35):
        
        # Assigning a Call to a Attribute (line 35):
        
        # Call to randrange(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'MUX_SIZE' (line 35)
        MUX_SIZE_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 31), 'MUX_SIZE', False)
        # Getting the type of 'DATA_SIZE' (line 35)
        DATA_SIZE_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 42), 'DATA_SIZE', False)
        # Applying the binary operator '+' (line 35)
        result_add_52 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 31), '+', MUX_SIZE_50, DATA_SIZE_51)
        
        # Processing the call keyword arguments (line 35)
        kwargs_53 = {}
        # Getting the type of 'randrange' (line 35)
        randrange_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 21), 'randrange', False)
        # Calling randrange(args, kwargs) (line 35)
        randrange_call_result_54 = invoke(stypy.reporting.localization.Localization(__file__, 35, 21), randrange_49, *[result_add_52], **kwargs_53)
        
        # Getting the type of 'node' (line 35)
        node_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'node')
        # Setting the type of the member 'value' of a type (line 35)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), node_55, 'value', randrange_call_result_54)
        # SSA branch for the else part of an if statement (line 32)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Attribute (line 37):
        
        # Assigning a Call to a Attribute (line 37):
        
        # Call to randint(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'OPCODE_AND' (line 37)
        OPCODE_AND_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 30), 'OPCODE_AND', False)
        # Getting the type of 'OPCODE_IF' (line 37)
        OPCODE_IF_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 42), 'OPCODE_IF', False)
        # Processing the call keyword arguments (line 37)
        kwargs_59 = {}
        # Getting the type of 'randint' (line 37)
        randint_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 22), 'randint', False)
        # Calling randint(args, kwargs) (line 37)
        randint_call_result_60 = invoke(stypy.reporting.localization.Localization(__file__, 37, 22), randint_56, *[OPCODE_AND_57, OPCODE_IF_58], **kwargs_59)
        
        # Getting the type of 'node' (line 37)
        node_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'node')
        # Setting the type of the member 'opcode' of a type (line 37)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), node_61, 'opcode', randint_call_result_60)
        
        # Assigning a Num to a Attribute (line 38):
        
        # Assigning a Num to a Attribute (line 38):
        int_62 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 21), 'int')
        # Getting the type of 'node' (line 38)
        node_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'node')
        # Setting the type of the member 'value' of a type (line 38)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), node_63, 'value', int_62)
        
        # Assigning a Call to a Attribute (line 39):
        
        # Assigning a Call to a Attribute (line 39):
        
        # Call to tuple(...): (line 39)
        # Processing the call arguments (line 39)
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 39)
        # Processing the call arguments (line 39)
        
        # Obtaining the type of the subscript
        # Getting the type of 'node' (line 39)
        node_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 61), 'node', False)
        # Obtaining the member 'opcode' of a type (line 39)
        opcode_70 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 61), node_69, 'opcode')
        # Getting the type of 'ARG_NUM' (line 39)
        ARG_NUM_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 53), 'ARG_NUM', False)
        # Obtaining the member '__getitem__' of a type (line 39)
        getitem___72 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 53), ARG_NUM_71, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 39)
        subscript_call_result_73 = invoke(stypy.reporting.localization.Localization(__file__, 39, 53), getitem___72, opcode_70)
        
        # Processing the call keyword arguments (line 39)
        kwargs_74 = {}
        # Getting the type of 'range' (line 39)
        range_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 47), 'range', False)
        # Calling range(args, kwargs) (line 39)
        range_call_result_75 = invoke(stypy.reporting.localization.Localization(__file__, 39, 47), range_68, *[subscript_call_result_73], **kwargs_74)
        
        comprehension_76 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 27), range_call_result_75)
        # Assigning a type to the variable '_' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 27), '_', comprehension_76)
        
        # Call to TreeNode(...): (line 39)
        # Processing the call keyword arguments (line 39)
        kwargs_66 = {}
        # Getting the type of 'TreeNode' (line 39)
        TreeNode_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 27), 'TreeNode', False)
        # Calling TreeNode(args, kwargs) (line 39)
        TreeNode_call_result_67 = invoke(stypy.reporting.localization.Localization(__file__, 39, 27), TreeNode_65, *[], **kwargs_66)
        
        list_77 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 27), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 27), list_77, TreeNode_call_result_67)
        # Processing the call keyword arguments (line 39)
        kwargs_78 = {}
        # Getting the type of 'tuple' (line 39)
        tuple_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 'tuple', False)
        # Calling tuple(args, kwargs) (line 39)
        tuple_call_result_79 = invoke(stypy.reporting.localization.Localization(__file__, 39, 20), tuple_64, *[list_77], **kwargs_78)
        
        # Getting the type of 'node' (line 39)
        node_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'node')
        # Setting the type of the member 'args' of a type (line 39)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), node_80, 'args', tuple_call_result_79)
        
        # Getting the type of 'node' (line 40)
        node_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'node')
        # Obtaining the member 'args' of a type (line 40)
        args_82 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 19), node_81, 'args')
        # Assigning a type to the variable 'args_82' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'args_82', args_82)
        # Testing if the for loop is going to be iterated (line 40)
        # Testing the type of a for loop iterable (line 40)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 40, 8), args_82)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 40, 8), args_82):
            # Getting the type of the for loop variable (line 40)
            for_loop_var_83 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 40, 8), args_82)
            # Assigning a type to the variable 'arg' (line 40)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'arg', for_loop_var_83)
            # SSA begins for a for statement (line 40)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to make_random_genome(...): (line 41)
            # Processing the call arguments (line 41)
            # Getting the type of 'arg' (line 41)
            arg_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 31), 'arg', False)
            # Getting the type of 'depth' (line 41)
            depth_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 36), 'depth', False)
            int_87 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 42), 'int')
            # Applying the binary operator '+' (line 41)
            result_add_88 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 36), '+', depth_86, int_87)
            
            # Processing the call keyword arguments (line 41)
            kwargs_89 = {}
            # Getting the type of 'make_random_genome' (line 41)
            make_random_genome_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'make_random_genome', False)
            # Calling make_random_genome(args, kwargs) (line 41)
            make_random_genome_call_result_90 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), make_random_genome_84, *[arg_85, result_add_88], **kwargs_89)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for if statement (line 32)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'make_random_genome(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'make_random_genome' in the type store
    # Getting the type of 'stypy_return_type' (line 31)
    stypy_return_type_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_91)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'make_random_genome'
    return stypy_return_type_91

# Assigning a type to the variable 'make_random_genome' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'make_random_genome', make_random_genome)
# Declaration of the 'TreeNode' class

class TreeNode:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'OPCODE_NONE' (line 44)
        OPCODE_NONE_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 30), 'OPCODE_NONE')
        int_93 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 49), 'int')
        # Getting the type of 'None' (line 44)
        None_94 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 58), 'None')
        defaults = [OPCODE_NONE_92, int_93, None_94]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 44, 4, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TreeNode.__init__', ['opcode', 'value', 'args'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['opcode', 'value', 'args'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 45):
        
        # Assigning a Name to a Attribute (line 45):
        # Getting the type of 'opcode' (line 45)
        opcode_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 22), 'opcode')
        # Getting the type of 'self' (line 45)
        self_96 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'self')
        # Setting the type of the member 'opcode' of a type (line 45)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), self_96, 'opcode', opcode_95)
        
        # Assigning a Name to a Attribute (line 46):
        
        # Assigning a Name to a Attribute (line 46):
        # Getting the type of 'value' (line 46)
        value_97 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 21), 'value')
        # Getting the type of 'self' (line 46)
        self_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'self')
        # Setting the type of the member 'value' of a type (line 46)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), self_98, 'value', value_97)
        
        # Assigning a Name to a Attribute (line 47):
        
        # Assigning a Name to a Attribute (line 47):
        # Getting the type of 'args' (line 47)
        args_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 20), 'args')
        # Getting the type of 'self' (line 47)
        self_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'self')
        # Setting the type of the member 'args' of a type (line 47)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), self_100, 'args', args_99)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def mutate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mutate'
        module_type_store = module_type_store.open_function_context('mutate', 49, 4, False)
        # Assigning a type to the variable 'self' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TreeNode.mutate.__dict__.__setitem__('stypy_localization', localization)
        TreeNode.mutate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TreeNode.mutate.__dict__.__setitem__('stypy_type_store', module_type_store)
        TreeNode.mutate.__dict__.__setitem__('stypy_function_name', 'TreeNode.mutate')
        TreeNode.mutate.__dict__.__setitem__('stypy_param_names_list', [])
        TreeNode.mutate.__dict__.__setitem__('stypy_varargs_param_name', None)
        TreeNode.mutate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TreeNode.mutate.__dict__.__setitem__('stypy_call_defaults', defaults)
        TreeNode.mutate.__dict__.__setitem__('stypy_call_varargs', varargs)
        TreeNode.mutate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TreeNode.mutate.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TreeNode.mutate', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mutate', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mutate(...)' code ##################

        str_101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 8), 'str', 'Mutate this node.')
        
        # Getting the type of 'self' (line 52)
        self_102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'self')
        # Obtaining the member 'opcode' of a type (line 52)
        opcode_103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 11), self_102, 'opcode')
        # Getting the type of 'OPCODE_NONE' (line 52)
        OPCODE_NONE_104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 26), 'OPCODE_NONE')
        # Applying the binary operator '==' (line 52)
        result_eq_105 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 11), '==', opcode_103, OPCODE_NONE_104)
        
        # Testing if the type of an if condition is none (line 52)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 52, 8), result_eq_105):
            pass
        else:
            
            # Testing the type of an if condition (line 52)
            if_condition_106 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 52, 8), result_eq_105)
            # Assigning a type to the variable 'if_condition_106' (line 52)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'if_condition_106', if_condition_106)
            # SSA begins for if statement (line 52)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Assigning a type to the variable 'stypy_return_type' (line 53)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'stypy_return_type', types.NoneType)
            # SSA join for if statement (line 52)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to random(...): (line 55)
        # Processing the call keyword arguments (line 55)
        kwargs_108 = {}
        # Getting the type of 'random' (line 55)
        random_107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 11), 'random', False)
        # Calling random(args, kwargs) (line 55)
        random_call_result_109 = invoke(stypy.reporting.localization.Localization(__file__, 55, 11), random_107, *[], **kwargs_108)
        
        float_110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 22), 'float')
        # Applying the binary operator '>' (line 55)
        result_gt_111 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 11), '>', random_call_result_109, float_110)
        
        # Testing if the type of an if condition is none (line 55)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 55, 8), result_gt_111):
            
            # Call to make_random_genome(...): (line 60)
            # Processing the call arguments (line 60)
            # Getting the type of 'self' (line 60)
            self_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 31), 'self', False)
            # Getting the type of 'MAX_DEPTH' (line 60)
            MAX_DEPTH_120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 37), 'MAX_DEPTH', False)
            int_121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 47), 'int')
            # Applying the binary operator '-' (line 60)
            result_sub_122 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 37), '-', MAX_DEPTH_120, int_121)
            
            # Processing the call keyword arguments (line 60)
            kwargs_123 = {}
            # Getting the type of 'make_random_genome' (line 60)
            make_random_genome_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'make_random_genome', False)
            # Calling make_random_genome(args, kwargs) (line 60)
            make_random_genome_call_result_124 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), make_random_genome_118, *[self_119, result_sub_122], **kwargs_123)
            
        else:
            
            # Testing the type of an if condition (line 55)
            if_condition_112 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 55, 8), result_gt_111)
            # Assigning a type to the variable 'if_condition_112' (line 55)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'if_condition_112', if_condition_112)
            # SSA begins for if statement (line 55)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to make_random_genome(...): (line 57)
            # Processing the call arguments (line 57)
            # Getting the type of 'self' (line 57)
            self_114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 31), 'self', False)
            # Getting the type of 'MAX_DEPTH' (line 57)
            MAX_DEPTH_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 37), 'MAX_DEPTH', False)
            # Processing the call keyword arguments (line 57)
            kwargs_116 = {}
            # Getting the type of 'make_random_genome' (line 57)
            make_random_genome_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'make_random_genome', False)
            # Calling make_random_genome(args, kwargs) (line 57)
            make_random_genome_call_result_117 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), make_random_genome_113, *[self_114, MAX_DEPTH_115], **kwargs_116)
            
            # SSA branch for the else part of an if statement (line 55)
            module_type_store.open_ssa_branch('else')
            
            # Call to make_random_genome(...): (line 60)
            # Processing the call arguments (line 60)
            # Getting the type of 'self' (line 60)
            self_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 31), 'self', False)
            # Getting the type of 'MAX_DEPTH' (line 60)
            MAX_DEPTH_120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 37), 'MAX_DEPTH', False)
            int_121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 47), 'int')
            # Applying the binary operator '-' (line 60)
            result_sub_122 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 37), '-', MAX_DEPTH_120, int_121)
            
            # Processing the call keyword arguments (line 60)
            kwargs_123 = {}
            # Getting the type of 'make_random_genome' (line 60)
            make_random_genome_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'make_random_genome', False)
            # Calling make_random_genome(args, kwargs) (line 60)
            make_random_genome_call_result_124 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), make_random_genome_118, *[self_119, result_sub_122], **kwargs_123)
            
            # SSA join for if statement (line 55)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'mutate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mutate' in the type store
        # Getting the type of 'stypy_return_type' (line 49)
        stypy_return_type_125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_125)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mutate'
        return stypy_return_type_125


    @norecursion
    def execute(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'execute'
        module_type_store = module_type_store.open_function_context('execute', 63, 4, False)
        # Assigning a type to the variable 'self' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TreeNode.execute.__dict__.__setitem__('stypy_localization', localization)
        TreeNode.execute.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TreeNode.execute.__dict__.__setitem__('stypy_type_store', module_type_store)
        TreeNode.execute.__dict__.__setitem__('stypy_function_name', 'TreeNode.execute')
        TreeNode.execute.__dict__.__setitem__('stypy_param_names_list', ['input'])
        TreeNode.execute.__dict__.__setitem__('stypy_varargs_param_name', None)
        TreeNode.execute.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TreeNode.execute.__dict__.__setitem__('stypy_call_defaults', defaults)
        TreeNode.execute.__dict__.__setitem__('stypy_call_varargs', varargs)
        TreeNode.execute.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TreeNode.execute.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TreeNode.execute', ['input'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'execute', localization, ['input'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'execute(...)' code ##################

        
        # Getting the type of 'self' (line 64)
        self_126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 'self')
        # Obtaining the member 'opcode' of a type (line 64)
        opcode_127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 11), self_126, 'opcode')
        # Getting the type of 'OPCODE_NONE' (line 64)
        OPCODE_NONE_128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 26), 'OPCODE_NONE')
        # Applying the binary operator '==' (line 64)
        result_eq_129 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 11), '==', opcode_127, OPCODE_NONE_128)
        
        # Testing if the type of an if condition is none (line 64)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 64, 8), result_eq_129):
            
            # Getting the type of 'self' (line 66)
            self_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 13), 'self')
            # Obtaining the member 'opcode' of a type (line 66)
            opcode_141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 13), self_140, 'opcode')
            # Getting the type of 'OPCODE_AND' (line 66)
            OPCODE_AND_142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 28), 'OPCODE_AND')
            # Applying the binary operator '==' (line 66)
            result_eq_143 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 13), '==', opcode_141, OPCODE_AND_142)
            
            # Testing if the type of an if condition is none (line 66)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 66, 13), result_eq_143):
                
                # Getting the type of 'self' (line 69)
                self_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'self')
                # Obtaining the member 'opcode' of a type (line 69)
                opcode_165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 13), self_164, 'opcode')
                # Getting the type of 'OPCODE_OR' (line 69)
                OPCODE_OR_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 28), 'OPCODE_OR')
                # Applying the binary operator '==' (line 69)
                result_eq_167 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 13), '==', opcode_165, OPCODE_OR_166)
                
                # Testing if the type of an if condition is none (line 69)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 69, 13), result_eq_167):
                    
                    # Getting the type of 'self' (line 72)
                    self_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'self')
                    # Obtaining the member 'opcode' of a type (line 72)
                    opcode_189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 13), self_188, 'opcode')
                    # Getting the type of 'OPCODE_NOT' (line 72)
                    OPCODE_NOT_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 28), 'OPCODE_NOT')
                    # Applying the binary operator '==' (line 72)
                    result_eq_191 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 13), '==', opcode_189, OPCODE_NOT_190)
                    
                    # Testing if the type of an if condition is none (line 72)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 72, 13), result_eq_191):
                        
                        # Getting the type of 'self' (line 74)
                        self_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 74)
                        opcode_205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 13), self_204, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 74)
                        OPCODE_IF_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 74)
                        result_eq_207 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 13), '==', opcode_205, OPCODE_IF_206)
                        
                        # Testing if the type of an if condition is none (line 74)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 74)
                            if_condition_208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207)
                            # Assigning a type to the variable 'if_condition_208' (line 74)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'if_condition_208', if_condition_208)
                            # SSA begins for if statement (line 74)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to execute(...): (line 75)
                            # Processing the call arguments (line 75)
                            # Getting the type of 'input' (line 75)
                            input_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 36), 'input', False)
                            # Processing the call keyword arguments (line 75)
                            kwargs_216 = {}
                            
                            # Obtaining the type of the subscript
                            int_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 25), 'int')
                            # Getting the type of 'self' (line 75)
                            self_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'self', False)
                            # Obtaining the member 'args' of a type (line 75)
                            args_211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), self_210, 'args')
                            # Obtaining the member '__getitem__' of a type (line 75)
                            getitem___212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), args_211, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                            subscript_call_result_213 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), getitem___212, int_209)
                            
                            # Obtaining the member 'execute' of a type (line 75)
                            execute_214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), subscript_call_result_213, 'execute')
                            # Calling execute(args, kwargs) (line 75)
                            execute_call_result_217 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), execute_214, *[input_215], **kwargs_216)
                            
                            # Testing if the type of an if condition is none (line 75)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217):
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                            else:
                                
                                # Testing the type of an if condition (line 75)
                                if_condition_218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217)
                                # Assigning a type to the variable 'if_condition_218' (line 75)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'if_condition_218', if_condition_218)
                                # SSA begins for if statement (line 75)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to execute(...): (line 76)
                                # Processing the call arguments (line 76)
                                # Getting the type of 'input' (line 76)
                                input_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 44), 'input', False)
                                # Processing the call keyword arguments (line 76)
                                kwargs_226 = {}
                                
                                # Obtaining the type of the subscript
                                int_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 33), 'int')
                                # Getting the type of 'self' (line 76)
                                self_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 76)
                                args_221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), self_220, 'args')
                                # Obtaining the member '__getitem__' of a type (line 76)
                                getitem___222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), args_221, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 76)
                                subscript_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), getitem___222, int_219)
                                
                                # Obtaining the member 'execute' of a type (line 76)
                                execute_224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), subscript_call_result_223, 'execute')
                                # Calling execute(args, kwargs) (line 76)
                                execute_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), execute_224, *[input_225], **kwargs_226)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 76)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'stypy_return_type', execute_call_result_227)
                                # SSA branch for the else part of an if statement (line 75)
                                module_type_store.open_ssa_branch('else')
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                                # SSA join for if statement (line 75)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 74)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 72)
                        if_condition_192 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 13), result_eq_191)
                        # Assigning a type to the variable 'if_condition_192' (line 72)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'if_condition_192', if_condition_192)
                        # SSA begins for if statement (line 72)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        int_193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 19), 'int')
                        
                        # Call to execute(...): (line 73)
                        # Processing the call arguments (line 73)
                        # Getting the type of 'input' (line 73)
                        input_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 44), 'input', False)
                        # Processing the call keyword arguments (line 73)
                        kwargs_201 = {}
                        
                        # Obtaining the type of the subscript
                        int_194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 33), 'int')
                        # Getting the type of 'self' (line 73)
                        self_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'self', False)
                        # Obtaining the member 'args' of a type (line 73)
                        args_196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), self_195, 'args')
                        # Obtaining the member '__getitem__' of a type (line 73)
                        getitem___197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), args_196, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 73)
                        subscript_call_result_198 = invoke(stypy.reporting.localization.Localization(__file__, 73, 23), getitem___197, int_194)
                        
                        # Obtaining the member 'execute' of a type (line 73)
                        execute_199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), subscript_call_result_198, 'execute')
                        # Calling execute(args, kwargs) (line 73)
                        execute_call_result_202 = invoke(stypy.reporting.localization.Localization(__file__, 73, 23), execute_199, *[input_200], **kwargs_201)
                        
                        # Applying the binary operator '^' (line 73)
                        result_xor_203 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 19), '^', int_193, execute_call_result_202)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 73)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'stypy_return_type', result_xor_203)
                        # SSA branch for the else part of an if statement (line 72)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'self' (line 74)
                        self_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 74)
                        opcode_205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 13), self_204, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 74)
                        OPCODE_IF_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 74)
                        result_eq_207 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 13), '==', opcode_205, OPCODE_IF_206)
                        
                        # Testing if the type of an if condition is none (line 74)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 74)
                            if_condition_208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207)
                            # Assigning a type to the variable 'if_condition_208' (line 74)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'if_condition_208', if_condition_208)
                            # SSA begins for if statement (line 74)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to execute(...): (line 75)
                            # Processing the call arguments (line 75)
                            # Getting the type of 'input' (line 75)
                            input_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 36), 'input', False)
                            # Processing the call keyword arguments (line 75)
                            kwargs_216 = {}
                            
                            # Obtaining the type of the subscript
                            int_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 25), 'int')
                            # Getting the type of 'self' (line 75)
                            self_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'self', False)
                            # Obtaining the member 'args' of a type (line 75)
                            args_211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), self_210, 'args')
                            # Obtaining the member '__getitem__' of a type (line 75)
                            getitem___212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), args_211, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                            subscript_call_result_213 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), getitem___212, int_209)
                            
                            # Obtaining the member 'execute' of a type (line 75)
                            execute_214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), subscript_call_result_213, 'execute')
                            # Calling execute(args, kwargs) (line 75)
                            execute_call_result_217 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), execute_214, *[input_215], **kwargs_216)
                            
                            # Testing if the type of an if condition is none (line 75)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217):
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                            else:
                                
                                # Testing the type of an if condition (line 75)
                                if_condition_218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217)
                                # Assigning a type to the variable 'if_condition_218' (line 75)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'if_condition_218', if_condition_218)
                                # SSA begins for if statement (line 75)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to execute(...): (line 76)
                                # Processing the call arguments (line 76)
                                # Getting the type of 'input' (line 76)
                                input_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 44), 'input', False)
                                # Processing the call keyword arguments (line 76)
                                kwargs_226 = {}
                                
                                # Obtaining the type of the subscript
                                int_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 33), 'int')
                                # Getting the type of 'self' (line 76)
                                self_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 76)
                                args_221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), self_220, 'args')
                                # Obtaining the member '__getitem__' of a type (line 76)
                                getitem___222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), args_221, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 76)
                                subscript_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), getitem___222, int_219)
                                
                                # Obtaining the member 'execute' of a type (line 76)
                                execute_224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), subscript_call_result_223, 'execute')
                                # Calling execute(args, kwargs) (line 76)
                                execute_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), execute_224, *[input_225], **kwargs_226)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 76)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'stypy_return_type', execute_call_result_227)
                                # SSA branch for the else part of an if statement (line 75)
                                module_type_store.open_ssa_branch('else')
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                                # SSA join for if statement (line 75)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 74)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 72)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 69)
                    if_condition_168 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 13), result_eq_167)
                    # Assigning a type to the variable 'if_condition_168' (line 69)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'if_condition_168', if_condition_168)
                    # SSA begins for if statement (line 69)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to execute(...): (line 70)
                    # Processing the call arguments (line 70)
                    # Getting the type of 'input' (line 70)
                    input_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 40), 'input', False)
                    # Processing the call keyword arguments (line 70)
                    kwargs_176 = {}
                    
                    # Obtaining the type of the subscript
                    int_169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 29), 'int')
                    # Getting the type of 'self' (line 70)
                    self_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 19), 'self', False)
                    # Obtaining the member 'args' of a type (line 70)
                    args_171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 19), self_170, 'args')
                    # Obtaining the member '__getitem__' of a type (line 70)
                    getitem___172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 19), args_171, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 70)
                    subscript_call_result_173 = invoke(stypy.reporting.localization.Localization(__file__, 70, 19), getitem___172, int_169)
                    
                    # Obtaining the member 'execute' of a type (line 70)
                    execute_174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 19), subscript_call_result_173, 'execute')
                    # Calling execute(args, kwargs) (line 70)
                    execute_call_result_177 = invoke(stypy.reporting.localization.Localization(__file__, 70, 19), execute_174, *[input_175], **kwargs_176)
                    
                    
                    # Call to execute(...): (line 71)
                    # Processing the call arguments (line 71)
                    # Getting the type of 'input' (line 71)
                    input_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 40), 'input', False)
                    # Processing the call keyword arguments (line 71)
                    kwargs_185 = {}
                    
                    # Obtaining the type of the subscript
                    int_178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 29), 'int')
                    # Getting the type of 'self' (line 71)
                    self_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'self', False)
                    # Obtaining the member 'args' of a type (line 71)
                    args_180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 19), self_179, 'args')
                    # Obtaining the member '__getitem__' of a type (line 71)
                    getitem___181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 19), args_180, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 71)
                    subscript_call_result_182 = invoke(stypy.reporting.localization.Localization(__file__, 71, 19), getitem___181, int_178)
                    
                    # Obtaining the member 'execute' of a type (line 71)
                    execute_183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 19), subscript_call_result_182, 'execute')
                    # Calling execute(args, kwargs) (line 71)
                    execute_call_result_186 = invoke(stypy.reporting.localization.Localization(__file__, 71, 19), execute_183, *[input_184], **kwargs_185)
                    
                    # Applying the binary operator '|' (line 70)
                    result_or__187 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 19), '|', execute_call_result_177, execute_call_result_186)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 70)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'stypy_return_type', result_or__187)
                    # SSA branch for the else part of an if statement (line 69)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'self' (line 72)
                    self_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'self')
                    # Obtaining the member 'opcode' of a type (line 72)
                    opcode_189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 13), self_188, 'opcode')
                    # Getting the type of 'OPCODE_NOT' (line 72)
                    OPCODE_NOT_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 28), 'OPCODE_NOT')
                    # Applying the binary operator '==' (line 72)
                    result_eq_191 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 13), '==', opcode_189, OPCODE_NOT_190)
                    
                    # Testing if the type of an if condition is none (line 72)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 72, 13), result_eq_191):
                        
                        # Getting the type of 'self' (line 74)
                        self_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 74)
                        opcode_205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 13), self_204, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 74)
                        OPCODE_IF_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 74)
                        result_eq_207 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 13), '==', opcode_205, OPCODE_IF_206)
                        
                        # Testing if the type of an if condition is none (line 74)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 74)
                            if_condition_208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207)
                            # Assigning a type to the variable 'if_condition_208' (line 74)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'if_condition_208', if_condition_208)
                            # SSA begins for if statement (line 74)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to execute(...): (line 75)
                            # Processing the call arguments (line 75)
                            # Getting the type of 'input' (line 75)
                            input_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 36), 'input', False)
                            # Processing the call keyword arguments (line 75)
                            kwargs_216 = {}
                            
                            # Obtaining the type of the subscript
                            int_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 25), 'int')
                            # Getting the type of 'self' (line 75)
                            self_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'self', False)
                            # Obtaining the member 'args' of a type (line 75)
                            args_211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), self_210, 'args')
                            # Obtaining the member '__getitem__' of a type (line 75)
                            getitem___212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), args_211, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                            subscript_call_result_213 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), getitem___212, int_209)
                            
                            # Obtaining the member 'execute' of a type (line 75)
                            execute_214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), subscript_call_result_213, 'execute')
                            # Calling execute(args, kwargs) (line 75)
                            execute_call_result_217 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), execute_214, *[input_215], **kwargs_216)
                            
                            # Testing if the type of an if condition is none (line 75)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217):
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                            else:
                                
                                # Testing the type of an if condition (line 75)
                                if_condition_218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217)
                                # Assigning a type to the variable 'if_condition_218' (line 75)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'if_condition_218', if_condition_218)
                                # SSA begins for if statement (line 75)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to execute(...): (line 76)
                                # Processing the call arguments (line 76)
                                # Getting the type of 'input' (line 76)
                                input_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 44), 'input', False)
                                # Processing the call keyword arguments (line 76)
                                kwargs_226 = {}
                                
                                # Obtaining the type of the subscript
                                int_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 33), 'int')
                                # Getting the type of 'self' (line 76)
                                self_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 76)
                                args_221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), self_220, 'args')
                                # Obtaining the member '__getitem__' of a type (line 76)
                                getitem___222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), args_221, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 76)
                                subscript_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), getitem___222, int_219)
                                
                                # Obtaining the member 'execute' of a type (line 76)
                                execute_224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), subscript_call_result_223, 'execute')
                                # Calling execute(args, kwargs) (line 76)
                                execute_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), execute_224, *[input_225], **kwargs_226)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 76)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'stypy_return_type', execute_call_result_227)
                                # SSA branch for the else part of an if statement (line 75)
                                module_type_store.open_ssa_branch('else')
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                                # SSA join for if statement (line 75)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 74)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 72)
                        if_condition_192 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 13), result_eq_191)
                        # Assigning a type to the variable 'if_condition_192' (line 72)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'if_condition_192', if_condition_192)
                        # SSA begins for if statement (line 72)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        int_193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 19), 'int')
                        
                        # Call to execute(...): (line 73)
                        # Processing the call arguments (line 73)
                        # Getting the type of 'input' (line 73)
                        input_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 44), 'input', False)
                        # Processing the call keyword arguments (line 73)
                        kwargs_201 = {}
                        
                        # Obtaining the type of the subscript
                        int_194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 33), 'int')
                        # Getting the type of 'self' (line 73)
                        self_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'self', False)
                        # Obtaining the member 'args' of a type (line 73)
                        args_196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), self_195, 'args')
                        # Obtaining the member '__getitem__' of a type (line 73)
                        getitem___197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), args_196, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 73)
                        subscript_call_result_198 = invoke(stypy.reporting.localization.Localization(__file__, 73, 23), getitem___197, int_194)
                        
                        # Obtaining the member 'execute' of a type (line 73)
                        execute_199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), subscript_call_result_198, 'execute')
                        # Calling execute(args, kwargs) (line 73)
                        execute_call_result_202 = invoke(stypy.reporting.localization.Localization(__file__, 73, 23), execute_199, *[input_200], **kwargs_201)
                        
                        # Applying the binary operator '^' (line 73)
                        result_xor_203 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 19), '^', int_193, execute_call_result_202)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 73)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'stypy_return_type', result_xor_203)
                        # SSA branch for the else part of an if statement (line 72)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'self' (line 74)
                        self_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 74)
                        opcode_205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 13), self_204, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 74)
                        OPCODE_IF_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 74)
                        result_eq_207 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 13), '==', opcode_205, OPCODE_IF_206)
                        
                        # Testing if the type of an if condition is none (line 74)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 74)
                            if_condition_208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207)
                            # Assigning a type to the variable 'if_condition_208' (line 74)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'if_condition_208', if_condition_208)
                            # SSA begins for if statement (line 74)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to execute(...): (line 75)
                            # Processing the call arguments (line 75)
                            # Getting the type of 'input' (line 75)
                            input_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 36), 'input', False)
                            # Processing the call keyword arguments (line 75)
                            kwargs_216 = {}
                            
                            # Obtaining the type of the subscript
                            int_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 25), 'int')
                            # Getting the type of 'self' (line 75)
                            self_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'self', False)
                            # Obtaining the member 'args' of a type (line 75)
                            args_211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), self_210, 'args')
                            # Obtaining the member '__getitem__' of a type (line 75)
                            getitem___212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), args_211, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                            subscript_call_result_213 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), getitem___212, int_209)
                            
                            # Obtaining the member 'execute' of a type (line 75)
                            execute_214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), subscript_call_result_213, 'execute')
                            # Calling execute(args, kwargs) (line 75)
                            execute_call_result_217 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), execute_214, *[input_215], **kwargs_216)
                            
                            # Testing if the type of an if condition is none (line 75)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217):
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                            else:
                                
                                # Testing the type of an if condition (line 75)
                                if_condition_218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217)
                                # Assigning a type to the variable 'if_condition_218' (line 75)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'if_condition_218', if_condition_218)
                                # SSA begins for if statement (line 75)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to execute(...): (line 76)
                                # Processing the call arguments (line 76)
                                # Getting the type of 'input' (line 76)
                                input_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 44), 'input', False)
                                # Processing the call keyword arguments (line 76)
                                kwargs_226 = {}
                                
                                # Obtaining the type of the subscript
                                int_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 33), 'int')
                                # Getting the type of 'self' (line 76)
                                self_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 76)
                                args_221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), self_220, 'args')
                                # Obtaining the member '__getitem__' of a type (line 76)
                                getitem___222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), args_221, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 76)
                                subscript_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), getitem___222, int_219)
                                
                                # Obtaining the member 'execute' of a type (line 76)
                                execute_224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), subscript_call_result_223, 'execute')
                                # Calling execute(args, kwargs) (line 76)
                                execute_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), execute_224, *[input_225], **kwargs_226)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 76)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'stypy_return_type', execute_call_result_227)
                                # SSA branch for the else part of an if statement (line 75)
                                module_type_store.open_ssa_branch('else')
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                                # SSA join for if statement (line 75)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 74)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 72)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 69)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 66)
                if_condition_144 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 13), result_eq_143)
                # Assigning a type to the variable 'if_condition_144' (line 66)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 13), 'if_condition_144', if_condition_144)
                # SSA begins for if statement (line 66)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to execute(...): (line 67)
                # Processing the call arguments (line 67)
                # Getting the type of 'input' (line 67)
                input_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 40), 'input', False)
                # Processing the call keyword arguments (line 67)
                kwargs_152 = {}
                
                # Obtaining the type of the subscript
                int_145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 29), 'int')
                # Getting the type of 'self' (line 67)
                self_146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 19), 'self', False)
                # Obtaining the member 'args' of a type (line 67)
                args_147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 19), self_146, 'args')
                # Obtaining the member '__getitem__' of a type (line 67)
                getitem___148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 19), args_147, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 67)
                subscript_call_result_149 = invoke(stypy.reporting.localization.Localization(__file__, 67, 19), getitem___148, int_145)
                
                # Obtaining the member 'execute' of a type (line 67)
                execute_150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 19), subscript_call_result_149, 'execute')
                # Calling execute(args, kwargs) (line 67)
                execute_call_result_153 = invoke(stypy.reporting.localization.Localization(__file__, 67, 19), execute_150, *[input_151], **kwargs_152)
                
                
                # Call to execute(...): (line 68)
                # Processing the call arguments (line 68)
                # Getting the type of 'input' (line 68)
                input_160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 40), 'input', False)
                # Processing the call keyword arguments (line 68)
                kwargs_161 = {}
                
                # Obtaining the type of the subscript
                int_154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 29), 'int')
                # Getting the type of 'self' (line 68)
                self_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 19), 'self', False)
                # Obtaining the member 'args' of a type (line 68)
                args_156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 19), self_155, 'args')
                # Obtaining the member '__getitem__' of a type (line 68)
                getitem___157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 19), args_156, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 68)
                subscript_call_result_158 = invoke(stypy.reporting.localization.Localization(__file__, 68, 19), getitem___157, int_154)
                
                # Obtaining the member 'execute' of a type (line 68)
                execute_159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 19), subscript_call_result_158, 'execute')
                # Calling execute(args, kwargs) (line 68)
                execute_call_result_162 = invoke(stypy.reporting.localization.Localization(__file__, 68, 19), execute_159, *[input_160], **kwargs_161)
                
                # Applying the binary operator '&' (line 67)
                result_and__163 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 19), '&', execute_call_result_153, execute_call_result_162)
                
                # Assigning a type to the variable 'stypy_return_type' (line 67)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'stypy_return_type', result_and__163)
                # SSA branch for the else part of an if statement (line 66)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'self' (line 69)
                self_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'self')
                # Obtaining the member 'opcode' of a type (line 69)
                opcode_165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 13), self_164, 'opcode')
                # Getting the type of 'OPCODE_OR' (line 69)
                OPCODE_OR_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 28), 'OPCODE_OR')
                # Applying the binary operator '==' (line 69)
                result_eq_167 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 13), '==', opcode_165, OPCODE_OR_166)
                
                # Testing if the type of an if condition is none (line 69)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 69, 13), result_eq_167):
                    
                    # Getting the type of 'self' (line 72)
                    self_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'self')
                    # Obtaining the member 'opcode' of a type (line 72)
                    opcode_189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 13), self_188, 'opcode')
                    # Getting the type of 'OPCODE_NOT' (line 72)
                    OPCODE_NOT_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 28), 'OPCODE_NOT')
                    # Applying the binary operator '==' (line 72)
                    result_eq_191 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 13), '==', opcode_189, OPCODE_NOT_190)
                    
                    # Testing if the type of an if condition is none (line 72)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 72, 13), result_eq_191):
                        
                        # Getting the type of 'self' (line 74)
                        self_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 74)
                        opcode_205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 13), self_204, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 74)
                        OPCODE_IF_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 74)
                        result_eq_207 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 13), '==', opcode_205, OPCODE_IF_206)
                        
                        # Testing if the type of an if condition is none (line 74)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 74)
                            if_condition_208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207)
                            # Assigning a type to the variable 'if_condition_208' (line 74)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'if_condition_208', if_condition_208)
                            # SSA begins for if statement (line 74)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to execute(...): (line 75)
                            # Processing the call arguments (line 75)
                            # Getting the type of 'input' (line 75)
                            input_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 36), 'input', False)
                            # Processing the call keyword arguments (line 75)
                            kwargs_216 = {}
                            
                            # Obtaining the type of the subscript
                            int_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 25), 'int')
                            # Getting the type of 'self' (line 75)
                            self_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'self', False)
                            # Obtaining the member 'args' of a type (line 75)
                            args_211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), self_210, 'args')
                            # Obtaining the member '__getitem__' of a type (line 75)
                            getitem___212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), args_211, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                            subscript_call_result_213 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), getitem___212, int_209)
                            
                            # Obtaining the member 'execute' of a type (line 75)
                            execute_214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), subscript_call_result_213, 'execute')
                            # Calling execute(args, kwargs) (line 75)
                            execute_call_result_217 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), execute_214, *[input_215], **kwargs_216)
                            
                            # Testing if the type of an if condition is none (line 75)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217):
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                            else:
                                
                                # Testing the type of an if condition (line 75)
                                if_condition_218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217)
                                # Assigning a type to the variable 'if_condition_218' (line 75)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'if_condition_218', if_condition_218)
                                # SSA begins for if statement (line 75)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to execute(...): (line 76)
                                # Processing the call arguments (line 76)
                                # Getting the type of 'input' (line 76)
                                input_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 44), 'input', False)
                                # Processing the call keyword arguments (line 76)
                                kwargs_226 = {}
                                
                                # Obtaining the type of the subscript
                                int_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 33), 'int')
                                # Getting the type of 'self' (line 76)
                                self_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 76)
                                args_221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), self_220, 'args')
                                # Obtaining the member '__getitem__' of a type (line 76)
                                getitem___222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), args_221, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 76)
                                subscript_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), getitem___222, int_219)
                                
                                # Obtaining the member 'execute' of a type (line 76)
                                execute_224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), subscript_call_result_223, 'execute')
                                # Calling execute(args, kwargs) (line 76)
                                execute_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), execute_224, *[input_225], **kwargs_226)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 76)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'stypy_return_type', execute_call_result_227)
                                # SSA branch for the else part of an if statement (line 75)
                                module_type_store.open_ssa_branch('else')
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                                # SSA join for if statement (line 75)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 74)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 72)
                        if_condition_192 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 13), result_eq_191)
                        # Assigning a type to the variable 'if_condition_192' (line 72)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'if_condition_192', if_condition_192)
                        # SSA begins for if statement (line 72)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        int_193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 19), 'int')
                        
                        # Call to execute(...): (line 73)
                        # Processing the call arguments (line 73)
                        # Getting the type of 'input' (line 73)
                        input_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 44), 'input', False)
                        # Processing the call keyword arguments (line 73)
                        kwargs_201 = {}
                        
                        # Obtaining the type of the subscript
                        int_194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 33), 'int')
                        # Getting the type of 'self' (line 73)
                        self_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'self', False)
                        # Obtaining the member 'args' of a type (line 73)
                        args_196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), self_195, 'args')
                        # Obtaining the member '__getitem__' of a type (line 73)
                        getitem___197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), args_196, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 73)
                        subscript_call_result_198 = invoke(stypy.reporting.localization.Localization(__file__, 73, 23), getitem___197, int_194)
                        
                        # Obtaining the member 'execute' of a type (line 73)
                        execute_199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), subscript_call_result_198, 'execute')
                        # Calling execute(args, kwargs) (line 73)
                        execute_call_result_202 = invoke(stypy.reporting.localization.Localization(__file__, 73, 23), execute_199, *[input_200], **kwargs_201)
                        
                        # Applying the binary operator '^' (line 73)
                        result_xor_203 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 19), '^', int_193, execute_call_result_202)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 73)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'stypy_return_type', result_xor_203)
                        # SSA branch for the else part of an if statement (line 72)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'self' (line 74)
                        self_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 74)
                        opcode_205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 13), self_204, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 74)
                        OPCODE_IF_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 74)
                        result_eq_207 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 13), '==', opcode_205, OPCODE_IF_206)
                        
                        # Testing if the type of an if condition is none (line 74)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 74)
                            if_condition_208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207)
                            # Assigning a type to the variable 'if_condition_208' (line 74)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'if_condition_208', if_condition_208)
                            # SSA begins for if statement (line 74)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to execute(...): (line 75)
                            # Processing the call arguments (line 75)
                            # Getting the type of 'input' (line 75)
                            input_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 36), 'input', False)
                            # Processing the call keyword arguments (line 75)
                            kwargs_216 = {}
                            
                            # Obtaining the type of the subscript
                            int_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 25), 'int')
                            # Getting the type of 'self' (line 75)
                            self_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'self', False)
                            # Obtaining the member 'args' of a type (line 75)
                            args_211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), self_210, 'args')
                            # Obtaining the member '__getitem__' of a type (line 75)
                            getitem___212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), args_211, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                            subscript_call_result_213 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), getitem___212, int_209)
                            
                            # Obtaining the member 'execute' of a type (line 75)
                            execute_214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), subscript_call_result_213, 'execute')
                            # Calling execute(args, kwargs) (line 75)
                            execute_call_result_217 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), execute_214, *[input_215], **kwargs_216)
                            
                            # Testing if the type of an if condition is none (line 75)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217):
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                            else:
                                
                                # Testing the type of an if condition (line 75)
                                if_condition_218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217)
                                # Assigning a type to the variable 'if_condition_218' (line 75)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'if_condition_218', if_condition_218)
                                # SSA begins for if statement (line 75)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to execute(...): (line 76)
                                # Processing the call arguments (line 76)
                                # Getting the type of 'input' (line 76)
                                input_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 44), 'input', False)
                                # Processing the call keyword arguments (line 76)
                                kwargs_226 = {}
                                
                                # Obtaining the type of the subscript
                                int_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 33), 'int')
                                # Getting the type of 'self' (line 76)
                                self_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 76)
                                args_221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), self_220, 'args')
                                # Obtaining the member '__getitem__' of a type (line 76)
                                getitem___222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), args_221, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 76)
                                subscript_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), getitem___222, int_219)
                                
                                # Obtaining the member 'execute' of a type (line 76)
                                execute_224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), subscript_call_result_223, 'execute')
                                # Calling execute(args, kwargs) (line 76)
                                execute_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), execute_224, *[input_225], **kwargs_226)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 76)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'stypy_return_type', execute_call_result_227)
                                # SSA branch for the else part of an if statement (line 75)
                                module_type_store.open_ssa_branch('else')
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                                # SSA join for if statement (line 75)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 74)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 72)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 69)
                    if_condition_168 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 13), result_eq_167)
                    # Assigning a type to the variable 'if_condition_168' (line 69)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'if_condition_168', if_condition_168)
                    # SSA begins for if statement (line 69)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to execute(...): (line 70)
                    # Processing the call arguments (line 70)
                    # Getting the type of 'input' (line 70)
                    input_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 40), 'input', False)
                    # Processing the call keyword arguments (line 70)
                    kwargs_176 = {}
                    
                    # Obtaining the type of the subscript
                    int_169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 29), 'int')
                    # Getting the type of 'self' (line 70)
                    self_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 19), 'self', False)
                    # Obtaining the member 'args' of a type (line 70)
                    args_171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 19), self_170, 'args')
                    # Obtaining the member '__getitem__' of a type (line 70)
                    getitem___172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 19), args_171, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 70)
                    subscript_call_result_173 = invoke(stypy.reporting.localization.Localization(__file__, 70, 19), getitem___172, int_169)
                    
                    # Obtaining the member 'execute' of a type (line 70)
                    execute_174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 19), subscript_call_result_173, 'execute')
                    # Calling execute(args, kwargs) (line 70)
                    execute_call_result_177 = invoke(stypy.reporting.localization.Localization(__file__, 70, 19), execute_174, *[input_175], **kwargs_176)
                    
                    
                    # Call to execute(...): (line 71)
                    # Processing the call arguments (line 71)
                    # Getting the type of 'input' (line 71)
                    input_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 40), 'input', False)
                    # Processing the call keyword arguments (line 71)
                    kwargs_185 = {}
                    
                    # Obtaining the type of the subscript
                    int_178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 29), 'int')
                    # Getting the type of 'self' (line 71)
                    self_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'self', False)
                    # Obtaining the member 'args' of a type (line 71)
                    args_180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 19), self_179, 'args')
                    # Obtaining the member '__getitem__' of a type (line 71)
                    getitem___181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 19), args_180, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 71)
                    subscript_call_result_182 = invoke(stypy.reporting.localization.Localization(__file__, 71, 19), getitem___181, int_178)
                    
                    # Obtaining the member 'execute' of a type (line 71)
                    execute_183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 19), subscript_call_result_182, 'execute')
                    # Calling execute(args, kwargs) (line 71)
                    execute_call_result_186 = invoke(stypy.reporting.localization.Localization(__file__, 71, 19), execute_183, *[input_184], **kwargs_185)
                    
                    # Applying the binary operator '|' (line 70)
                    result_or__187 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 19), '|', execute_call_result_177, execute_call_result_186)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 70)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'stypy_return_type', result_or__187)
                    # SSA branch for the else part of an if statement (line 69)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'self' (line 72)
                    self_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'self')
                    # Obtaining the member 'opcode' of a type (line 72)
                    opcode_189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 13), self_188, 'opcode')
                    # Getting the type of 'OPCODE_NOT' (line 72)
                    OPCODE_NOT_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 28), 'OPCODE_NOT')
                    # Applying the binary operator '==' (line 72)
                    result_eq_191 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 13), '==', opcode_189, OPCODE_NOT_190)
                    
                    # Testing if the type of an if condition is none (line 72)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 72, 13), result_eq_191):
                        
                        # Getting the type of 'self' (line 74)
                        self_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 74)
                        opcode_205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 13), self_204, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 74)
                        OPCODE_IF_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 74)
                        result_eq_207 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 13), '==', opcode_205, OPCODE_IF_206)
                        
                        # Testing if the type of an if condition is none (line 74)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 74)
                            if_condition_208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207)
                            # Assigning a type to the variable 'if_condition_208' (line 74)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'if_condition_208', if_condition_208)
                            # SSA begins for if statement (line 74)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to execute(...): (line 75)
                            # Processing the call arguments (line 75)
                            # Getting the type of 'input' (line 75)
                            input_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 36), 'input', False)
                            # Processing the call keyword arguments (line 75)
                            kwargs_216 = {}
                            
                            # Obtaining the type of the subscript
                            int_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 25), 'int')
                            # Getting the type of 'self' (line 75)
                            self_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'self', False)
                            # Obtaining the member 'args' of a type (line 75)
                            args_211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), self_210, 'args')
                            # Obtaining the member '__getitem__' of a type (line 75)
                            getitem___212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), args_211, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                            subscript_call_result_213 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), getitem___212, int_209)
                            
                            # Obtaining the member 'execute' of a type (line 75)
                            execute_214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), subscript_call_result_213, 'execute')
                            # Calling execute(args, kwargs) (line 75)
                            execute_call_result_217 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), execute_214, *[input_215], **kwargs_216)
                            
                            # Testing if the type of an if condition is none (line 75)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217):
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                            else:
                                
                                # Testing the type of an if condition (line 75)
                                if_condition_218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217)
                                # Assigning a type to the variable 'if_condition_218' (line 75)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'if_condition_218', if_condition_218)
                                # SSA begins for if statement (line 75)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to execute(...): (line 76)
                                # Processing the call arguments (line 76)
                                # Getting the type of 'input' (line 76)
                                input_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 44), 'input', False)
                                # Processing the call keyword arguments (line 76)
                                kwargs_226 = {}
                                
                                # Obtaining the type of the subscript
                                int_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 33), 'int')
                                # Getting the type of 'self' (line 76)
                                self_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 76)
                                args_221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), self_220, 'args')
                                # Obtaining the member '__getitem__' of a type (line 76)
                                getitem___222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), args_221, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 76)
                                subscript_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), getitem___222, int_219)
                                
                                # Obtaining the member 'execute' of a type (line 76)
                                execute_224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), subscript_call_result_223, 'execute')
                                # Calling execute(args, kwargs) (line 76)
                                execute_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), execute_224, *[input_225], **kwargs_226)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 76)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'stypy_return_type', execute_call_result_227)
                                # SSA branch for the else part of an if statement (line 75)
                                module_type_store.open_ssa_branch('else')
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                                # SSA join for if statement (line 75)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 74)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 72)
                        if_condition_192 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 13), result_eq_191)
                        # Assigning a type to the variable 'if_condition_192' (line 72)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'if_condition_192', if_condition_192)
                        # SSA begins for if statement (line 72)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        int_193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 19), 'int')
                        
                        # Call to execute(...): (line 73)
                        # Processing the call arguments (line 73)
                        # Getting the type of 'input' (line 73)
                        input_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 44), 'input', False)
                        # Processing the call keyword arguments (line 73)
                        kwargs_201 = {}
                        
                        # Obtaining the type of the subscript
                        int_194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 33), 'int')
                        # Getting the type of 'self' (line 73)
                        self_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'self', False)
                        # Obtaining the member 'args' of a type (line 73)
                        args_196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), self_195, 'args')
                        # Obtaining the member '__getitem__' of a type (line 73)
                        getitem___197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), args_196, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 73)
                        subscript_call_result_198 = invoke(stypy.reporting.localization.Localization(__file__, 73, 23), getitem___197, int_194)
                        
                        # Obtaining the member 'execute' of a type (line 73)
                        execute_199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), subscript_call_result_198, 'execute')
                        # Calling execute(args, kwargs) (line 73)
                        execute_call_result_202 = invoke(stypy.reporting.localization.Localization(__file__, 73, 23), execute_199, *[input_200], **kwargs_201)
                        
                        # Applying the binary operator '^' (line 73)
                        result_xor_203 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 19), '^', int_193, execute_call_result_202)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 73)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'stypy_return_type', result_xor_203)
                        # SSA branch for the else part of an if statement (line 72)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'self' (line 74)
                        self_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 74)
                        opcode_205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 13), self_204, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 74)
                        OPCODE_IF_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 74)
                        result_eq_207 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 13), '==', opcode_205, OPCODE_IF_206)
                        
                        # Testing if the type of an if condition is none (line 74)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 74)
                            if_condition_208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207)
                            # Assigning a type to the variable 'if_condition_208' (line 74)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'if_condition_208', if_condition_208)
                            # SSA begins for if statement (line 74)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to execute(...): (line 75)
                            # Processing the call arguments (line 75)
                            # Getting the type of 'input' (line 75)
                            input_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 36), 'input', False)
                            # Processing the call keyword arguments (line 75)
                            kwargs_216 = {}
                            
                            # Obtaining the type of the subscript
                            int_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 25), 'int')
                            # Getting the type of 'self' (line 75)
                            self_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'self', False)
                            # Obtaining the member 'args' of a type (line 75)
                            args_211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), self_210, 'args')
                            # Obtaining the member '__getitem__' of a type (line 75)
                            getitem___212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), args_211, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                            subscript_call_result_213 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), getitem___212, int_209)
                            
                            # Obtaining the member 'execute' of a type (line 75)
                            execute_214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), subscript_call_result_213, 'execute')
                            # Calling execute(args, kwargs) (line 75)
                            execute_call_result_217 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), execute_214, *[input_215], **kwargs_216)
                            
                            # Testing if the type of an if condition is none (line 75)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217):
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                            else:
                                
                                # Testing the type of an if condition (line 75)
                                if_condition_218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217)
                                # Assigning a type to the variable 'if_condition_218' (line 75)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'if_condition_218', if_condition_218)
                                # SSA begins for if statement (line 75)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to execute(...): (line 76)
                                # Processing the call arguments (line 76)
                                # Getting the type of 'input' (line 76)
                                input_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 44), 'input', False)
                                # Processing the call keyword arguments (line 76)
                                kwargs_226 = {}
                                
                                # Obtaining the type of the subscript
                                int_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 33), 'int')
                                # Getting the type of 'self' (line 76)
                                self_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 76)
                                args_221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), self_220, 'args')
                                # Obtaining the member '__getitem__' of a type (line 76)
                                getitem___222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), args_221, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 76)
                                subscript_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), getitem___222, int_219)
                                
                                # Obtaining the member 'execute' of a type (line 76)
                                execute_224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), subscript_call_result_223, 'execute')
                                # Calling execute(args, kwargs) (line 76)
                                execute_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), execute_224, *[input_225], **kwargs_226)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 76)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'stypy_return_type', execute_call_result_227)
                                # SSA branch for the else part of an if statement (line 75)
                                module_type_store.open_ssa_branch('else')
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                                # SSA join for if statement (line 75)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 74)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 72)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 69)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 66)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 64)
            if_condition_130 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 8), result_eq_129)
            # Assigning a type to the variable 'if_condition_130' (line 64)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'if_condition_130', if_condition_130)
            # SSA begins for if statement (line 64)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'input' (line 65)
            input_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 20), 'input')
            int_132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 29), 'int')
            # Getting the type of 'self' (line 65)
            self_133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 34), 'self')
            # Obtaining the member 'value' of a type (line 65)
            value_134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 34), self_133, 'value')
            # Applying the binary operator '<<' (line 65)
            result_lshift_135 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 29), '<<', int_132, value_134)
            
            # Applying the binary operator '&' (line 65)
            result_and__136 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 20), '&', input_131, result_lshift_135)
            
            # Getting the type of 'self' (line 65)
            self_137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 50), 'self')
            # Obtaining the member 'value' of a type (line 65)
            value_138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 50), self_137, 'value')
            # Applying the binary operator '>>' (line 65)
            result_rshift_139 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 19), '>>', result_and__136, value_138)
            
            # Assigning a type to the variable 'stypy_return_type' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'stypy_return_type', result_rshift_139)
            # SSA branch for the else part of an if statement (line 64)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'self' (line 66)
            self_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 13), 'self')
            # Obtaining the member 'opcode' of a type (line 66)
            opcode_141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 13), self_140, 'opcode')
            # Getting the type of 'OPCODE_AND' (line 66)
            OPCODE_AND_142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 28), 'OPCODE_AND')
            # Applying the binary operator '==' (line 66)
            result_eq_143 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 13), '==', opcode_141, OPCODE_AND_142)
            
            # Testing if the type of an if condition is none (line 66)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 66, 13), result_eq_143):
                
                # Getting the type of 'self' (line 69)
                self_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'self')
                # Obtaining the member 'opcode' of a type (line 69)
                opcode_165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 13), self_164, 'opcode')
                # Getting the type of 'OPCODE_OR' (line 69)
                OPCODE_OR_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 28), 'OPCODE_OR')
                # Applying the binary operator '==' (line 69)
                result_eq_167 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 13), '==', opcode_165, OPCODE_OR_166)
                
                # Testing if the type of an if condition is none (line 69)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 69, 13), result_eq_167):
                    
                    # Getting the type of 'self' (line 72)
                    self_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'self')
                    # Obtaining the member 'opcode' of a type (line 72)
                    opcode_189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 13), self_188, 'opcode')
                    # Getting the type of 'OPCODE_NOT' (line 72)
                    OPCODE_NOT_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 28), 'OPCODE_NOT')
                    # Applying the binary operator '==' (line 72)
                    result_eq_191 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 13), '==', opcode_189, OPCODE_NOT_190)
                    
                    # Testing if the type of an if condition is none (line 72)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 72, 13), result_eq_191):
                        
                        # Getting the type of 'self' (line 74)
                        self_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 74)
                        opcode_205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 13), self_204, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 74)
                        OPCODE_IF_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 74)
                        result_eq_207 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 13), '==', opcode_205, OPCODE_IF_206)
                        
                        # Testing if the type of an if condition is none (line 74)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 74)
                            if_condition_208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207)
                            # Assigning a type to the variable 'if_condition_208' (line 74)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'if_condition_208', if_condition_208)
                            # SSA begins for if statement (line 74)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to execute(...): (line 75)
                            # Processing the call arguments (line 75)
                            # Getting the type of 'input' (line 75)
                            input_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 36), 'input', False)
                            # Processing the call keyword arguments (line 75)
                            kwargs_216 = {}
                            
                            # Obtaining the type of the subscript
                            int_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 25), 'int')
                            # Getting the type of 'self' (line 75)
                            self_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'self', False)
                            # Obtaining the member 'args' of a type (line 75)
                            args_211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), self_210, 'args')
                            # Obtaining the member '__getitem__' of a type (line 75)
                            getitem___212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), args_211, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                            subscript_call_result_213 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), getitem___212, int_209)
                            
                            # Obtaining the member 'execute' of a type (line 75)
                            execute_214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), subscript_call_result_213, 'execute')
                            # Calling execute(args, kwargs) (line 75)
                            execute_call_result_217 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), execute_214, *[input_215], **kwargs_216)
                            
                            # Testing if the type of an if condition is none (line 75)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217):
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                            else:
                                
                                # Testing the type of an if condition (line 75)
                                if_condition_218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217)
                                # Assigning a type to the variable 'if_condition_218' (line 75)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'if_condition_218', if_condition_218)
                                # SSA begins for if statement (line 75)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to execute(...): (line 76)
                                # Processing the call arguments (line 76)
                                # Getting the type of 'input' (line 76)
                                input_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 44), 'input', False)
                                # Processing the call keyword arguments (line 76)
                                kwargs_226 = {}
                                
                                # Obtaining the type of the subscript
                                int_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 33), 'int')
                                # Getting the type of 'self' (line 76)
                                self_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 76)
                                args_221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), self_220, 'args')
                                # Obtaining the member '__getitem__' of a type (line 76)
                                getitem___222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), args_221, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 76)
                                subscript_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), getitem___222, int_219)
                                
                                # Obtaining the member 'execute' of a type (line 76)
                                execute_224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), subscript_call_result_223, 'execute')
                                # Calling execute(args, kwargs) (line 76)
                                execute_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), execute_224, *[input_225], **kwargs_226)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 76)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'stypy_return_type', execute_call_result_227)
                                # SSA branch for the else part of an if statement (line 75)
                                module_type_store.open_ssa_branch('else')
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                                # SSA join for if statement (line 75)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 74)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 72)
                        if_condition_192 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 13), result_eq_191)
                        # Assigning a type to the variable 'if_condition_192' (line 72)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'if_condition_192', if_condition_192)
                        # SSA begins for if statement (line 72)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        int_193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 19), 'int')
                        
                        # Call to execute(...): (line 73)
                        # Processing the call arguments (line 73)
                        # Getting the type of 'input' (line 73)
                        input_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 44), 'input', False)
                        # Processing the call keyword arguments (line 73)
                        kwargs_201 = {}
                        
                        # Obtaining the type of the subscript
                        int_194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 33), 'int')
                        # Getting the type of 'self' (line 73)
                        self_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'self', False)
                        # Obtaining the member 'args' of a type (line 73)
                        args_196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), self_195, 'args')
                        # Obtaining the member '__getitem__' of a type (line 73)
                        getitem___197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), args_196, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 73)
                        subscript_call_result_198 = invoke(stypy.reporting.localization.Localization(__file__, 73, 23), getitem___197, int_194)
                        
                        # Obtaining the member 'execute' of a type (line 73)
                        execute_199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), subscript_call_result_198, 'execute')
                        # Calling execute(args, kwargs) (line 73)
                        execute_call_result_202 = invoke(stypy.reporting.localization.Localization(__file__, 73, 23), execute_199, *[input_200], **kwargs_201)
                        
                        # Applying the binary operator '^' (line 73)
                        result_xor_203 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 19), '^', int_193, execute_call_result_202)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 73)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'stypy_return_type', result_xor_203)
                        # SSA branch for the else part of an if statement (line 72)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'self' (line 74)
                        self_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 74)
                        opcode_205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 13), self_204, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 74)
                        OPCODE_IF_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 74)
                        result_eq_207 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 13), '==', opcode_205, OPCODE_IF_206)
                        
                        # Testing if the type of an if condition is none (line 74)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 74)
                            if_condition_208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207)
                            # Assigning a type to the variable 'if_condition_208' (line 74)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'if_condition_208', if_condition_208)
                            # SSA begins for if statement (line 74)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to execute(...): (line 75)
                            # Processing the call arguments (line 75)
                            # Getting the type of 'input' (line 75)
                            input_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 36), 'input', False)
                            # Processing the call keyword arguments (line 75)
                            kwargs_216 = {}
                            
                            # Obtaining the type of the subscript
                            int_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 25), 'int')
                            # Getting the type of 'self' (line 75)
                            self_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'self', False)
                            # Obtaining the member 'args' of a type (line 75)
                            args_211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), self_210, 'args')
                            # Obtaining the member '__getitem__' of a type (line 75)
                            getitem___212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), args_211, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                            subscript_call_result_213 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), getitem___212, int_209)
                            
                            # Obtaining the member 'execute' of a type (line 75)
                            execute_214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), subscript_call_result_213, 'execute')
                            # Calling execute(args, kwargs) (line 75)
                            execute_call_result_217 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), execute_214, *[input_215], **kwargs_216)
                            
                            # Testing if the type of an if condition is none (line 75)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217):
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                            else:
                                
                                # Testing the type of an if condition (line 75)
                                if_condition_218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217)
                                # Assigning a type to the variable 'if_condition_218' (line 75)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'if_condition_218', if_condition_218)
                                # SSA begins for if statement (line 75)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to execute(...): (line 76)
                                # Processing the call arguments (line 76)
                                # Getting the type of 'input' (line 76)
                                input_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 44), 'input', False)
                                # Processing the call keyword arguments (line 76)
                                kwargs_226 = {}
                                
                                # Obtaining the type of the subscript
                                int_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 33), 'int')
                                # Getting the type of 'self' (line 76)
                                self_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 76)
                                args_221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), self_220, 'args')
                                # Obtaining the member '__getitem__' of a type (line 76)
                                getitem___222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), args_221, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 76)
                                subscript_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), getitem___222, int_219)
                                
                                # Obtaining the member 'execute' of a type (line 76)
                                execute_224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), subscript_call_result_223, 'execute')
                                # Calling execute(args, kwargs) (line 76)
                                execute_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), execute_224, *[input_225], **kwargs_226)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 76)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'stypy_return_type', execute_call_result_227)
                                # SSA branch for the else part of an if statement (line 75)
                                module_type_store.open_ssa_branch('else')
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                                # SSA join for if statement (line 75)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 74)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 72)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 69)
                    if_condition_168 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 13), result_eq_167)
                    # Assigning a type to the variable 'if_condition_168' (line 69)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'if_condition_168', if_condition_168)
                    # SSA begins for if statement (line 69)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to execute(...): (line 70)
                    # Processing the call arguments (line 70)
                    # Getting the type of 'input' (line 70)
                    input_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 40), 'input', False)
                    # Processing the call keyword arguments (line 70)
                    kwargs_176 = {}
                    
                    # Obtaining the type of the subscript
                    int_169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 29), 'int')
                    # Getting the type of 'self' (line 70)
                    self_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 19), 'self', False)
                    # Obtaining the member 'args' of a type (line 70)
                    args_171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 19), self_170, 'args')
                    # Obtaining the member '__getitem__' of a type (line 70)
                    getitem___172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 19), args_171, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 70)
                    subscript_call_result_173 = invoke(stypy.reporting.localization.Localization(__file__, 70, 19), getitem___172, int_169)
                    
                    # Obtaining the member 'execute' of a type (line 70)
                    execute_174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 19), subscript_call_result_173, 'execute')
                    # Calling execute(args, kwargs) (line 70)
                    execute_call_result_177 = invoke(stypy.reporting.localization.Localization(__file__, 70, 19), execute_174, *[input_175], **kwargs_176)
                    
                    
                    # Call to execute(...): (line 71)
                    # Processing the call arguments (line 71)
                    # Getting the type of 'input' (line 71)
                    input_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 40), 'input', False)
                    # Processing the call keyword arguments (line 71)
                    kwargs_185 = {}
                    
                    # Obtaining the type of the subscript
                    int_178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 29), 'int')
                    # Getting the type of 'self' (line 71)
                    self_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'self', False)
                    # Obtaining the member 'args' of a type (line 71)
                    args_180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 19), self_179, 'args')
                    # Obtaining the member '__getitem__' of a type (line 71)
                    getitem___181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 19), args_180, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 71)
                    subscript_call_result_182 = invoke(stypy.reporting.localization.Localization(__file__, 71, 19), getitem___181, int_178)
                    
                    # Obtaining the member 'execute' of a type (line 71)
                    execute_183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 19), subscript_call_result_182, 'execute')
                    # Calling execute(args, kwargs) (line 71)
                    execute_call_result_186 = invoke(stypy.reporting.localization.Localization(__file__, 71, 19), execute_183, *[input_184], **kwargs_185)
                    
                    # Applying the binary operator '|' (line 70)
                    result_or__187 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 19), '|', execute_call_result_177, execute_call_result_186)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 70)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'stypy_return_type', result_or__187)
                    # SSA branch for the else part of an if statement (line 69)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'self' (line 72)
                    self_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'self')
                    # Obtaining the member 'opcode' of a type (line 72)
                    opcode_189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 13), self_188, 'opcode')
                    # Getting the type of 'OPCODE_NOT' (line 72)
                    OPCODE_NOT_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 28), 'OPCODE_NOT')
                    # Applying the binary operator '==' (line 72)
                    result_eq_191 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 13), '==', opcode_189, OPCODE_NOT_190)
                    
                    # Testing if the type of an if condition is none (line 72)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 72, 13), result_eq_191):
                        
                        # Getting the type of 'self' (line 74)
                        self_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 74)
                        opcode_205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 13), self_204, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 74)
                        OPCODE_IF_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 74)
                        result_eq_207 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 13), '==', opcode_205, OPCODE_IF_206)
                        
                        # Testing if the type of an if condition is none (line 74)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 74)
                            if_condition_208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207)
                            # Assigning a type to the variable 'if_condition_208' (line 74)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'if_condition_208', if_condition_208)
                            # SSA begins for if statement (line 74)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to execute(...): (line 75)
                            # Processing the call arguments (line 75)
                            # Getting the type of 'input' (line 75)
                            input_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 36), 'input', False)
                            # Processing the call keyword arguments (line 75)
                            kwargs_216 = {}
                            
                            # Obtaining the type of the subscript
                            int_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 25), 'int')
                            # Getting the type of 'self' (line 75)
                            self_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'self', False)
                            # Obtaining the member 'args' of a type (line 75)
                            args_211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), self_210, 'args')
                            # Obtaining the member '__getitem__' of a type (line 75)
                            getitem___212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), args_211, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                            subscript_call_result_213 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), getitem___212, int_209)
                            
                            # Obtaining the member 'execute' of a type (line 75)
                            execute_214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), subscript_call_result_213, 'execute')
                            # Calling execute(args, kwargs) (line 75)
                            execute_call_result_217 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), execute_214, *[input_215], **kwargs_216)
                            
                            # Testing if the type of an if condition is none (line 75)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217):
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                            else:
                                
                                # Testing the type of an if condition (line 75)
                                if_condition_218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217)
                                # Assigning a type to the variable 'if_condition_218' (line 75)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'if_condition_218', if_condition_218)
                                # SSA begins for if statement (line 75)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to execute(...): (line 76)
                                # Processing the call arguments (line 76)
                                # Getting the type of 'input' (line 76)
                                input_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 44), 'input', False)
                                # Processing the call keyword arguments (line 76)
                                kwargs_226 = {}
                                
                                # Obtaining the type of the subscript
                                int_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 33), 'int')
                                # Getting the type of 'self' (line 76)
                                self_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 76)
                                args_221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), self_220, 'args')
                                # Obtaining the member '__getitem__' of a type (line 76)
                                getitem___222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), args_221, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 76)
                                subscript_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), getitem___222, int_219)
                                
                                # Obtaining the member 'execute' of a type (line 76)
                                execute_224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), subscript_call_result_223, 'execute')
                                # Calling execute(args, kwargs) (line 76)
                                execute_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), execute_224, *[input_225], **kwargs_226)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 76)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'stypy_return_type', execute_call_result_227)
                                # SSA branch for the else part of an if statement (line 75)
                                module_type_store.open_ssa_branch('else')
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                                # SSA join for if statement (line 75)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 74)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 72)
                        if_condition_192 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 13), result_eq_191)
                        # Assigning a type to the variable 'if_condition_192' (line 72)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'if_condition_192', if_condition_192)
                        # SSA begins for if statement (line 72)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        int_193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 19), 'int')
                        
                        # Call to execute(...): (line 73)
                        # Processing the call arguments (line 73)
                        # Getting the type of 'input' (line 73)
                        input_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 44), 'input', False)
                        # Processing the call keyword arguments (line 73)
                        kwargs_201 = {}
                        
                        # Obtaining the type of the subscript
                        int_194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 33), 'int')
                        # Getting the type of 'self' (line 73)
                        self_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'self', False)
                        # Obtaining the member 'args' of a type (line 73)
                        args_196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), self_195, 'args')
                        # Obtaining the member '__getitem__' of a type (line 73)
                        getitem___197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), args_196, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 73)
                        subscript_call_result_198 = invoke(stypy.reporting.localization.Localization(__file__, 73, 23), getitem___197, int_194)
                        
                        # Obtaining the member 'execute' of a type (line 73)
                        execute_199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), subscript_call_result_198, 'execute')
                        # Calling execute(args, kwargs) (line 73)
                        execute_call_result_202 = invoke(stypy.reporting.localization.Localization(__file__, 73, 23), execute_199, *[input_200], **kwargs_201)
                        
                        # Applying the binary operator '^' (line 73)
                        result_xor_203 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 19), '^', int_193, execute_call_result_202)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 73)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'stypy_return_type', result_xor_203)
                        # SSA branch for the else part of an if statement (line 72)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'self' (line 74)
                        self_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 74)
                        opcode_205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 13), self_204, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 74)
                        OPCODE_IF_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 74)
                        result_eq_207 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 13), '==', opcode_205, OPCODE_IF_206)
                        
                        # Testing if the type of an if condition is none (line 74)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 74)
                            if_condition_208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207)
                            # Assigning a type to the variable 'if_condition_208' (line 74)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'if_condition_208', if_condition_208)
                            # SSA begins for if statement (line 74)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to execute(...): (line 75)
                            # Processing the call arguments (line 75)
                            # Getting the type of 'input' (line 75)
                            input_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 36), 'input', False)
                            # Processing the call keyword arguments (line 75)
                            kwargs_216 = {}
                            
                            # Obtaining the type of the subscript
                            int_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 25), 'int')
                            # Getting the type of 'self' (line 75)
                            self_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'self', False)
                            # Obtaining the member 'args' of a type (line 75)
                            args_211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), self_210, 'args')
                            # Obtaining the member '__getitem__' of a type (line 75)
                            getitem___212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), args_211, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                            subscript_call_result_213 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), getitem___212, int_209)
                            
                            # Obtaining the member 'execute' of a type (line 75)
                            execute_214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), subscript_call_result_213, 'execute')
                            # Calling execute(args, kwargs) (line 75)
                            execute_call_result_217 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), execute_214, *[input_215], **kwargs_216)
                            
                            # Testing if the type of an if condition is none (line 75)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217):
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                            else:
                                
                                # Testing the type of an if condition (line 75)
                                if_condition_218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217)
                                # Assigning a type to the variable 'if_condition_218' (line 75)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'if_condition_218', if_condition_218)
                                # SSA begins for if statement (line 75)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to execute(...): (line 76)
                                # Processing the call arguments (line 76)
                                # Getting the type of 'input' (line 76)
                                input_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 44), 'input', False)
                                # Processing the call keyword arguments (line 76)
                                kwargs_226 = {}
                                
                                # Obtaining the type of the subscript
                                int_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 33), 'int')
                                # Getting the type of 'self' (line 76)
                                self_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 76)
                                args_221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), self_220, 'args')
                                # Obtaining the member '__getitem__' of a type (line 76)
                                getitem___222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), args_221, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 76)
                                subscript_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), getitem___222, int_219)
                                
                                # Obtaining the member 'execute' of a type (line 76)
                                execute_224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), subscript_call_result_223, 'execute')
                                # Calling execute(args, kwargs) (line 76)
                                execute_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), execute_224, *[input_225], **kwargs_226)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 76)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'stypy_return_type', execute_call_result_227)
                                # SSA branch for the else part of an if statement (line 75)
                                module_type_store.open_ssa_branch('else')
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                                # SSA join for if statement (line 75)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 74)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 72)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 69)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 66)
                if_condition_144 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 13), result_eq_143)
                # Assigning a type to the variable 'if_condition_144' (line 66)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 13), 'if_condition_144', if_condition_144)
                # SSA begins for if statement (line 66)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to execute(...): (line 67)
                # Processing the call arguments (line 67)
                # Getting the type of 'input' (line 67)
                input_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 40), 'input', False)
                # Processing the call keyword arguments (line 67)
                kwargs_152 = {}
                
                # Obtaining the type of the subscript
                int_145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 29), 'int')
                # Getting the type of 'self' (line 67)
                self_146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 19), 'self', False)
                # Obtaining the member 'args' of a type (line 67)
                args_147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 19), self_146, 'args')
                # Obtaining the member '__getitem__' of a type (line 67)
                getitem___148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 19), args_147, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 67)
                subscript_call_result_149 = invoke(stypy.reporting.localization.Localization(__file__, 67, 19), getitem___148, int_145)
                
                # Obtaining the member 'execute' of a type (line 67)
                execute_150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 19), subscript_call_result_149, 'execute')
                # Calling execute(args, kwargs) (line 67)
                execute_call_result_153 = invoke(stypy.reporting.localization.Localization(__file__, 67, 19), execute_150, *[input_151], **kwargs_152)
                
                
                # Call to execute(...): (line 68)
                # Processing the call arguments (line 68)
                # Getting the type of 'input' (line 68)
                input_160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 40), 'input', False)
                # Processing the call keyword arguments (line 68)
                kwargs_161 = {}
                
                # Obtaining the type of the subscript
                int_154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 29), 'int')
                # Getting the type of 'self' (line 68)
                self_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 19), 'self', False)
                # Obtaining the member 'args' of a type (line 68)
                args_156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 19), self_155, 'args')
                # Obtaining the member '__getitem__' of a type (line 68)
                getitem___157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 19), args_156, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 68)
                subscript_call_result_158 = invoke(stypy.reporting.localization.Localization(__file__, 68, 19), getitem___157, int_154)
                
                # Obtaining the member 'execute' of a type (line 68)
                execute_159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 19), subscript_call_result_158, 'execute')
                # Calling execute(args, kwargs) (line 68)
                execute_call_result_162 = invoke(stypy.reporting.localization.Localization(__file__, 68, 19), execute_159, *[input_160], **kwargs_161)
                
                # Applying the binary operator '&' (line 67)
                result_and__163 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 19), '&', execute_call_result_153, execute_call_result_162)
                
                # Assigning a type to the variable 'stypy_return_type' (line 67)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'stypy_return_type', result_and__163)
                # SSA branch for the else part of an if statement (line 66)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'self' (line 69)
                self_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'self')
                # Obtaining the member 'opcode' of a type (line 69)
                opcode_165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 13), self_164, 'opcode')
                # Getting the type of 'OPCODE_OR' (line 69)
                OPCODE_OR_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 28), 'OPCODE_OR')
                # Applying the binary operator '==' (line 69)
                result_eq_167 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 13), '==', opcode_165, OPCODE_OR_166)
                
                # Testing if the type of an if condition is none (line 69)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 69, 13), result_eq_167):
                    
                    # Getting the type of 'self' (line 72)
                    self_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'self')
                    # Obtaining the member 'opcode' of a type (line 72)
                    opcode_189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 13), self_188, 'opcode')
                    # Getting the type of 'OPCODE_NOT' (line 72)
                    OPCODE_NOT_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 28), 'OPCODE_NOT')
                    # Applying the binary operator '==' (line 72)
                    result_eq_191 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 13), '==', opcode_189, OPCODE_NOT_190)
                    
                    # Testing if the type of an if condition is none (line 72)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 72, 13), result_eq_191):
                        
                        # Getting the type of 'self' (line 74)
                        self_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 74)
                        opcode_205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 13), self_204, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 74)
                        OPCODE_IF_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 74)
                        result_eq_207 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 13), '==', opcode_205, OPCODE_IF_206)
                        
                        # Testing if the type of an if condition is none (line 74)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 74)
                            if_condition_208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207)
                            # Assigning a type to the variable 'if_condition_208' (line 74)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'if_condition_208', if_condition_208)
                            # SSA begins for if statement (line 74)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to execute(...): (line 75)
                            # Processing the call arguments (line 75)
                            # Getting the type of 'input' (line 75)
                            input_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 36), 'input', False)
                            # Processing the call keyword arguments (line 75)
                            kwargs_216 = {}
                            
                            # Obtaining the type of the subscript
                            int_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 25), 'int')
                            # Getting the type of 'self' (line 75)
                            self_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'self', False)
                            # Obtaining the member 'args' of a type (line 75)
                            args_211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), self_210, 'args')
                            # Obtaining the member '__getitem__' of a type (line 75)
                            getitem___212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), args_211, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                            subscript_call_result_213 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), getitem___212, int_209)
                            
                            # Obtaining the member 'execute' of a type (line 75)
                            execute_214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), subscript_call_result_213, 'execute')
                            # Calling execute(args, kwargs) (line 75)
                            execute_call_result_217 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), execute_214, *[input_215], **kwargs_216)
                            
                            # Testing if the type of an if condition is none (line 75)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217):
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                            else:
                                
                                # Testing the type of an if condition (line 75)
                                if_condition_218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217)
                                # Assigning a type to the variable 'if_condition_218' (line 75)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'if_condition_218', if_condition_218)
                                # SSA begins for if statement (line 75)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to execute(...): (line 76)
                                # Processing the call arguments (line 76)
                                # Getting the type of 'input' (line 76)
                                input_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 44), 'input', False)
                                # Processing the call keyword arguments (line 76)
                                kwargs_226 = {}
                                
                                # Obtaining the type of the subscript
                                int_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 33), 'int')
                                # Getting the type of 'self' (line 76)
                                self_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 76)
                                args_221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), self_220, 'args')
                                # Obtaining the member '__getitem__' of a type (line 76)
                                getitem___222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), args_221, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 76)
                                subscript_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), getitem___222, int_219)
                                
                                # Obtaining the member 'execute' of a type (line 76)
                                execute_224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), subscript_call_result_223, 'execute')
                                # Calling execute(args, kwargs) (line 76)
                                execute_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), execute_224, *[input_225], **kwargs_226)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 76)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'stypy_return_type', execute_call_result_227)
                                # SSA branch for the else part of an if statement (line 75)
                                module_type_store.open_ssa_branch('else')
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                                # SSA join for if statement (line 75)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 74)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 72)
                        if_condition_192 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 13), result_eq_191)
                        # Assigning a type to the variable 'if_condition_192' (line 72)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'if_condition_192', if_condition_192)
                        # SSA begins for if statement (line 72)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        int_193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 19), 'int')
                        
                        # Call to execute(...): (line 73)
                        # Processing the call arguments (line 73)
                        # Getting the type of 'input' (line 73)
                        input_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 44), 'input', False)
                        # Processing the call keyword arguments (line 73)
                        kwargs_201 = {}
                        
                        # Obtaining the type of the subscript
                        int_194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 33), 'int')
                        # Getting the type of 'self' (line 73)
                        self_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'self', False)
                        # Obtaining the member 'args' of a type (line 73)
                        args_196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), self_195, 'args')
                        # Obtaining the member '__getitem__' of a type (line 73)
                        getitem___197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), args_196, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 73)
                        subscript_call_result_198 = invoke(stypy.reporting.localization.Localization(__file__, 73, 23), getitem___197, int_194)
                        
                        # Obtaining the member 'execute' of a type (line 73)
                        execute_199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), subscript_call_result_198, 'execute')
                        # Calling execute(args, kwargs) (line 73)
                        execute_call_result_202 = invoke(stypy.reporting.localization.Localization(__file__, 73, 23), execute_199, *[input_200], **kwargs_201)
                        
                        # Applying the binary operator '^' (line 73)
                        result_xor_203 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 19), '^', int_193, execute_call_result_202)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 73)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'stypy_return_type', result_xor_203)
                        # SSA branch for the else part of an if statement (line 72)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'self' (line 74)
                        self_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 74)
                        opcode_205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 13), self_204, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 74)
                        OPCODE_IF_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 74)
                        result_eq_207 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 13), '==', opcode_205, OPCODE_IF_206)
                        
                        # Testing if the type of an if condition is none (line 74)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 74)
                            if_condition_208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207)
                            # Assigning a type to the variable 'if_condition_208' (line 74)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'if_condition_208', if_condition_208)
                            # SSA begins for if statement (line 74)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to execute(...): (line 75)
                            # Processing the call arguments (line 75)
                            # Getting the type of 'input' (line 75)
                            input_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 36), 'input', False)
                            # Processing the call keyword arguments (line 75)
                            kwargs_216 = {}
                            
                            # Obtaining the type of the subscript
                            int_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 25), 'int')
                            # Getting the type of 'self' (line 75)
                            self_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'self', False)
                            # Obtaining the member 'args' of a type (line 75)
                            args_211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), self_210, 'args')
                            # Obtaining the member '__getitem__' of a type (line 75)
                            getitem___212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), args_211, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                            subscript_call_result_213 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), getitem___212, int_209)
                            
                            # Obtaining the member 'execute' of a type (line 75)
                            execute_214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), subscript_call_result_213, 'execute')
                            # Calling execute(args, kwargs) (line 75)
                            execute_call_result_217 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), execute_214, *[input_215], **kwargs_216)
                            
                            # Testing if the type of an if condition is none (line 75)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217):
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                            else:
                                
                                # Testing the type of an if condition (line 75)
                                if_condition_218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217)
                                # Assigning a type to the variable 'if_condition_218' (line 75)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'if_condition_218', if_condition_218)
                                # SSA begins for if statement (line 75)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to execute(...): (line 76)
                                # Processing the call arguments (line 76)
                                # Getting the type of 'input' (line 76)
                                input_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 44), 'input', False)
                                # Processing the call keyword arguments (line 76)
                                kwargs_226 = {}
                                
                                # Obtaining the type of the subscript
                                int_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 33), 'int')
                                # Getting the type of 'self' (line 76)
                                self_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 76)
                                args_221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), self_220, 'args')
                                # Obtaining the member '__getitem__' of a type (line 76)
                                getitem___222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), args_221, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 76)
                                subscript_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), getitem___222, int_219)
                                
                                # Obtaining the member 'execute' of a type (line 76)
                                execute_224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), subscript_call_result_223, 'execute')
                                # Calling execute(args, kwargs) (line 76)
                                execute_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), execute_224, *[input_225], **kwargs_226)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 76)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'stypy_return_type', execute_call_result_227)
                                # SSA branch for the else part of an if statement (line 75)
                                module_type_store.open_ssa_branch('else')
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                                # SSA join for if statement (line 75)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 74)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 72)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 69)
                    if_condition_168 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 13), result_eq_167)
                    # Assigning a type to the variable 'if_condition_168' (line 69)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'if_condition_168', if_condition_168)
                    # SSA begins for if statement (line 69)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to execute(...): (line 70)
                    # Processing the call arguments (line 70)
                    # Getting the type of 'input' (line 70)
                    input_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 40), 'input', False)
                    # Processing the call keyword arguments (line 70)
                    kwargs_176 = {}
                    
                    # Obtaining the type of the subscript
                    int_169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 29), 'int')
                    # Getting the type of 'self' (line 70)
                    self_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 19), 'self', False)
                    # Obtaining the member 'args' of a type (line 70)
                    args_171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 19), self_170, 'args')
                    # Obtaining the member '__getitem__' of a type (line 70)
                    getitem___172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 19), args_171, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 70)
                    subscript_call_result_173 = invoke(stypy.reporting.localization.Localization(__file__, 70, 19), getitem___172, int_169)
                    
                    # Obtaining the member 'execute' of a type (line 70)
                    execute_174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 19), subscript_call_result_173, 'execute')
                    # Calling execute(args, kwargs) (line 70)
                    execute_call_result_177 = invoke(stypy.reporting.localization.Localization(__file__, 70, 19), execute_174, *[input_175], **kwargs_176)
                    
                    
                    # Call to execute(...): (line 71)
                    # Processing the call arguments (line 71)
                    # Getting the type of 'input' (line 71)
                    input_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 40), 'input', False)
                    # Processing the call keyword arguments (line 71)
                    kwargs_185 = {}
                    
                    # Obtaining the type of the subscript
                    int_178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 29), 'int')
                    # Getting the type of 'self' (line 71)
                    self_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'self', False)
                    # Obtaining the member 'args' of a type (line 71)
                    args_180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 19), self_179, 'args')
                    # Obtaining the member '__getitem__' of a type (line 71)
                    getitem___181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 19), args_180, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 71)
                    subscript_call_result_182 = invoke(stypy.reporting.localization.Localization(__file__, 71, 19), getitem___181, int_178)
                    
                    # Obtaining the member 'execute' of a type (line 71)
                    execute_183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 19), subscript_call_result_182, 'execute')
                    # Calling execute(args, kwargs) (line 71)
                    execute_call_result_186 = invoke(stypy.reporting.localization.Localization(__file__, 71, 19), execute_183, *[input_184], **kwargs_185)
                    
                    # Applying the binary operator '|' (line 70)
                    result_or__187 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 19), '|', execute_call_result_177, execute_call_result_186)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 70)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'stypy_return_type', result_or__187)
                    # SSA branch for the else part of an if statement (line 69)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'self' (line 72)
                    self_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'self')
                    # Obtaining the member 'opcode' of a type (line 72)
                    opcode_189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 13), self_188, 'opcode')
                    # Getting the type of 'OPCODE_NOT' (line 72)
                    OPCODE_NOT_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 28), 'OPCODE_NOT')
                    # Applying the binary operator '==' (line 72)
                    result_eq_191 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 13), '==', opcode_189, OPCODE_NOT_190)
                    
                    # Testing if the type of an if condition is none (line 72)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 72, 13), result_eq_191):
                        
                        # Getting the type of 'self' (line 74)
                        self_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 74)
                        opcode_205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 13), self_204, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 74)
                        OPCODE_IF_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 74)
                        result_eq_207 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 13), '==', opcode_205, OPCODE_IF_206)
                        
                        # Testing if the type of an if condition is none (line 74)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 74)
                            if_condition_208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207)
                            # Assigning a type to the variable 'if_condition_208' (line 74)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'if_condition_208', if_condition_208)
                            # SSA begins for if statement (line 74)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to execute(...): (line 75)
                            # Processing the call arguments (line 75)
                            # Getting the type of 'input' (line 75)
                            input_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 36), 'input', False)
                            # Processing the call keyword arguments (line 75)
                            kwargs_216 = {}
                            
                            # Obtaining the type of the subscript
                            int_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 25), 'int')
                            # Getting the type of 'self' (line 75)
                            self_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'self', False)
                            # Obtaining the member 'args' of a type (line 75)
                            args_211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), self_210, 'args')
                            # Obtaining the member '__getitem__' of a type (line 75)
                            getitem___212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), args_211, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                            subscript_call_result_213 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), getitem___212, int_209)
                            
                            # Obtaining the member 'execute' of a type (line 75)
                            execute_214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), subscript_call_result_213, 'execute')
                            # Calling execute(args, kwargs) (line 75)
                            execute_call_result_217 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), execute_214, *[input_215], **kwargs_216)
                            
                            # Testing if the type of an if condition is none (line 75)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217):
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                            else:
                                
                                # Testing the type of an if condition (line 75)
                                if_condition_218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217)
                                # Assigning a type to the variable 'if_condition_218' (line 75)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'if_condition_218', if_condition_218)
                                # SSA begins for if statement (line 75)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to execute(...): (line 76)
                                # Processing the call arguments (line 76)
                                # Getting the type of 'input' (line 76)
                                input_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 44), 'input', False)
                                # Processing the call keyword arguments (line 76)
                                kwargs_226 = {}
                                
                                # Obtaining the type of the subscript
                                int_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 33), 'int')
                                # Getting the type of 'self' (line 76)
                                self_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 76)
                                args_221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), self_220, 'args')
                                # Obtaining the member '__getitem__' of a type (line 76)
                                getitem___222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), args_221, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 76)
                                subscript_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), getitem___222, int_219)
                                
                                # Obtaining the member 'execute' of a type (line 76)
                                execute_224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), subscript_call_result_223, 'execute')
                                # Calling execute(args, kwargs) (line 76)
                                execute_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), execute_224, *[input_225], **kwargs_226)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 76)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'stypy_return_type', execute_call_result_227)
                                # SSA branch for the else part of an if statement (line 75)
                                module_type_store.open_ssa_branch('else')
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                                # SSA join for if statement (line 75)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 74)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 72)
                        if_condition_192 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 13), result_eq_191)
                        # Assigning a type to the variable 'if_condition_192' (line 72)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'if_condition_192', if_condition_192)
                        # SSA begins for if statement (line 72)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        int_193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 19), 'int')
                        
                        # Call to execute(...): (line 73)
                        # Processing the call arguments (line 73)
                        # Getting the type of 'input' (line 73)
                        input_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 44), 'input', False)
                        # Processing the call keyword arguments (line 73)
                        kwargs_201 = {}
                        
                        # Obtaining the type of the subscript
                        int_194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 33), 'int')
                        # Getting the type of 'self' (line 73)
                        self_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'self', False)
                        # Obtaining the member 'args' of a type (line 73)
                        args_196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), self_195, 'args')
                        # Obtaining the member '__getitem__' of a type (line 73)
                        getitem___197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), args_196, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 73)
                        subscript_call_result_198 = invoke(stypy.reporting.localization.Localization(__file__, 73, 23), getitem___197, int_194)
                        
                        # Obtaining the member 'execute' of a type (line 73)
                        execute_199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), subscript_call_result_198, 'execute')
                        # Calling execute(args, kwargs) (line 73)
                        execute_call_result_202 = invoke(stypy.reporting.localization.Localization(__file__, 73, 23), execute_199, *[input_200], **kwargs_201)
                        
                        # Applying the binary operator '^' (line 73)
                        result_xor_203 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 19), '^', int_193, execute_call_result_202)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 73)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'stypy_return_type', result_xor_203)
                        # SSA branch for the else part of an if statement (line 72)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'self' (line 74)
                        self_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 74)
                        opcode_205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 13), self_204, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 74)
                        OPCODE_IF_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 74)
                        result_eq_207 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 13), '==', opcode_205, OPCODE_IF_206)
                        
                        # Testing if the type of an if condition is none (line 74)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 74)
                            if_condition_208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 13), result_eq_207)
                            # Assigning a type to the variable 'if_condition_208' (line 74)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'if_condition_208', if_condition_208)
                            # SSA begins for if statement (line 74)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to execute(...): (line 75)
                            # Processing the call arguments (line 75)
                            # Getting the type of 'input' (line 75)
                            input_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 36), 'input', False)
                            # Processing the call keyword arguments (line 75)
                            kwargs_216 = {}
                            
                            # Obtaining the type of the subscript
                            int_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 25), 'int')
                            # Getting the type of 'self' (line 75)
                            self_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'self', False)
                            # Obtaining the member 'args' of a type (line 75)
                            args_211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), self_210, 'args')
                            # Obtaining the member '__getitem__' of a type (line 75)
                            getitem___212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), args_211, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                            subscript_call_result_213 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), getitem___212, int_209)
                            
                            # Obtaining the member 'execute' of a type (line 75)
                            execute_214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), subscript_call_result_213, 'execute')
                            # Calling execute(args, kwargs) (line 75)
                            execute_call_result_217 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), execute_214, *[input_215], **kwargs_216)
                            
                            # Testing if the type of an if condition is none (line 75)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217):
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                            else:
                                
                                # Testing the type of an if condition (line 75)
                                if_condition_218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 12), execute_call_result_217)
                                # Assigning a type to the variable 'if_condition_218' (line 75)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'if_condition_218', if_condition_218)
                                # SSA begins for if statement (line 75)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to execute(...): (line 76)
                                # Processing the call arguments (line 76)
                                # Getting the type of 'input' (line 76)
                                input_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 44), 'input', False)
                                # Processing the call keyword arguments (line 76)
                                kwargs_226 = {}
                                
                                # Obtaining the type of the subscript
                                int_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 33), 'int')
                                # Getting the type of 'self' (line 76)
                                self_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 76)
                                args_221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), self_220, 'args')
                                # Obtaining the member '__getitem__' of a type (line 76)
                                getitem___222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), args_221, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 76)
                                subscript_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), getitem___222, int_219)
                                
                                # Obtaining the member 'execute' of a type (line 76)
                                execute_224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), subscript_call_result_223, 'execute')
                                # Calling execute(args, kwargs) (line 76)
                                execute_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 76, 23), execute_224, *[input_225], **kwargs_226)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 76)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'stypy_return_type', execute_call_result_227)
                                # SSA branch for the else part of an if statement (line 75)
                                module_type_store.open_ssa_branch('else')
                                
                                # Call to execute(...): (line 78)
                                # Processing the call arguments (line 78)
                                # Getting the type of 'input' (line 78)
                                input_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'input', False)
                                # Processing the call keyword arguments (line 78)
                                kwargs_235 = {}
                                
                                # Obtaining the type of the subscript
                                int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'int')
                                # Getting the type of 'self' (line 78)
                                self_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self', False)
                                # Obtaining the member 'args' of a type (line 78)
                                args_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_229, 'args')
                                # Obtaining the member '__getitem__' of a type (line 78)
                                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), args_230, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), getitem___231, int_228)
                                
                                # Obtaining the member 'execute' of a type (line 78)
                                execute_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), subscript_call_result_232, 'execute')
                                # Calling execute(args, kwargs) (line 78)
                                execute_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), execute_233, *[input_234], **kwargs_235)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 78)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'stypy_return_type', execute_call_result_236)
                                # SSA join for if statement (line 75)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 74)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 72)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 69)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 66)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 64)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'execute(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'execute' in the type store
        # Getting the type of 'stypy_return_type' (line 63)
        stypy_return_type_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_237)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'execute'
        return stypy_return_type_237


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 80, 4, False)
        # Assigning a type to the variable 'self' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TreeNode.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        TreeNode.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TreeNode.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TreeNode.stypy__str__.__dict__.__setitem__('stypy_function_name', 'TreeNode.stypy__str__')
        TreeNode.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        TreeNode.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TreeNode.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TreeNode.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TreeNode.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TreeNode.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TreeNode.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TreeNode.stypy__str__', [], None, None, defaults, varargs, kwargs)

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

        
        # Getting the type of 'self' (line 81)
        self_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'self')
        # Obtaining the member 'opcode' of a type (line 81)
        opcode_239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 11), self_238, 'opcode')
        # Getting the type of 'OPCODE_NONE' (line 81)
        OPCODE_NONE_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 26), 'OPCODE_NONE')
        # Applying the binary operator '==' (line 81)
        result_eq_241 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 11), '==', opcode_239, OPCODE_NONE_240)
        
        # Testing if the type of an if condition is none (line 81)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 81, 8), result_eq_241):
            
            # Getting the type of 'self' (line 83)
            self_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 13), 'self')
            # Obtaining the member 'opcode' of a type (line 83)
            opcode_248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 13), self_247, 'opcode')
            # Getting the type of 'OPCODE_AND' (line 83)
            OPCODE_AND_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 28), 'OPCODE_AND')
            # Applying the binary operator '==' (line 83)
            result_eq_250 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 13), '==', opcode_248, OPCODE_AND_249)
            
            # Testing if the type of an if condition is none (line 83)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 83, 13), result_eq_250):
                
                # Getting the type of 'self' (line 85)
                self_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 13), 'self')
                # Obtaining the member 'opcode' of a type (line 85)
                opcode_257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 13), self_256, 'opcode')
                # Getting the type of 'OPCODE_OR' (line 85)
                OPCODE_OR_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 28), 'OPCODE_OR')
                # Applying the binary operator '==' (line 85)
                result_eq_259 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 13), '==', opcode_257, OPCODE_OR_258)
                
                # Testing if the type of an if condition is none (line 85)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 85, 13), result_eq_259):
                    
                    # Getting the type of 'self' (line 87)
                    self_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'self')
                    # Obtaining the member 'opcode' of a type (line 87)
                    opcode_266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 13), self_265, 'opcode')
                    # Getting the type of 'OPCODE_NOT' (line 87)
                    OPCODE_NOT_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 28), 'OPCODE_NOT')
                    # Applying the binary operator '==' (line 87)
                    result_eq_268 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 13), '==', opcode_266, OPCODE_NOT_267)
                    
                    # Testing if the type of an if condition is none (line 87)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 87, 13), result_eq_268):
                        
                        # Getting the type of 'self' (line 89)
                        self_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 89)
                        opcode_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 13), self_274, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 89)
                        OPCODE_IF_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 89)
                        result_eq_277 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 13), '==', opcode_275, OPCODE_IF_276)
                        
                        # Testing if the type of an if condition is none (line 89)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 89)
                            if_condition_278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277)
                            # Assigning a type to the variable 'if_condition_278' (line 89)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'if_condition_278', if_condition_278)
                            # SSA begins for if statement (line 89)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a BinOp to a Name (line 90):
                            
                            # Assigning a BinOp to a Name (line 90):
                            str_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 21), 'str', '(if %s then %s else %s)')
                            # Getting the type of 'self' (line 90)
                            self_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 49), 'self')
                            # Obtaining the member 'args' of a type (line 90)
                            args_281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 49), self_280, 'args')
                            # Applying the binary operator '%' (line 90)
                            result_mod_282 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 21), '%', str_279, args_281)
                            
                            # Assigning a type to the variable 'output' (line 90)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'output', result_mod_282)
                            # SSA join for if statement (line 89)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 87)
                        if_condition_269 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 13), result_eq_268)
                        # Assigning a type to the variable 'if_condition_269' (line 87)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'if_condition_269', if_condition_269)
                        # SSA begins for if statement (line 87)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a BinOp to a Name (line 88):
                        
                        # Assigning a BinOp to a Name (line 88):
                        str_270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 21), 'str', '(not %s)')
                        # Getting the type of 'self' (line 88)
                        self_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 34), 'self')
                        # Obtaining the member 'args' of a type (line 88)
                        args_272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 34), self_271, 'args')
                        # Applying the binary operator '%' (line 88)
                        result_mod_273 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 21), '%', str_270, args_272)
                        
                        # Assigning a type to the variable 'output' (line 88)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'output', result_mod_273)
                        # SSA branch for the else part of an if statement (line 87)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'self' (line 89)
                        self_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 89)
                        opcode_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 13), self_274, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 89)
                        OPCODE_IF_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 89)
                        result_eq_277 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 13), '==', opcode_275, OPCODE_IF_276)
                        
                        # Testing if the type of an if condition is none (line 89)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 89)
                            if_condition_278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277)
                            # Assigning a type to the variable 'if_condition_278' (line 89)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'if_condition_278', if_condition_278)
                            # SSA begins for if statement (line 89)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a BinOp to a Name (line 90):
                            
                            # Assigning a BinOp to a Name (line 90):
                            str_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 21), 'str', '(if %s then %s else %s)')
                            # Getting the type of 'self' (line 90)
                            self_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 49), 'self')
                            # Obtaining the member 'args' of a type (line 90)
                            args_281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 49), self_280, 'args')
                            # Applying the binary operator '%' (line 90)
                            result_mod_282 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 21), '%', str_279, args_281)
                            
                            # Assigning a type to the variable 'output' (line 90)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'output', result_mod_282)
                            # SSA join for if statement (line 89)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 87)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 85)
                    if_condition_260 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 13), result_eq_259)
                    # Assigning a type to the variable 'if_condition_260' (line 85)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 13), 'if_condition_260', if_condition_260)
                    # SSA begins for if statement (line 85)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a BinOp to a Name (line 86):
                    
                    # Assigning a BinOp to a Name (line 86):
                    str_261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 21), 'str', '(or %s %s)')
                    # Getting the type of 'self' (line 86)
                    self_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 36), 'self')
                    # Obtaining the member 'args' of a type (line 86)
                    args_263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 36), self_262, 'args')
                    # Applying the binary operator '%' (line 86)
                    result_mod_264 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 21), '%', str_261, args_263)
                    
                    # Assigning a type to the variable 'output' (line 86)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'output', result_mod_264)
                    # SSA branch for the else part of an if statement (line 85)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'self' (line 87)
                    self_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'self')
                    # Obtaining the member 'opcode' of a type (line 87)
                    opcode_266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 13), self_265, 'opcode')
                    # Getting the type of 'OPCODE_NOT' (line 87)
                    OPCODE_NOT_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 28), 'OPCODE_NOT')
                    # Applying the binary operator '==' (line 87)
                    result_eq_268 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 13), '==', opcode_266, OPCODE_NOT_267)
                    
                    # Testing if the type of an if condition is none (line 87)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 87, 13), result_eq_268):
                        
                        # Getting the type of 'self' (line 89)
                        self_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 89)
                        opcode_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 13), self_274, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 89)
                        OPCODE_IF_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 89)
                        result_eq_277 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 13), '==', opcode_275, OPCODE_IF_276)
                        
                        # Testing if the type of an if condition is none (line 89)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 89)
                            if_condition_278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277)
                            # Assigning a type to the variable 'if_condition_278' (line 89)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'if_condition_278', if_condition_278)
                            # SSA begins for if statement (line 89)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a BinOp to a Name (line 90):
                            
                            # Assigning a BinOp to a Name (line 90):
                            str_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 21), 'str', '(if %s then %s else %s)')
                            # Getting the type of 'self' (line 90)
                            self_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 49), 'self')
                            # Obtaining the member 'args' of a type (line 90)
                            args_281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 49), self_280, 'args')
                            # Applying the binary operator '%' (line 90)
                            result_mod_282 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 21), '%', str_279, args_281)
                            
                            # Assigning a type to the variable 'output' (line 90)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'output', result_mod_282)
                            # SSA join for if statement (line 89)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 87)
                        if_condition_269 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 13), result_eq_268)
                        # Assigning a type to the variable 'if_condition_269' (line 87)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'if_condition_269', if_condition_269)
                        # SSA begins for if statement (line 87)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a BinOp to a Name (line 88):
                        
                        # Assigning a BinOp to a Name (line 88):
                        str_270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 21), 'str', '(not %s)')
                        # Getting the type of 'self' (line 88)
                        self_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 34), 'self')
                        # Obtaining the member 'args' of a type (line 88)
                        args_272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 34), self_271, 'args')
                        # Applying the binary operator '%' (line 88)
                        result_mod_273 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 21), '%', str_270, args_272)
                        
                        # Assigning a type to the variable 'output' (line 88)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'output', result_mod_273)
                        # SSA branch for the else part of an if statement (line 87)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'self' (line 89)
                        self_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 89)
                        opcode_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 13), self_274, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 89)
                        OPCODE_IF_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 89)
                        result_eq_277 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 13), '==', opcode_275, OPCODE_IF_276)
                        
                        # Testing if the type of an if condition is none (line 89)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 89)
                            if_condition_278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277)
                            # Assigning a type to the variable 'if_condition_278' (line 89)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'if_condition_278', if_condition_278)
                            # SSA begins for if statement (line 89)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a BinOp to a Name (line 90):
                            
                            # Assigning a BinOp to a Name (line 90):
                            str_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 21), 'str', '(if %s then %s else %s)')
                            # Getting the type of 'self' (line 90)
                            self_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 49), 'self')
                            # Obtaining the member 'args' of a type (line 90)
                            args_281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 49), self_280, 'args')
                            # Applying the binary operator '%' (line 90)
                            result_mod_282 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 21), '%', str_279, args_281)
                            
                            # Assigning a type to the variable 'output' (line 90)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'output', result_mod_282)
                            # SSA join for if statement (line 89)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 87)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 85)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 83)
                if_condition_251 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 13), result_eq_250)
                # Assigning a type to the variable 'if_condition_251' (line 83)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 13), 'if_condition_251', if_condition_251)
                # SSA begins for if statement (line 83)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a BinOp to a Name (line 84):
                
                # Assigning a BinOp to a Name (line 84):
                str_252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 21), 'str', '(and %s %s)')
                # Getting the type of 'self' (line 84)
                self_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 37), 'self')
                # Obtaining the member 'args' of a type (line 84)
                args_254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 37), self_253, 'args')
                # Applying the binary operator '%' (line 84)
                result_mod_255 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 21), '%', str_252, args_254)
                
                # Assigning a type to the variable 'output' (line 84)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'output', result_mod_255)
                # SSA branch for the else part of an if statement (line 83)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'self' (line 85)
                self_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 13), 'self')
                # Obtaining the member 'opcode' of a type (line 85)
                opcode_257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 13), self_256, 'opcode')
                # Getting the type of 'OPCODE_OR' (line 85)
                OPCODE_OR_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 28), 'OPCODE_OR')
                # Applying the binary operator '==' (line 85)
                result_eq_259 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 13), '==', opcode_257, OPCODE_OR_258)
                
                # Testing if the type of an if condition is none (line 85)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 85, 13), result_eq_259):
                    
                    # Getting the type of 'self' (line 87)
                    self_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'self')
                    # Obtaining the member 'opcode' of a type (line 87)
                    opcode_266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 13), self_265, 'opcode')
                    # Getting the type of 'OPCODE_NOT' (line 87)
                    OPCODE_NOT_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 28), 'OPCODE_NOT')
                    # Applying the binary operator '==' (line 87)
                    result_eq_268 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 13), '==', opcode_266, OPCODE_NOT_267)
                    
                    # Testing if the type of an if condition is none (line 87)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 87, 13), result_eq_268):
                        
                        # Getting the type of 'self' (line 89)
                        self_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 89)
                        opcode_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 13), self_274, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 89)
                        OPCODE_IF_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 89)
                        result_eq_277 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 13), '==', opcode_275, OPCODE_IF_276)
                        
                        # Testing if the type of an if condition is none (line 89)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 89)
                            if_condition_278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277)
                            # Assigning a type to the variable 'if_condition_278' (line 89)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'if_condition_278', if_condition_278)
                            # SSA begins for if statement (line 89)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a BinOp to a Name (line 90):
                            
                            # Assigning a BinOp to a Name (line 90):
                            str_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 21), 'str', '(if %s then %s else %s)')
                            # Getting the type of 'self' (line 90)
                            self_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 49), 'self')
                            # Obtaining the member 'args' of a type (line 90)
                            args_281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 49), self_280, 'args')
                            # Applying the binary operator '%' (line 90)
                            result_mod_282 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 21), '%', str_279, args_281)
                            
                            # Assigning a type to the variable 'output' (line 90)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'output', result_mod_282)
                            # SSA join for if statement (line 89)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 87)
                        if_condition_269 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 13), result_eq_268)
                        # Assigning a type to the variable 'if_condition_269' (line 87)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'if_condition_269', if_condition_269)
                        # SSA begins for if statement (line 87)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a BinOp to a Name (line 88):
                        
                        # Assigning a BinOp to a Name (line 88):
                        str_270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 21), 'str', '(not %s)')
                        # Getting the type of 'self' (line 88)
                        self_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 34), 'self')
                        # Obtaining the member 'args' of a type (line 88)
                        args_272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 34), self_271, 'args')
                        # Applying the binary operator '%' (line 88)
                        result_mod_273 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 21), '%', str_270, args_272)
                        
                        # Assigning a type to the variable 'output' (line 88)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'output', result_mod_273)
                        # SSA branch for the else part of an if statement (line 87)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'self' (line 89)
                        self_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 89)
                        opcode_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 13), self_274, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 89)
                        OPCODE_IF_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 89)
                        result_eq_277 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 13), '==', opcode_275, OPCODE_IF_276)
                        
                        # Testing if the type of an if condition is none (line 89)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 89)
                            if_condition_278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277)
                            # Assigning a type to the variable 'if_condition_278' (line 89)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'if_condition_278', if_condition_278)
                            # SSA begins for if statement (line 89)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a BinOp to a Name (line 90):
                            
                            # Assigning a BinOp to a Name (line 90):
                            str_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 21), 'str', '(if %s then %s else %s)')
                            # Getting the type of 'self' (line 90)
                            self_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 49), 'self')
                            # Obtaining the member 'args' of a type (line 90)
                            args_281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 49), self_280, 'args')
                            # Applying the binary operator '%' (line 90)
                            result_mod_282 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 21), '%', str_279, args_281)
                            
                            # Assigning a type to the variable 'output' (line 90)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'output', result_mod_282)
                            # SSA join for if statement (line 89)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 87)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 85)
                    if_condition_260 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 13), result_eq_259)
                    # Assigning a type to the variable 'if_condition_260' (line 85)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 13), 'if_condition_260', if_condition_260)
                    # SSA begins for if statement (line 85)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a BinOp to a Name (line 86):
                    
                    # Assigning a BinOp to a Name (line 86):
                    str_261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 21), 'str', '(or %s %s)')
                    # Getting the type of 'self' (line 86)
                    self_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 36), 'self')
                    # Obtaining the member 'args' of a type (line 86)
                    args_263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 36), self_262, 'args')
                    # Applying the binary operator '%' (line 86)
                    result_mod_264 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 21), '%', str_261, args_263)
                    
                    # Assigning a type to the variable 'output' (line 86)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'output', result_mod_264)
                    # SSA branch for the else part of an if statement (line 85)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'self' (line 87)
                    self_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'self')
                    # Obtaining the member 'opcode' of a type (line 87)
                    opcode_266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 13), self_265, 'opcode')
                    # Getting the type of 'OPCODE_NOT' (line 87)
                    OPCODE_NOT_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 28), 'OPCODE_NOT')
                    # Applying the binary operator '==' (line 87)
                    result_eq_268 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 13), '==', opcode_266, OPCODE_NOT_267)
                    
                    # Testing if the type of an if condition is none (line 87)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 87, 13), result_eq_268):
                        
                        # Getting the type of 'self' (line 89)
                        self_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 89)
                        opcode_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 13), self_274, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 89)
                        OPCODE_IF_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 89)
                        result_eq_277 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 13), '==', opcode_275, OPCODE_IF_276)
                        
                        # Testing if the type of an if condition is none (line 89)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 89)
                            if_condition_278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277)
                            # Assigning a type to the variable 'if_condition_278' (line 89)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'if_condition_278', if_condition_278)
                            # SSA begins for if statement (line 89)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a BinOp to a Name (line 90):
                            
                            # Assigning a BinOp to a Name (line 90):
                            str_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 21), 'str', '(if %s then %s else %s)')
                            # Getting the type of 'self' (line 90)
                            self_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 49), 'self')
                            # Obtaining the member 'args' of a type (line 90)
                            args_281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 49), self_280, 'args')
                            # Applying the binary operator '%' (line 90)
                            result_mod_282 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 21), '%', str_279, args_281)
                            
                            # Assigning a type to the variable 'output' (line 90)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'output', result_mod_282)
                            # SSA join for if statement (line 89)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 87)
                        if_condition_269 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 13), result_eq_268)
                        # Assigning a type to the variable 'if_condition_269' (line 87)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'if_condition_269', if_condition_269)
                        # SSA begins for if statement (line 87)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a BinOp to a Name (line 88):
                        
                        # Assigning a BinOp to a Name (line 88):
                        str_270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 21), 'str', '(not %s)')
                        # Getting the type of 'self' (line 88)
                        self_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 34), 'self')
                        # Obtaining the member 'args' of a type (line 88)
                        args_272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 34), self_271, 'args')
                        # Applying the binary operator '%' (line 88)
                        result_mod_273 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 21), '%', str_270, args_272)
                        
                        # Assigning a type to the variable 'output' (line 88)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'output', result_mod_273)
                        # SSA branch for the else part of an if statement (line 87)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'self' (line 89)
                        self_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 89)
                        opcode_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 13), self_274, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 89)
                        OPCODE_IF_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 89)
                        result_eq_277 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 13), '==', opcode_275, OPCODE_IF_276)
                        
                        # Testing if the type of an if condition is none (line 89)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 89)
                            if_condition_278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277)
                            # Assigning a type to the variable 'if_condition_278' (line 89)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'if_condition_278', if_condition_278)
                            # SSA begins for if statement (line 89)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a BinOp to a Name (line 90):
                            
                            # Assigning a BinOp to a Name (line 90):
                            str_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 21), 'str', '(if %s then %s else %s)')
                            # Getting the type of 'self' (line 90)
                            self_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 49), 'self')
                            # Obtaining the member 'args' of a type (line 90)
                            args_281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 49), self_280, 'args')
                            # Applying the binary operator '%' (line 90)
                            result_mod_282 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 21), '%', str_279, args_281)
                            
                            # Assigning a type to the variable 'output' (line 90)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'output', result_mod_282)
                            # SSA join for if statement (line 89)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 87)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 85)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 83)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 81)
            if_condition_242 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 8), result_eq_241)
            # Assigning a type to the variable 'if_condition_242' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'if_condition_242', if_condition_242)
            # SSA begins for if statement (line 81)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 82):
            
            # Assigning a BinOp to a Name (line 82):
            str_243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 21), 'str', '(bit %s)')
            # Getting the type of 'self' (line 82)
            self_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 34), 'self')
            # Obtaining the member 'value' of a type (line 82)
            value_245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 34), self_244, 'value')
            # Applying the binary operator '%' (line 82)
            result_mod_246 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 21), '%', str_243, value_245)
            
            # Assigning a type to the variable 'output' (line 82)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'output', result_mod_246)
            # SSA branch for the else part of an if statement (line 81)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'self' (line 83)
            self_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 13), 'self')
            # Obtaining the member 'opcode' of a type (line 83)
            opcode_248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 13), self_247, 'opcode')
            # Getting the type of 'OPCODE_AND' (line 83)
            OPCODE_AND_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 28), 'OPCODE_AND')
            # Applying the binary operator '==' (line 83)
            result_eq_250 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 13), '==', opcode_248, OPCODE_AND_249)
            
            # Testing if the type of an if condition is none (line 83)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 83, 13), result_eq_250):
                
                # Getting the type of 'self' (line 85)
                self_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 13), 'self')
                # Obtaining the member 'opcode' of a type (line 85)
                opcode_257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 13), self_256, 'opcode')
                # Getting the type of 'OPCODE_OR' (line 85)
                OPCODE_OR_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 28), 'OPCODE_OR')
                # Applying the binary operator '==' (line 85)
                result_eq_259 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 13), '==', opcode_257, OPCODE_OR_258)
                
                # Testing if the type of an if condition is none (line 85)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 85, 13), result_eq_259):
                    
                    # Getting the type of 'self' (line 87)
                    self_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'self')
                    # Obtaining the member 'opcode' of a type (line 87)
                    opcode_266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 13), self_265, 'opcode')
                    # Getting the type of 'OPCODE_NOT' (line 87)
                    OPCODE_NOT_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 28), 'OPCODE_NOT')
                    # Applying the binary operator '==' (line 87)
                    result_eq_268 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 13), '==', opcode_266, OPCODE_NOT_267)
                    
                    # Testing if the type of an if condition is none (line 87)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 87, 13), result_eq_268):
                        
                        # Getting the type of 'self' (line 89)
                        self_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 89)
                        opcode_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 13), self_274, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 89)
                        OPCODE_IF_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 89)
                        result_eq_277 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 13), '==', opcode_275, OPCODE_IF_276)
                        
                        # Testing if the type of an if condition is none (line 89)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 89)
                            if_condition_278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277)
                            # Assigning a type to the variable 'if_condition_278' (line 89)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'if_condition_278', if_condition_278)
                            # SSA begins for if statement (line 89)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a BinOp to a Name (line 90):
                            
                            # Assigning a BinOp to a Name (line 90):
                            str_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 21), 'str', '(if %s then %s else %s)')
                            # Getting the type of 'self' (line 90)
                            self_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 49), 'self')
                            # Obtaining the member 'args' of a type (line 90)
                            args_281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 49), self_280, 'args')
                            # Applying the binary operator '%' (line 90)
                            result_mod_282 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 21), '%', str_279, args_281)
                            
                            # Assigning a type to the variable 'output' (line 90)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'output', result_mod_282)
                            # SSA join for if statement (line 89)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 87)
                        if_condition_269 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 13), result_eq_268)
                        # Assigning a type to the variable 'if_condition_269' (line 87)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'if_condition_269', if_condition_269)
                        # SSA begins for if statement (line 87)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a BinOp to a Name (line 88):
                        
                        # Assigning a BinOp to a Name (line 88):
                        str_270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 21), 'str', '(not %s)')
                        # Getting the type of 'self' (line 88)
                        self_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 34), 'self')
                        # Obtaining the member 'args' of a type (line 88)
                        args_272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 34), self_271, 'args')
                        # Applying the binary operator '%' (line 88)
                        result_mod_273 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 21), '%', str_270, args_272)
                        
                        # Assigning a type to the variable 'output' (line 88)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'output', result_mod_273)
                        # SSA branch for the else part of an if statement (line 87)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'self' (line 89)
                        self_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 89)
                        opcode_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 13), self_274, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 89)
                        OPCODE_IF_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 89)
                        result_eq_277 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 13), '==', opcode_275, OPCODE_IF_276)
                        
                        # Testing if the type of an if condition is none (line 89)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 89)
                            if_condition_278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277)
                            # Assigning a type to the variable 'if_condition_278' (line 89)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'if_condition_278', if_condition_278)
                            # SSA begins for if statement (line 89)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a BinOp to a Name (line 90):
                            
                            # Assigning a BinOp to a Name (line 90):
                            str_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 21), 'str', '(if %s then %s else %s)')
                            # Getting the type of 'self' (line 90)
                            self_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 49), 'self')
                            # Obtaining the member 'args' of a type (line 90)
                            args_281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 49), self_280, 'args')
                            # Applying the binary operator '%' (line 90)
                            result_mod_282 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 21), '%', str_279, args_281)
                            
                            # Assigning a type to the variable 'output' (line 90)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'output', result_mod_282)
                            # SSA join for if statement (line 89)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 87)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 85)
                    if_condition_260 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 13), result_eq_259)
                    # Assigning a type to the variable 'if_condition_260' (line 85)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 13), 'if_condition_260', if_condition_260)
                    # SSA begins for if statement (line 85)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a BinOp to a Name (line 86):
                    
                    # Assigning a BinOp to a Name (line 86):
                    str_261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 21), 'str', '(or %s %s)')
                    # Getting the type of 'self' (line 86)
                    self_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 36), 'self')
                    # Obtaining the member 'args' of a type (line 86)
                    args_263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 36), self_262, 'args')
                    # Applying the binary operator '%' (line 86)
                    result_mod_264 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 21), '%', str_261, args_263)
                    
                    # Assigning a type to the variable 'output' (line 86)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'output', result_mod_264)
                    # SSA branch for the else part of an if statement (line 85)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'self' (line 87)
                    self_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'self')
                    # Obtaining the member 'opcode' of a type (line 87)
                    opcode_266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 13), self_265, 'opcode')
                    # Getting the type of 'OPCODE_NOT' (line 87)
                    OPCODE_NOT_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 28), 'OPCODE_NOT')
                    # Applying the binary operator '==' (line 87)
                    result_eq_268 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 13), '==', opcode_266, OPCODE_NOT_267)
                    
                    # Testing if the type of an if condition is none (line 87)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 87, 13), result_eq_268):
                        
                        # Getting the type of 'self' (line 89)
                        self_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 89)
                        opcode_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 13), self_274, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 89)
                        OPCODE_IF_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 89)
                        result_eq_277 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 13), '==', opcode_275, OPCODE_IF_276)
                        
                        # Testing if the type of an if condition is none (line 89)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 89)
                            if_condition_278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277)
                            # Assigning a type to the variable 'if_condition_278' (line 89)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'if_condition_278', if_condition_278)
                            # SSA begins for if statement (line 89)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a BinOp to a Name (line 90):
                            
                            # Assigning a BinOp to a Name (line 90):
                            str_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 21), 'str', '(if %s then %s else %s)')
                            # Getting the type of 'self' (line 90)
                            self_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 49), 'self')
                            # Obtaining the member 'args' of a type (line 90)
                            args_281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 49), self_280, 'args')
                            # Applying the binary operator '%' (line 90)
                            result_mod_282 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 21), '%', str_279, args_281)
                            
                            # Assigning a type to the variable 'output' (line 90)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'output', result_mod_282)
                            # SSA join for if statement (line 89)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 87)
                        if_condition_269 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 13), result_eq_268)
                        # Assigning a type to the variable 'if_condition_269' (line 87)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'if_condition_269', if_condition_269)
                        # SSA begins for if statement (line 87)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a BinOp to a Name (line 88):
                        
                        # Assigning a BinOp to a Name (line 88):
                        str_270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 21), 'str', '(not %s)')
                        # Getting the type of 'self' (line 88)
                        self_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 34), 'self')
                        # Obtaining the member 'args' of a type (line 88)
                        args_272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 34), self_271, 'args')
                        # Applying the binary operator '%' (line 88)
                        result_mod_273 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 21), '%', str_270, args_272)
                        
                        # Assigning a type to the variable 'output' (line 88)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'output', result_mod_273)
                        # SSA branch for the else part of an if statement (line 87)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'self' (line 89)
                        self_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 89)
                        opcode_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 13), self_274, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 89)
                        OPCODE_IF_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 89)
                        result_eq_277 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 13), '==', opcode_275, OPCODE_IF_276)
                        
                        # Testing if the type of an if condition is none (line 89)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 89)
                            if_condition_278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277)
                            # Assigning a type to the variable 'if_condition_278' (line 89)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'if_condition_278', if_condition_278)
                            # SSA begins for if statement (line 89)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a BinOp to a Name (line 90):
                            
                            # Assigning a BinOp to a Name (line 90):
                            str_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 21), 'str', '(if %s then %s else %s)')
                            # Getting the type of 'self' (line 90)
                            self_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 49), 'self')
                            # Obtaining the member 'args' of a type (line 90)
                            args_281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 49), self_280, 'args')
                            # Applying the binary operator '%' (line 90)
                            result_mod_282 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 21), '%', str_279, args_281)
                            
                            # Assigning a type to the variable 'output' (line 90)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'output', result_mod_282)
                            # SSA join for if statement (line 89)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 87)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 85)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 83)
                if_condition_251 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 13), result_eq_250)
                # Assigning a type to the variable 'if_condition_251' (line 83)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 13), 'if_condition_251', if_condition_251)
                # SSA begins for if statement (line 83)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a BinOp to a Name (line 84):
                
                # Assigning a BinOp to a Name (line 84):
                str_252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 21), 'str', '(and %s %s)')
                # Getting the type of 'self' (line 84)
                self_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 37), 'self')
                # Obtaining the member 'args' of a type (line 84)
                args_254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 37), self_253, 'args')
                # Applying the binary operator '%' (line 84)
                result_mod_255 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 21), '%', str_252, args_254)
                
                # Assigning a type to the variable 'output' (line 84)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'output', result_mod_255)
                # SSA branch for the else part of an if statement (line 83)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'self' (line 85)
                self_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 13), 'self')
                # Obtaining the member 'opcode' of a type (line 85)
                opcode_257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 13), self_256, 'opcode')
                # Getting the type of 'OPCODE_OR' (line 85)
                OPCODE_OR_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 28), 'OPCODE_OR')
                # Applying the binary operator '==' (line 85)
                result_eq_259 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 13), '==', opcode_257, OPCODE_OR_258)
                
                # Testing if the type of an if condition is none (line 85)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 85, 13), result_eq_259):
                    
                    # Getting the type of 'self' (line 87)
                    self_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'self')
                    # Obtaining the member 'opcode' of a type (line 87)
                    opcode_266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 13), self_265, 'opcode')
                    # Getting the type of 'OPCODE_NOT' (line 87)
                    OPCODE_NOT_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 28), 'OPCODE_NOT')
                    # Applying the binary operator '==' (line 87)
                    result_eq_268 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 13), '==', opcode_266, OPCODE_NOT_267)
                    
                    # Testing if the type of an if condition is none (line 87)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 87, 13), result_eq_268):
                        
                        # Getting the type of 'self' (line 89)
                        self_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 89)
                        opcode_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 13), self_274, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 89)
                        OPCODE_IF_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 89)
                        result_eq_277 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 13), '==', opcode_275, OPCODE_IF_276)
                        
                        # Testing if the type of an if condition is none (line 89)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 89)
                            if_condition_278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277)
                            # Assigning a type to the variable 'if_condition_278' (line 89)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'if_condition_278', if_condition_278)
                            # SSA begins for if statement (line 89)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a BinOp to a Name (line 90):
                            
                            # Assigning a BinOp to a Name (line 90):
                            str_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 21), 'str', '(if %s then %s else %s)')
                            # Getting the type of 'self' (line 90)
                            self_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 49), 'self')
                            # Obtaining the member 'args' of a type (line 90)
                            args_281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 49), self_280, 'args')
                            # Applying the binary operator '%' (line 90)
                            result_mod_282 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 21), '%', str_279, args_281)
                            
                            # Assigning a type to the variable 'output' (line 90)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'output', result_mod_282)
                            # SSA join for if statement (line 89)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 87)
                        if_condition_269 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 13), result_eq_268)
                        # Assigning a type to the variable 'if_condition_269' (line 87)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'if_condition_269', if_condition_269)
                        # SSA begins for if statement (line 87)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a BinOp to a Name (line 88):
                        
                        # Assigning a BinOp to a Name (line 88):
                        str_270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 21), 'str', '(not %s)')
                        # Getting the type of 'self' (line 88)
                        self_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 34), 'self')
                        # Obtaining the member 'args' of a type (line 88)
                        args_272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 34), self_271, 'args')
                        # Applying the binary operator '%' (line 88)
                        result_mod_273 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 21), '%', str_270, args_272)
                        
                        # Assigning a type to the variable 'output' (line 88)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'output', result_mod_273)
                        # SSA branch for the else part of an if statement (line 87)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'self' (line 89)
                        self_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 89)
                        opcode_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 13), self_274, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 89)
                        OPCODE_IF_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 89)
                        result_eq_277 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 13), '==', opcode_275, OPCODE_IF_276)
                        
                        # Testing if the type of an if condition is none (line 89)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 89)
                            if_condition_278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277)
                            # Assigning a type to the variable 'if_condition_278' (line 89)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'if_condition_278', if_condition_278)
                            # SSA begins for if statement (line 89)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a BinOp to a Name (line 90):
                            
                            # Assigning a BinOp to a Name (line 90):
                            str_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 21), 'str', '(if %s then %s else %s)')
                            # Getting the type of 'self' (line 90)
                            self_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 49), 'self')
                            # Obtaining the member 'args' of a type (line 90)
                            args_281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 49), self_280, 'args')
                            # Applying the binary operator '%' (line 90)
                            result_mod_282 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 21), '%', str_279, args_281)
                            
                            # Assigning a type to the variable 'output' (line 90)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'output', result_mod_282)
                            # SSA join for if statement (line 89)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 87)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 85)
                    if_condition_260 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 13), result_eq_259)
                    # Assigning a type to the variable 'if_condition_260' (line 85)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 13), 'if_condition_260', if_condition_260)
                    # SSA begins for if statement (line 85)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a BinOp to a Name (line 86):
                    
                    # Assigning a BinOp to a Name (line 86):
                    str_261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 21), 'str', '(or %s %s)')
                    # Getting the type of 'self' (line 86)
                    self_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 36), 'self')
                    # Obtaining the member 'args' of a type (line 86)
                    args_263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 36), self_262, 'args')
                    # Applying the binary operator '%' (line 86)
                    result_mod_264 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 21), '%', str_261, args_263)
                    
                    # Assigning a type to the variable 'output' (line 86)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'output', result_mod_264)
                    # SSA branch for the else part of an if statement (line 85)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'self' (line 87)
                    self_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'self')
                    # Obtaining the member 'opcode' of a type (line 87)
                    opcode_266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 13), self_265, 'opcode')
                    # Getting the type of 'OPCODE_NOT' (line 87)
                    OPCODE_NOT_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 28), 'OPCODE_NOT')
                    # Applying the binary operator '==' (line 87)
                    result_eq_268 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 13), '==', opcode_266, OPCODE_NOT_267)
                    
                    # Testing if the type of an if condition is none (line 87)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 87, 13), result_eq_268):
                        
                        # Getting the type of 'self' (line 89)
                        self_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 89)
                        opcode_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 13), self_274, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 89)
                        OPCODE_IF_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 89)
                        result_eq_277 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 13), '==', opcode_275, OPCODE_IF_276)
                        
                        # Testing if the type of an if condition is none (line 89)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 89)
                            if_condition_278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277)
                            # Assigning a type to the variable 'if_condition_278' (line 89)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'if_condition_278', if_condition_278)
                            # SSA begins for if statement (line 89)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a BinOp to a Name (line 90):
                            
                            # Assigning a BinOp to a Name (line 90):
                            str_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 21), 'str', '(if %s then %s else %s)')
                            # Getting the type of 'self' (line 90)
                            self_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 49), 'self')
                            # Obtaining the member 'args' of a type (line 90)
                            args_281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 49), self_280, 'args')
                            # Applying the binary operator '%' (line 90)
                            result_mod_282 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 21), '%', str_279, args_281)
                            
                            # Assigning a type to the variable 'output' (line 90)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'output', result_mod_282)
                            # SSA join for if statement (line 89)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 87)
                        if_condition_269 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 13), result_eq_268)
                        # Assigning a type to the variable 'if_condition_269' (line 87)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'if_condition_269', if_condition_269)
                        # SSA begins for if statement (line 87)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a BinOp to a Name (line 88):
                        
                        # Assigning a BinOp to a Name (line 88):
                        str_270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 21), 'str', '(not %s)')
                        # Getting the type of 'self' (line 88)
                        self_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 34), 'self')
                        # Obtaining the member 'args' of a type (line 88)
                        args_272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 34), self_271, 'args')
                        # Applying the binary operator '%' (line 88)
                        result_mod_273 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 21), '%', str_270, args_272)
                        
                        # Assigning a type to the variable 'output' (line 88)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'output', result_mod_273)
                        # SSA branch for the else part of an if statement (line 87)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'self' (line 89)
                        self_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'self')
                        # Obtaining the member 'opcode' of a type (line 89)
                        opcode_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 13), self_274, 'opcode')
                        # Getting the type of 'OPCODE_IF' (line 89)
                        OPCODE_IF_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 28), 'OPCODE_IF')
                        # Applying the binary operator '==' (line 89)
                        result_eq_277 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 13), '==', opcode_275, OPCODE_IF_276)
                        
                        # Testing if the type of an if condition is none (line 89)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 89)
                            if_condition_278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 13), result_eq_277)
                            # Assigning a type to the variable 'if_condition_278' (line 89)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'if_condition_278', if_condition_278)
                            # SSA begins for if statement (line 89)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a BinOp to a Name (line 90):
                            
                            # Assigning a BinOp to a Name (line 90):
                            str_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 21), 'str', '(if %s then %s else %s)')
                            # Getting the type of 'self' (line 90)
                            self_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 49), 'self')
                            # Obtaining the member 'args' of a type (line 90)
                            args_281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 49), self_280, 'args')
                            # Applying the binary operator '%' (line 90)
                            result_mod_282 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 21), '%', str_279, args_281)
                            
                            # Assigning a type to the variable 'output' (line 90)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'output', result_mod_282)
                            # SSA join for if statement (line 89)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 87)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 85)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 83)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 81)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'output' (line 92)
        output_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 15), 'output')
        # Assigning a type to the variable 'stypy_return_type' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'stypy_return_type', output_283)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 80)
        stypy_return_type_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_284)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_284


# Assigning a type to the variable 'TreeNode' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'TreeNode', TreeNode)
# Declaration of the 'Individual' class

class Individual:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 95)
        None_285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 30), 'None')
        defaults = [None_285]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 95, 4, False)
        # Assigning a type to the variable 'self' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Individual.__init__', ['genome'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['genome'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, (-1)), 'str', '\n        Initialise the multiplexer with a genome and data size.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 99)
        # Getting the type of 'genome' (line 99)
        genome_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 11), 'genome')
        # Getting the type of 'None' (line 99)
        None_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 21), 'None')
        
        (may_be_289, more_types_in_union_290) = may_be_none(genome_287, None_288)

        if may_be_289:

            if more_types_in_union_290:
                # Runtime conditional SSA (line 99)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 100):
            
            # Assigning a Call to a Attribute (line 100):
            
            # Call to TreeNode(...): (line 100)
            # Processing the call keyword arguments (line 100)
            kwargs_292 = {}
            # Getting the type of 'TreeNode' (line 100)
            TreeNode_291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 26), 'TreeNode', False)
            # Calling TreeNode(args, kwargs) (line 100)
            TreeNode_call_result_293 = invoke(stypy.reporting.localization.Localization(__file__, 100, 26), TreeNode_291, *[], **kwargs_292)
            
            # Getting the type of 'self' (line 100)
            self_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'self')
            # Setting the type of the member 'genome' of a type (line 100)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), self_294, 'genome', TreeNode_call_result_293)
            
            # Call to make_random_genome(...): (line 101)
            # Processing the call arguments (line 101)
            # Getting the type of 'self' (line 101)
            self_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 31), 'self', False)
            # Obtaining the member 'genome' of a type (line 101)
            genome_297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 31), self_296, 'genome')
            int_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 44), 'int')
            # Processing the call keyword arguments (line 101)
            kwargs_299 = {}
            # Getting the type of 'make_random_genome' (line 101)
            make_random_genome_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'make_random_genome', False)
            # Calling make_random_genome(args, kwargs) (line 101)
            make_random_genome_call_result_300 = invoke(stypy.reporting.localization.Localization(__file__, 101, 12), make_random_genome_295, *[genome_297, int_298], **kwargs_299)
            

            if more_types_in_union_290:
                # Runtime conditional SSA for else branch (line 99)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_289) or more_types_in_union_290):
            
            # Assigning a Name to a Attribute (line 103):
            
            # Assigning a Name to a Attribute (line 103):
            # Getting the type of 'genome' (line 103)
            genome_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 26), 'genome')
            # Getting the type of 'self' (line 103)
            self_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'self')
            # Setting the type of the member 'genome' of a type (line 103)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), self_302, 'genome', genome_301)

            if (may_be_289 and more_types_in_union_290):
                # SSA join for if statement (line 99)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Num to a Attribute (line 105):
        
        # Assigning a Num to a Attribute (line 105):
        float_303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 23), 'float')
        # Getting the type of 'self' (line 105)
        self_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'self')
        # Setting the type of the member 'fitness' of a type (line 105)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 8), self_304, 'fitness', float_303)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 107, 4, False)
        # Assigning a type to the variable 'self' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Individual.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        Individual.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Individual.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Individual.stypy__str__.__dict__.__setitem__('stypy_function_name', 'Individual.stypy__str__')
        Individual.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        Individual.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Individual.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Individual.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Individual.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Individual.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Individual.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Individual.stypy__str__', [], None, None, defaults, varargs, kwargs)

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

        str_305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 8), 'str', 'Represent this individual.')
        str_306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 15), 'str', 'Genome: %s, fitness %s.')
        
        # Obtaining an instance of the builtin type 'tuple' (line 109)
        tuple_307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 109)
        # Adding element type (line 109)
        # Getting the type of 'self' (line 109)
        self_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 44), 'self')
        # Obtaining the member 'genome' of a type (line 109)
        genome_309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 44), self_308, 'genome')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 44), tuple_307, genome_309)
        # Adding element type (line 109)
        # Getting the type of 'self' (line 109)
        self_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 57), 'self')
        # Obtaining the member 'fitness' of a type (line 109)
        fitness_311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 57), self_310, 'fitness')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 44), tuple_307, fitness_311)
        
        # Applying the binary operator '%' (line 109)
        result_mod_312 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 15), '%', str_306, tuple_307)
        
        # Assigning a type to the variable 'stypy_return_type' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'stypy_return_type', result_mod_312)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 107)
        stypy_return_type_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_313)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_313


    @norecursion
    def copy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'copy'
        module_type_store = module_type_store.open_function_context('copy', 111, 4, False)
        # Assigning a type to the variable 'self' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Individual.copy.__dict__.__setitem__('stypy_localization', localization)
        Individual.copy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Individual.copy.__dict__.__setitem__('stypy_type_store', module_type_store)
        Individual.copy.__dict__.__setitem__('stypy_function_name', 'Individual.copy')
        Individual.copy.__dict__.__setitem__('stypy_param_names_list', [])
        Individual.copy.__dict__.__setitem__('stypy_varargs_param_name', None)
        Individual.copy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Individual.copy.__dict__.__setitem__('stypy_call_defaults', defaults)
        Individual.copy.__dict__.__setitem__('stypy_call_varargs', varargs)
        Individual.copy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Individual.copy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Individual.copy', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to Individual(...): (line 112)
        # Processing the call arguments (line 112)
        
        # Call to deepcopy(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'self' (line 112)
        self_317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 40), 'self', False)
        # Obtaining the member 'genome' of a type (line 112)
        genome_318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 40), self_317, 'genome')
        # Processing the call keyword arguments (line 112)
        kwargs_319 = {}
        # Getting the type of 'copy' (line 112)
        copy_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 26), 'copy', False)
        # Obtaining the member 'deepcopy' of a type (line 112)
        deepcopy_316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 26), copy_315, 'deepcopy')
        # Calling deepcopy(args, kwargs) (line 112)
        deepcopy_call_result_320 = invoke(stypy.reporting.localization.Localization(__file__, 112, 26), deepcopy_316, *[genome_318], **kwargs_319)
        
        # Processing the call keyword arguments (line 112)
        kwargs_321 = {}
        # Getting the type of 'Individual' (line 112)
        Individual_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 15), 'Individual', False)
        # Calling Individual(args, kwargs) (line 112)
        Individual_call_result_322 = invoke(stypy.reporting.localization.Localization(__file__, 112, 15), Individual_314, *[deepcopy_call_result_320], **kwargs_321)
        
        # Assigning a type to the variable 'stypy_return_type' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'stypy_return_type', Individual_call_result_322)
        
        # ################# End of 'copy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'copy' in the type store
        # Getting the type of 'stypy_return_type' (line 111)
        stypy_return_type_323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_323)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'copy'
        return stypy_return_type_323


    @norecursion
    def mutate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mutate'
        module_type_store = module_type_store.open_function_context('mutate', 114, 4, False)
        # Assigning a type to the variable 'self' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Individual.mutate.__dict__.__setitem__('stypy_localization', localization)
        Individual.mutate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Individual.mutate.__dict__.__setitem__('stypy_type_store', module_type_store)
        Individual.mutate.__dict__.__setitem__('stypy_function_name', 'Individual.mutate')
        Individual.mutate.__dict__.__setitem__('stypy_param_names_list', [])
        Individual.mutate.__dict__.__setitem__('stypy_varargs_param_name', None)
        Individual.mutate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Individual.mutate.__dict__.__setitem__('stypy_call_defaults', defaults)
        Individual.mutate.__dict__.__setitem__('stypy_call_varargs', varargs)
        Individual.mutate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Individual.mutate.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Individual.mutate', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mutate', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mutate(...)' code ##################

        str_324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 8), 'str', 'Mutate this individual.')
        # Getting the type of 'self' (line 116)
        self_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 11), 'self')
        # Obtaining the member 'genome' of a type (line 116)
        genome_326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 11), self_325, 'genome')
        # Obtaining the member 'args' of a type (line 116)
        args_327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 11), genome_326, 'args')
        # Testing if the type of an if condition is none (line 116)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 116, 8), args_327):
            pass
        else:
            
            # Testing the type of an if condition (line 116)
            if_condition_328 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 8), args_327)
            # Assigning a type to the variable 'if_condition_328' (line 116)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'if_condition_328', if_condition_328)
            # SSA begins for if statement (line 116)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Tuple (line 117):
            
            # Assigning a Call to a Name:
            
            # Call to get_random_node(...): (line 117)
            # Processing the call keyword arguments (line 117)
            kwargs_331 = {}
            # Getting the type of 'self' (line 117)
            self_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 27), 'self', False)
            # Obtaining the member 'get_random_node' of a type (line 117)
            get_random_node_330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 27), self_329, 'get_random_node')
            # Calling get_random_node(args, kwargs) (line 117)
            get_random_node_call_result_332 = invoke(stypy.reporting.localization.Localization(__file__, 117, 27), get_random_node_330, *[], **kwargs_331)
            
            # Assigning a type to the variable 'call_assignment_1' (line 117)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'call_assignment_1', get_random_node_call_result_332)
            
            # Assigning a Call to a Name (line 117):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_1' (line 117)
            call_assignment_1_333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'call_assignment_1', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_334 = stypy_get_value_from_tuple(call_assignment_1_333, 2, 0)
            
            # Assigning a type to the variable 'call_assignment_2' (line 117)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'call_assignment_2', stypy_get_value_from_tuple_call_result_334)
            
            # Assigning a Name to a Name (line 117):
            # Getting the type of 'call_assignment_2' (line 117)
            call_assignment_2_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'call_assignment_2')
            # Assigning a type to the variable 'node' (line 117)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'node', call_assignment_2_335)
            
            # Assigning a Call to a Name (line 117):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_1' (line 117)
            call_assignment_1_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'call_assignment_1', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_337 = stypy_get_value_from_tuple(call_assignment_1_336, 2, 1)
            
            # Assigning a type to the variable 'call_assignment_3' (line 117)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'call_assignment_3', stypy_get_value_from_tuple_call_result_337)
            
            # Assigning a Name to a Name (line 117):
            # Getting the type of 'call_assignment_3' (line 117)
            call_assignment_3_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'call_assignment_3')
            # Assigning a type to the variable 'choice' (line 117)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 18), 'choice', call_assignment_3_338)
            
            # Call to mutate(...): (line 118)
            # Processing the call keyword arguments (line 118)
            kwargs_345 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'choice' (line 118)
            choice_339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 22), 'choice', False)
            # Getting the type of 'node' (line 118)
            node_340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'node', False)
            # Obtaining the member 'args' of a type (line 118)
            args_341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 12), node_340, 'args')
            # Obtaining the member '__getitem__' of a type (line 118)
            getitem___342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 12), args_341, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 118)
            subscript_call_result_343 = invoke(stypy.reporting.localization.Localization(__file__, 118, 12), getitem___342, choice_339)
            
            # Obtaining the member 'mutate' of a type (line 118)
            mutate_344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 12), subscript_call_result_343, 'mutate')
            # Calling mutate(args, kwargs) (line 118)
            mutate_call_result_346 = invoke(stypy.reporting.localization.Localization(__file__, 118, 12), mutate_344, *[], **kwargs_345)
            
            # SSA join for if statement (line 116)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'mutate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mutate' in the type store
        # Getting the type of 'stypy_return_type' (line 114)
        stypy_return_type_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_347)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mutate'
        return stypy_return_type_347


    @norecursion
    def get_random_node(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'MAX_DEPTH' (line 120)
        MAX_DEPTH_348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 40), 'MAX_DEPTH')
        defaults = [MAX_DEPTH_348]
        # Create a new context for function 'get_random_node'
        module_type_store = module_type_store.open_function_context('get_random_node', 120, 4, False)
        # Assigning a type to the variable 'self' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Individual.get_random_node.__dict__.__setitem__('stypy_localization', localization)
        Individual.get_random_node.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Individual.get_random_node.__dict__.__setitem__('stypy_type_store', module_type_store)
        Individual.get_random_node.__dict__.__setitem__('stypy_function_name', 'Individual.get_random_node')
        Individual.get_random_node.__dict__.__setitem__('stypy_param_names_list', ['max_depth'])
        Individual.get_random_node.__dict__.__setitem__('stypy_varargs_param_name', None)
        Individual.get_random_node.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Individual.get_random_node.__dict__.__setitem__('stypy_call_defaults', defaults)
        Individual.get_random_node.__dict__.__setitem__('stypy_call_varargs', varargs)
        Individual.get_random_node.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Individual.get_random_node.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Individual.get_random_node', ['max_depth'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_random_node', localization, ['max_depth'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_random_node(...)' code ##################

        str_349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 8), 'str', 'Get a random node from the tree.')
        
        # Assigning a Attribute to a Name (line 122):
        
        # Assigning a Attribute to a Name (line 122):
        # Getting the type of 'self' (line 122)
        self_350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 15), 'self')
        # Obtaining the member 'genome' of a type (line 122)
        genome_351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 15), self_350, 'genome')
        # Assigning a type to the variable 'root' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'root', genome_351)
        
        # Assigning a Name to a Name (line 123):
        
        # Assigning a Name to a Name (line 123):
        # Getting the type of 'root' (line 123)
        root_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 24), 'root')
        # Assigning a type to the variable 'previous_root' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'previous_root', root_352)
        
        # Assigning a Num to a Name (line 124):
        
        # Assigning a Num to a Name (line 124):
        int_353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 17), 'int')
        # Assigning a type to the variable 'choice' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'choice', int_353)
        
        
        # Call to range(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'max_depth' (line 125)
        max_depth_355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 29), 'max_depth', False)
        # Processing the call keyword arguments (line 125)
        kwargs_356 = {}
        # Getting the type of 'range' (line 125)
        range_354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 23), 'range', False)
        # Calling range(args, kwargs) (line 125)
        range_call_result_357 = invoke(stypy.reporting.localization.Localization(__file__, 125, 23), range_354, *[max_depth_355], **kwargs_356)
        
        # Assigning a type to the variable 'range_call_result_357' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'range_call_result_357', range_call_result_357)
        # Testing if the for loop is going to be iterated (line 125)
        # Testing the type of a for loop iterable (line 125)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 125, 8), range_call_result_357)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 125, 8), range_call_result_357):
            # Getting the type of the for loop variable (line 125)
            for_loop_var_358 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 125, 8), range_call_result_357)
            # Assigning a type to the variable 'counter' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'counter', for_loop_var_358)
            # SSA begins for a for statement (line 125)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Evaluating a boolean operation
            # Getting the type of 'root' (line 126)
            root_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 15), 'root')
            # Obtaining the member 'args' of a type (line 126)
            args_360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 15), root_359, 'args')
            
            
            # Call to random(...): (line 126)
            # Processing the call keyword arguments (line 126)
            kwargs_362 = {}
            # Getting the type of 'random' (line 126)
            random_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 29), 'random', False)
            # Calling random(args, kwargs) (line 126)
            random_call_result_363 = invoke(stypy.reporting.localization.Localization(__file__, 126, 29), random_361, *[], **kwargs_362)
            
            int_364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 40), 'int')
            # Getting the type of 'MAX_DEPTH' (line 126)
            MAX_DEPTH_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 44), 'MAX_DEPTH')
            # Applying the binary operator 'div' (line 126)
            result_div_366 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 40), 'div', int_364, MAX_DEPTH_365)
            
            # Applying the binary operator '>' (line 126)
            result_gt_367 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 29), '>', random_call_result_363, result_div_366)
            
            # Applying the binary operator 'and' (line 126)
            result_and_keyword_368 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 15), 'and', args_360, result_gt_367)
            
            # Testing if the type of an if condition is none (line 126)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 126, 12), result_and_keyword_368):
                pass
            else:
                
                # Testing the type of an if condition (line 126)
                if_condition_369 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 126, 12), result_and_keyword_368)
                # Assigning a type to the variable 'if_condition_369' (line 126)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'if_condition_369', if_condition_369)
                # SSA begins for if statement (line 126)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 127):
                
                # Assigning a Name to a Name (line 127):
                # Getting the type of 'root' (line 127)
                root_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 32), 'root')
                # Assigning a type to the variable 'previous_root' (line 127)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'previous_root', root_370)
                
                # Assigning a Call to a Name (line 128):
                
                # Assigning a Call to a Name (line 128):
                
                # Call to randrange(...): (line 128)
                # Processing the call arguments (line 128)
                
                # Call to len(...): (line 128)
                # Processing the call arguments (line 128)
                # Getting the type of 'root' (line 128)
                root_373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 39), 'root', False)
                # Obtaining the member 'args' of a type (line 128)
                args_374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 39), root_373, 'args')
                # Processing the call keyword arguments (line 128)
                kwargs_375 = {}
                # Getting the type of 'len' (line 128)
                len_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 35), 'len', False)
                # Calling len(args, kwargs) (line 128)
                len_call_result_376 = invoke(stypy.reporting.localization.Localization(__file__, 128, 35), len_372, *[args_374], **kwargs_375)
                
                # Processing the call keyword arguments (line 128)
                kwargs_377 = {}
                # Getting the type of 'randrange' (line 128)
                randrange_371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 25), 'randrange', False)
                # Calling randrange(args, kwargs) (line 128)
                randrange_call_result_378 = invoke(stypy.reporting.localization.Localization(__file__, 128, 25), randrange_371, *[len_call_result_376], **kwargs_377)
                
                # Assigning a type to the variable 'choice' (line 128)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'choice', randrange_call_result_378)
                
                # Assigning a Subscript to a Name (line 129):
                
                # Assigning a Subscript to a Name (line 129):
                
                # Obtaining the type of the subscript
                # Getting the type of 'choice' (line 129)
                choice_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 33), 'choice')
                # Getting the type of 'root' (line 129)
                root_380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 23), 'root')
                # Obtaining the member 'args' of a type (line 129)
                args_381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 23), root_380, 'args')
                # Obtaining the member '__getitem__' of a type (line 129)
                getitem___382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 23), args_381, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 129)
                subscript_call_result_383 = invoke(stypy.reporting.localization.Localization(__file__, 129, 23), getitem___382, choice_379)
                
                # Assigning a type to the variable 'root' (line 129)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'root', subscript_call_result_383)
                # SSA branch for the else part of an if statement (line 126)
                module_type_store.open_ssa_branch('else')
                # SSA join for if statement (line 126)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Obtaining an instance of the builtin type 'tuple' (line 132)
        tuple_384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 132)
        # Adding element type (line 132)
        # Getting the type of 'previous_root' (line 132)
        previous_root_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'previous_root')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 16), tuple_384, previous_root_385)
        # Adding element type (line 132)
        # Getting the type of 'choice' (line 132)
        choice_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 31), 'choice')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 16), tuple_384, choice_386)
        
        # Assigning a type to the variable 'stypy_return_type' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'stypy_return_type', tuple_384)
        
        # ################# End of 'get_random_node(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_random_node' in the type store
        # Getting the type of 'stypy_return_type' (line 120)
        stypy_return_type_387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_387)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_random_node'
        return stypy_return_type_387


    @norecursion
    def update_fitness(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 134)
        False_388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 39), 'False')
        defaults = [False_388]
        # Create a new context for function 'update_fitness'
        module_type_store = module_type_store.open_function_context('update_fitness', 134, 4, False)
        # Assigning a type to the variable 'self' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Individual.update_fitness.__dict__.__setitem__('stypy_localization', localization)
        Individual.update_fitness.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Individual.update_fitness.__dict__.__setitem__('stypy_type_store', module_type_store)
        Individual.update_fitness.__dict__.__setitem__('stypy_function_name', 'Individual.update_fitness')
        Individual.update_fitness.__dict__.__setitem__('stypy_param_names_list', ['full_test'])
        Individual.update_fitness.__dict__.__setitem__('stypy_varargs_param_name', None)
        Individual.update_fitness.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Individual.update_fitness.__dict__.__setitem__('stypy_call_defaults', defaults)
        Individual.update_fitness.__dict__.__setitem__('stypy_call_varargs', varargs)
        Individual.update_fitness.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Individual.update_fitness.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Individual.update_fitness', ['full_test'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update_fitness', localization, ['full_test'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update_fitness(...)' code ##################

        str_389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 8), 'str', "Calculate the individual's fitness and update it.")
        
        # Assigning a Num to a Name (line 136):
        
        # Assigning a Num to a Name (line 136):
        int_390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 18), 'int')
        # Assigning a type to the variable 'correct' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'correct', int_390)
        # Getting the type of 'full_test' (line 137)
        full_test_391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 11), 'full_test')
        # Testing if the type of an if condition is none (line 137)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 137, 8), full_test_391):
            
            
            # Call to range(...): (line 152)
            # Processing the call arguments (line 152)
            # Getting the type of 'DATA_SIZE' (line 152)
            DATA_SIZE_442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 29), 'DATA_SIZE', False)
            # Processing the call keyword arguments (line 152)
            kwargs_443 = {}
            # Getting the type of 'range' (line 152)
            range_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 23), 'range', False)
            # Calling range(args, kwargs) (line 152)
            range_call_result_444 = invoke(stypy.reporting.localization.Localization(__file__, 152, 23), range_441, *[DATA_SIZE_442], **kwargs_443)
            
            # Assigning a type to the variable 'range_call_result_444' (line 152)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'range_call_result_444', range_call_result_444)
            # Testing if the for loop is going to be iterated (line 152)
            # Testing the type of a for loop iterable (line 152)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 152, 12), range_call_result_444)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 152, 12), range_call_result_444):
                # Getting the type of the for loop variable (line 152)
                for_loop_var_445 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 152, 12), range_call_result_444)
                # Assigning a type to the variable 'mux' (line 152)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'mux', for_loop_var_445)
                # SSA begins for a for statement (line 152)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Call to range(...): (line 153)
                # Processing the call arguments (line 153)
                int_447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 34), 'int')
                # Getting the type of 'DATA_SIZE' (line 153)
                DATA_SIZE_448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 39), 'DATA_SIZE', False)
                # Applying the binary operator '<<' (line 153)
                result_lshift_449 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 34), '<<', int_447, DATA_SIZE_448)
                
                # Processing the call keyword arguments (line 153)
                kwargs_450 = {}
                # Getting the type of 'range' (line 153)
                range_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 28), 'range', False)
                # Calling range(args, kwargs) (line 153)
                range_call_result_451 = invoke(stypy.reporting.localization.Localization(__file__, 153, 28), range_446, *[result_lshift_449], **kwargs_450)
                
                # Assigning a type to the variable 'range_call_result_451' (line 153)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'range_call_result_451', range_call_result_451)
                # Testing if the for loop is going to be iterated (line 153)
                # Testing the type of a for loop iterable (line 153)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 153, 16), range_call_result_451)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 153, 16), range_call_result_451):
                    # Getting the type of the for loop variable (line 153)
                    for_loop_var_452 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 153, 16), range_call_result_451)
                    # Assigning a type to the variable 'data' (line 153)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'data', for_loop_var_452)
                    # SSA begins for a for statement (line 153)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Assigning a BinOp to a Name (line 154):
                    
                    # Assigning a BinOp to a Name (line 154):
                    # Getting the type of 'data' (line 154)
                    data_453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 29), 'data')
                    int_454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 37), 'int')
                    # Applying the binary operator '<<' (line 154)
                    result_lshift_455 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 29), '<<', data_453, int_454)
                    
                    # Getting the type of 'mux' (line 154)
                    mux_456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 42), 'mux')
                    # Applying the binary operator '|' (line 154)
                    result_or__457 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 28), '|', result_lshift_455, mux_456)
                    
                    # Assigning a type to the variable 'input' (line 154)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'input', result_or__457)
                    
                    # Assigning a Call to a Name (line 155):
                    
                    # Assigning a Call to a Name (line 155):
                    
                    # Call to execute(...): (line 155)
                    # Processing the call arguments (line 155)
                    # Getting the type of 'input' (line 155)
                    input_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 49), 'input', False)
                    # Processing the call keyword arguments (line 155)
                    kwargs_462 = {}
                    # Getting the type of 'self' (line 155)
                    self_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 29), 'self', False)
                    # Obtaining the member 'genome' of a type (line 155)
                    genome_459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 29), self_458, 'genome')
                    # Obtaining the member 'execute' of a type (line 155)
                    execute_460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 29), genome_459, 'execute')
                    # Calling execute(args, kwargs) (line 155)
                    execute_call_result_463 = invoke(stypy.reporting.localization.Localization(__file__, 155, 29), execute_460, *[input_461], **kwargs_462)
                    
                    # Assigning a type to the variable 'output' (line 155)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 20), 'output', execute_call_result_463)
                    
                    # Assigning a BinOp to a Name (line 158):
                    
                    # Assigning a BinOp to a Name (line 158):
                    # Getting the type of 'data' (line 158)
                    data_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 38), 'data')
                    int_465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 46), 'int')
                    # Getting the type of 'mux' (line 158)
                    mux_466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 51), 'mux')
                    # Applying the binary operator '<<' (line 158)
                    result_lshift_467 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 46), '<<', int_465, mux_466)
                    
                    # Applying the binary operator '&' (line 158)
                    result_and__468 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 38), '&', data_464, result_lshift_467)
                    
                    # Getting the type of 'mux' (line 158)
                    mux_469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 60), 'mux')
                    # Applying the binary operator '>>' (line 158)
                    result_rshift_470 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 37), '>>', result_and__468, mux_469)
                    
                    # Assigning a type to the variable 'correct_output' (line 158)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 20), 'correct_output', result_rshift_470)
                    
                    # Getting the type of 'output' (line 159)
                    output_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 23), 'output')
                    # Getting the type of 'correct_output' (line 159)
                    correct_output_472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 33), 'correct_output')
                    # Applying the binary operator '==' (line 159)
                    result_eq_473 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 23), '==', output_471, correct_output_472)
                    
                    # Testing if the type of an if condition is none (line 159)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 159, 20), result_eq_473):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 159)
                        if_condition_474 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 20), result_eq_473)
                        # Assigning a type to the variable 'if_condition_474' (line 159)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 20), 'if_condition_474', if_condition_474)
                        # SSA begins for if statement (line 159)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Getting the type of 'correct' (line 160)
                        correct_475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'correct')
                        int_476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 35), 'int')
                        # Applying the binary operator '+=' (line 160)
                        result_iadd_477 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 24), '+=', correct_475, int_476)
                        # Assigning a type to the variable 'correct' (line 160)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'correct', result_iadd_477)
                        
                        # SSA join for if statement (line 159)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a BinOp to a Name (line 161):
            
            # Assigning a BinOp to a Name (line 161):
            int_478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 21), 'int')
            # Getting the type of 'DATA_SIZE' (line 161)
            DATA_SIZE_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 26), 'DATA_SIZE')
            # Applying the binary operator '<<' (line 161)
            result_lshift_480 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 21), '<<', int_478, DATA_SIZE_479)
            
            # Getting the type of 'DATA_SIZE' (line 161)
            DATA_SIZE_481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 39), 'DATA_SIZE')
            # Applying the binary operator '*' (line 161)
            result_mul_482 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 20), '*', result_lshift_480, DATA_SIZE_481)
            
            # Assigning a type to the variable 'total' (line 161)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'total', result_mul_482)
        else:
            
            # Testing the type of an if condition (line 137)
            if_condition_392 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 8), full_test_391)
            # Assigning a type to the variable 'if_condition_392' (line 137)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'if_condition_392', if_condition_392)
            # SSA begins for if statement (line 137)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 138):
            
            # Assigning a BinOp to a Name (line 138):
            int_393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 20), 'int')
            # Getting the type of 'DATA_SIZE' (line 138)
            DATA_SIZE_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 25), 'DATA_SIZE')
            # Applying the binary operator '<<' (line 138)
            result_lshift_395 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 20), '<<', int_393, DATA_SIZE_394)
            
            int_396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 38), 'int')
            # Applying the binary operator '-' (line 138)
            result_sub_397 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 19), '-', result_lshift_395, int_396)
            
            # Assigning a type to the variable 'data' (line 138)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'data', result_sub_397)
            
            
            # Call to range(...): (line 139)
            # Processing the call arguments (line 139)
            # Getting the type of 'DATA_SIZE' (line 139)
            DATA_SIZE_399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 29), 'DATA_SIZE', False)
            # Processing the call keyword arguments (line 139)
            kwargs_400 = {}
            # Getting the type of 'range' (line 139)
            range_398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 23), 'range', False)
            # Calling range(args, kwargs) (line 139)
            range_call_result_401 = invoke(stypy.reporting.localization.Localization(__file__, 139, 23), range_398, *[DATA_SIZE_399], **kwargs_400)
            
            # Assigning a type to the variable 'range_call_result_401' (line 139)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'range_call_result_401', range_call_result_401)
            # Testing if the for loop is going to be iterated (line 139)
            # Testing the type of a for loop iterable (line 139)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 139, 12), range_call_result_401)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 139, 12), range_call_result_401):
                # Getting the type of the for loop variable (line 139)
                for_loop_var_402 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 139, 12), range_call_result_401)
                # Assigning a type to the variable 'mux' (line 139)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'mux', for_loop_var_402)
                # SSA begins for a for statement (line 139)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Call to range(...): (line 140)
                # Processing the call arguments (line 140)
                int_404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 31), 'int')
                # Processing the call keyword arguments (line 140)
                kwargs_405 = {}
                # Getting the type of 'range' (line 140)
                range_403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 25), 'range', False)
                # Calling range(args, kwargs) (line 140)
                range_call_result_406 = invoke(stypy.reporting.localization.Localization(__file__, 140, 25), range_403, *[int_404], **kwargs_405)
                
                # Assigning a type to the variable 'range_call_result_406' (line 140)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 16), 'range_call_result_406', range_call_result_406)
                # Testing if the for loop is going to be iterated (line 140)
                # Testing the type of a for loop iterable (line 140)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 140, 16), range_call_result_406)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 140, 16), range_call_result_406):
                    # Getting the type of the for loop variable (line 140)
                    for_loop_var_407 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 140, 16), range_call_result_406)
                    # Assigning a type to the variable '_' (line 140)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 16), '_', for_loop_var_407)
                    # SSA begins for a for statement (line 140)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Getting the type of 'data' (line 142)
                    data_408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 20), 'data')
                    int_409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 29), 'int')
                    # Getting the type of 'mux' (line 142)
                    mux_410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 34), 'mux')
                    # Applying the binary operator '<<' (line 142)
                    result_lshift_411 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 29), '<<', int_409, mux_410)
                    
                    # Applying the binary operator '^=' (line 142)
                    result_ixor_412 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 20), '^=', data_408, result_lshift_411)
                    # Assigning a type to the variable 'data' (line 142)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 20), 'data', result_ixor_412)
                    
                    
                    # Assigning a BinOp to a Name (line 143):
                    
                    # Assigning a BinOp to a Name (line 143):
                    # Getting the type of 'data' (line 143)
                    data_413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 29), 'data')
                    int_414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 37), 'int')
                    # Applying the binary operator '<<' (line 143)
                    result_lshift_415 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 29), '<<', data_413, int_414)
                    
                    # Getting the type of 'mux' (line 143)
                    mux_416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 42), 'mux')
                    # Applying the binary operator '|' (line 143)
                    result_or__417 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 28), '|', result_lshift_415, mux_416)
                    
                    # Assigning a type to the variable 'input' (line 143)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 20), 'input', result_or__417)
                    
                    # Assigning a Call to a Name (line 144):
                    
                    # Assigning a Call to a Name (line 144):
                    
                    # Call to execute(...): (line 144)
                    # Processing the call arguments (line 144)
                    # Getting the type of 'input' (line 144)
                    input_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 49), 'input', False)
                    # Processing the call keyword arguments (line 144)
                    kwargs_422 = {}
                    # Getting the type of 'self' (line 144)
                    self_418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 29), 'self', False)
                    # Obtaining the member 'genome' of a type (line 144)
                    genome_419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 29), self_418, 'genome')
                    # Obtaining the member 'execute' of a type (line 144)
                    execute_420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 29), genome_419, 'execute')
                    # Calling execute(args, kwargs) (line 144)
                    execute_call_result_423 = invoke(stypy.reporting.localization.Localization(__file__, 144, 29), execute_420, *[input_421], **kwargs_422)
                    
                    # Assigning a type to the variable 'output' (line 144)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 20), 'output', execute_call_result_423)
                    
                    # Assigning a BinOp to a Name (line 147):
                    
                    # Assigning a BinOp to a Name (line 147):
                    # Getting the type of 'data' (line 147)
                    data_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 38), 'data')
                    int_425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 46), 'int')
                    # Getting the type of 'mux' (line 147)
                    mux_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 51), 'mux')
                    # Applying the binary operator '<<' (line 147)
                    result_lshift_427 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 46), '<<', int_425, mux_426)
                    
                    # Applying the binary operator '&' (line 147)
                    result_and__428 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 38), '&', data_424, result_lshift_427)
                    
                    # Getting the type of 'mux' (line 147)
                    mux_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 60), 'mux')
                    # Applying the binary operator '>>' (line 147)
                    result_rshift_430 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 37), '>>', result_and__428, mux_429)
                    
                    # Assigning a type to the variable 'correct_output' (line 147)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 20), 'correct_output', result_rshift_430)
                    
                    # Getting the type of 'output' (line 148)
                    output_431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 23), 'output')
                    # Getting the type of 'correct_output' (line 148)
                    correct_output_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 33), 'correct_output')
                    # Applying the binary operator '==' (line 148)
                    result_eq_433 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 23), '==', output_431, correct_output_432)
                    
                    # Testing if the type of an if condition is none (line 148)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 148, 20), result_eq_433):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 148)
                        if_condition_434 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 20), result_eq_433)
                        # Assigning a type to the variable 'if_condition_434' (line 148)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 20), 'if_condition_434', if_condition_434)
                        # SSA begins for if statement (line 148)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Getting the type of 'correct' (line 149)
                        correct_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 24), 'correct')
                        int_436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 35), 'int')
                        # Applying the binary operator '+=' (line 149)
                        result_iadd_437 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 24), '+=', correct_435, int_436)
                        # Assigning a type to the variable 'correct' (line 149)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 24), 'correct', result_iadd_437)
                        
                        # SSA join for if statement (line 148)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a BinOp to a Name (line 150):
            
            # Assigning a BinOp to a Name (line 150):
            # Getting the type of 'DATA_SIZE' (line 150)
            DATA_SIZE_438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 20), 'DATA_SIZE')
            int_439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 32), 'int')
            # Applying the binary operator '*' (line 150)
            result_mul_440 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 20), '*', DATA_SIZE_438, int_439)
            
            # Assigning a type to the variable 'total' (line 150)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'total', result_mul_440)
            # SSA branch for the else part of an if statement (line 137)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to range(...): (line 152)
            # Processing the call arguments (line 152)
            # Getting the type of 'DATA_SIZE' (line 152)
            DATA_SIZE_442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 29), 'DATA_SIZE', False)
            # Processing the call keyword arguments (line 152)
            kwargs_443 = {}
            # Getting the type of 'range' (line 152)
            range_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 23), 'range', False)
            # Calling range(args, kwargs) (line 152)
            range_call_result_444 = invoke(stypy.reporting.localization.Localization(__file__, 152, 23), range_441, *[DATA_SIZE_442], **kwargs_443)
            
            # Assigning a type to the variable 'range_call_result_444' (line 152)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'range_call_result_444', range_call_result_444)
            # Testing if the for loop is going to be iterated (line 152)
            # Testing the type of a for loop iterable (line 152)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 152, 12), range_call_result_444)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 152, 12), range_call_result_444):
                # Getting the type of the for loop variable (line 152)
                for_loop_var_445 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 152, 12), range_call_result_444)
                # Assigning a type to the variable 'mux' (line 152)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'mux', for_loop_var_445)
                # SSA begins for a for statement (line 152)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Call to range(...): (line 153)
                # Processing the call arguments (line 153)
                int_447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 34), 'int')
                # Getting the type of 'DATA_SIZE' (line 153)
                DATA_SIZE_448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 39), 'DATA_SIZE', False)
                # Applying the binary operator '<<' (line 153)
                result_lshift_449 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 34), '<<', int_447, DATA_SIZE_448)
                
                # Processing the call keyword arguments (line 153)
                kwargs_450 = {}
                # Getting the type of 'range' (line 153)
                range_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 28), 'range', False)
                # Calling range(args, kwargs) (line 153)
                range_call_result_451 = invoke(stypy.reporting.localization.Localization(__file__, 153, 28), range_446, *[result_lshift_449], **kwargs_450)
                
                # Assigning a type to the variable 'range_call_result_451' (line 153)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'range_call_result_451', range_call_result_451)
                # Testing if the for loop is going to be iterated (line 153)
                # Testing the type of a for loop iterable (line 153)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 153, 16), range_call_result_451)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 153, 16), range_call_result_451):
                    # Getting the type of the for loop variable (line 153)
                    for_loop_var_452 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 153, 16), range_call_result_451)
                    # Assigning a type to the variable 'data' (line 153)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'data', for_loop_var_452)
                    # SSA begins for a for statement (line 153)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Assigning a BinOp to a Name (line 154):
                    
                    # Assigning a BinOp to a Name (line 154):
                    # Getting the type of 'data' (line 154)
                    data_453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 29), 'data')
                    int_454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 37), 'int')
                    # Applying the binary operator '<<' (line 154)
                    result_lshift_455 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 29), '<<', data_453, int_454)
                    
                    # Getting the type of 'mux' (line 154)
                    mux_456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 42), 'mux')
                    # Applying the binary operator '|' (line 154)
                    result_or__457 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 28), '|', result_lshift_455, mux_456)
                    
                    # Assigning a type to the variable 'input' (line 154)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'input', result_or__457)
                    
                    # Assigning a Call to a Name (line 155):
                    
                    # Assigning a Call to a Name (line 155):
                    
                    # Call to execute(...): (line 155)
                    # Processing the call arguments (line 155)
                    # Getting the type of 'input' (line 155)
                    input_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 49), 'input', False)
                    # Processing the call keyword arguments (line 155)
                    kwargs_462 = {}
                    # Getting the type of 'self' (line 155)
                    self_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 29), 'self', False)
                    # Obtaining the member 'genome' of a type (line 155)
                    genome_459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 29), self_458, 'genome')
                    # Obtaining the member 'execute' of a type (line 155)
                    execute_460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 29), genome_459, 'execute')
                    # Calling execute(args, kwargs) (line 155)
                    execute_call_result_463 = invoke(stypy.reporting.localization.Localization(__file__, 155, 29), execute_460, *[input_461], **kwargs_462)
                    
                    # Assigning a type to the variable 'output' (line 155)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 20), 'output', execute_call_result_463)
                    
                    # Assigning a BinOp to a Name (line 158):
                    
                    # Assigning a BinOp to a Name (line 158):
                    # Getting the type of 'data' (line 158)
                    data_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 38), 'data')
                    int_465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 46), 'int')
                    # Getting the type of 'mux' (line 158)
                    mux_466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 51), 'mux')
                    # Applying the binary operator '<<' (line 158)
                    result_lshift_467 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 46), '<<', int_465, mux_466)
                    
                    # Applying the binary operator '&' (line 158)
                    result_and__468 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 38), '&', data_464, result_lshift_467)
                    
                    # Getting the type of 'mux' (line 158)
                    mux_469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 60), 'mux')
                    # Applying the binary operator '>>' (line 158)
                    result_rshift_470 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 37), '>>', result_and__468, mux_469)
                    
                    # Assigning a type to the variable 'correct_output' (line 158)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 20), 'correct_output', result_rshift_470)
                    
                    # Getting the type of 'output' (line 159)
                    output_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 23), 'output')
                    # Getting the type of 'correct_output' (line 159)
                    correct_output_472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 33), 'correct_output')
                    # Applying the binary operator '==' (line 159)
                    result_eq_473 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 23), '==', output_471, correct_output_472)
                    
                    # Testing if the type of an if condition is none (line 159)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 159, 20), result_eq_473):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 159)
                        if_condition_474 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 20), result_eq_473)
                        # Assigning a type to the variable 'if_condition_474' (line 159)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 20), 'if_condition_474', if_condition_474)
                        # SSA begins for if statement (line 159)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Getting the type of 'correct' (line 160)
                        correct_475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'correct')
                        int_476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 35), 'int')
                        # Applying the binary operator '+=' (line 160)
                        result_iadd_477 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 24), '+=', correct_475, int_476)
                        # Assigning a type to the variable 'correct' (line 160)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'correct', result_iadd_477)
                        
                        # SSA join for if statement (line 159)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a BinOp to a Name (line 161):
            
            # Assigning a BinOp to a Name (line 161):
            int_478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 21), 'int')
            # Getting the type of 'DATA_SIZE' (line 161)
            DATA_SIZE_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 26), 'DATA_SIZE')
            # Applying the binary operator '<<' (line 161)
            result_lshift_480 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 21), '<<', int_478, DATA_SIZE_479)
            
            # Getting the type of 'DATA_SIZE' (line 161)
            DATA_SIZE_481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 39), 'DATA_SIZE')
            # Applying the binary operator '*' (line 161)
            result_mul_482 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 20), '*', result_lshift_480, DATA_SIZE_481)
            
            # Assigning a type to the variable 'total' (line 161)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'total', result_mul_482)
            # SSA join for if statement (line 137)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Attribute (line 163):
        
        # Assigning a BinOp to a Attribute (line 163):
        float_483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 24), 'float')
        # Getting the type of 'correct' (line 163)
        correct_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 30), 'correct')
        # Applying the binary operator '*' (line 163)
        result_mul_485 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 24), '*', float_483, correct_484)
        
        # Getting the type of 'total' (line 163)
        total_486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 41), 'total')
        # Applying the binary operator 'div' (line 163)
        result_div_487 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 23), 'div', result_mul_485, total_486)
        
        # Getting the type of 'self' (line 163)
        self_488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'self')
        # Setting the type of the member 'fitness' of a type (line 163)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 8), self_488, 'fitness', result_div_487)
        # Getting the type of 'self' (line 164)
        self_489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 15), 'self')
        # Obtaining the member 'fitness' of a type (line 164)
        fitness_490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 15), self_489, 'fitness')
        # Assigning a type to the variable 'stypy_return_type' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'stypy_return_type', fitness_490)
        
        # ################# End of 'update_fitness(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update_fitness' in the type store
        # Getting the type of 'stypy_return_type' (line 134)
        stypy_return_type_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_491)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update_fitness'
        return stypy_return_type_491


# Assigning a type to the variable 'Individual' (line 94)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'Individual', Individual)
# Declaration of the 'Pool' class

class Pool:
    
    # Assigning a Num to a Name (line 167):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 169, 4, False)
        # Assigning a type to the variable 'self' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Pool.__init__', [], None, None, defaults, varargs, kwargs)

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

        str_492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 8), 'str', 'Initialise the pool.')
        
        # Assigning a ListComp to a Attribute (line 171):
        
        # Assigning a ListComp to a Attribute (line 171):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'Pool' (line 171)
        Pool_497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 55), 'Pool', False)
        # Obtaining the member 'population_size' of a type (line 171)
        population_size_498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 55), Pool_497, 'population_size')
        # Processing the call keyword arguments (line 171)
        kwargs_499 = {}
        # Getting the type of 'range' (line 171)
        range_496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 49), 'range', False)
        # Calling range(args, kwargs) (line 171)
        range_call_result_500 = invoke(stypy.reporting.localization.Localization(__file__, 171, 49), range_496, *[population_size_498], **kwargs_499)
        
        comprehension_501 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 27), range_call_result_500)
        # Assigning a type to the variable '_' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 27), '_', comprehension_501)
        
        # Call to Individual(...): (line 171)
        # Processing the call keyword arguments (line 171)
        kwargs_494 = {}
        # Getting the type of 'Individual' (line 171)
        Individual_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 27), 'Individual', False)
        # Calling Individual(args, kwargs) (line 171)
        Individual_call_result_495 = invoke(stypy.reporting.localization.Localization(__file__, 171, 27), Individual_493, *[], **kwargs_494)
        
        list_502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 27), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 27), list_502, Individual_call_result_495)
        # Getting the type of 'self' (line 171)
        self_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'self')
        # Setting the type of the member 'population' of a type (line 171)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), self_503, 'population', list_502)
        
        # Assigning a Num to a Attribute (line 172):
        
        # Assigning a Num to a Attribute (line 172):
        int_504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 21), 'int')
        # Getting the type of 'self' (line 172)
        self_505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'self')
        # Setting the type of the member 'epoch' of a type (line 172)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), self_505, 'epoch', int_504)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def crossover(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'crossover'
        module_type_store = module_type_store.open_function_context('crossover', 174, 4, False)
        # Assigning a type to the variable 'self' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Pool.crossover.__dict__.__setitem__('stypy_localization', localization)
        Pool.crossover.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Pool.crossover.__dict__.__setitem__('stypy_type_store', module_type_store)
        Pool.crossover.__dict__.__setitem__('stypy_function_name', 'Pool.crossover')
        Pool.crossover.__dict__.__setitem__('stypy_param_names_list', ['father', 'mother'])
        Pool.crossover.__dict__.__setitem__('stypy_varargs_param_name', None)
        Pool.crossover.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Pool.crossover.__dict__.__setitem__('stypy_call_defaults', defaults)
        Pool.crossover.__dict__.__setitem__('stypy_call_varargs', varargs)
        Pool.crossover.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Pool.crossover.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Pool.crossover', ['father', 'mother'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'crossover', localization, ['father', 'mother'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'crossover(...)' code ##################

        
        # Assigning a Call to a Name (line 175):
        
        # Assigning a Call to a Name (line 175):
        
        # Call to copy(...): (line 175)
        # Processing the call keyword arguments (line 175)
        kwargs_508 = {}
        # Getting the type of 'father' (line 175)
        father_506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 14), 'father', False)
        # Obtaining the member 'copy' of a type (line 175)
        copy_507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 14), father_506, 'copy')
        # Calling copy(args, kwargs) (line 175)
        copy_call_result_509 = invoke(stypy.reporting.localization.Localization(__file__, 175, 14), copy_507, *[], **kwargs_508)
        
        # Assigning a type to the variable 'son' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'son', copy_call_result_509)
        
        # Assigning a Call to a Name (line 176):
        
        # Assigning a Call to a Name (line 176):
        
        # Call to copy(...): (line 176)
        # Processing the call keyword arguments (line 176)
        kwargs_512 = {}
        # Getting the type of 'mother' (line 176)
        mother_510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 19), 'mother', False)
        # Obtaining the member 'copy' of a type (line 176)
        copy_511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 19), mother_510, 'copy')
        # Calling copy(args, kwargs) (line 176)
        copy_call_result_513 = invoke(stypy.reporting.localization.Localization(__file__, 176, 19), copy_511, *[], **kwargs_512)
        
        # Assigning a type to the variable 'daughter' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'daughter', copy_call_result_513)
        
        # Assigning a Call to a Tuple (line 177):
        
        # Assigning a Call to a Name:
        
        # Call to get_random_node(...): (line 177)
        # Processing the call keyword arguments (line 177)
        kwargs_516 = {}
        # Getting the type of 'son' (line 177)
        son_514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 31), 'son', False)
        # Obtaining the member 'get_random_node' of a type (line 177)
        get_random_node_515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 31), son_514, 'get_random_node')
        # Calling get_random_node(args, kwargs) (line 177)
        get_random_node_call_result_517 = invoke(stypy.reporting.localization.Localization(__file__, 177, 31), get_random_node_515, *[], **kwargs_516)
        
        # Assigning a type to the variable 'call_assignment_4' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'call_assignment_4', get_random_node_call_result_517)
        
        # Assigning a Call to a Name (line 177):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4' (line 177)
        call_assignment_4_518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'call_assignment_4', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_519 = stypy_get_value_from_tuple(call_assignment_4_518, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_5' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'call_assignment_5', stypy_get_value_from_tuple_call_result_519)
        
        # Assigning a Name to a Name (line 177):
        # Getting the type of 'call_assignment_5' (line 177)
        call_assignment_5_520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'call_assignment_5')
        # Assigning a type to the variable 'son_node' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'son_node', call_assignment_5_520)
        
        # Assigning a Call to a Name (line 177):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4' (line 177)
        call_assignment_4_521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'call_assignment_4', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_522 = stypy_get_value_from_tuple(call_assignment_4_521, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_6' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'call_assignment_6', stypy_get_value_from_tuple_call_result_522)
        
        # Assigning a Name to a Name (line 177):
        # Getting the type of 'call_assignment_6' (line 177)
        call_assignment_6_523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'call_assignment_6')
        # Assigning a type to the variable 'son_choice' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 18), 'son_choice', call_assignment_6_523)
        
        # Assigning a Call to a Tuple (line 178):
        
        # Assigning a Call to a Name:
        
        # Call to get_random_node(...): (line 178)
        # Processing the call keyword arguments (line 178)
        kwargs_526 = {}
        # Getting the type of 'daughter' (line 178)
        daughter_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 41), 'daughter', False)
        # Obtaining the member 'get_random_node' of a type (line 178)
        get_random_node_525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 41), daughter_524, 'get_random_node')
        # Calling get_random_node(args, kwargs) (line 178)
        get_random_node_call_result_527 = invoke(stypy.reporting.localization.Localization(__file__, 178, 41), get_random_node_525, *[], **kwargs_526)
        
        # Assigning a type to the variable 'call_assignment_7' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'call_assignment_7', get_random_node_call_result_527)
        
        # Assigning a Call to a Name (line 178):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_7' (line 178)
        call_assignment_7_528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'call_assignment_7', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_529 = stypy_get_value_from_tuple(call_assignment_7_528, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_8' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'call_assignment_8', stypy_get_value_from_tuple_call_result_529)
        
        # Assigning a Name to a Name (line 178):
        # Getting the type of 'call_assignment_8' (line 178)
        call_assignment_8_530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'call_assignment_8')
        # Assigning a type to the variable 'daughter_node' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'daughter_node', call_assignment_8_530)
        
        # Assigning a Call to a Name (line 178):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_7' (line 178)
        call_assignment_7_531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'call_assignment_7', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_532 = stypy_get_value_from_tuple(call_assignment_7_531, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_9' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'call_assignment_9', stypy_get_value_from_tuple_call_result_532)
        
        # Assigning a Name to a Name (line 178):
        # Getting the type of 'call_assignment_9' (line 178)
        call_assignment_9_533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'call_assignment_9')
        # Assigning a type to the variable 'daughter_choice' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 23), 'daughter_choice', call_assignment_9_533)
        
        # Evaluating a boolean operation
        # Getting the type of 'son_node' (line 179)
        son_node_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 11), 'son_node')
        # Obtaining the member 'args' of a type (line 179)
        args_535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 11), son_node_534, 'args')
        # Getting the type of 'daughter_node' (line 179)
        daughter_node_536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 29), 'daughter_node')
        # Obtaining the member 'args' of a type (line 179)
        args_537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 29), daughter_node_536, 'args')
        # Applying the binary operator 'and' (line 179)
        result_and_keyword_538 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 11), 'and', args_535, args_537)
        
        # Testing if the type of an if condition is none (line 179)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 179, 8), result_and_keyword_538):
            pass
        else:
            
            # Testing the type of an if condition (line 179)
            if_condition_539 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 179, 8), result_and_keyword_538)
            # Assigning a type to the variable 'if_condition_539' (line 179)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'if_condition_539', if_condition_539)
            # SSA begins for if statement (line 179)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Name (line 180):
            
            # Assigning a Subscript to a Name (line 180):
            
            # Obtaining the type of the subscript
            # Getting the type of 'son_choice' (line 180)
            son_choice_540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 38), 'son_choice')
            # Getting the type of 'son_node' (line 180)
            son_node_541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 24), 'son_node')
            # Obtaining the member 'args' of a type (line 180)
            args_542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 24), son_node_541, 'args')
            # Obtaining the member '__getitem__' of a type (line 180)
            getitem___543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 24), args_542, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 180)
            subscript_call_result_544 = invoke(stypy.reporting.localization.Localization(__file__, 180, 24), getitem___543, son_choice_540)
            
            # Assigning a type to the variable 'temp_node' (line 180)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'temp_node', subscript_call_result_544)
            
            # Assigning a BinOp to a Attribute (line 181):
            
            # Assigning a BinOp to a Attribute (line 181):
            
            # Obtaining the type of the subscript
            # Getting the type of 'son_choice' (line 181)
            son_choice_545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 43), 'son_choice')
            slice_546 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 181, 28), None, son_choice_545, None)
            # Getting the type of 'son_node' (line 181)
            son_node_547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 28), 'son_node')
            # Obtaining the member 'args' of a type (line 181)
            args_548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 28), son_node_547, 'args')
            # Obtaining the member '__getitem__' of a type (line 181)
            getitem___549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 28), args_548, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 181)
            subscript_call_result_550 = invoke(stypy.reporting.localization.Localization(__file__, 181, 28), getitem___549, slice_546)
            
            
            # Obtaining an instance of the builtin type 'tuple' (line 181)
            tuple_551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 58), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 181)
            # Adding element type (line 181)
            
            # Obtaining the type of the subscript
            # Getting the type of 'daughter_choice' (line 181)
            daughter_choice_552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 77), 'daughter_choice')
            # Getting the type of 'daughter_node' (line 181)
            daughter_node_553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 58), 'daughter_node')
            # Obtaining the member 'args' of a type (line 181)
            args_554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 58), daughter_node_553, 'args')
            # Obtaining the member '__getitem__' of a type (line 181)
            getitem___555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 58), args_554, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 181)
            subscript_call_result_556 = invoke(stypy.reporting.localization.Localization(__file__, 181, 58), getitem___555, daughter_choice_552)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 58), tuple_551, subscript_call_result_556)
            
            # Applying the binary operator '+' (line 181)
            result_add_557 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 28), '+', subscript_call_result_550, tuple_551)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'son_choice' (line 181)
            son_choice_558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 113), 'son_choice')
            int_559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 124), 'int')
            # Applying the binary operator '+' (line 181)
            result_add_560 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 113), '+', son_choice_558, int_559)
            
            slice_561 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 181, 99), result_add_560, None, None)
            # Getting the type of 'son_node' (line 181)
            son_node_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 99), 'son_node')
            # Obtaining the member 'args' of a type (line 181)
            args_563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 99), son_node_562, 'args')
            # Obtaining the member '__getitem__' of a type (line 181)
            getitem___564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 99), args_563, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 181)
            subscript_call_result_565 = invoke(stypy.reporting.localization.Localization(__file__, 181, 99), getitem___564, slice_561)
            
            # Applying the binary operator '+' (line 181)
            result_add_566 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 97), '+', result_add_557, subscript_call_result_565)
            
            # Getting the type of 'son_node' (line 181)
            son_node_567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'son_node')
            # Setting the type of the member 'args' of a type (line 181)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 12), son_node_567, 'args', result_add_566)
            
            # Assigning a BinOp to a Attribute (line 182):
            
            # Assigning a BinOp to a Attribute (line 182):
            
            # Obtaining the type of the subscript
            # Getting the type of 'daughter_choice' (line 182)
            daughter_choice_568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 53), 'daughter_choice')
            slice_569 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 182, 33), None, daughter_choice_568, None)
            # Getting the type of 'daughter_node' (line 182)
            daughter_node_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 33), 'daughter_node')
            # Obtaining the member 'args' of a type (line 182)
            args_571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 33), daughter_node_570, 'args')
            # Obtaining the member '__getitem__' of a type (line 182)
            getitem___572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 33), args_571, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 182)
            subscript_call_result_573 = invoke(stypy.reporting.localization.Localization(__file__, 182, 33), getitem___572, slice_569)
            
            
            # Obtaining an instance of the builtin type 'tuple' (line 182)
            tuple_574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 73), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 182)
            # Adding element type (line 182)
            # Getting the type of 'temp_node' (line 182)
            temp_node_575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 73), 'temp_node')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 73), tuple_574, temp_node_575)
            
            # Applying the binary operator '+' (line 182)
            result_add_576 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 33), '+', subscript_call_result_573, tuple_574)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'daughter_choice' (line 182)
            daughter_choice_577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 107), 'daughter_choice')
            int_578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 123), 'int')
            # Applying the binary operator '+' (line 182)
            result_add_579 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 107), '+', daughter_choice_577, int_578)
            
            slice_580 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 182, 88), result_add_579, None, None)
            # Getting the type of 'daughter_node' (line 182)
            daughter_node_581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 88), 'daughter_node')
            # Obtaining the member 'args' of a type (line 182)
            args_582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 88), daughter_node_581, 'args')
            # Obtaining the member '__getitem__' of a type (line 182)
            getitem___583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 88), args_582, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 182)
            subscript_call_result_584 = invoke(stypy.reporting.localization.Localization(__file__, 182, 88), getitem___583, slice_580)
            
            # Applying the binary operator '+' (line 182)
            result_add_585 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 86), '+', result_add_576, subscript_call_result_584)
            
            # Getting the type of 'daughter_node' (line 182)
            daughter_node_586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'daughter_node')
            # Setting the type of the member 'args' of a type (line 182)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 12), daughter_node_586, 'args', result_add_585)
            # SSA join for if statement (line 179)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Obtaining an instance of the builtin type 'tuple' (line 183)
        tuple_587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 183)
        # Adding element type (line 183)
        # Getting the type of 'son' (line 183)
        son_588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 15), 'son')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 15), tuple_587, son_588)
        # Adding element type (line 183)
        # Getting the type of 'daughter' (line 183)
        daughter_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 20), 'daughter')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 15), tuple_587, daughter_589)
        
        # Assigning a type to the variable 'stypy_return_type' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'stypy_return_type', tuple_587)
        
        # ################# End of 'crossover(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'crossover' in the type store
        # Getting the type of 'stypy_return_type' (line 174)
        stypy_return_type_590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_590)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'crossover'
        return stypy_return_type_590


    @norecursion
    def advance_epoch(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'advance_epoch'
        module_type_store = module_type_store.open_function_context('advance_epoch', 185, 4, False)
        # Assigning a type to the variable 'self' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Pool.advance_epoch.__dict__.__setitem__('stypy_localization', localization)
        Pool.advance_epoch.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Pool.advance_epoch.__dict__.__setitem__('stypy_type_store', module_type_store)
        Pool.advance_epoch.__dict__.__setitem__('stypy_function_name', 'Pool.advance_epoch')
        Pool.advance_epoch.__dict__.__setitem__('stypy_param_names_list', [])
        Pool.advance_epoch.__dict__.__setitem__('stypy_varargs_param_name', None)
        Pool.advance_epoch.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Pool.advance_epoch.__dict__.__setitem__('stypy_call_defaults', defaults)
        Pool.advance_epoch.__dict__.__setitem__('stypy_call_varargs', varargs)
        Pool.advance_epoch.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Pool.advance_epoch.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Pool.advance_epoch', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'advance_epoch', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'advance_epoch(...)' code ##################

        str_591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 8), 'str', 'Pass the time.')
        
        # Call to sort(...): (line 188)
        # Processing the call keyword arguments (line 188)
        # Getting the type of 'fitness' (line 188)
        fitness_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 33), 'fitness', False)
        keyword_596 = fitness_595
        # Getting the type of 'True' (line 188)
        True_597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 50), 'True', False)
        keyword_598 = True_597
        kwargs_599 = {'reverse': keyword_598, 'key': keyword_596}
        # Getting the type of 'self' (line 188)
        self_592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'self', False)
        # Obtaining the member 'population' of a type (line 188)
        population_593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 8), self_592, 'population')
        # Obtaining the member 'sort' of a type (line 188)
        sort_594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 8), population_593, 'sort')
        # Calling sort(args, kwargs) (line 188)
        sort_call_result_600 = invoke(stypy.reporting.localization.Localization(__file__, 188, 8), sort_594, *[], **kwargs_599)
        
        
        # Assigning a List to a Name (line 189):
        
        # Assigning a List to a Name (line 189):
        
        # Obtaining an instance of the builtin type 'list' (line 189)
        list_601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 189)
        
        # Assigning a type to the variable 'new_population' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'new_population', list_601)
        
        # Assigning a Call to a Name (line 192):
        
        # Assigning a Call to a Name (line 192):
        
        # Call to int(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'Pool' (line 192)
        Pool_603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 20), 'Pool', False)
        # Obtaining the member 'population_size' of a type (line 192)
        population_size_604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 20), Pool_603, 'population_size')
        float_605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 43), 'float')
        # Applying the binary operator '*' (line 192)
        result_mul_606 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 20), '*', population_size_604, float_605)
        
        # Processing the call keyword arguments (line 192)
        kwargs_607 = {}
        # Getting the type of 'int' (line 192)
        int_602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 16), 'int', False)
        # Calling int(args, kwargs) (line 192)
        int_call_result_608 = invoke(stypy.reporting.localization.Localization(__file__, 192, 16), int_602, *[result_mul_606], **kwargs_607)
        
        # Assigning a type to the variable 'iters' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'iters', int_call_result_608)
        
        
        # Call to range(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'iters' (line 193)
        iters_610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 29), 'iters', False)
        # Processing the call keyword arguments (line 193)
        kwargs_611 = {}
        # Getting the type of 'range' (line 193)
        range_609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 23), 'range', False)
        # Calling range(args, kwargs) (line 193)
        range_call_result_612 = invoke(stypy.reporting.localization.Localization(__file__, 193, 23), range_609, *[iters_610], **kwargs_611)
        
        # Assigning a type to the variable 'range_call_result_612' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'range_call_result_612', range_call_result_612)
        # Testing if the for loop is going to be iterated (line 193)
        # Testing the type of a for loop iterable (line 193)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 193, 8), range_call_result_612)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 193, 8), range_call_result_612):
            # Getting the type of the for loop variable (line 193)
            for_loop_var_613 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 193, 8), range_call_result_612)
            # Assigning a type to the variable 'counter' (line 193)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'counter', for_loop_var_613)
            # SSA begins for a for statement (line 193)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 194):
            
            # Assigning a Call to a Name (line 194):
            
            # Call to copy(...): (line 194)
            # Processing the call keyword arguments (line 194)
            kwargs_620 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'counter' (line 194)
            counter_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 45), 'counter', False)
            # Getting the type of 'self' (line 194)
            self_615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 29), 'self', False)
            # Obtaining the member 'population' of a type (line 194)
            population_616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 29), self_615, 'population')
            # Obtaining the member '__getitem__' of a type (line 194)
            getitem___617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 29), population_616, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 194)
            subscript_call_result_618 = invoke(stypy.reporting.localization.Localization(__file__, 194, 29), getitem___617, counter_614)
            
            # Obtaining the member 'copy' of a type (line 194)
            copy_619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 29), subscript_call_result_618, 'copy')
            # Calling copy(args, kwargs) (line 194)
            copy_call_result_621 = invoke(stypy.reporting.localization.Localization(__file__, 194, 29), copy_619, *[], **kwargs_620)
            
            # Assigning a type to the variable 'new_individual' (line 194)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'new_individual', copy_call_result_621)
            
            # Call to append(...): (line 195)
            # Processing the call arguments (line 195)
            # Getting the type of 'new_individual' (line 195)
            new_individual_624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 34), 'new_individual', False)
            # Processing the call keyword arguments (line 195)
            kwargs_625 = {}
            # Getting the type of 'new_population' (line 195)
            new_population_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'new_population', False)
            # Obtaining the member 'append' of a type (line 195)
            append_623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 12), new_population_622, 'append')
            # Calling append(args, kwargs) (line 195)
            append_call_result_626 = invoke(stypy.reporting.localization.Localization(__file__, 195, 12), append_623, *[new_individual_624], **kwargs_625)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Name (line 198):
        
        # Assigning a Call to a Name (line 198):
        
        # Call to int(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'Pool' (line 198)
        Pool_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 20), 'Pool', False)
        # Obtaining the member 'population_size' of a type (line 198)
        population_size_629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 20), Pool_628, 'population_size')
        float_630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 43), 'float')
        # Applying the binary operator '*' (line 198)
        result_mul_631 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 20), '*', population_size_629, float_630)
        
        # Processing the call keyword arguments (line 198)
        kwargs_632 = {}
        # Getting the type of 'int' (line 198)
        int_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'int', False)
        # Calling int(args, kwargs) (line 198)
        int_call_result_633 = invoke(stypy.reporting.localization.Localization(__file__, 198, 16), int_627, *[result_mul_631], **kwargs_632)
        
        # Assigning a type to the variable 'iters' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'iters', int_call_result_633)
        
        
        # Call to range(...): (line 199)
        # Processing the call arguments (line 199)
        int_635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 29), 'int')
        # Getting the type of 'iters' (line 199)
        iters_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 32), 'iters', False)
        int_637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 39), 'int')
        # Processing the call keyword arguments (line 199)
        kwargs_638 = {}
        # Getting the type of 'range' (line 199)
        range_634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 23), 'range', False)
        # Calling range(args, kwargs) (line 199)
        range_call_result_639 = invoke(stypy.reporting.localization.Localization(__file__, 199, 23), range_634, *[int_635, iters_636, int_637], **kwargs_638)
        
        # Assigning a type to the variable 'range_call_result_639' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'range_call_result_639', range_call_result_639)
        # Testing if the for loop is going to be iterated (line 199)
        # Testing the type of a for loop iterable (line 199)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 199, 8), range_call_result_639)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 199, 8), range_call_result_639):
            # Getting the type of the for loop variable (line 199)
            for_loop_var_640 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 199, 8), range_call_result_639)
            # Assigning a type to the variable 'counter' (line 199)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'counter', for_loop_var_640)
            # SSA begins for a for statement (line 199)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Name (line 201):
            
            # Assigning a Subscript to a Name (line 201):
            
            # Obtaining the type of the subscript
            
            # Call to int(...): (line 201)
            # Processing the call arguments (line 201)
            
            # Call to triangular(...): (line 201)
            # Processing the call arguments (line 201)
            int_643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 52), 'int')
            # Getting the type of 'iters' (line 201)
            iters_644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 55), 'iters', False)
            int_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 62), 'int')
            # Processing the call keyword arguments (line 201)
            kwargs_646 = {}
            # Getting the type of 'triangular' (line 201)
            triangular_642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 41), 'triangular', False)
            # Calling triangular(args, kwargs) (line 201)
            triangular_call_result_647 = invoke(stypy.reporting.localization.Localization(__file__, 201, 41), triangular_642, *[int_643, iters_644, int_645], **kwargs_646)
            
            # Processing the call keyword arguments (line 201)
            kwargs_648 = {}
            # Getting the type of 'int' (line 201)
            int_641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 37), 'int', False)
            # Calling int(args, kwargs) (line 201)
            int_call_result_649 = invoke(stypy.reporting.localization.Localization(__file__, 201, 37), int_641, *[triangular_call_result_647], **kwargs_648)
            
            # Getting the type of 'self' (line 201)
            self_650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 21), 'self')
            # Obtaining the member 'population' of a type (line 201)
            population_651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 21), self_650, 'population')
            # Obtaining the member '__getitem__' of a type (line 201)
            getitem___652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 21), population_651, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 201)
            subscript_call_result_653 = invoke(stypy.reporting.localization.Localization(__file__, 201, 21), getitem___652, int_call_result_649)
            
            # Assigning a type to the variable 'father' (line 201)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'father', subscript_call_result_653)
            
            # Assigning a Subscript to a Name (line 202):
            
            # Assigning a Subscript to a Name (line 202):
            
            # Obtaining the type of the subscript
            
            # Call to int(...): (line 202)
            # Processing the call arguments (line 202)
            
            # Call to triangular(...): (line 202)
            # Processing the call arguments (line 202)
            int_656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 52), 'int')
            # Getting the type of 'iters' (line 202)
            iters_657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 55), 'iters', False)
            int_658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 62), 'int')
            # Processing the call keyword arguments (line 202)
            kwargs_659 = {}
            # Getting the type of 'triangular' (line 202)
            triangular_655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 41), 'triangular', False)
            # Calling triangular(args, kwargs) (line 202)
            triangular_call_result_660 = invoke(stypy.reporting.localization.Localization(__file__, 202, 41), triangular_655, *[int_656, iters_657, int_658], **kwargs_659)
            
            # Processing the call keyword arguments (line 202)
            kwargs_661 = {}
            # Getting the type of 'int' (line 202)
            int_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 37), 'int', False)
            # Calling int(args, kwargs) (line 202)
            int_call_result_662 = invoke(stypy.reporting.localization.Localization(__file__, 202, 37), int_654, *[triangular_call_result_660], **kwargs_661)
            
            # Getting the type of 'self' (line 202)
            self_663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 21), 'self')
            # Obtaining the member 'population' of a type (line 202)
            population_664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 21), self_663, 'population')
            # Obtaining the member '__getitem__' of a type (line 202)
            getitem___665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 21), population_664, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 202)
            subscript_call_result_666 = invoke(stypy.reporting.localization.Localization(__file__, 202, 21), getitem___665, int_call_result_662)
            
            # Assigning a type to the variable 'mother' (line 202)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'mother', subscript_call_result_666)
            
            # Assigning a Call to a Name (line 203):
            
            # Assigning a Call to a Name (line 203):
            
            # Call to crossover(...): (line 203)
            # Processing the call arguments (line 203)
            # Getting the type of 'father' (line 203)
            father_669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 38), 'father', False)
            # Getting the type of 'mother' (line 203)
            mother_670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 46), 'mother', False)
            # Processing the call keyword arguments (line 203)
            kwargs_671 = {}
            # Getting the type of 'self' (line 203)
            self_667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 23), 'self', False)
            # Obtaining the member 'crossover' of a type (line 203)
            crossover_668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 23), self_667, 'crossover')
            # Calling crossover(args, kwargs) (line 203)
            crossover_call_result_672 = invoke(stypy.reporting.localization.Localization(__file__, 203, 23), crossover_668, *[father_669, mother_670], **kwargs_671)
            
            # Assigning a type to the variable 'children' (line 203)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'children', crossover_call_result_672)
            
            # Call to mutate(...): (line 204)
            # Processing the call keyword arguments (line 204)
            kwargs_678 = {}
            
            # Obtaining the type of the subscript
            int_673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 21), 'int')
            # Getting the type of 'children' (line 204)
            children_674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'children', False)
            # Obtaining the member '__getitem__' of a type (line 204)
            getitem___675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 12), children_674, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 204)
            subscript_call_result_676 = invoke(stypy.reporting.localization.Localization(__file__, 204, 12), getitem___675, int_673)
            
            # Obtaining the member 'mutate' of a type (line 204)
            mutate_677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 12), subscript_call_result_676, 'mutate')
            # Calling mutate(args, kwargs) (line 204)
            mutate_call_result_679 = invoke(stypy.reporting.localization.Localization(__file__, 204, 12), mutate_677, *[], **kwargs_678)
            
            
            # Getting the type of 'new_population' (line 205)
            new_population_680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'new_population')
            # Getting the type of 'children' (line 205)
            children_681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 30), 'children')
            # Applying the binary operator '+=' (line 205)
            result_iadd_682 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 12), '+=', new_population_680, children_681)
            # Assigning a type to the variable 'new_population' (line 205)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'new_population', result_iadd_682)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Name to a Attribute (line 207):
        
        # Assigning a Name to a Attribute (line 207):
        # Getting the type of 'new_population' (line 207)
        new_population_683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 26), 'new_population')
        # Getting the type of 'self' (line 207)
        self_684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'self')
        # Setting the type of the member 'population' of a type (line 207)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), self_684, 'population', new_population_683)
        
        # Getting the type of 'self' (line 208)
        self_685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 22), 'self')
        # Obtaining the member 'population' of a type (line 208)
        population_686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 22), self_685, 'population')
        # Assigning a type to the variable 'population_686' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'population_686', population_686)
        # Testing if the for loop is going to be iterated (line 208)
        # Testing the type of a for loop iterable (line 208)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 208, 8), population_686)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 208, 8), population_686):
            # Getting the type of the for loop variable (line 208)
            for_loop_var_687 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 208, 8), population_686)
            # Assigning a type to the variable 'person' (line 208)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'person', for_loop_var_687)
            # SSA begins for a for statement (line 208)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to update_fitness(...): (line 209)
            # Processing the call keyword arguments (line 209)
            kwargs_690 = {}
            # Getting the type of 'person' (line 209)
            person_688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'person', False)
            # Obtaining the member 'update_fitness' of a type (line 209)
            update_fitness_689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 12), person_688, 'update_fitness')
            # Calling update_fitness(args, kwargs) (line 209)
            update_fitness_call_result_691 = invoke(stypy.reporting.localization.Localization(__file__, 209, 12), update_fitness_689, *[], **kwargs_690)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 'self' (line 210)
        self_692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'self')
        # Obtaining the member 'epoch' of a type (line 210)
        epoch_693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 8), self_692, 'epoch')
        int_694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 22), 'int')
        # Applying the binary operator '+=' (line 210)
        result_iadd_695 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 8), '+=', epoch_693, int_694)
        # Getting the type of 'self' (line 210)
        self_696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'self')
        # Setting the type of the member 'epoch' of a type (line 210)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 8), self_696, 'epoch', result_iadd_695)
        
        
        # ################# End of 'advance_epoch(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'advance_epoch' in the type store
        # Getting the type of 'stypy_return_type' (line 185)
        stypy_return_type_697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_697)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'advance_epoch'
        return stypy_return_type_697


    @norecursion
    def get_best_individual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_best_individual'
        module_type_store = module_type_store.open_function_context('get_best_individual', 212, 4, False)
        # Assigning a type to the variable 'self' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Pool.get_best_individual.__dict__.__setitem__('stypy_localization', localization)
        Pool.get_best_individual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Pool.get_best_individual.__dict__.__setitem__('stypy_type_store', module_type_store)
        Pool.get_best_individual.__dict__.__setitem__('stypy_function_name', 'Pool.get_best_individual')
        Pool.get_best_individual.__dict__.__setitem__('stypy_param_names_list', [])
        Pool.get_best_individual.__dict__.__setitem__('stypy_varargs_param_name', None)
        Pool.get_best_individual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Pool.get_best_individual.__dict__.__setitem__('stypy_call_defaults', defaults)
        Pool.get_best_individual.__dict__.__setitem__('stypy_call_varargs', varargs)
        Pool.get_best_individual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Pool.get_best_individual.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Pool.get_best_individual', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_best_individual', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_best_individual(...)' code ##################

        str_698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 8), 'str', 'Get the best individual of this pool.')
        
        # Call to max(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'self' (line 214)
        self_700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 19), 'self', False)
        # Obtaining the member 'population' of a type (line 214)
        population_701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 19), self_700, 'population')
        # Processing the call keyword arguments (line 214)
        # Getting the type of 'fitness' (line 214)
        fitness_702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 40), 'fitness', False)
        keyword_703 = fitness_702
        kwargs_704 = {'key': keyword_703}
        # Getting the type of 'max' (line 214)
        max_699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 15), 'max', False)
        # Calling max(args, kwargs) (line 214)
        max_call_result_705 = invoke(stypy.reporting.localization.Localization(__file__, 214, 15), max_699, *[population_701], **kwargs_704)
        
        # Assigning a type to the variable 'stypy_return_type' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'stypy_return_type', max_call_result_705)
        
        # ################# End of 'get_best_individual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_best_individual' in the type store
        # Getting the type of 'stypy_return_type' (line 212)
        stypy_return_type_706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_706)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_best_individual'
        return stypy_return_type_706


# Assigning a type to the variable 'Pool' (line 166)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), 'Pool', Pool)

# Assigning a Num to a Name (line 167):
int_707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 22), 'int')
# Getting the type of 'Pool'
Pool_708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Pool')
# Setting the type of the member 'population_size' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Pool_708, 'population_size', int_707)

@norecursion
def main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'main'
    module_type_store = module_type_store.open_function_context('main', 217, 0, False)
    
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

    
    # Assigning a Call to a Name (line 218):
    
    # Assigning a Call to a Name (line 218):
    
    # Call to Pool(...): (line 218)
    # Processing the call keyword arguments (line 218)
    kwargs_710 = {}
    # Getting the type of 'Pool' (line 218)
    Pool_709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 11), 'Pool', False)
    # Calling Pool(args, kwargs) (line 218)
    Pool_call_result_711 = invoke(stypy.reporting.localization.Localization(__file__, 218, 11), Pool_709, *[], **kwargs_710)
    
    # Assigning a type to the variable 'pool' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'pool', Pool_call_result_711)
    
    # Assigning a Call to a Name (line 219):
    
    # Assigning a Call to a Name (line 219):
    
    # Call to time(...): (line 219)
    # Processing the call keyword arguments (line 219)
    kwargs_714 = {}
    # Getting the type of 'time' (line 219)
    time_712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 17), 'time', False)
    # Obtaining the member 'time' of a type (line 219)
    time_713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 17), time_712, 'time')
    # Calling time(args, kwargs) (line 219)
    time_call_result_715 = invoke(stypy.reporting.localization.Localization(__file__, 219, 17), time_713, *[], **kwargs_714)
    
    # Assigning a type to the variable 'start_time' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'start_time', time_call_result_715)
    
    
    # Call to range(...): (line 220)
    # Processing the call arguments (line 220)
    int_717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 23), 'int')
    # Processing the call keyword arguments (line 220)
    kwargs_718 = {}
    # Getting the type of 'range' (line 220)
    range_716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 17), 'range', False)
    # Calling range(args, kwargs) (line 220)
    range_call_result_719 = invoke(stypy.reporting.localization.Localization(__file__, 220, 17), range_716, *[int_717], **kwargs_718)
    
    # Assigning a type to the variable 'range_call_result_719' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'range_call_result_719', range_call_result_719)
    # Testing if the for loop is going to be iterated (line 220)
    # Testing the type of a for loop iterable (line 220)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 220, 4), range_call_result_719)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 220, 4), range_call_result_719):
        # Getting the type of the for loop variable (line 220)
        for_loop_var_720 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 220, 4), range_call_result_719)
        # Assigning a type to the variable 'epoch' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'epoch', for_loop_var_720)
        # SSA begins for a for statement (line 220)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to advance_epoch(...): (line 221)
        # Processing the call keyword arguments (line 221)
        kwargs_723 = {}
        # Getting the type of 'pool' (line 221)
        pool_721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'pool', False)
        # Obtaining the member 'advance_epoch' of a type (line 221)
        advance_epoch_722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), pool_721, 'advance_epoch')
        # Calling advance_epoch(args, kwargs) (line 221)
        advance_epoch_call_result_724 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), advance_epoch_722, *[], **kwargs_723)
        
        
        # Assigning a Call to a Name (line 222):
        
        # Assigning a Call to a Name (line 222):
        
        # Call to get_best_individual(...): (line 222)
        # Processing the call keyword arguments (line 222)
        kwargs_727 = {}
        # Getting the type of 'pool' (line 222)
        pool_725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 26), 'pool', False)
        # Obtaining the member 'get_best_individual' of a type (line 222)
        get_best_individual_726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 26), pool_725, 'get_best_individual')
        # Calling get_best_individual(args, kwargs) (line 222)
        get_best_individual_call_result_728 = invoke(stypy.reporting.localization.Localization(__file__, 222, 26), get_best_individual_726, *[], **kwargs_727)
        
        # Assigning a type to the variable 'best_individual' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'best_individual', get_best_individual_call_result_728)
        
        # Getting the type of 'epoch' (line 223)
        epoch_729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 15), 'epoch')
        int_730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 23), 'int')
        # Applying the binary operator '%' (line 223)
        result_mod_731 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 15), '%', epoch_729, int_730)
        
        # Applying the 'not' unary operator (line 223)
        result_not__732 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 11), 'not', result_mod_731)
        
        # Testing if the type of an if condition is none (line 223)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 223, 8), result_not__732):
            pass
        else:
            
            # Testing the type of an if condition (line 223)
            if_condition_733 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 8), result_not__732)
            # Assigning a type to the variable 'if_condition_733' (line 223)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'if_condition_733', if_condition_733)
            # SSA begins for if statement (line 223)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            pass
            # SSA join for if statement (line 223)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 217)
    stypy_return_type_734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_734)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_734

# Assigning a type to the variable 'main' (line 217)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 0), 'main', main)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 229, 0, False)
    
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

    
    # Call to main(...): (line 230)
    # Processing the call keyword arguments (line 230)
    kwargs_736 = {}
    # Getting the type of 'main' (line 230)
    main_735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'main', False)
    # Calling main(args, kwargs) (line 230)
    main_call_result_737 = invoke(stypy.reporting.localization.Localization(__file__, 230, 4), main_735, *[], **kwargs_736)
    
    # Getting the type of 'True' (line 231)
    True_738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'stypy_return_type', True_738)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 229)
    stypy_return_type_739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_739)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_739

# Assigning a type to the variable 'run' (line 229)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 0), 'run', run)

# Call to run(...): (line 233)
# Processing the call keyword arguments (line 233)
kwargs_741 = {}
# Getting the type of 'run' (line 233)
run_740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 0), 'run', False)
# Calling run(args, kwargs) (line 233)
run_call_result_742 = invoke(stypy.reporting.localization.Localization(__file__, 233, 0), run_740, *[], **kwargs_741)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

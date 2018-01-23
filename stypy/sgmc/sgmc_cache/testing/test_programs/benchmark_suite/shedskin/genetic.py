
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # (c) Bearophile
2: #
3: # genetic algorithm
4: 
5: from random import random, randint, choice
6: from math import sin, pi
7: from copy import copy 
8: 
9: infiniteNeg = -1e302
10: 
11: 
12: class Individual:
13:     def __init__(self, ngenes):
14:         self.ngenes = ngenes
15:         self.genome = [random()<0.5 for i in xrange(ngenes)]
16:         self.fitness = infiniteNeg
17:     def bin2dec(self, inf=0, sup=0): 
18:         if sup == 0: sup = self.ngenes - 1 
19:         result = 0
20:         for i in xrange(inf, sup+1):
21:             if self.genome[i]:
22:                 result += 1 << (i-inf)
23:         return result
24:     def computeFitness(self):
25:         self.fitness = self.fitnessFun(self.computeValuesGenome())
26:     def __repr__(self):
27:         return "".join([str(int(gene)) for gene in self.genome])
28: 
29:     def fitnessFun(self, x):
30:         return x + abs(sin(32*x))
31:     def computeValuesGenome(self, xMin=0, xMax=pi):
32:         scaleFactor = (xMax-xMin) / (1<<self.ngenes)
33:         return self.bin2dec() * scaleFactor
34: 
35: 
36: class SGA:
37:     def __init__(self):
38:         self.popSize = 200            # Ex. 200
39:         self.genomeSize = 16          # Ex. 16
40:         self.generationsMax = 16      # Ex. 100
41:         self.crossingOverProb = 0.75  # In [0,1] ex. 0.75
42:         self.selectivePressure = 0.75 # In [0,1] ex. 0.75
43:         self.geneMutationProb = 0.005  # Ex. 0.005
44: 
45:     def generateRandomPop(self):
46:         self.population = [Individual(self.genomeSize) for i in xrange(self.popSize)]
47: 
48:     def computeFitnessPop(self):
49:         for individual in self.population:
50:             individual.computeFitness()
51: 
52:     def mutatePop(self):
53:         nmutations = int(round(self.popSize * self.genomeSize * self.geneMutationProb))
54:         for i in xrange(nmutations):
55:             individual = choice(self.population) 
56:             gene = randint(0, self.genomeSize-1)
57:             individual.genome[gene] = not individual.genome[gene] 
58: 
59:     def tounamentSelectionPop(self):
60:         pop2 = []
61:         for i in xrange(self.popSize):
62:             individual1 = choice(self.population) 
63:             individual2 = choice(self.population)
64:             if random() < self.selectivePressure:
65:                 if individual1.fitness > individual2.fitness:
66:                     pop2.append(individual1)
67:                 else:
68:                     pop2.append(individual2)
69:             else:
70:                 if individual1.fitness > individual2.fitness:
71:                     pop2.append(individual2)
72:                 else:
73:                     pop2.append(individual1)
74:         return pop2 # fixed
75: 
76:     def crossingOverPop(self):
77:         nCrossingOver = int(round(self.popSize * self.crossingOverProb))
78:         for i in xrange(nCrossingOver):
79:             ind1 = choice(self.population) 
80:             ind2 = choice(self.population) 
81:             crossPosition = randint(0, self.genomeSize-1)
82:             for j in xrange(crossPosition+1):
83:                 ind1.genome[j], ind2.genome[j] = ind2.genome[j], ind1.genome[j]
84: 
85:     def showGeneration_bestIndFind(self):
86:         fitnessTot = 0.0
87:         bestIndividualGeneration = self.population[0]
88:         for individual in self.population:
89:             fitnessTot += individual.fitness
90:             if individual.fitness > bestIndividualGeneration.fitness:
91:                 bestIndividualGeneration = individual
92:         if self.bestIndividual.fitness < bestIndividualGeneration.fitness:
93:             self.bestIndividual = copy(bestIndividualGeneration) 
94: 
95: 
96:     def run(self):
97:         self.generateRandomPop()
98:         self.bestIndividual = Individual(self.genomeSize)
99:         for self.generation in xrange(1, self.generationsMax+1):
100:             if self.generation % 300 == 0:
101:                 pass#print 'generation', self.generation
102:             self.computeFitnessPop()
103:             self.showGeneration_bestIndFind()
104:             self.population = self.tounamentSelectionPop()  
105:             self.mutatePop()
106:             self.crossingOverPop()
107: 
108: def run():
109:     sga = SGA()
110:     sga.generationsMax = 3000
111:     sga.genomeSize = 20
112:     sga.popSize = 30
113:     sga.geneMutationProb = 0.01
114:     sga.run()
115:     return True
116: 
117: run()
118: 
119: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from random import random, randint, choice' statement (line 5)
try:
    from random import random, randint, choice

except:
    random = UndefinedType
    randint = UndefinedType
    choice = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'random', None, module_type_store, ['random', 'randint', 'choice'], [random, randint, choice])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from math import sin, pi' statement (line 6)
try:
    from math import sin, pi

except:
    sin = UndefinedType
    pi = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'math', None, module_type_store, ['sin', 'pi'], [sin, pi])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from copy import copy' statement (line 7)
try:
    from copy import copy

except:
    copy = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'copy', None, module_type_store, ['copy'], [copy])


# Assigning a Num to a Name (line 9):

# Assigning a Num to a Name (line 9):
float_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 14), 'float')
# Assigning a type to the variable 'infiniteNeg' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'infiniteNeg', float_3)
# Declaration of the 'Individual' class

class Individual:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 13, 4, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Individual.__init__', ['ngenes'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['ngenes'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 14):
        
        # Assigning a Name to a Attribute (line 14):
        # Getting the type of 'ngenes' (line 14)
        ngenes_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 22), 'ngenes')
        # Getting the type of 'self' (line 14)
        self_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'self')
        # Setting the type of the member 'ngenes' of a type (line 14)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), self_5, 'ngenes', ngenes_4)
        
        # Assigning a ListComp to a Attribute (line 15):
        
        # Assigning a ListComp to a Attribute (line 15):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to xrange(...): (line 15)
        # Processing the call arguments (line 15)
        # Getting the type of 'ngenes' (line 15)
        ngenes_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 52), 'ngenes', False)
        # Processing the call keyword arguments (line 15)
        kwargs_13 = {}
        # Getting the type of 'xrange' (line 15)
        xrange_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 45), 'xrange', False)
        # Calling xrange(args, kwargs) (line 15)
        xrange_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 15, 45), xrange_11, *[ngenes_12], **kwargs_13)
        
        comprehension_15 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 23), xrange_call_result_14)
        # Assigning a type to the variable 'i' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 23), 'i', comprehension_15)
        
        
        # Call to random(...): (line 15)
        # Processing the call keyword arguments (line 15)
        kwargs_7 = {}
        # Getting the type of 'random' (line 15)
        random_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 23), 'random', False)
        # Calling random(args, kwargs) (line 15)
        random_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 15, 23), random_6, *[], **kwargs_7)
        
        float_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 32), 'float')
        # Applying the binary operator '<' (line 15)
        result_lt_10 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 23), '<', random_call_result_8, float_9)
        
        list_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 23), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 23), list_16, result_lt_10)
        # Getting the type of 'self' (line 15)
        self_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'self')
        # Setting the type of the member 'genome' of a type (line 15)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), self_17, 'genome', list_16)
        
        # Assigning a Name to a Attribute (line 16):
        
        # Assigning a Name to a Attribute (line 16):
        # Getting the type of 'infiniteNeg' (line 16)
        infiniteNeg_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 23), 'infiniteNeg')
        # Getting the type of 'self' (line 16)
        self_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'self')
        # Setting the type of the member 'fitness' of a type (line 16)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), self_19, 'fitness', infiniteNeg_18)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def bin2dec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 26), 'int')
        int_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 33), 'int')
        defaults = [int_20, int_21]
        # Create a new context for function 'bin2dec'
        module_type_store = module_type_store.open_function_context('bin2dec', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Individual.bin2dec.__dict__.__setitem__('stypy_localization', localization)
        Individual.bin2dec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Individual.bin2dec.__dict__.__setitem__('stypy_type_store', module_type_store)
        Individual.bin2dec.__dict__.__setitem__('stypy_function_name', 'Individual.bin2dec')
        Individual.bin2dec.__dict__.__setitem__('stypy_param_names_list', ['inf', 'sup'])
        Individual.bin2dec.__dict__.__setitem__('stypy_varargs_param_name', None)
        Individual.bin2dec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Individual.bin2dec.__dict__.__setitem__('stypy_call_defaults', defaults)
        Individual.bin2dec.__dict__.__setitem__('stypy_call_varargs', varargs)
        Individual.bin2dec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Individual.bin2dec.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Individual.bin2dec', ['inf', 'sup'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'bin2dec', localization, ['inf', 'sup'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'bin2dec(...)' code ##################

        
        
        # Getting the type of 'sup' (line 18)
        sup_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'sup')
        int_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 18), 'int')
        # Applying the binary operator '==' (line 18)
        result_eq_24 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 11), '==', sup_22, int_23)
        
        # Testing the type of an if condition (line 18)
        if_condition_25 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 18, 8), result_eq_24)
        # Assigning a type to the variable 'if_condition_25' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'if_condition_25', if_condition_25)
        # SSA begins for if statement (line 18)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 18):
        
        # Assigning a BinOp to a Name (line 18):
        # Getting the type of 'self' (line 18)
        self_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 27), 'self')
        # Obtaining the member 'ngenes' of a type (line 18)
        ngenes_27 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 27), self_26, 'ngenes')
        int_28 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 41), 'int')
        # Applying the binary operator '-' (line 18)
        result_sub_29 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 27), '-', ngenes_27, int_28)
        
        # Assigning a type to the variable 'sup' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 21), 'sup', result_sub_29)
        # SSA join for if statement (line 18)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Num to a Name (line 19):
        
        # Assigning a Num to a Name (line 19):
        int_30 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 17), 'int')
        # Assigning a type to the variable 'result' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'result', int_30)
        
        
        # Call to xrange(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'inf' (line 20)
        inf_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 24), 'inf', False)
        # Getting the type of 'sup' (line 20)
        sup_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 29), 'sup', False)
        int_34 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 33), 'int')
        # Applying the binary operator '+' (line 20)
        result_add_35 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 29), '+', sup_33, int_34)
        
        # Processing the call keyword arguments (line 20)
        kwargs_36 = {}
        # Getting the type of 'xrange' (line 20)
        xrange_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 20)
        xrange_call_result_37 = invoke(stypy.reporting.localization.Localization(__file__, 20, 17), xrange_31, *[inf_32, result_add_35], **kwargs_36)
        
        # Testing the type of a for loop iterable (line 20)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 20, 8), xrange_call_result_37)
        # Getting the type of the for loop variable (line 20)
        for_loop_var_38 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 20, 8), xrange_call_result_37)
        # Assigning a type to the variable 'i' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'i', for_loop_var_38)
        # SSA begins for a for statement (line 20)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 21)
        i_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 27), 'i')
        # Getting the type of 'self' (line 21)
        self_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'self')
        # Obtaining the member 'genome' of a type (line 21)
        genome_41 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 15), self_40, 'genome')
        # Obtaining the member '__getitem__' of a type (line 21)
        getitem___42 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 15), genome_41, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 21)
        subscript_call_result_43 = invoke(stypy.reporting.localization.Localization(__file__, 21, 15), getitem___42, i_39)
        
        # Testing the type of an if condition (line 21)
        if_condition_44 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 21, 12), subscript_call_result_43)
        # Assigning a type to the variable 'if_condition_44' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'if_condition_44', if_condition_44)
        # SSA begins for if statement (line 21)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'result' (line 22)
        result_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'result')
        int_46 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 26), 'int')
        # Getting the type of 'i' (line 22)
        i_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 32), 'i')
        # Getting the type of 'inf' (line 22)
        inf_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 34), 'inf')
        # Applying the binary operator '-' (line 22)
        result_sub_49 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 32), '-', i_47, inf_48)
        
        # Applying the binary operator '<<' (line 22)
        result_lshift_50 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 26), '<<', int_46, result_sub_49)
        
        # Applying the binary operator '+=' (line 22)
        result_iadd_51 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 16), '+=', result_45, result_lshift_50)
        # Assigning a type to the variable 'result' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'result', result_iadd_51)
        
        # SSA join for if statement (line 21)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'result' (line 23)
        result_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'stypy_return_type', result_52)
        
        # ################# End of 'bin2dec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'bin2dec' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_53)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'bin2dec'
        return stypy_return_type_53


    @norecursion
    def computeFitness(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'computeFitness'
        module_type_store = module_type_store.open_function_context('computeFitness', 24, 4, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Individual.computeFitness.__dict__.__setitem__('stypy_localization', localization)
        Individual.computeFitness.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Individual.computeFitness.__dict__.__setitem__('stypy_type_store', module_type_store)
        Individual.computeFitness.__dict__.__setitem__('stypy_function_name', 'Individual.computeFitness')
        Individual.computeFitness.__dict__.__setitem__('stypy_param_names_list', [])
        Individual.computeFitness.__dict__.__setitem__('stypy_varargs_param_name', None)
        Individual.computeFitness.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Individual.computeFitness.__dict__.__setitem__('stypy_call_defaults', defaults)
        Individual.computeFitness.__dict__.__setitem__('stypy_call_varargs', varargs)
        Individual.computeFitness.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Individual.computeFitness.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Individual.computeFitness', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'computeFitness', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'computeFitness(...)' code ##################

        
        # Assigning a Call to a Attribute (line 25):
        
        # Assigning a Call to a Attribute (line 25):
        
        # Call to fitnessFun(...): (line 25)
        # Processing the call arguments (line 25)
        
        # Call to computeValuesGenome(...): (line 25)
        # Processing the call keyword arguments (line 25)
        kwargs_58 = {}
        # Getting the type of 'self' (line 25)
        self_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 39), 'self', False)
        # Obtaining the member 'computeValuesGenome' of a type (line 25)
        computeValuesGenome_57 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 39), self_56, 'computeValuesGenome')
        # Calling computeValuesGenome(args, kwargs) (line 25)
        computeValuesGenome_call_result_59 = invoke(stypy.reporting.localization.Localization(__file__, 25, 39), computeValuesGenome_57, *[], **kwargs_58)
        
        # Processing the call keyword arguments (line 25)
        kwargs_60 = {}
        # Getting the type of 'self' (line 25)
        self_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 23), 'self', False)
        # Obtaining the member 'fitnessFun' of a type (line 25)
        fitnessFun_55 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 23), self_54, 'fitnessFun')
        # Calling fitnessFun(args, kwargs) (line 25)
        fitnessFun_call_result_61 = invoke(stypy.reporting.localization.Localization(__file__, 25, 23), fitnessFun_55, *[computeValuesGenome_call_result_59], **kwargs_60)
        
        # Getting the type of 'self' (line 25)
        self_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self')
        # Setting the type of the member 'fitness' of a type (line 25)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_62, 'fitness', fitnessFun_call_result_61)
        
        # ################# End of 'computeFitness(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'computeFitness' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_63)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'computeFitness'
        return stypy_return_type_63


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 26, 4, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Individual.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        Individual.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Individual.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Individual.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'Individual.stypy__repr__')
        Individual.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        Individual.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Individual.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Individual.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Individual.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Individual.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Individual.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Individual.stypy__repr__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__repr__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__repr__(...)' code ##################

        
        # Call to join(...): (line 27)
        # Processing the call arguments (line 27)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 27)
        self_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 51), 'self', False)
        # Obtaining the member 'genome' of a type (line 27)
        genome_74 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 51), self_73, 'genome')
        comprehension_75 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 24), genome_74)
        # Assigning a type to the variable 'gene' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 24), 'gene', comprehension_75)
        
        # Call to str(...): (line 27)
        # Processing the call arguments (line 27)
        
        # Call to int(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'gene' (line 27)
        gene_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 32), 'gene', False)
        # Processing the call keyword arguments (line 27)
        kwargs_69 = {}
        # Getting the type of 'int' (line 27)
        int_67 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 28), 'int', False)
        # Calling int(args, kwargs) (line 27)
        int_call_result_70 = invoke(stypy.reporting.localization.Localization(__file__, 27, 28), int_67, *[gene_68], **kwargs_69)
        
        # Processing the call keyword arguments (line 27)
        kwargs_71 = {}
        # Getting the type of 'str' (line 27)
        str_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 24), 'str', False)
        # Calling str(args, kwargs) (line 27)
        str_call_result_72 = invoke(stypy.reporting.localization.Localization(__file__, 27, 24), str_66, *[int_call_result_70], **kwargs_71)
        
        list_76 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 24), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 24), list_76, str_call_result_72)
        # Processing the call keyword arguments (line 27)
        kwargs_77 = {}
        str_64 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 15), 'str', '')
        # Obtaining the member 'join' of a type (line 27)
        join_65 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 15), str_64, 'join')
        # Calling join(args, kwargs) (line 27)
        join_call_result_78 = invoke(stypy.reporting.localization.Localization(__file__, 27, 15), join_65, *[list_76], **kwargs_77)
        
        # Assigning a type to the variable 'stypy_return_type' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'stypy_return_type', join_call_result_78)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 26)
        stypy_return_type_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_79)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_79


    @norecursion
    def fitnessFun(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fitnessFun'
        module_type_store = module_type_store.open_function_context('fitnessFun', 29, 4, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Individual.fitnessFun.__dict__.__setitem__('stypy_localization', localization)
        Individual.fitnessFun.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Individual.fitnessFun.__dict__.__setitem__('stypy_type_store', module_type_store)
        Individual.fitnessFun.__dict__.__setitem__('stypy_function_name', 'Individual.fitnessFun')
        Individual.fitnessFun.__dict__.__setitem__('stypy_param_names_list', ['x'])
        Individual.fitnessFun.__dict__.__setitem__('stypy_varargs_param_name', None)
        Individual.fitnessFun.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Individual.fitnessFun.__dict__.__setitem__('stypy_call_defaults', defaults)
        Individual.fitnessFun.__dict__.__setitem__('stypy_call_varargs', varargs)
        Individual.fitnessFun.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Individual.fitnessFun.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Individual.fitnessFun', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fitnessFun', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fitnessFun(...)' code ##################

        # Getting the type of 'x' (line 30)
        x_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 15), 'x')
        
        # Call to abs(...): (line 30)
        # Processing the call arguments (line 30)
        
        # Call to sin(...): (line 30)
        # Processing the call arguments (line 30)
        int_83 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 27), 'int')
        # Getting the type of 'x' (line 30)
        x_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 30), 'x', False)
        # Applying the binary operator '*' (line 30)
        result_mul_85 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 27), '*', int_83, x_84)
        
        # Processing the call keyword arguments (line 30)
        kwargs_86 = {}
        # Getting the type of 'sin' (line 30)
        sin_82 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 23), 'sin', False)
        # Calling sin(args, kwargs) (line 30)
        sin_call_result_87 = invoke(stypy.reporting.localization.Localization(__file__, 30, 23), sin_82, *[result_mul_85], **kwargs_86)
        
        # Processing the call keyword arguments (line 30)
        kwargs_88 = {}
        # Getting the type of 'abs' (line 30)
        abs_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 19), 'abs', False)
        # Calling abs(args, kwargs) (line 30)
        abs_call_result_89 = invoke(stypy.reporting.localization.Localization(__file__, 30, 19), abs_81, *[sin_call_result_87], **kwargs_88)
        
        # Applying the binary operator '+' (line 30)
        result_add_90 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 15), '+', x_80, abs_call_result_89)
        
        # Assigning a type to the variable 'stypy_return_type' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'stypy_return_type', result_add_90)
        
        # ################# End of 'fitnessFun(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fitnessFun' in the type store
        # Getting the type of 'stypy_return_type' (line 29)
        stypy_return_type_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_91)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fitnessFun'
        return stypy_return_type_91


    @norecursion
    def computeValuesGenome(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_92 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 39), 'int')
        # Getting the type of 'pi' (line 31)
        pi_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 47), 'pi')
        defaults = [int_92, pi_93]
        # Create a new context for function 'computeValuesGenome'
        module_type_store = module_type_store.open_function_context('computeValuesGenome', 31, 4, False)
        # Assigning a type to the variable 'self' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Individual.computeValuesGenome.__dict__.__setitem__('stypy_localization', localization)
        Individual.computeValuesGenome.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Individual.computeValuesGenome.__dict__.__setitem__('stypy_type_store', module_type_store)
        Individual.computeValuesGenome.__dict__.__setitem__('stypy_function_name', 'Individual.computeValuesGenome')
        Individual.computeValuesGenome.__dict__.__setitem__('stypy_param_names_list', ['xMin', 'xMax'])
        Individual.computeValuesGenome.__dict__.__setitem__('stypy_varargs_param_name', None)
        Individual.computeValuesGenome.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Individual.computeValuesGenome.__dict__.__setitem__('stypy_call_defaults', defaults)
        Individual.computeValuesGenome.__dict__.__setitem__('stypy_call_varargs', varargs)
        Individual.computeValuesGenome.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Individual.computeValuesGenome.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Individual.computeValuesGenome', ['xMin', 'xMax'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'computeValuesGenome', localization, ['xMin', 'xMax'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'computeValuesGenome(...)' code ##################

        
        # Assigning a BinOp to a Name (line 32):
        
        # Assigning a BinOp to a Name (line 32):
        # Getting the type of 'xMax' (line 32)
        xMax_94 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 23), 'xMax')
        # Getting the type of 'xMin' (line 32)
        xMin_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 28), 'xMin')
        # Applying the binary operator '-' (line 32)
        result_sub_96 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 23), '-', xMax_94, xMin_95)
        
        int_97 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 37), 'int')
        # Getting the type of 'self' (line 32)
        self_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 40), 'self')
        # Obtaining the member 'ngenes' of a type (line 32)
        ngenes_99 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 40), self_98, 'ngenes')
        # Applying the binary operator '<<' (line 32)
        result_lshift_100 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 37), '<<', int_97, ngenes_99)
        
        # Applying the binary operator 'div' (line 32)
        result_div_101 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 22), 'div', result_sub_96, result_lshift_100)
        
        # Assigning a type to the variable 'scaleFactor' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'scaleFactor', result_div_101)
        
        # Call to bin2dec(...): (line 33)
        # Processing the call keyword arguments (line 33)
        kwargs_104 = {}
        # Getting the type of 'self' (line 33)
        self_102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'self', False)
        # Obtaining the member 'bin2dec' of a type (line 33)
        bin2dec_103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 15), self_102, 'bin2dec')
        # Calling bin2dec(args, kwargs) (line 33)
        bin2dec_call_result_105 = invoke(stypy.reporting.localization.Localization(__file__, 33, 15), bin2dec_103, *[], **kwargs_104)
        
        # Getting the type of 'scaleFactor' (line 33)
        scaleFactor_106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 32), 'scaleFactor')
        # Applying the binary operator '*' (line 33)
        result_mul_107 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 15), '*', bin2dec_call_result_105, scaleFactor_106)
        
        # Assigning a type to the variable 'stypy_return_type' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'stypy_return_type', result_mul_107)
        
        # ################# End of 'computeValuesGenome(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'computeValuesGenome' in the type store
        # Getting the type of 'stypy_return_type' (line 31)
        stypy_return_type_108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_108)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'computeValuesGenome'
        return stypy_return_type_108


# Assigning a type to the variable 'Individual' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'Individual', Individual)
# Declaration of the 'SGA' class

class SGA:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 37, 4, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SGA.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Num to a Attribute (line 38):
        
        # Assigning a Num to a Attribute (line 38):
        int_109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 23), 'int')
        # Getting the type of 'self' (line 38)
        self_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self')
        # Setting the type of the member 'popSize' of a type (line 38)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_110, 'popSize', int_109)
        
        # Assigning a Num to a Attribute (line 39):
        
        # Assigning a Num to a Attribute (line 39):
        int_111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 26), 'int')
        # Getting the type of 'self' (line 39)
        self_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'self')
        # Setting the type of the member 'genomeSize' of a type (line 39)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), self_112, 'genomeSize', int_111)
        
        # Assigning a Num to a Attribute (line 40):
        
        # Assigning a Num to a Attribute (line 40):
        int_113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 30), 'int')
        # Getting the type of 'self' (line 40)
        self_114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'self')
        # Setting the type of the member 'generationsMax' of a type (line 40)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), self_114, 'generationsMax', int_113)
        
        # Assigning a Num to a Attribute (line 41):
        
        # Assigning a Num to a Attribute (line 41):
        float_115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 32), 'float')
        # Getting the type of 'self' (line 41)
        self_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'self')
        # Setting the type of the member 'crossingOverProb' of a type (line 41)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), self_116, 'crossingOverProb', float_115)
        
        # Assigning a Num to a Attribute (line 42):
        
        # Assigning a Num to a Attribute (line 42):
        float_117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 33), 'float')
        # Getting the type of 'self' (line 42)
        self_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'self')
        # Setting the type of the member 'selectivePressure' of a type (line 42)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), self_118, 'selectivePressure', float_117)
        
        # Assigning a Num to a Attribute (line 43):
        
        # Assigning a Num to a Attribute (line 43):
        float_119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 32), 'float')
        # Getting the type of 'self' (line 43)
        self_120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self')
        # Setting the type of the member 'geneMutationProb' of a type (line 43)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_120, 'geneMutationProb', float_119)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def generateRandomPop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'generateRandomPop'
        module_type_store = module_type_store.open_function_context('generateRandomPop', 45, 4, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SGA.generateRandomPop.__dict__.__setitem__('stypy_localization', localization)
        SGA.generateRandomPop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SGA.generateRandomPop.__dict__.__setitem__('stypy_type_store', module_type_store)
        SGA.generateRandomPop.__dict__.__setitem__('stypy_function_name', 'SGA.generateRandomPop')
        SGA.generateRandomPop.__dict__.__setitem__('stypy_param_names_list', [])
        SGA.generateRandomPop.__dict__.__setitem__('stypy_varargs_param_name', None)
        SGA.generateRandomPop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SGA.generateRandomPop.__dict__.__setitem__('stypy_call_defaults', defaults)
        SGA.generateRandomPop.__dict__.__setitem__('stypy_call_varargs', varargs)
        SGA.generateRandomPop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SGA.generateRandomPop.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SGA.generateRandomPop', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'generateRandomPop', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'generateRandomPop(...)' code ##################

        
        # Assigning a ListComp to a Attribute (line 46):
        
        # Assigning a ListComp to a Attribute (line 46):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to xrange(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'self' (line 46)
        self_127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 71), 'self', False)
        # Obtaining the member 'popSize' of a type (line 46)
        popSize_128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 71), self_127, 'popSize')
        # Processing the call keyword arguments (line 46)
        kwargs_129 = {}
        # Getting the type of 'xrange' (line 46)
        xrange_126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 64), 'xrange', False)
        # Calling xrange(args, kwargs) (line 46)
        xrange_call_result_130 = invoke(stypy.reporting.localization.Localization(__file__, 46, 64), xrange_126, *[popSize_128], **kwargs_129)
        
        comprehension_131 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 27), xrange_call_result_130)
        # Assigning a type to the variable 'i' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 27), 'i', comprehension_131)
        
        # Call to Individual(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'self' (line 46)
        self_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 38), 'self', False)
        # Obtaining the member 'genomeSize' of a type (line 46)
        genomeSize_123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 38), self_122, 'genomeSize')
        # Processing the call keyword arguments (line 46)
        kwargs_124 = {}
        # Getting the type of 'Individual' (line 46)
        Individual_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 27), 'Individual', False)
        # Calling Individual(args, kwargs) (line 46)
        Individual_call_result_125 = invoke(stypy.reporting.localization.Localization(__file__, 46, 27), Individual_121, *[genomeSize_123], **kwargs_124)
        
        list_132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 27), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 27), list_132, Individual_call_result_125)
        # Getting the type of 'self' (line 46)
        self_133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'self')
        # Setting the type of the member 'population' of a type (line 46)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), self_133, 'population', list_132)
        
        # ################# End of 'generateRandomPop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'generateRandomPop' in the type store
        # Getting the type of 'stypy_return_type' (line 45)
        stypy_return_type_134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'generateRandomPop'
        return stypy_return_type_134


    @norecursion
    def computeFitnessPop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'computeFitnessPop'
        module_type_store = module_type_store.open_function_context('computeFitnessPop', 48, 4, False)
        # Assigning a type to the variable 'self' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SGA.computeFitnessPop.__dict__.__setitem__('stypy_localization', localization)
        SGA.computeFitnessPop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SGA.computeFitnessPop.__dict__.__setitem__('stypy_type_store', module_type_store)
        SGA.computeFitnessPop.__dict__.__setitem__('stypy_function_name', 'SGA.computeFitnessPop')
        SGA.computeFitnessPop.__dict__.__setitem__('stypy_param_names_list', [])
        SGA.computeFitnessPop.__dict__.__setitem__('stypy_varargs_param_name', None)
        SGA.computeFitnessPop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SGA.computeFitnessPop.__dict__.__setitem__('stypy_call_defaults', defaults)
        SGA.computeFitnessPop.__dict__.__setitem__('stypy_call_varargs', varargs)
        SGA.computeFitnessPop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SGA.computeFitnessPop.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SGA.computeFitnessPop', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'computeFitnessPop', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'computeFitnessPop(...)' code ##################

        
        # Getting the type of 'self' (line 49)
        self_135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 26), 'self')
        # Obtaining the member 'population' of a type (line 49)
        population_136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 26), self_135, 'population')
        # Testing the type of a for loop iterable (line 49)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 49, 8), population_136)
        # Getting the type of the for loop variable (line 49)
        for_loop_var_137 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 49, 8), population_136)
        # Assigning a type to the variable 'individual' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'individual', for_loop_var_137)
        # SSA begins for a for statement (line 49)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to computeFitness(...): (line 50)
        # Processing the call keyword arguments (line 50)
        kwargs_140 = {}
        # Getting the type of 'individual' (line 50)
        individual_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'individual', False)
        # Obtaining the member 'computeFitness' of a type (line 50)
        computeFitness_139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), individual_138, 'computeFitness')
        # Calling computeFitness(args, kwargs) (line 50)
        computeFitness_call_result_141 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), computeFitness_139, *[], **kwargs_140)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'computeFitnessPop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'computeFitnessPop' in the type store
        # Getting the type of 'stypy_return_type' (line 48)
        stypy_return_type_142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_142)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'computeFitnessPop'
        return stypy_return_type_142


    @norecursion
    def mutatePop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mutatePop'
        module_type_store = module_type_store.open_function_context('mutatePop', 52, 4, False)
        # Assigning a type to the variable 'self' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SGA.mutatePop.__dict__.__setitem__('stypy_localization', localization)
        SGA.mutatePop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SGA.mutatePop.__dict__.__setitem__('stypy_type_store', module_type_store)
        SGA.mutatePop.__dict__.__setitem__('stypy_function_name', 'SGA.mutatePop')
        SGA.mutatePop.__dict__.__setitem__('stypy_param_names_list', [])
        SGA.mutatePop.__dict__.__setitem__('stypy_varargs_param_name', None)
        SGA.mutatePop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SGA.mutatePop.__dict__.__setitem__('stypy_call_defaults', defaults)
        SGA.mutatePop.__dict__.__setitem__('stypy_call_varargs', varargs)
        SGA.mutatePop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SGA.mutatePop.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SGA.mutatePop', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mutatePop', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mutatePop(...)' code ##################

        
        # Assigning a Call to a Name (line 53):
        
        # Assigning a Call to a Name (line 53):
        
        # Call to int(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Call to round(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'self' (line 53)
        self_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 31), 'self', False)
        # Obtaining the member 'popSize' of a type (line 53)
        popSize_146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 31), self_145, 'popSize')
        # Getting the type of 'self' (line 53)
        self_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 46), 'self', False)
        # Obtaining the member 'genomeSize' of a type (line 53)
        genomeSize_148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 46), self_147, 'genomeSize')
        # Applying the binary operator '*' (line 53)
        result_mul_149 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 31), '*', popSize_146, genomeSize_148)
        
        # Getting the type of 'self' (line 53)
        self_150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 64), 'self', False)
        # Obtaining the member 'geneMutationProb' of a type (line 53)
        geneMutationProb_151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 64), self_150, 'geneMutationProb')
        # Applying the binary operator '*' (line 53)
        result_mul_152 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 62), '*', result_mul_149, geneMutationProb_151)
        
        # Processing the call keyword arguments (line 53)
        kwargs_153 = {}
        # Getting the type of 'round' (line 53)
        round_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 25), 'round', False)
        # Calling round(args, kwargs) (line 53)
        round_call_result_154 = invoke(stypy.reporting.localization.Localization(__file__, 53, 25), round_144, *[result_mul_152], **kwargs_153)
        
        # Processing the call keyword arguments (line 53)
        kwargs_155 = {}
        # Getting the type of 'int' (line 53)
        int_143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 21), 'int', False)
        # Calling int(args, kwargs) (line 53)
        int_call_result_156 = invoke(stypy.reporting.localization.Localization(__file__, 53, 21), int_143, *[round_call_result_154], **kwargs_155)
        
        # Assigning a type to the variable 'nmutations' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'nmutations', int_call_result_156)
        
        
        # Call to xrange(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'nmutations' (line 54)
        nmutations_158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'nmutations', False)
        # Processing the call keyword arguments (line 54)
        kwargs_159 = {}
        # Getting the type of 'xrange' (line 54)
        xrange_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 54)
        xrange_call_result_160 = invoke(stypy.reporting.localization.Localization(__file__, 54, 17), xrange_157, *[nmutations_158], **kwargs_159)
        
        # Testing the type of a for loop iterable (line 54)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 54, 8), xrange_call_result_160)
        # Getting the type of the for loop variable (line 54)
        for_loop_var_161 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 54, 8), xrange_call_result_160)
        # Assigning a type to the variable 'i' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'i', for_loop_var_161)
        # SSA begins for a for statement (line 54)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 55):
        
        # Assigning a Call to a Name (line 55):
        
        # Call to choice(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'self' (line 55)
        self_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 32), 'self', False)
        # Obtaining the member 'population' of a type (line 55)
        population_164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 32), self_163, 'population')
        # Processing the call keyword arguments (line 55)
        kwargs_165 = {}
        # Getting the type of 'choice' (line 55)
        choice_162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 25), 'choice', False)
        # Calling choice(args, kwargs) (line 55)
        choice_call_result_166 = invoke(stypy.reporting.localization.Localization(__file__, 55, 25), choice_162, *[population_164], **kwargs_165)
        
        # Assigning a type to the variable 'individual' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'individual', choice_call_result_166)
        
        # Assigning a Call to a Name (line 56):
        
        # Assigning a Call to a Name (line 56):
        
        # Call to randint(...): (line 56)
        # Processing the call arguments (line 56)
        int_168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 27), 'int')
        # Getting the type of 'self' (line 56)
        self_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 30), 'self', False)
        # Obtaining the member 'genomeSize' of a type (line 56)
        genomeSize_170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 30), self_169, 'genomeSize')
        int_171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 46), 'int')
        # Applying the binary operator '-' (line 56)
        result_sub_172 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 30), '-', genomeSize_170, int_171)
        
        # Processing the call keyword arguments (line 56)
        kwargs_173 = {}
        # Getting the type of 'randint' (line 56)
        randint_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 19), 'randint', False)
        # Calling randint(args, kwargs) (line 56)
        randint_call_result_174 = invoke(stypy.reporting.localization.Localization(__file__, 56, 19), randint_167, *[int_168, result_sub_172], **kwargs_173)
        
        # Assigning a type to the variable 'gene' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'gene', randint_call_result_174)
        
        # Assigning a UnaryOp to a Subscript (line 57):
        
        # Assigning a UnaryOp to a Subscript (line 57):
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'gene' (line 57)
        gene_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 60), 'gene')
        # Getting the type of 'individual' (line 57)
        individual_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 42), 'individual')
        # Obtaining the member 'genome' of a type (line 57)
        genome_177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 42), individual_176, 'genome')
        # Obtaining the member '__getitem__' of a type (line 57)
        getitem___178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 42), genome_177, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 57)
        subscript_call_result_179 = invoke(stypy.reporting.localization.Localization(__file__, 57, 42), getitem___178, gene_175)
        
        # Applying the 'not' unary operator (line 57)
        result_not__180 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 38), 'not', subscript_call_result_179)
        
        # Getting the type of 'individual' (line 57)
        individual_181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'individual')
        # Obtaining the member 'genome' of a type (line 57)
        genome_182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), individual_181, 'genome')
        # Getting the type of 'gene' (line 57)
        gene_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 30), 'gene')
        # Storing an element on a container (line 57)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 12), genome_182, (gene_183, result_not__180))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'mutatePop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mutatePop' in the type store
        # Getting the type of 'stypy_return_type' (line 52)
        stypy_return_type_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_184)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mutatePop'
        return stypy_return_type_184


    @norecursion
    def tounamentSelectionPop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tounamentSelectionPop'
        module_type_store = module_type_store.open_function_context('tounamentSelectionPop', 59, 4, False)
        # Assigning a type to the variable 'self' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SGA.tounamentSelectionPop.__dict__.__setitem__('stypy_localization', localization)
        SGA.tounamentSelectionPop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SGA.tounamentSelectionPop.__dict__.__setitem__('stypy_type_store', module_type_store)
        SGA.tounamentSelectionPop.__dict__.__setitem__('stypy_function_name', 'SGA.tounamentSelectionPop')
        SGA.tounamentSelectionPop.__dict__.__setitem__('stypy_param_names_list', [])
        SGA.tounamentSelectionPop.__dict__.__setitem__('stypy_varargs_param_name', None)
        SGA.tounamentSelectionPop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SGA.tounamentSelectionPop.__dict__.__setitem__('stypy_call_defaults', defaults)
        SGA.tounamentSelectionPop.__dict__.__setitem__('stypy_call_varargs', varargs)
        SGA.tounamentSelectionPop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SGA.tounamentSelectionPop.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SGA.tounamentSelectionPop', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tounamentSelectionPop', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tounamentSelectionPop(...)' code ##################

        
        # Assigning a List to a Name (line 60):
        
        # Assigning a List to a Name (line 60):
        
        # Obtaining an instance of the builtin type 'list' (line 60)
        list_185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 60)
        
        # Assigning a type to the variable 'pop2' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'pop2', list_185)
        
        
        # Call to xrange(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'self' (line 61)
        self_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'self', False)
        # Obtaining the member 'popSize' of a type (line 61)
        popSize_188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 24), self_187, 'popSize')
        # Processing the call keyword arguments (line 61)
        kwargs_189 = {}
        # Getting the type of 'xrange' (line 61)
        xrange_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 61)
        xrange_call_result_190 = invoke(stypy.reporting.localization.Localization(__file__, 61, 17), xrange_186, *[popSize_188], **kwargs_189)
        
        # Testing the type of a for loop iterable (line 61)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 61, 8), xrange_call_result_190)
        # Getting the type of the for loop variable (line 61)
        for_loop_var_191 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 61, 8), xrange_call_result_190)
        # Assigning a type to the variable 'i' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'i', for_loop_var_191)
        # SSA begins for a for statement (line 61)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 62):
        
        # Assigning a Call to a Name (line 62):
        
        # Call to choice(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'self' (line 62)
        self_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 33), 'self', False)
        # Obtaining the member 'population' of a type (line 62)
        population_194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 33), self_193, 'population')
        # Processing the call keyword arguments (line 62)
        kwargs_195 = {}
        # Getting the type of 'choice' (line 62)
        choice_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 26), 'choice', False)
        # Calling choice(args, kwargs) (line 62)
        choice_call_result_196 = invoke(stypy.reporting.localization.Localization(__file__, 62, 26), choice_192, *[population_194], **kwargs_195)
        
        # Assigning a type to the variable 'individual1' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'individual1', choice_call_result_196)
        
        # Assigning a Call to a Name (line 63):
        
        # Assigning a Call to a Name (line 63):
        
        # Call to choice(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'self' (line 63)
        self_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 33), 'self', False)
        # Obtaining the member 'population' of a type (line 63)
        population_199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 33), self_198, 'population')
        # Processing the call keyword arguments (line 63)
        kwargs_200 = {}
        # Getting the type of 'choice' (line 63)
        choice_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 26), 'choice', False)
        # Calling choice(args, kwargs) (line 63)
        choice_call_result_201 = invoke(stypy.reporting.localization.Localization(__file__, 63, 26), choice_197, *[population_199], **kwargs_200)
        
        # Assigning a type to the variable 'individual2' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'individual2', choice_call_result_201)
        
        
        
        # Call to random(...): (line 64)
        # Processing the call keyword arguments (line 64)
        kwargs_203 = {}
        # Getting the type of 'random' (line 64)
        random_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'random', False)
        # Calling random(args, kwargs) (line 64)
        random_call_result_204 = invoke(stypy.reporting.localization.Localization(__file__, 64, 15), random_202, *[], **kwargs_203)
        
        # Getting the type of 'self' (line 64)
        self_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 26), 'self')
        # Obtaining the member 'selectivePressure' of a type (line 64)
        selectivePressure_206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 26), self_205, 'selectivePressure')
        # Applying the binary operator '<' (line 64)
        result_lt_207 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 15), '<', random_call_result_204, selectivePressure_206)
        
        # Testing the type of an if condition (line 64)
        if_condition_208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 12), result_lt_207)
        # Assigning a type to the variable 'if_condition_208' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'if_condition_208', if_condition_208)
        # SSA begins for if statement (line 64)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'individual1' (line 65)
        individual1_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 'individual1')
        # Obtaining the member 'fitness' of a type (line 65)
        fitness_210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 19), individual1_209, 'fitness')
        # Getting the type of 'individual2' (line 65)
        individual2_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 41), 'individual2')
        # Obtaining the member 'fitness' of a type (line 65)
        fitness_212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 41), individual2_211, 'fitness')
        # Applying the binary operator '>' (line 65)
        result_gt_213 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 19), '>', fitness_210, fitness_212)
        
        # Testing the type of an if condition (line 65)
        if_condition_214 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 65, 16), result_gt_213)
        # Assigning a type to the variable 'if_condition_214' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'if_condition_214', if_condition_214)
        # SSA begins for if statement (line 65)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'individual1' (line 66)
        individual1_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 32), 'individual1', False)
        # Processing the call keyword arguments (line 66)
        kwargs_218 = {}
        # Getting the type of 'pop2' (line 66)
        pop2_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'pop2', False)
        # Obtaining the member 'append' of a type (line 66)
        append_216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 20), pop2_215, 'append')
        # Calling append(args, kwargs) (line 66)
        append_call_result_219 = invoke(stypy.reporting.localization.Localization(__file__, 66, 20), append_216, *[individual1_217], **kwargs_218)
        
        # SSA branch for the else part of an if statement (line 65)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'individual2' (line 68)
        individual2_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 32), 'individual2', False)
        # Processing the call keyword arguments (line 68)
        kwargs_223 = {}
        # Getting the type of 'pop2' (line 68)
        pop2_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 20), 'pop2', False)
        # Obtaining the member 'append' of a type (line 68)
        append_221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 20), pop2_220, 'append')
        # Calling append(args, kwargs) (line 68)
        append_call_result_224 = invoke(stypy.reporting.localization.Localization(__file__, 68, 20), append_221, *[individual2_222], **kwargs_223)
        
        # SSA join for if statement (line 65)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 64)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'individual1' (line 70)
        individual1_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 19), 'individual1')
        # Obtaining the member 'fitness' of a type (line 70)
        fitness_226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 19), individual1_225, 'fitness')
        # Getting the type of 'individual2' (line 70)
        individual2_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 41), 'individual2')
        # Obtaining the member 'fitness' of a type (line 70)
        fitness_228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 41), individual2_227, 'fitness')
        # Applying the binary operator '>' (line 70)
        result_gt_229 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 19), '>', fitness_226, fitness_228)
        
        # Testing the type of an if condition (line 70)
        if_condition_230 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 16), result_gt_229)
        # Assigning a type to the variable 'if_condition_230' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'if_condition_230', if_condition_230)
        # SSA begins for if statement (line 70)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'individual2' (line 71)
        individual2_233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 32), 'individual2', False)
        # Processing the call keyword arguments (line 71)
        kwargs_234 = {}
        # Getting the type of 'pop2' (line 71)
        pop2_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 20), 'pop2', False)
        # Obtaining the member 'append' of a type (line 71)
        append_232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 20), pop2_231, 'append')
        # Calling append(args, kwargs) (line 71)
        append_call_result_235 = invoke(stypy.reporting.localization.Localization(__file__, 71, 20), append_232, *[individual2_233], **kwargs_234)
        
        # SSA branch for the else part of an if statement (line 70)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'individual1' (line 73)
        individual1_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 32), 'individual1', False)
        # Processing the call keyword arguments (line 73)
        kwargs_239 = {}
        # Getting the type of 'pop2' (line 73)
        pop2_236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 20), 'pop2', False)
        # Obtaining the member 'append' of a type (line 73)
        append_237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 20), pop2_236, 'append')
        # Calling append(args, kwargs) (line 73)
        append_call_result_240 = invoke(stypy.reporting.localization.Localization(__file__, 73, 20), append_237, *[individual1_238], **kwargs_239)
        
        # SSA join for if statement (line 70)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 64)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'pop2' (line 74)
        pop2_241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 15), 'pop2')
        # Assigning a type to the variable 'stypy_return_type' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'stypy_return_type', pop2_241)
        
        # ################# End of 'tounamentSelectionPop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tounamentSelectionPop' in the type store
        # Getting the type of 'stypy_return_type' (line 59)
        stypy_return_type_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_242)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tounamentSelectionPop'
        return stypy_return_type_242


    @norecursion
    def crossingOverPop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'crossingOverPop'
        module_type_store = module_type_store.open_function_context('crossingOverPop', 76, 4, False)
        # Assigning a type to the variable 'self' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SGA.crossingOverPop.__dict__.__setitem__('stypy_localization', localization)
        SGA.crossingOverPop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SGA.crossingOverPop.__dict__.__setitem__('stypy_type_store', module_type_store)
        SGA.crossingOverPop.__dict__.__setitem__('stypy_function_name', 'SGA.crossingOverPop')
        SGA.crossingOverPop.__dict__.__setitem__('stypy_param_names_list', [])
        SGA.crossingOverPop.__dict__.__setitem__('stypy_varargs_param_name', None)
        SGA.crossingOverPop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SGA.crossingOverPop.__dict__.__setitem__('stypy_call_defaults', defaults)
        SGA.crossingOverPop.__dict__.__setitem__('stypy_call_varargs', varargs)
        SGA.crossingOverPop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SGA.crossingOverPop.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SGA.crossingOverPop', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'crossingOverPop', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'crossingOverPop(...)' code ##################

        
        # Assigning a Call to a Name (line 77):
        
        # Assigning a Call to a Name (line 77):
        
        # Call to int(...): (line 77)
        # Processing the call arguments (line 77)
        
        # Call to round(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'self' (line 77)
        self_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 34), 'self', False)
        # Obtaining the member 'popSize' of a type (line 77)
        popSize_246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 34), self_245, 'popSize')
        # Getting the type of 'self' (line 77)
        self_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 49), 'self', False)
        # Obtaining the member 'crossingOverProb' of a type (line 77)
        crossingOverProb_248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 49), self_247, 'crossingOverProb')
        # Applying the binary operator '*' (line 77)
        result_mul_249 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 34), '*', popSize_246, crossingOverProb_248)
        
        # Processing the call keyword arguments (line 77)
        kwargs_250 = {}
        # Getting the type of 'round' (line 77)
        round_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 28), 'round', False)
        # Calling round(args, kwargs) (line 77)
        round_call_result_251 = invoke(stypy.reporting.localization.Localization(__file__, 77, 28), round_244, *[result_mul_249], **kwargs_250)
        
        # Processing the call keyword arguments (line 77)
        kwargs_252 = {}
        # Getting the type of 'int' (line 77)
        int_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 24), 'int', False)
        # Calling int(args, kwargs) (line 77)
        int_call_result_253 = invoke(stypy.reporting.localization.Localization(__file__, 77, 24), int_243, *[round_call_result_251], **kwargs_252)
        
        # Assigning a type to the variable 'nCrossingOver' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'nCrossingOver', int_call_result_253)
        
        
        # Call to xrange(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'nCrossingOver' (line 78)
        nCrossingOver_255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 24), 'nCrossingOver', False)
        # Processing the call keyword arguments (line 78)
        kwargs_256 = {}
        # Getting the type of 'xrange' (line 78)
        xrange_254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 78)
        xrange_call_result_257 = invoke(stypy.reporting.localization.Localization(__file__, 78, 17), xrange_254, *[nCrossingOver_255], **kwargs_256)
        
        # Testing the type of a for loop iterable (line 78)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 78, 8), xrange_call_result_257)
        # Getting the type of the for loop variable (line 78)
        for_loop_var_258 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 78, 8), xrange_call_result_257)
        # Assigning a type to the variable 'i' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'i', for_loop_var_258)
        # SSA begins for a for statement (line 78)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 79):
        
        # Assigning a Call to a Name (line 79):
        
        # Call to choice(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'self' (line 79)
        self_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 26), 'self', False)
        # Obtaining the member 'population' of a type (line 79)
        population_261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 26), self_260, 'population')
        # Processing the call keyword arguments (line 79)
        kwargs_262 = {}
        # Getting the type of 'choice' (line 79)
        choice_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 19), 'choice', False)
        # Calling choice(args, kwargs) (line 79)
        choice_call_result_263 = invoke(stypy.reporting.localization.Localization(__file__, 79, 19), choice_259, *[population_261], **kwargs_262)
        
        # Assigning a type to the variable 'ind1' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'ind1', choice_call_result_263)
        
        # Assigning a Call to a Name (line 80):
        
        # Assigning a Call to a Name (line 80):
        
        # Call to choice(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'self' (line 80)
        self_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 26), 'self', False)
        # Obtaining the member 'population' of a type (line 80)
        population_266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 26), self_265, 'population')
        # Processing the call keyword arguments (line 80)
        kwargs_267 = {}
        # Getting the type of 'choice' (line 80)
        choice_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 19), 'choice', False)
        # Calling choice(args, kwargs) (line 80)
        choice_call_result_268 = invoke(stypy.reporting.localization.Localization(__file__, 80, 19), choice_264, *[population_266], **kwargs_267)
        
        # Assigning a type to the variable 'ind2' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'ind2', choice_call_result_268)
        
        # Assigning a Call to a Name (line 81):
        
        # Assigning a Call to a Name (line 81):
        
        # Call to randint(...): (line 81)
        # Processing the call arguments (line 81)
        int_270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 36), 'int')
        # Getting the type of 'self' (line 81)
        self_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 39), 'self', False)
        # Obtaining the member 'genomeSize' of a type (line 81)
        genomeSize_272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 39), self_271, 'genomeSize')
        int_273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 55), 'int')
        # Applying the binary operator '-' (line 81)
        result_sub_274 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 39), '-', genomeSize_272, int_273)
        
        # Processing the call keyword arguments (line 81)
        kwargs_275 = {}
        # Getting the type of 'randint' (line 81)
        randint_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 28), 'randint', False)
        # Calling randint(args, kwargs) (line 81)
        randint_call_result_276 = invoke(stypy.reporting.localization.Localization(__file__, 81, 28), randint_269, *[int_270, result_sub_274], **kwargs_275)
        
        # Assigning a type to the variable 'crossPosition' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'crossPosition', randint_call_result_276)
        
        
        # Call to xrange(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'crossPosition' (line 82)
        crossPosition_278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 28), 'crossPosition', False)
        int_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 42), 'int')
        # Applying the binary operator '+' (line 82)
        result_add_280 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 28), '+', crossPosition_278, int_279)
        
        # Processing the call keyword arguments (line 82)
        kwargs_281 = {}
        # Getting the type of 'xrange' (line 82)
        xrange_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 21), 'xrange', False)
        # Calling xrange(args, kwargs) (line 82)
        xrange_call_result_282 = invoke(stypy.reporting.localization.Localization(__file__, 82, 21), xrange_277, *[result_add_280], **kwargs_281)
        
        # Testing the type of a for loop iterable (line 82)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 82, 12), xrange_call_result_282)
        # Getting the type of the for loop variable (line 82)
        for_loop_var_283 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 82, 12), xrange_call_result_282)
        # Assigning a type to the variable 'j' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'j', for_loop_var_283)
        # SSA begins for a for statement (line 82)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Tuple to a Tuple (line 83):
        
        # Assigning a Subscript to a Name (line 83):
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 83)
        j_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 61), 'j')
        # Getting the type of 'ind2' (line 83)
        ind2_285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 49), 'ind2')
        # Obtaining the member 'genome' of a type (line 83)
        genome_286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 49), ind2_285, 'genome')
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 49), genome_286, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_288 = invoke(stypy.reporting.localization.Localization(__file__, 83, 49), getitem___287, j_284)
        
        # Assigning a type to the variable 'tuple_assignment_1' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'tuple_assignment_1', subscript_call_result_288)
        
        # Assigning a Subscript to a Name (line 83):
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 83)
        j_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 77), 'j')
        # Getting the type of 'ind1' (line 83)
        ind1_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 65), 'ind1')
        # Obtaining the member 'genome' of a type (line 83)
        genome_291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 65), ind1_290, 'genome')
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 65), genome_291, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_293 = invoke(stypy.reporting.localization.Localization(__file__, 83, 65), getitem___292, j_289)
        
        # Assigning a type to the variable 'tuple_assignment_2' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'tuple_assignment_2', subscript_call_result_293)
        
        # Assigning a Name to a Subscript (line 83):
        # Getting the type of 'tuple_assignment_1' (line 83)
        tuple_assignment_1_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'tuple_assignment_1')
        # Getting the type of 'ind1' (line 83)
        ind1_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'ind1')
        # Obtaining the member 'genome' of a type (line 83)
        genome_296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 16), ind1_295, 'genome')
        # Getting the type of 'j' (line 83)
        j_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 28), 'j')
        # Storing an element on a container (line 83)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 16), genome_296, (j_297, tuple_assignment_1_294))
        
        # Assigning a Name to a Subscript (line 83):
        # Getting the type of 'tuple_assignment_2' (line 83)
        tuple_assignment_2_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'tuple_assignment_2')
        # Getting the type of 'ind2' (line 83)
        ind2_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 32), 'ind2')
        # Obtaining the member 'genome' of a type (line 83)
        genome_300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 32), ind2_299, 'genome')
        # Getting the type of 'j' (line 83)
        j_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 44), 'j')
        # Storing an element on a container (line 83)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 32), genome_300, (j_301, tuple_assignment_2_298))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'crossingOverPop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'crossingOverPop' in the type store
        # Getting the type of 'stypy_return_type' (line 76)
        stypy_return_type_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_302)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'crossingOverPop'
        return stypy_return_type_302


    @norecursion
    def showGeneration_bestIndFind(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'showGeneration_bestIndFind'
        module_type_store = module_type_store.open_function_context('showGeneration_bestIndFind', 85, 4, False)
        # Assigning a type to the variable 'self' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SGA.showGeneration_bestIndFind.__dict__.__setitem__('stypy_localization', localization)
        SGA.showGeneration_bestIndFind.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SGA.showGeneration_bestIndFind.__dict__.__setitem__('stypy_type_store', module_type_store)
        SGA.showGeneration_bestIndFind.__dict__.__setitem__('stypy_function_name', 'SGA.showGeneration_bestIndFind')
        SGA.showGeneration_bestIndFind.__dict__.__setitem__('stypy_param_names_list', [])
        SGA.showGeneration_bestIndFind.__dict__.__setitem__('stypy_varargs_param_name', None)
        SGA.showGeneration_bestIndFind.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SGA.showGeneration_bestIndFind.__dict__.__setitem__('stypy_call_defaults', defaults)
        SGA.showGeneration_bestIndFind.__dict__.__setitem__('stypy_call_varargs', varargs)
        SGA.showGeneration_bestIndFind.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SGA.showGeneration_bestIndFind.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SGA.showGeneration_bestIndFind', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'showGeneration_bestIndFind', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'showGeneration_bestIndFind(...)' code ##################

        
        # Assigning a Num to a Name (line 86):
        
        # Assigning a Num to a Name (line 86):
        float_303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 21), 'float')
        # Assigning a type to the variable 'fitnessTot' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'fitnessTot', float_303)
        
        # Assigning a Subscript to a Name (line 87):
        
        # Assigning a Subscript to a Name (line 87):
        
        # Obtaining the type of the subscript
        int_304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 51), 'int')
        # Getting the type of 'self' (line 87)
        self_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 35), 'self')
        # Obtaining the member 'population' of a type (line 87)
        population_306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 35), self_305, 'population')
        # Obtaining the member '__getitem__' of a type (line 87)
        getitem___307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 35), population_306, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 87)
        subscript_call_result_308 = invoke(stypy.reporting.localization.Localization(__file__, 87, 35), getitem___307, int_304)
        
        # Assigning a type to the variable 'bestIndividualGeneration' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'bestIndividualGeneration', subscript_call_result_308)
        
        # Getting the type of 'self' (line 88)
        self_309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 26), 'self')
        # Obtaining the member 'population' of a type (line 88)
        population_310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 26), self_309, 'population')
        # Testing the type of a for loop iterable (line 88)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 88, 8), population_310)
        # Getting the type of the for loop variable (line 88)
        for_loop_var_311 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 88, 8), population_310)
        # Assigning a type to the variable 'individual' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'individual', for_loop_var_311)
        # SSA begins for a for statement (line 88)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'fitnessTot' (line 89)
        fitnessTot_312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'fitnessTot')
        # Getting the type of 'individual' (line 89)
        individual_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 26), 'individual')
        # Obtaining the member 'fitness' of a type (line 89)
        fitness_314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 26), individual_313, 'fitness')
        # Applying the binary operator '+=' (line 89)
        result_iadd_315 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 12), '+=', fitnessTot_312, fitness_314)
        # Assigning a type to the variable 'fitnessTot' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'fitnessTot', result_iadd_315)
        
        
        
        # Getting the type of 'individual' (line 90)
        individual_316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 15), 'individual')
        # Obtaining the member 'fitness' of a type (line 90)
        fitness_317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 15), individual_316, 'fitness')
        # Getting the type of 'bestIndividualGeneration' (line 90)
        bestIndividualGeneration_318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 36), 'bestIndividualGeneration')
        # Obtaining the member 'fitness' of a type (line 90)
        fitness_319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 36), bestIndividualGeneration_318, 'fitness')
        # Applying the binary operator '>' (line 90)
        result_gt_320 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 15), '>', fitness_317, fitness_319)
        
        # Testing the type of an if condition (line 90)
        if_condition_321 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 12), result_gt_320)
        # Assigning a type to the variable 'if_condition_321' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'if_condition_321', if_condition_321)
        # SSA begins for if statement (line 90)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 91):
        
        # Assigning a Name to a Name (line 91):
        # Getting the type of 'individual' (line 91)
        individual_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 43), 'individual')
        # Assigning a type to the variable 'bestIndividualGeneration' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'bestIndividualGeneration', individual_322)
        # SSA join for if statement (line 90)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 92)
        self_323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'self')
        # Obtaining the member 'bestIndividual' of a type (line 92)
        bestIndividual_324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 11), self_323, 'bestIndividual')
        # Obtaining the member 'fitness' of a type (line 92)
        fitness_325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 11), bestIndividual_324, 'fitness')
        # Getting the type of 'bestIndividualGeneration' (line 92)
        bestIndividualGeneration_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 41), 'bestIndividualGeneration')
        # Obtaining the member 'fitness' of a type (line 92)
        fitness_327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 41), bestIndividualGeneration_326, 'fitness')
        # Applying the binary operator '<' (line 92)
        result_lt_328 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 11), '<', fitness_325, fitness_327)
        
        # Testing the type of an if condition (line 92)
        if_condition_329 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 8), result_lt_328)
        # Assigning a type to the variable 'if_condition_329' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'if_condition_329', if_condition_329)
        # SSA begins for if statement (line 92)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 93):
        
        # Assigning a Call to a Attribute (line 93):
        
        # Call to copy(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'bestIndividualGeneration' (line 93)
        bestIndividualGeneration_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 39), 'bestIndividualGeneration', False)
        # Processing the call keyword arguments (line 93)
        kwargs_332 = {}
        # Getting the type of 'copy' (line 93)
        copy_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 34), 'copy', False)
        # Calling copy(args, kwargs) (line 93)
        copy_call_result_333 = invoke(stypy.reporting.localization.Localization(__file__, 93, 34), copy_330, *[bestIndividualGeneration_331], **kwargs_332)
        
        # Getting the type of 'self' (line 93)
        self_334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'self')
        # Setting the type of the member 'bestIndividual' of a type (line 93)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 12), self_334, 'bestIndividual', copy_call_result_333)
        # SSA join for if statement (line 92)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'showGeneration_bestIndFind(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'showGeneration_bestIndFind' in the type store
        # Getting the type of 'stypy_return_type' (line 85)
        stypy_return_type_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_335)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'showGeneration_bestIndFind'
        return stypy_return_type_335


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 96, 4, False)
        # Assigning a type to the variable 'self' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SGA.run.__dict__.__setitem__('stypy_localization', localization)
        SGA.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SGA.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        SGA.run.__dict__.__setitem__('stypy_function_name', 'SGA.run')
        SGA.run.__dict__.__setitem__('stypy_param_names_list', [])
        SGA.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        SGA.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SGA.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        SGA.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        SGA.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SGA.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SGA.run', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to generateRandomPop(...): (line 97)
        # Processing the call keyword arguments (line 97)
        kwargs_338 = {}
        # Getting the type of 'self' (line 97)
        self_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'self', False)
        # Obtaining the member 'generateRandomPop' of a type (line 97)
        generateRandomPop_337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), self_336, 'generateRandomPop')
        # Calling generateRandomPop(args, kwargs) (line 97)
        generateRandomPop_call_result_339 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), generateRandomPop_337, *[], **kwargs_338)
        
        
        # Assigning a Call to a Attribute (line 98):
        
        # Assigning a Call to a Attribute (line 98):
        
        # Call to Individual(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'self' (line 98)
        self_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 41), 'self', False)
        # Obtaining the member 'genomeSize' of a type (line 98)
        genomeSize_342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 41), self_341, 'genomeSize')
        # Processing the call keyword arguments (line 98)
        kwargs_343 = {}
        # Getting the type of 'Individual' (line 98)
        Individual_340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 30), 'Individual', False)
        # Calling Individual(args, kwargs) (line 98)
        Individual_call_result_344 = invoke(stypy.reporting.localization.Localization(__file__, 98, 30), Individual_340, *[genomeSize_342], **kwargs_343)
        
        # Getting the type of 'self' (line 98)
        self_345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'self')
        # Setting the type of the member 'bestIndividual' of a type (line 98)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), self_345, 'bestIndividual', Individual_call_result_344)
        
        
        # Call to xrange(...): (line 99)
        # Processing the call arguments (line 99)
        int_347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 38), 'int')
        # Getting the type of 'self' (line 99)
        self_348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 41), 'self', False)
        # Obtaining the member 'generationsMax' of a type (line 99)
        generationsMax_349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 41), self_348, 'generationsMax')
        int_350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 61), 'int')
        # Applying the binary operator '+' (line 99)
        result_add_351 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 41), '+', generationsMax_349, int_350)
        
        # Processing the call keyword arguments (line 99)
        kwargs_352 = {}
        # Getting the type of 'xrange' (line 99)
        xrange_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 31), 'xrange', False)
        # Calling xrange(args, kwargs) (line 99)
        xrange_call_result_353 = invoke(stypy.reporting.localization.Localization(__file__, 99, 31), xrange_346, *[int_347, result_add_351], **kwargs_352)
        
        # Testing the type of a for loop iterable (line 99)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 99, 8), xrange_call_result_353)
        # Getting the type of the for loop variable (line 99)
        for_loop_var_354 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 99, 8), xrange_call_result_353)
        # Getting the type of 'self' (line 99)
        self_355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'self')
        # Setting the type of the member 'generation' of a type (line 99)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 8), self_355, 'generation', for_loop_var_354)
        # SSA begins for a for statement (line 99)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'self' (line 100)
        self_356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 15), 'self')
        # Obtaining the member 'generation' of a type (line 100)
        generation_357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 15), self_356, 'generation')
        int_358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 33), 'int')
        # Applying the binary operator '%' (line 100)
        result_mod_359 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 15), '%', generation_357, int_358)
        
        int_360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 40), 'int')
        # Applying the binary operator '==' (line 100)
        result_eq_361 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 15), '==', result_mod_359, int_360)
        
        # Testing the type of an if condition (line 100)
        if_condition_362 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 12), result_eq_361)
        # Assigning a type to the variable 'if_condition_362' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'if_condition_362', if_condition_362)
        # SSA begins for if statement (line 100)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA join for if statement (line 100)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to computeFitnessPop(...): (line 102)
        # Processing the call keyword arguments (line 102)
        kwargs_365 = {}
        # Getting the type of 'self' (line 102)
        self_363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'self', False)
        # Obtaining the member 'computeFitnessPop' of a type (line 102)
        computeFitnessPop_364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), self_363, 'computeFitnessPop')
        # Calling computeFitnessPop(args, kwargs) (line 102)
        computeFitnessPop_call_result_366 = invoke(stypy.reporting.localization.Localization(__file__, 102, 12), computeFitnessPop_364, *[], **kwargs_365)
        
        
        # Call to showGeneration_bestIndFind(...): (line 103)
        # Processing the call keyword arguments (line 103)
        kwargs_369 = {}
        # Getting the type of 'self' (line 103)
        self_367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'self', False)
        # Obtaining the member 'showGeneration_bestIndFind' of a type (line 103)
        showGeneration_bestIndFind_368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), self_367, 'showGeneration_bestIndFind')
        # Calling showGeneration_bestIndFind(args, kwargs) (line 103)
        showGeneration_bestIndFind_call_result_370 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), showGeneration_bestIndFind_368, *[], **kwargs_369)
        
        
        # Assigning a Call to a Attribute (line 104):
        
        # Assigning a Call to a Attribute (line 104):
        
        # Call to tounamentSelectionPop(...): (line 104)
        # Processing the call keyword arguments (line 104)
        kwargs_373 = {}
        # Getting the type of 'self' (line 104)
        self_371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 30), 'self', False)
        # Obtaining the member 'tounamentSelectionPop' of a type (line 104)
        tounamentSelectionPop_372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 30), self_371, 'tounamentSelectionPop')
        # Calling tounamentSelectionPop(args, kwargs) (line 104)
        tounamentSelectionPop_call_result_374 = invoke(stypy.reporting.localization.Localization(__file__, 104, 30), tounamentSelectionPop_372, *[], **kwargs_373)
        
        # Getting the type of 'self' (line 104)
        self_375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'self')
        # Setting the type of the member 'population' of a type (line 104)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 12), self_375, 'population', tounamentSelectionPop_call_result_374)
        
        # Call to mutatePop(...): (line 105)
        # Processing the call keyword arguments (line 105)
        kwargs_378 = {}
        # Getting the type of 'self' (line 105)
        self_376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'self', False)
        # Obtaining the member 'mutatePop' of a type (line 105)
        mutatePop_377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 12), self_376, 'mutatePop')
        # Calling mutatePop(args, kwargs) (line 105)
        mutatePop_call_result_379 = invoke(stypy.reporting.localization.Localization(__file__, 105, 12), mutatePop_377, *[], **kwargs_378)
        
        
        # Call to crossingOverPop(...): (line 106)
        # Processing the call keyword arguments (line 106)
        kwargs_382 = {}
        # Getting the type of 'self' (line 106)
        self_380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'self', False)
        # Obtaining the member 'crossingOverPop' of a type (line 106)
        crossingOverPop_381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 12), self_380, 'crossingOverPop')
        # Calling crossingOverPop(args, kwargs) (line 106)
        crossingOverPop_call_result_383 = invoke(stypy.reporting.localization.Localization(__file__, 106, 12), crossingOverPop_381, *[], **kwargs_382)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 96)
        stypy_return_type_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_384)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_384


# Assigning a type to the variable 'SGA' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'SGA', SGA)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 108, 0, False)
    
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

    
    # Assigning a Call to a Name (line 109):
    
    # Assigning a Call to a Name (line 109):
    
    # Call to SGA(...): (line 109)
    # Processing the call keyword arguments (line 109)
    kwargs_386 = {}
    # Getting the type of 'SGA' (line 109)
    SGA_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 10), 'SGA', False)
    # Calling SGA(args, kwargs) (line 109)
    SGA_call_result_387 = invoke(stypy.reporting.localization.Localization(__file__, 109, 10), SGA_385, *[], **kwargs_386)
    
    # Assigning a type to the variable 'sga' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'sga', SGA_call_result_387)
    
    # Assigning a Num to a Attribute (line 110):
    
    # Assigning a Num to a Attribute (line 110):
    int_388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 25), 'int')
    # Getting the type of 'sga' (line 110)
    sga_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'sga')
    # Setting the type of the member 'generationsMax' of a type (line 110)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 4), sga_389, 'generationsMax', int_388)
    
    # Assigning a Num to a Attribute (line 111):
    
    # Assigning a Num to a Attribute (line 111):
    int_390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 21), 'int')
    # Getting the type of 'sga' (line 111)
    sga_391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'sga')
    # Setting the type of the member 'genomeSize' of a type (line 111)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 4), sga_391, 'genomeSize', int_390)
    
    # Assigning a Num to a Attribute (line 112):
    
    # Assigning a Num to a Attribute (line 112):
    int_392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 18), 'int')
    # Getting the type of 'sga' (line 112)
    sga_393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'sga')
    # Setting the type of the member 'popSize' of a type (line 112)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 4), sga_393, 'popSize', int_392)
    
    # Assigning a Num to a Attribute (line 113):
    
    # Assigning a Num to a Attribute (line 113):
    float_394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 27), 'float')
    # Getting the type of 'sga' (line 113)
    sga_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'sga')
    # Setting the type of the member 'geneMutationProb' of a type (line 113)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 4), sga_395, 'geneMutationProb', float_394)
    
    # Call to run(...): (line 114)
    # Processing the call keyword arguments (line 114)
    kwargs_398 = {}
    # Getting the type of 'sga' (line 114)
    sga_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'sga', False)
    # Obtaining the member 'run' of a type (line 114)
    run_397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 4), sga_396, 'run')
    # Calling run(args, kwargs) (line 114)
    run_call_result_399 = invoke(stypy.reporting.localization.Localization(__file__, 114, 4), run_397, *[], **kwargs_398)
    
    # Getting the type of 'True' (line 115)
    True_400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type', True_400)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 108)
    stypy_return_type_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_401)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_401

# Assigning a type to the variable 'run' (line 108)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'run', run)

# Call to run(...): (line 117)
# Processing the call keyword arguments (line 117)
kwargs_403 = {}
# Getting the type of 'run' (line 117)
run_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'run', False)
# Calling run(args, kwargs) (line 117)
run_call_result_404 = invoke(stypy.reporting.localization.Localization(__file__, 117, 0), run_402, *[], **kwargs_403)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

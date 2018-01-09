
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: 
3: # Adatron SVM with polynomial kernel
4: # placed in the public domain by Stavros Korokithakis
5: 
6: import sys
7: from math import exp
8: import os
9: 
10: def Relative(path):
11:     return os.path.join(os.path.dirname(__file__), path)
12: 
13: CYTOSOLIC = 0
14: EXTRACELLULAR = 1
15: NUCLEAR = 2
16: MITOCHONDRIAL = 3
17: BLIND = 4
18: 
19: D = 5.0
20: 
21: LENGTH = 50
22: 
23: PROTEINS = []
24: 
25: AMINOACIDS = "ACDEFGHIKLMNPQRSTVWY"
26: 
27: class Protein:
28:     def __init__(self, name, mass, isoelectric_point, size, sequence, type):
29:         self.name = name
30:         self.mass = mass
31:         self.isoelectric_point = isoelectric_point
32:         self.size = size
33:         self.sequence = sequence
34:         self.type = type
35:         self.extract_composition()
36: 
37:     def extract_composition(self):
38:         self.local_composition = dict(((x, 0.0) for x in AMINOACIDS))
39:         for counter in range(LENGTH):
40:             self.local_composition[self.sequence[counter]] += 1.0 / LENGTH
41:         self.global_composition = dict(((x, 0.0) for x in AMINOACIDS))
42:         for aminoacid in self.sequence:
43:             self.global_composition[aminoacid] += 1.0 / len(self.sequence)
44: 
45:     def create_vector(self):
46:         vector = []
47:         for key, value in sorted(self.local_composition.items()):
48:             vector.append(value)
49:         for key in sorted(self.global_composition.keys()):
50:             vector.append(value)
51:         return vector
52: 
53: 
54: def load_file(filename, type):
55:     global PROTEINS
56:     protfile = open(filename)
57:     for line in protfile:
58:         if line.startswith("name"):
59:             continue
60:         name, mass, isoelectric_point, size, sequence = line.strip().split("\t")
61:         protein = Protein(name, mass, isoelectric_point, size, sequence, type)
62:         PROTEINS.append(protein)
63:     protfile.close()
64: 
65: 
66: def create_tables():
67:     '''Create the feature and label tables.'''
68:     feature_table = []
69:     label_table = []
70: 
71:     for protein in PROTEINS:
72:         feature_table.append(protein.create_vector())
73: 
74:     for protein in PROTEINS:
75:         if protein.type == BLIND:
76:             continue
77:         labels = [-1] * 4
78:         # Invert the sign of the label our protein belongs to.
79:         labels[protein.type] *= -1
80:         label_table.append(labels)
81: 
82:     return feature_table, label_table
83: 
84: 
85: def create_kernel_table(feature_table):
86:     kernel_table = []
87:     for row in feature_table:
88:         kernel_row = []
89:         for candidate in feature_table:
90:             difference = 0.0
91:             for counter in range(len(row)):
92:                 difference += (row[counter] - candidate[counter]) ** 2
93:             kernel_row.append(exp(-D*difference))
94:         kernel_table.append(kernel_row)
95:     return kernel_table
96: 
97: 
98: def train_adatron(kernel_table, label_table, h, c):
99:     tolerance = 0.5
100:     alphas = [([0.0] * len(kernel_table)) for _ in range(len(label_table[0]))]
101:     betas = [([0.0] * len(kernel_table)) for _ in range(len(label_table[0]))]
102:     bias = [0.0] * len(label_table[0])
103:     labelalphas = [0.0] * len(kernel_table)
104:     max_differences = [(0.0, 0)] * len(label_table[0])
105:     for iteration in range(10*len(kernel_table)):
106:         #print "Starting iteration %s..." % iteration
107:         if iteration == 20: # XXX shedskin test
108:             return alphas, bias
109:         for klass in range(len(label_table[0])):
110:             max_differences[klass] = (0.0, 0)
111:             for elem in range(len(kernel_table)):
112:                 labelalphas[elem] = label_table[elem][klass] * alphas[klass][elem]
113:             for col_counter in range(len(kernel_table)):
114:                 prediction = 0.0
115:                 for row_counter in range(len(kernel_table)):
116:                     prediction += kernel_table[col_counter][row_counter] * \
117:                                  labelalphas[row_counter]
118:                 g = 1.0 - ((prediction + bias[klass]) * label_table[col_counter][klass])
119:                 betas[klass][col_counter] = min(max((alphas[klass][col_counter] + h * g), 0.0), c)
120:                 difference = abs(alphas[klass][col_counter] - betas[klass][col_counter])
121:                 if difference > max_differences[klass][0]:
122:                     max_differences[klass] = (difference, col_counter)
123: 
124:             if all([max_difference[0] < tolerance for max_difference in max_differences]):
125:                 return alphas, bias
126:             else:
127:                 alphas[klass][max_differences[klass][1]] = betas[klass][max_differences[klass][1]]
128:                 element_sum = 0.0
129:                 for element_counter in range(len(kernel_table)):
130:                     element_sum += label_table[element_counter][klass] * alphas[klass][element_counter] / 4
131:                 bias[klass] = bias[klass] + element_sum
132: 
133: def calculate_error(alphas, bias, kernel_table, label_table):
134:     prediction = 0.0
135:     predictions = [([0.0] * len(kernel_table)) for _ in range(len(label_table[0]))]
136:     for klass in range(len(label_table[0])):
137:         for col_counter in range(len(kernel_table)):
138:             for row_counter in range(len(kernel_table)):
139:                 prediction += kernel_table[col_counter][row_counter] * \
140:                               label_table[row_counter][klass] * alphas[klass][row_counter]
141:             predictions[klass][col_counter] = prediction + bias[klass]
142: 
143:     for col_counter in range(len(kernel_table)):
144:         current_predictions = []
145:         error = 0
146:         for row_counter in range(len(label_table[0])):
147:             current_predictions.append(predictions[row_counter][col_counter])
148: 
149:         predicted_class = current_predictions.index(max(current_predictions))
150: 
151:         if label_table[col_counter][predicted_class] < 0:
152:             error += 1
153: 
154:         return 1.0 * error / len(kernel_table)
155: 
156: 
157: def main():
158:     for filename, type in [(Relative("testdata/c.txt"), CYTOSOLIC), (Relative("testdata/e.txt"), EXTRACELLULAR), (Relative("testdata/n.txt"), NUCLEAR), (Relative("testdata/m.txt"), MITOCHONDRIAL)]:#, ("b.txt", BLIND)]:
159:         load_file(filename, type)
160:     #print "Creating feature tables..."
161:     feature_table, label_table = create_tables()
162:     #import pickle
163:     #print "Loading kernel table..."
164:     #kernel_file = file("kernel_table.txt")
165:     #kernel_table = pickle.load(kernel_file)
166:     #kernel_file.close()
167:     #print "Creating kernel table..."
168:     kernel_table = create_kernel_table(feature_table)
169:     #print "Training SVM..."
170:     alphas, bias = train_adatron(kernel_table, label_table, 1.0, 3.0)
171:     #print calculate_error(alphas, bias, kernel_table, label_table)
172:     calculate_error(alphas, bias, kernel_table, label_table)
173: 
174: 
175: def run():
176:     main()
177:     return True
178: 
179: run()

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import sys' statement (line 6)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from math import exp' statement (line 7)
try:
    from math import exp

except:
    exp = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'math', None, module_type_store, ['exp'], [exp])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import os' statement (line 8)
import os

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'os', os, module_type_store)


@norecursion
def Relative(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Relative'
    module_type_store = module_type_store.open_function_context('Relative', 10, 0, False)
    
    # Passed parameters checking function
    Relative.stypy_localization = localization
    Relative.stypy_type_of_self = None
    Relative.stypy_type_store = module_type_store
    Relative.stypy_function_name = 'Relative'
    Relative.stypy_param_names_list = ['path']
    Relative.stypy_varargs_param_name = None
    Relative.stypy_kwargs_param_name = None
    Relative.stypy_call_defaults = defaults
    Relative.stypy_call_varargs = varargs
    Relative.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'Relative', ['path'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'Relative', localization, ['path'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'Relative(...)' code ##################

    
    # Call to join(...): (line 11)
    # Processing the call arguments (line 11)
    
    # Call to dirname(...): (line 11)
    # Processing the call arguments (line 11)
    # Getting the type of '__file__' (line 11)
    file___16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 40), '__file__', False)
    # Processing the call keyword arguments (line 11)
    kwargs_17 = {}
    # Getting the type of 'os' (line 11)
    os_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 11)
    path_14 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 24), os_13, 'path')
    # Obtaining the member 'dirname' of a type (line 11)
    dirname_15 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 24), path_14, 'dirname')
    # Calling dirname(args, kwargs) (line 11)
    dirname_call_result_18 = invoke(stypy.reporting.localization.Localization(__file__, 11, 24), dirname_15, *[file___16], **kwargs_17)
    
    # Getting the type of 'path' (line 11)
    path_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 51), 'path', False)
    # Processing the call keyword arguments (line 11)
    kwargs_20 = {}
    # Getting the type of 'os' (line 11)
    os_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 11)
    path_11 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 11), os_10, 'path')
    # Obtaining the member 'join' of a type (line 11)
    join_12 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 11), path_11, 'join')
    # Calling join(args, kwargs) (line 11)
    join_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 11, 11), join_12, *[dirname_call_result_18, path_19], **kwargs_20)
    
    # Assigning a type to the variable 'stypy_return_type' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'stypy_return_type', join_call_result_21)
    
    # ################# End of 'Relative(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Relative' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_22)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Relative'
    return stypy_return_type_22

# Assigning a type to the variable 'Relative' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'Relative', Relative)

# Assigning a Num to a Name (line 13):

# Assigning a Num to a Name (line 13):
int_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 12), 'int')
# Assigning a type to the variable 'CYTOSOLIC' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'CYTOSOLIC', int_23)

# Assigning a Num to a Name (line 14):

# Assigning a Num to a Name (line 14):
int_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 16), 'int')
# Assigning a type to the variable 'EXTRACELLULAR' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'EXTRACELLULAR', int_24)

# Assigning a Num to a Name (line 15):

# Assigning a Num to a Name (line 15):
int_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 10), 'int')
# Assigning a type to the variable 'NUCLEAR' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'NUCLEAR', int_25)

# Assigning a Num to a Name (line 16):

# Assigning a Num to a Name (line 16):
int_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 16), 'int')
# Assigning a type to the variable 'MITOCHONDRIAL' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'MITOCHONDRIAL', int_26)

# Assigning a Num to a Name (line 17):

# Assigning a Num to a Name (line 17):
int_27 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 8), 'int')
# Assigning a type to the variable 'BLIND' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'BLIND', int_27)

# Assigning a Num to a Name (line 19):

# Assigning a Num to a Name (line 19):
float_28 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 4), 'float')
# Assigning a type to the variable 'D' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'D', float_28)

# Assigning a Num to a Name (line 21):

# Assigning a Num to a Name (line 21):
int_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 9), 'int')
# Assigning a type to the variable 'LENGTH' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'LENGTH', int_29)

# Assigning a List to a Name (line 23):

# Assigning a List to a Name (line 23):

# Obtaining an instance of the builtin type 'list' (line 23)
list_30 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 11), 'list')
# Adding type elements to the builtin type 'list' instance (line 23)

# Assigning a type to the variable 'PROTEINS' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'PROTEINS', list_30)

# Assigning a Str to a Name (line 25):

# Assigning a Str to a Name (line 25):
str_31 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 13), 'str', 'ACDEFGHIKLMNPQRSTVWY')
# Assigning a type to the variable 'AMINOACIDS' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'AMINOACIDS', str_31)
# Declaration of the 'Protein' class

class Protein:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 28, 4, False)
        # Assigning a type to the variable 'self' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Protein.__init__', ['name', 'mass', 'isoelectric_point', 'size', 'sequence', 'type'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['name', 'mass', 'isoelectric_point', 'size', 'sequence', 'type'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 29):
        
        # Assigning a Name to a Attribute (line 29):
        # Getting the type of 'name' (line 29)
        name_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 20), 'name')
        # Getting the type of 'self' (line 29)
        self_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self')
        # Setting the type of the member 'name' of a type (line 29)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_33, 'name', name_32)
        
        # Assigning a Name to a Attribute (line 30):
        
        # Assigning a Name to a Attribute (line 30):
        # Getting the type of 'mass' (line 30)
        mass_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 20), 'mass')
        # Getting the type of 'self' (line 30)
        self_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self')
        # Setting the type of the member 'mass' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_35, 'mass', mass_34)
        
        # Assigning a Name to a Attribute (line 31):
        
        # Assigning a Name to a Attribute (line 31):
        # Getting the type of 'isoelectric_point' (line 31)
        isoelectric_point_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 33), 'isoelectric_point')
        # Getting the type of 'self' (line 31)
        self_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self')
        # Setting the type of the member 'isoelectric_point' of a type (line 31)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_37, 'isoelectric_point', isoelectric_point_36)
        
        # Assigning a Name to a Attribute (line 32):
        
        # Assigning a Name to a Attribute (line 32):
        # Getting the type of 'size' (line 32)
        size_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 20), 'size')
        # Getting the type of 'self' (line 32)
        self_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'self')
        # Setting the type of the member 'size' of a type (line 32)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), self_39, 'size', size_38)
        
        # Assigning a Name to a Attribute (line 33):
        
        # Assigning a Name to a Attribute (line 33):
        # Getting the type of 'sequence' (line 33)
        sequence_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 24), 'sequence')
        # Getting the type of 'self' (line 33)
        self_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self')
        # Setting the type of the member 'sequence' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_41, 'sequence', sequence_40)
        
        # Assigning a Name to a Attribute (line 34):
        
        # Assigning a Name to a Attribute (line 34):
        # Getting the type of 'type' (line 34)
        type_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 20), 'type')
        # Getting the type of 'self' (line 34)
        self_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self')
        # Setting the type of the member 'type' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_43, 'type', type_42)
        
        # Call to extract_composition(...): (line 35)
        # Processing the call keyword arguments (line 35)
        kwargs_46 = {}
        # Getting the type of 'self' (line 35)
        self_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'self', False)
        # Obtaining the member 'extract_composition' of a type (line 35)
        extract_composition_45 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), self_44, 'extract_composition')
        # Calling extract_composition(args, kwargs) (line 35)
        extract_composition_call_result_47 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), extract_composition_45, *[], **kwargs_46)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def extract_composition(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'extract_composition'
        module_type_store = module_type_store.open_function_context('extract_composition', 37, 4, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Protein.extract_composition.__dict__.__setitem__('stypy_localization', localization)
        Protein.extract_composition.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Protein.extract_composition.__dict__.__setitem__('stypy_type_store', module_type_store)
        Protein.extract_composition.__dict__.__setitem__('stypy_function_name', 'Protein.extract_composition')
        Protein.extract_composition.__dict__.__setitem__('stypy_param_names_list', [])
        Protein.extract_composition.__dict__.__setitem__('stypy_varargs_param_name', None)
        Protein.extract_composition.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Protein.extract_composition.__dict__.__setitem__('stypy_call_defaults', defaults)
        Protein.extract_composition.__dict__.__setitem__('stypy_call_varargs', varargs)
        Protein.extract_composition.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Protein.extract_composition.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Protein.extract_composition', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'extract_composition', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'extract_composition(...)' code ##################

        
        # Assigning a Call to a Attribute (line 38):
        
        # Assigning a Call to a Attribute (line 38):
        
        # Call to dict(...): (line 38)
        # Processing the call arguments (line 38)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 38, 39, True)
        # Calculating comprehension expression
        # Getting the type of 'AMINOACIDS' (line 38)
        AMINOACIDS_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 57), 'AMINOACIDS', False)
        comprehension_53 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 39), AMINOACIDS_52)
        # Assigning a type to the variable 'x' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 39), 'x', comprehension_53)
        
        # Obtaining an instance of the builtin type 'tuple' (line 38)
        tuple_49 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 38)
        # Adding element type (line 38)
        # Getting the type of 'x' (line 38)
        x_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 40), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 40), tuple_49, x_50)
        # Adding element type (line 38)
        float_51 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 40), tuple_49, float_51)
        
        list_54 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 39), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 39), list_54, tuple_49)
        # Processing the call keyword arguments (line 38)
        kwargs_55 = {}
        # Getting the type of 'dict' (line 38)
        dict_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 33), 'dict', False)
        # Calling dict(args, kwargs) (line 38)
        dict_call_result_56 = invoke(stypy.reporting.localization.Localization(__file__, 38, 33), dict_48, *[list_54], **kwargs_55)
        
        # Getting the type of 'self' (line 38)
        self_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self')
        # Setting the type of the member 'local_composition' of a type (line 38)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_57, 'local_composition', dict_call_result_56)
        
        
        # Call to range(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'LENGTH' (line 39)
        LENGTH_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 29), 'LENGTH', False)
        # Processing the call keyword arguments (line 39)
        kwargs_60 = {}
        # Getting the type of 'range' (line 39)
        range_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 23), 'range', False)
        # Calling range(args, kwargs) (line 39)
        range_call_result_61 = invoke(stypy.reporting.localization.Localization(__file__, 39, 23), range_58, *[LENGTH_59], **kwargs_60)
        
        # Testing the type of a for loop iterable (line 39)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 39, 8), range_call_result_61)
        # Getting the type of the for loop variable (line 39)
        for_loop_var_62 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 39, 8), range_call_result_61)
        # Assigning a type to the variable 'counter' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'counter', for_loop_var_62)
        # SSA begins for a for statement (line 39)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'self' (line 40)
        self_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'self')
        # Obtaining the member 'local_composition' of a type (line 40)
        local_composition_64 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 12), self_63, 'local_composition')
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        # Getting the type of 'counter' (line 40)
        counter_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 49), 'counter')
        # Getting the type of 'self' (line 40)
        self_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 35), 'self')
        # Obtaining the member 'sequence' of a type (line 40)
        sequence_67 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 35), self_66, 'sequence')
        # Obtaining the member '__getitem__' of a type (line 40)
        getitem___68 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 35), sequence_67, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 40)
        subscript_call_result_69 = invoke(stypy.reporting.localization.Localization(__file__, 40, 35), getitem___68, counter_65)
        
        # Getting the type of 'self' (line 40)
        self_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'self')
        # Obtaining the member 'local_composition' of a type (line 40)
        local_composition_71 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 12), self_70, 'local_composition')
        # Obtaining the member '__getitem__' of a type (line 40)
        getitem___72 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 12), local_composition_71, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 40)
        subscript_call_result_73 = invoke(stypy.reporting.localization.Localization(__file__, 40, 12), getitem___72, subscript_call_result_69)
        
        float_74 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 62), 'float')
        # Getting the type of 'LENGTH' (line 40)
        LENGTH_75 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 68), 'LENGTH')
        # Applying the binary operator 'div' (line 40)
        result_div_76 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 62), 'div', float_74, LENGTH_75)
        
        # Applying the binary operator '+=' (line 40)
        result_iadd_77 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 12), '+=', subscript_call_result_73, result_div_76)
        # Getting the type of 'self' (line 40)
        self_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'self')
        # Obtaining the member 'local_composition' of a type (line 40)
        local_composition_79 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 12), self_78, 'local_composition')
        
        # Obtaining the type of the subscript
        # Getting the type of 'counter' (line 40)
        counter_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 49), 'counter')
        # Getting the type of 'self' (line 40)
        self_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 35), 'self')
        # Obtaining the member 'sequence' of a type (line 40)
        sequence_82 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 35), self_81, 'sequence')
        # Obtaining the member '__getitem__' of a type (line 40)
        getitem___83 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 35), sequence_82, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 40)
        subscript_call_result_84 = invoke(stypy.reporting.localization.Localization(__file__, 40, 35), getitem___83, counter_80)
        
        # Storing an element on a container (line 40)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 12), local_composition_79, (subscript_call_result_84, result_iadd_77))
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 41):
        
        # Assigning a Call to a Attribute (line 41):
        
        # Call to dict(...): (line 41)
        # Processing the call arguments (line 41)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 41, 40, True)
        # Calculating comprehension expression
        # Getting the type of 'AMINOACIDS' (line 41)
        AMINOACIDS_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 58), 'AMINOACIDS', False)
        comprehension_90 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 40), AMINOACIDS_89)
        # Assigning a type to the variable 'x' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 40), 'x', comprehension_90)
        
        # Obtaining an instance of the builtin type 'tuple' (line 41)
        tuple_86 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 41)
        # Adding element type (line 41)
        # Getting the type of 'x' (line 41)
        x_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 41), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 41), tuple_86, x_87)
        # Adding element type (line 41)
        float_88 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 41), tuple_86, float_88)
        
        list_91 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 40), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 40), list_91, tuple_86)
        # Processing the call keyword arguments (line 41)
        kwargs_92 = {}
        # Getting the type of 'dict' (line 41)
        dict_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 34), 'dict', False)
        # Calling dict(args, kwargs) (line 41)
        dict_call_result_93 = invoke(stypy.reporting.localization.Localization(__file__, 41, 34), dict_85, *[list_91], **kwargs_92)
        
        # Getting the type of 'self' (line 41)
        self_94 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'self')
        # Setting the type of the member 'global_composition' of a type (line 41)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), self_94, 'global_composition', dict_call_result_93)
        
        # Getting the type of 'self' (line 42)
        self_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 25), 'self')
        # Obtaining the member 'sequence' of a type (line 42)
        sequence_96 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 25), self_95, 'sequence')
        # Testing the type of a for loop iterable (line 42)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 42, 8), sequence_96)
        # Getting the type of the for loop variable (line 42)
        for_loop_var_97 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 42, 8), sequence_96)
        # Assigning a type to the variable 'aminoacid' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'aminoacid', for_loop_var_97)
        # SSA begins for a for statement (line 42)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'self' (line 43)
        self_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'self')
        # Obtaining the member 'global_composition' of a type (line 43)
        global_composition_99 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 12), self_98, 'global_composition')
        
        # Obtaining the type of the subscript
        # Getting the type of 'aminoacid' (line 43)
        aminoacid_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 36), 'aminoacid')
        # Getting the type of 'self' (line 43)
        self_101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'self')
        # Obtaining the member 'global_composition' of a type (line 43)
        global_composition_102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 12), self_101, 'global_composition')
        # Obtaining the member '__getitem__' of a type (line 43)
        getitem___103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 12), global_composition_102, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 43)
        subscript_call_result_104 = invoke(stypy.reporting.localization.Localization(__file__, 43, 12), getitem___103, aminoacid_100)
        
        float_105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 50), 'float')
        
        # Call to len(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'self' (line 43)
        self_107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 60), 'self', False)
        # Obtaining the member 'sequence' of a type (line 43)
        sequence_108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 60), self_107, 'sequence')
        # Processing the call keyword arguments (line 43)
        kwargs_109 = {}
        # Getting the type of 'len' (line 43)
        len_106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 56), 'len', False)
        # Calling len(args, kwargs) (line 43)
        len_call_result_110 = invoke(stypy.reporting.localization.Localization(__file__, 43, 56), len_106, *[sequence_108], **kwargs_109)
        
        # Applying the binary operator 'div' (line 43)
        result_div_111 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 50), 'div', float_105, len_call_result_110)
        
        # Applying the binary operator '+=' (line 43)
        result_iadd_112 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 12), '+=', subscript_call_result_104, result_div_111)
        # Getting the type of 'self' (line 43)
        self_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'self')
        # Obtaining the member 'global_composition' of a type (line 43)
        global_composition_114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 12), self_113, 'global_composition')
        # Getting the type of 'aminoacid' (line 43)
        aminoacid_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 36), 'aminoacid')
        # Storing an element on a container (line 43)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 12), global_composition_114, (aminoacid_115, result_iadd_112))
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'extract_composition(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'extract_composition' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_116)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'extract_composition'
        return stypy_return_type_116


    @norecursion
    def create_vector(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_vector'
        module_type_store = module_type_store.open_function_context('create_vector', 45, 4, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Protein.create_vector.__dict__.__setitem__('stypy_localization', localization)
        Protein.create_vector.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Protein.create_vector.__dict__.__setitem__('stypy_type_store', module_type_store)
        Protein.create_vector.__dict__.__setitem__('stypy_function_name', 'Protein.create_vector')
        Protein.create_vector.__dict__.__setitem__('stypy_param_names_list', [])
        Protein.create_vector.__dict__.__setitem__('stypy_varargs_param_name', None)
        Protein.create_vector.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Protein.create_vector.__dict__.__setitem__('stypy_call_defaults', defaults)
        Protein.create_vector.__dict__.__setitem__('stypy_call_varargs', varargs)
        Protein.create_vector.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Protein.create_vector.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Protein.create_vector', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_vector', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_vector(...)' code ##################

        
        # Assigning a List to a Name (line 46):
        
        # Assigning a List to a Name (line 46):
        
        # Obtaining an instance of the builtin type 'list' (line 46)
        list_117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 46)
        
        # Assigning a type to the variable 'vector' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'vector', list_117)
        
        
        # Call to sorted(...): (line 47)
        # Processing the call arguments (line 47)
        
        # Call to items(...): (line 47)
        # Processing the call keyword arguments (line 47)
        kwargs_122 = {}
        # Getting the type of 'self' (line 47)
        self_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 33), 'self', False)
        # Obtaining the member 'local_composition' of a type (line 47)
        local_composition_120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 33), self_119, 'local_composition')
        # Obtaining the member 'items' of a type (line 47)
        items_121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 33), local_composition_120, 'items')
        # Calling items(args, kwargs) (line 47)
        items_call_result_123 = invoke(stypy.reporting.localization.Localization(__file__, 47, 33), items_121, *[], **kwargs_122)
        
        # Processing the call keyword arguments (line 47)
        kwargs_124 = {}
        # Getting the type of 'sorted' (line 47)
        sorted_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 26), 'sorted', False)
        # Calling sorted(args, kwargs) (line 47)
        sorted_call_result_125 = invoke(stypy.reporting.localization.Localization(__file__, 47, 26), sorted_118, *[items_call_result_123], **kwargs_124)
        
        # Testing the type of a for loop iterable (line 47)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 47, 8), sorted_call_result_125)
        # Getting the type of the for loop variable (line 47)
        for_loop_var_126 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 47, 8), sorted_call_result_125)
        # Assigning a type to the variable 'key' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 8), for_loop_var_126))
        # Assigning a type to the variable 'value' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 8), for_loop_var_126))
        # SSA begins for a for statement (line 47)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'value' (line 48)
        value_129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 26), 'value', False)
        # Processing the call keyword arguments (line 48)
        kwargs_130 = {}
        # Getting the type of 'vector' (line 48)
        vector_127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'vector', False)
        # Obtaining the member 'append' of a type (line 48)
        append_128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 12), vector_127, 'append')
        # Calling append(args, kwargs) (line 48)
        append_call_result_131 = invoke(stypy.reporting.localization.Localization(__file__, 48, 12), append_128, *[value_129], **kwargs_130)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to sorted(...): (line 49)
        # Processing the call arguments (line 49)
        
        # Call to keys(...): (line 49)
        # Processing the call keyword arguments (line 49)
        kwargs_136 = {}
        # Getting the type of 'self' (line 49)
        self_133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 26), 'self', False)
        # Obtaining the member 'global_composition' of a type (line 49)
        global_composition_134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 26), self_133, 'global_composition')
        # Obtaining the member 'keys' of a type (line 49)
        keys_135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 26), global_composition_134, 'keys')
        # Calling keys(args, kwargs) (line 49)
        keys_call_result_137 = invoke(stypy.reporting.localization.Localization(__file__, 49, 26), keys_135, *[], **kwargs_136)
        
        # Processing the call keyword arguments (line 49)
        kwargs_138 = {}
        # Getting the type of 'sorted' (line 49)
        sorted_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 'sorted', False)
        # Calling sorted(args, kwargs) (line 49)
        sorted_call_result_139 = invoke(stypy.reporting.localization.Localization(__file__, 49, 19), sorted_132, *[keys_call_result_137], **kwargs_138)
        
        # Testing the type of a for loop iterable (line 49)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 49, 8), sorted_call_result_139)
        # Getting the type of the for loop variable (line 49)
        for_loop_var_140 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 49, 8), sorted_call_result_139)
        # Assigning a type to the variable 'key' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'key', for_loop_var_140)
        # SSA begins for a for statement (line 49)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'value' (line 50)
        value_143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 26), 'value', False)
        # Processing the call keyword arguments (line 50)
        kwargs_144 = {}
        # Getting the type of 'vector' (line 50)
        vector_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'vector', False)
        # Obtaining the member 'append' of a type (line 50)
        append_142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), vector_141, 'append')
        # Calling append(args, kwargs) (line 50)
        append_call_result_145 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), append_142, *[value_143], **kwargs_144)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'vector' (line 51)
        vector_146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'vector')
        # Assigning a type to the variable 'stypy_return_type' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', vector_146)
        
        # ################# End of 'create_vector(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_vector' in the type store
        # Getting the type of 'stypy_return_type' (line 45)
        stypy_return_type_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_147)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_vector'
        return stypy_return_type_147


# Assigning a type to the variable 'Protein' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'Protein', Protein)

@norecursion
def load_file(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'load_file'
    module_type_store = module_type_store.open_function_context('load_file', 54, 0, False)
    
    # Passed parameters checking function
    load_file.stypy_localization = localization
    load_file.stypy_type_of_self = None
    load_file.stypy_type_store = module_type_store
    load_file.stypy_function_name = 'load_file'
    load_file.stypy_param_names_list = ['filename', 'type']
    load_file.stypy_varargs_param_name = None
    load_file.stypy_kwargs_param_name = None
    load_file.stypy_call_defaults = defaults
    load_file.stypy_call_varargs = varargs
    load_file.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'load_file', ['filename', 'type'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'load_file', localization, ['filename', 'type'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'load_file(...)' code ##################

    # Marking variables as global (line 55)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 55, 4), 'PROTEINS')
    
    # Assigning a Call to a Name (line 56):
    
    # Assigning a Call to a Name (line 56):
    
    # Call to open(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'filename' (line 56)
    filename_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 20), 'filename', False)
    # Processing the call keyword arguments (line 56)
    kwargs_150 = {}
    # Getting the type of 'open' (line 56)
    open_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 15), 'open', False)
    # Calling open(args, kwargs) (line 56)
    open_call_result_151 = invoke(stypy.reporting.localization.Localization(__file__, 56, 15), open_148, *[filename_149], **kwargs_150)
    
    # Assigning a type to the variable 'protfile' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'protfile', open_call_result_151)
    
    # Getting the type of 'protfile' (line 57)
    protfile_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'protfile')
    # Testing the type of a for loop iterable (line 57)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 57, 4), protfile_152)
    # Getting the type of the for loop variable (line 57)
    for_loop_var_153 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 57, 4), protfile_152)
    # Assigning a type to the variable 'line' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'line', for_loop_var_153)
    # SSA begins for a for statement (line 57)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to startswith(...): (line 58)
    # Processing the call arguments (line 58)
    str_156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 27), 'str', 'name')
    # Processing the call keyword arguments (line 58)
    kwargs_157 = {}
    # Getting the type of 'line' (line 58)
    line_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 11), 'line', False)
    # Obtaining the member 'startswith' of a type (line 58)
    startswith_155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 11), line_154, 'startswith')
    # Calling startswith(args, kwargs) (line 58)
    startswith_call_result_158 = invoke(stypy.reporting.localization.Localization(__file__, 58, 11), startswith_155, *[str_156], **kwargs_157)
    
    # Testing the type of an if condition (line 58)
    if_condition_159 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 8), startswith_call_result_158)
    # Assigning a type to the variable 'if_condition_159' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'if_condition_159', if_condition_159)
    # SSA begins for if statement (line 58)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 58)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 60):
    
    # Assigning a Subscript to a Name (line 60):
    
    # Obtaining the type of the subscript
    int_160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 8), 'int')
    
    # Call to split(...): (line 60)
    # Processing the call arguments (line 60)
    str_166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 75), 'str', '\t')
    # Processing the call keyword arguments (line 60)
    kwargs_167 = {}
    
    # Call to strip(...): (line 60)
    # Processing the call keyword arguments (line 60)
    kwargs_163 = {}
    # Getting the type of 'line' (line 60)
    line_161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 56), 'line', False)
    # Obtaining the member 'strip' of a type (line 60)
    strip_162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 56), line_161, 'strip')
    # Calling strip(args, kwargs) (line 60)
    strip_call_result_164 = invoke(stypy.reporting.localization.Localization(__file__, 60, 56), strip_162, *[], **kwargs_163)
    
    # Obtaining the member 'split' of a type (line 60)
    split_165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 56), strip_call_result_164, 'split')
    # Calling split(args, kwargs) (line 60)
    split_call_result_168 = invoke(stypy.reporting.localization.Localization(__file__, 60, 56), split_165, *[str_166], **kwargs_167)
    
    # Obtaining the member '__getitem__' of a type (line 60)
    getitem___169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), split_call_result_168, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 60)
    subscript_call_result_170 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), getitem___169, int_160)
    
    # Assigning a type to the variable 'tuple_var_assignment_1' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'tuple_var_assignment_1', subscript_call_result_170)
    
    # Assigning a Subscript to a Name (line 60):
    
    # Obtaining the type of the subscript
    int_171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 8), 'int')
    
    # Call to split(...): (line 60)
    # Processing the call arguments (line 60)
    str_177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 75), 'str', '\t')
    # Processing the call keyword arguments (line 60)
    kwargs_178 = {}
    
    # Call to strip(...): (line 60)
    # Processing the call keyword arguments (line 60)
    kwargs_174 = {}
    # Getting the type of 'line' (line 60)
    line_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 56), 'line', False)
    # Obtaining the member 'strip' of a type (line 60)
    strip_173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 56), line_172, 'strip')
    # Calling strip(args, kwargs) (line 60)
    strip_call_result_175 = invoke(stypy.reporting.localization.Localization(__file__, 60, 56), strip_173, *[], **kwargs_174)
    
    # Obtaining the member 'split' of a type (line 60)
    split_176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 56), strip_call_result_175, 'split')
    # Calling split(args, kwargs) (line 60)
    split_call_result_179 = invoke(stypy.reporting.localization.Localization(__file__, 60, 56), split_176, *[str_177], **kwargs_178)
    
    # Obtaining the member '__getitem__' of a type (line 60)
    getitem___180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), split_call_result_179, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 60)
    subscript_call_result_181 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), getitem___180, int_171)
    
    # Assigning a type to the variable 'tuple_var_assignment_2' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'tuple_var_assignment_2', subscript_call_result_181)
    
    # Assigning a Subscript to a Name (line 60):
    
    # Obtaining the type of the subscript
    int_182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 8), 'int')
    
    # Call to split(...): (line 60)
    # Processing the call arguments (line 60)
    str_188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 75), 'str', '\t')
    # Processing the call keyword arguments (line 60)
    kwargs_189 = {}
    
    # Call to strip(...): (line 60)
    # Processing the call keyword arguments (line 60)
    kwargs_185 = {}
    # Getting the type of 'line' (line 60)
    line_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 56), 'line', False)
    # Obtaining the member 'strip' of a type (line 60)
    strip_184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 56), line_183, 'strip')
    # Calling strip(args, kwargs) (line 60)
    strip_call_result_186 = invoke(stypy.reporting.localization.Localization(__file__, 60, 56), strip_184, *[], **kwargs_185)
    
    # Obtaining the member 'split' of a type (line 60)
    split_187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 56), strip_call_result_186, 'split')
    # Calling split(args, kwargs) (line 60)
    split_call_result_190 = invoke(stypy.reporting.localization.Localization(__file__, 60, 56), split_187, *[str_188], **kwargs_189)
    
    # Obtaining the member '__getitem__' of a type (line 60)
    getitem___191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), split_call_result_190, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 60)
    subscript_call_result_192 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), getitem___191, int_182)
    
    # Assigning a type to the variable 'tuple_var_assignment_3' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'tuple_var_assignment_3', subscript_call_result_192)
    
    # Assigning a Subscript to a Name (line 60):
    
    # Obtaining the type of the subscript
    int_193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 8), 'int')
    
    # Call to split(...): (line 60)
    # Processing the call arguments (line 60)
    str_199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 75), 'str', '\t')
    # Processing the call keyword arguments (line 60)
    kwargs_200 = {}
    
    # Call to strip(...): (line 60)
    # Processing the call keyword arguments (line 60)
    kwargs_196 = {}
    # Getting the type of 'line' (line 60)
    line_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 56), 'line', False)
    # Obtaining the member 'strip' of a type (line 60)
    strip_195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 56), line_194, 'strip')
    # Calling strip(args, kwargs) (line 60)
    strip_call_result_197 = invoke(stypy.reporting.localization.Localization(__file__, 60, 56), strip_195, *[], **kwargs_196)
    
    # Obtaining the member 'split' of a type (line 60)
    split_198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 56), strip_call_result_197, 'split')
    # Calling split(args, kwargs) (line 60)
    split_call_result_201 = invoke(stypy.reporting.localization.Localization(__file__, 60, 56), split_198, *[str_199], **kwargs_200)
    
    # Obtaining the member '__getitem__' of a type (line 60)
    getitem___202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), split_call_result_201, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 60)
    subscript_call_result_203 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), getitem___202, int_193)
    
    # Assigning a type to the variable 'tuple_var_assignment_4' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'tuple_var_assignment_4', subscript_call_result_203)
    
    # Assigning a Subscript to a Name (line 60):
    
    # Obtaining the type of the subscript
    int_204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 8), 'int')
    
    # Call to split(...): (line 60)
    # Processing the call arguments (line 60)
    str_210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 75), 'str', '\t')
    # Processing the call keyword arguments (line 60)
    kwargs_211 = {}
    
    # Call to strip(...): (line 60)
    # Processing the call keyword arguments (line 60)
    kwargs_207 = {}
    # Getting the type of 'line' (line 60)
    line_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 56), 'line', False)
    # Obtaining the member 'strip' of a type (line 60)
    strip_206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 56), line_205, 'strip')
    # Calling strip(args, kwargs) (line 60)
    strip_call_result_208 = invoke(stypy.reporting.localization.Localization(__file__, 60, 56), strip_206, *[], **kwargs_207)
    
    # Obtaining the member 'split' of a type (line 60)
    split_209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 56), strip_call_result_208, 'split')
    # Calling split(args, kwargs) (line 60)
    split_call_result_212 = invoke(stypy.reporting.localization.Localization(__file__, 60, 56), split_209, *[str_210], **kwargs_211)
    
    # Obtaining the member '__getitem__' of a type (line 60)
    getitem___213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), split_call_result_212, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 60)
    subscript_call_result_214 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), getitem___213, int_204)
    
    # Assigning a type to the variable 'tuple_var_assignment_5' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'tuple_var_assignment_5', subscript_call_result_214)
    
    # Assigning a Name to a Name (line 60):
    # Getting the type of 'tuple_var_assignment_1' (line 60)
    tuple_var_assignment_1_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'tuple_var_assignment_1')
    # Assigning a type to the variable 'name' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'name', tuple_var_assignment_1_215)
    
    # Assigning a Name to a Name (line 60):
    # Getting the type of 'tuple_var_assignment_2' (line 60)
    tuple_var_assignment_2_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'tuple_var_assignment_2')
    # Assigning a type to the variable 'mass' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 14), 'mass', tuple_var_assignment_2_216)
    
    # Assigning a Name to a Name (line 60):
    # Getting the type of 'tuple_var_assignment_3' (line 60)
    tuple_var_assignment_3_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'tuple_var_assignment_3')
    # Assigning a type to the variable 'isoelectric_point' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'isoelectric_point', tuple_var_assignment_3_217)
    
    # Assigning a Name to a Name (line 60):
    # Getting the type of 'tuple_var_assignment_4' (line 60)
    tuple_var_assignment_4_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'tuple_var_assignment_4')
    # Assigning a type to the variable 'size' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 39), 'size', tuple_var_assignment_4_218)
    
    # Assigning a Name to a Name (line 60):
    # Getting the type of 'tuple_var_assignment_5' (line 60)
    tuple_var_assignment_5_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'tuple_var_assignment_5')
    # Assigning a type to the variable 'sequence' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 45), 'sequence', tuple_var_assignment_5_219)
    
    # Assigning a Call to a Name (line 61):
    
    # Assigning a Call to a Name (line 61):
    
    # Call to Protein(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'name' (line 61)
    name_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 26), 'name', False)
    # Getting the type of 'mass' (line 61)
    mass_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 32), 'mass', False)
    # Getting the type of 'isoelectric_point' (line 61)
    isoelectric_point_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 38), 'isoelectric_point', False)
    # Getting the type of 'size' (line 61)
    size_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 57), 'size', False)
    # Getting the type of 'sequence' (line 61)
    sequence_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 63), 'sequence', False)
    # Getting the type of 'type' (line 61)
    type_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 73), 'type', False)
    # Processing the call keyword arguments (line 61)
    kwargs_227 = {}
    # Getting the type of 'Protein' (line 61)
    Protein_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 18), 'Protein', False)
    # Calling Protein(args, kwargs) (line 61)
    Protein_call_result_228 = invoke(stypy.reporting.localization.Localization(__file__, 61, 18), Protein_220, *[name_221, mass_222, isoelectric_point_223, size_224, sequence_225, type_226], **kwargs_227)
    
    # Assigning a type to the variable 'protein' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'protein', Protein_call_result_228)
    
    # Call to append(...): (line 62)
    # Processing the call arguments (line 62)
    # Getting the type of 'protein' (line 62)
    protein_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 24), 'protein', False)
    # Processing the call keyword arguments (line 62)
    kwargs_232 = {}
    # Getting the type of 'PROTEINS' (line 62)
    PROTEINS_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'PROTEINS', False)
    # Obtaining the member 'append' of a type (line 62)
    append_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), PROTEINS_229, 'append')
    # Calling append(args, kwargs) (line 62)
    append_call_result_233 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), append_230, *[protein_231], **kwargs_232)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to close(...): (line 63)
    # Processing the call keyword arguments (line 63)
    kwargs_236 = {}
    # Getting the type of 'protfile' (line 63)
    protfile_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'protfile', False)
    # Obtaining the member 'close' of a type (line 63)
    close_235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 4), protfile_234, 'close')
    # Calling close(args, kwargs) (line 63)
    close_call_result_237 = invoke(stypy.reporting.localization.Localization(__file__, 63, 4), close_235, *[], **kwargs_236)
    
    
    # ################# End of 'load_file(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'load_file' in the type store
    # Getting the type of 'stypy_return_type' (line 54)
    stypy_return_type_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_238)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'load_file'
    return stypy_return_type_238

# Assigning a type to the variable 'load_file' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'load_file', load_file)

@norecursion
def create_tables(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_tables'
    module_type_store = module_type_store.open_function_context('create_tables', 66, 0, False)
    
    # Passed parameters checking function
    create_tables.stypy_localization = localization
    create_tables.stypy_type_of_self = None
    create_tables.stypy_type_store = module_type_store
    create_tables.stypy_function_name = 'create_tables'
    create_tables.stypy_param_names_list = []
    create_tables.stypy_varargs_param_name = None
    create_tables.stypy_kwargs_param_name = None
    create_tables.stypy_call_defaults = defaults
    create_tables.stypy_call_varargs = varargs
    create_tables.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_tables', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_tables', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_tables(...)' code ##################

    str_239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 4), 'str', 'Create the feature and label tables.')
    
    # Assigning a List to a Name (line 68):
    
    # Assigning a List to a Name (line 68):
    
    # Obtaining an instance of the builtin type 'list' (line 68)
    list_240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 68)
    
    # Assigning a type to the variable 'feature_table' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'feature_table', list_240)
    
    # Assigning a List to a Name (line 69):
    
    # Assigning a List to a Name (line 69):
    
    # Obtaining an instance of the builtin type 'list' (line 69)
    list_241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 69)
    
    # Assigning a type to the variable 'label_table' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'label_table', list_241)
    
    # Getting the type of 'PROTEINS' (line 71)
    PROTEINS_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'PROTEINS')
    # Testing the type of a for loop iterable (line 71)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 71, 4), PROTEINS_242)
    # Getting the type of the for loop variable (line 71)
    for_loop_var_243 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 71, 4), PROTEINS_242)
    # Assigning a type to the variable 'protein' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'protein', for_loop_var_243)
    # SSA begins for a for statement (line 71)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 72)
    # Processing the call arguments (line 72)
    
    # Call to create_vector(...): (line 72)
    # Processing the call keyword arguments (line 72)
    kwargs_248 = {}
    # Getting the type of 'protein' (line 72)
    protein_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 29), 'protein', False)
    # Obtaining the member 'create_vector' of a type (line 72)
    create_vector_247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 29), protein_246, 'create_vector')
    # Calling create_vector(args, kwargs) (line 72)
    create_vector_call_result_249 = invoke(stypy.reporting.localization.Localization(__file__, 72, 29), create_vector_247, *[], **kwargs_248)
    
    # Processing the call keyword arguments (line 72)
    kwargs_250 = {}
    # Getting the type of 'feature_table' (line 72)
    feature_table_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'feature_table', False)
    # Obtaining the member 'append' of a type (line 72)
    append_245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), feature_table_244, 'append')
    # Calling append(args, kwargs) (line 72)
    append_call_result_251 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), append_245, *[create_vector_call_result_249], **kwargs_250)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'PROTEINS' (line 74)
    PROTEINS_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 19), 'PROTEINS')
    # Testing the type of a for loop iterable (line 74)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 74, 4), PROTEINS_252)
    # Getting the type of the for loop variable (line 74)
    for_loop_var_253 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 74, 4), PROTEINS_252)
    # Assigning a type to the variable 'protein' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'protein', for_loop_var_253)
    # SSA begins for a for statement (line 74)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'protein' (line 75)
    protein_254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 11), 'protein')
    # Obtaining the member 'type' of a type (line 75)
    type_255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 11), protein_254, 'type')
    # Getting the type of 'BLIND' (line 75)
    BLIND_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 27), 'BLIND')
    # Applying the binary operator '==' (line 75)
    result_eq_257 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 11), '==', type_255, BLIND_256)
    
    # Testing the type of an if condition (line 75)
    if_condition_258 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 8), result_eq_257)
    # Assigning a type to the variable 'if_condition_258' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'if_condition_258', if_condition_258)
    # SSA begins for if statement (line 75)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 75)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 77):
    
    # Assigning a BinOp to a Name (line 77):
    
    # Obtaining an instance of the builtin type 'list' (line 77)
    list_259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 77)
    # Adding element type (line 77)
    int_260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 17), list_259, int_260)
    
    int_261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 24), 'int')
    # Applying the binary operator '*' (line 77)
    result_mul_262 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 17), '*', list_259, int_261)
    
    # Assigning a type to the variable 'labels' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'labels', result_mul_262)
    
    # Getting the type of 'labels' (line 79)
    labels_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'labels')
    
    # Obtaining the type of the subscript
    # Getting the type of 'protein' (line 79)
    protein_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 15), 'protein')
    # Obtaining the member 'type' of a type (line 79)
    type_265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 15), protein_264, 'type')
    # Getting the type of 'labels' (line 79)
    labels_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'labels')
    # Obtaining the member '__getitem__' of a type (line 79)
    getitem___267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), labels_266, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 79)
    subscript_call_result_268 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), getitem___267, type_265)
    
    int_269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 32), 'int')
    # Applying the binary operator '*=' (line 79)
    result_imul_270 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 8), '*=', subscript_call_result_268, int_269)
    # Getting the type of 'labels' (line 79)
    labels_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'labels')
    # Getting the type of 'protein' (line 79)
    protein_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 15), 'protein')
    # Obtaining the member 'type' of a type (line 79)
    type_273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 15), protein_272, 'type')
    # Storing an element on a container (line 79)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 8), labels_271, (type_273, result_imul_270))
    
    
    # Call to append(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'labels' (line 80)
    labels_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 27), 'labels', False)
    # Processing the call keyword arguments (line 80)
    kwargs_277 = {}
    # Getting the type of 'label_table' (line 80)
    label_table_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'label_table', False)
    # Obtaining the member 'append' of a type (line 80)
    append_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), label_table_274, 'append')
    # Calling append(args, kwargs) (line 80)
    append_call_result_278 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), append_275, *[labels_276], **kwargs_277)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 82)
    tuple_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 82)
    # Adding element type (line 82)
    # Getting the type of 'feature_table' (line 82)
    feature_table_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 11), 'feature_table')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 11), tuple_279, feature_table_280)
    # Adding element type (line 82)
    # Getting the type of 'label_table' (line 82)
    label_table_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'label_table')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 11), tuple_279, label_table_281)
    
    # Assigning a type to the variable 'stypy_return_type' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type', tuple_279)
    
    # ################# End of 'create_tables(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_tables' in the type store
    # Getting the type of 'stypy_return_type' (line 66)
    stypy_return_type_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_282)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_tables'
    return stypy_return_type_282

# Assigning a type to the variable 'create_tables' (line 66)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'create_tables', create_tables)

@norecursion
def create_kernel_table(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_kernel_table'
    module_type_store = module_type_store.open_function_context('create_kernel_table', 85, 0, False)
    
    # Passed parameters checking function
    create_kernel_table.stypy_localization = localization
    create_kernel_table.stypy_type_of_self = None
    create_kernel_table.stypy_type_store = module_type_store
    create_kernel_table.stypy_function_name = 'create_kernel_table'
    create_kernel_table.stypy_param_names_list = ['feature_table']
    create_kernel_table.stypy_varargs_param_name = None
    create_kernel_table.stypy_kwargs_param_name = None
    create_kernel_table.stypy_call_defaults = defaults
    create_kernel_table.stypy_call_varargs = varargs
    create_kernel_table.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_kernel_table', ['feature_table'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_kernel_table', localization, ['feature_table'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_kernel_table(...)' code ##################

    
    # Assigning a List to a Name (line 86):
    
    # Assigning a List to a Name (line 86):
    
    # Obtaining an instance of the builtin type 'list' (line 86)
    list_283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 86)
    
    # Assigning a type to the variable 'kernel_table' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'kernel_table', list_283)
    
    # Getting the type of 'feature_table' (line 87)
    feature_table_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'feature_table')
    # Testing the type of a for loop iterable (line 87)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 87, 4), feature_table_284)
    # Getting the type of the for loop variable (line 87)
    for_loop_var_285 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 87, 4), feature_table_284)
    # Assigning a type to the variable 'row' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'row', for_loop_var_285)
    # SSA begins for a for statement (line 87)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a List to a Name (line 88):
    
    # Assigning a List to a Name (line 88):
    
    # Obtaining an instance of the builtin type 'list' (line 88)
    list_286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 88)
    
    # Assigning a type to the variable 'kernel_row' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'kernel_row', list_286)
    
    # Getting the type of 'feature_table' (line 89)
    feature_table_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 25), 'feature_table')
    # Testing the type of a for loop iterable (line 89)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 89, 8), feature_table_287)
    # Getting the type of the for loop variable (line 89)
    for_loop_var_288 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 89, 8), feature_table_287)
    # Assigning a type to the variable 'candidate' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'candidate', for_loop_var_288)
    # SSA begins for a for statement (line 89)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Num to a Name (line 90):
    
    # Assigning a Num to a Name (line 90):
    float_289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 25), 'float')
    # Assigning a type to the variable 'difference' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'difference', float_289)
    
    
    # Call to range(...): (line 91)
    # Processing the call arguments (line 91)
    
    # Call to len(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'row' (line 91)
    row_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 37), 'row', False)
    # Processing the call keyword arguments (line 91)
    kwargs_293 = {}
    # Getting the type of 'len' (line 91)
    len_291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 33), 'len', False)
    # Calling len(args, kwargs) (line 91)
    len_call_result_294 = invoke(stypy.reporting.localization.Localization(__file__, 91, 33), len_291, *[row_292], **kwargs_293)
    
    # Processing the call keyword arguments (line 91)
    kwargs_295 = {}
    # Getting the type of 'range' (line 91)
    range_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 27), 'range', False)
    # Calling range(args, kwargs) (line 91)
    range_call_result_296 = invoke(stypy.reporting.localization.Localization(__file__, 91, 27), range_290, *[len_call_result_294], **kwargs_295)
    
    # Testing the type of a for loop iterable (line 91)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 91, 12), range_call_result_296)
    # Getting the type of the for loop variable (line 91)
    for_loop_var_297 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 91, 12), range_call_result_296)
    # Assigning a type to the variable 'counter' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'counter', for_loop_var_297)
    # SSA begins for a for statement (line 91)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'difference' (line 92)
    difference_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'difference')
    
    # Obtaining the type of the subscript
    # Getting the type of 'counter' (line 92)
    counter_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 35), 'counter')
    # Getting the type of 'row' (line 92)
    row_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 31), 'row')
    # Obtaining the member '__getitem__' of a type (line 92)
    getitem___301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 31), row_300, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 92)
    subscript_call_result_302 = invoke(stypy.reporting.localization.Localization(__file__, 92, 31), getitem___301, counter_299)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'counter' (line 92)
    counter_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 56), 'counter')
    # Getting the type of 'candidate' (line 92)
    candidate_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 46), 'candidate')
    # Obtaining the member '__getitem__' of a type (line 92)
    getitem___305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 46), candidate_304, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 92)
    subscript_call_result_306 = invoke(stypy.reporting.localization.Localization(__file__, 92, 46), getitem___305, counter_303)
    
    # Applying the binary operator '-' (line 92)
    result_sub_307 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 31), '-', subscript_call_result_302, subscript_call_result_306)
    
    int_308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 69), 'int')
    # Applying the binary operator '**' (line 92)
    result_pow_309 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 30), '**', result_sub_307, int_308)
    
    # Applying the binary operator '+=' (line 92)
    result_iadd_310 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 16), '+=', difference_298, result_pow_309)
    # Assigning a type to the variable 'difference' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'difference', result_iadd_310)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 93)
    # Processing the call arguments (line 93)
    
    # Call to exp(...): (line 93)
    # Processing the call arguments (line 93)
    
    # Getting the type of 'D' (line 93)
    D_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 35), 'D', False)
    # Applying the 'usub' unary operator (line 93)
    result___neg___315 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 34), 'usub', D_314)
    
    # Getting the type of 'difference' (line 93)
    difference_316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 37), 'difference', False)
    # Applying the binary operator '*' (line 93)
    result_mul_317 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 34), '*', result___neg___315, difference_316)
    
    # Processing the call keyword arguments (line 93)
    kwargs_318 = {}
    # Getting the type of 'exp' (line 93)
    exp_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 30), 'exp', False)
    # Calling exp(args, kwargs) (line 93)
    exp_call_result_319 = invoke(stypy.reporting.localization.Localization(__file__, 93, 30), exp_313, *[result_mul_317], **kwargs_318)
    
    # Processing the call keyword arguments (line 93)
    kwargs_320 = {}
    # Getting the type of 'kernel_row' (line 93)
    kernel_row_311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'kernel_row', False)
    # Obtaining the member 'append' of a type (line 93)
    append_312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 12), kernel_row_311, 'append')
    # Calling append(args, kwargs) (line 93)
    append_call_result_321 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), append_312, *[exp_call_result_319], **kwargs_320)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 94)
    # Processing the call arguments (line 94)
    # Getting the type of 'kernel_row' (line 94)
    kernel_row_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 28), 'kernel_row', False)
    # Processing the call keyword arguments (line 94)
    kwargs_325 = {}
    # Getting the type of 'kernel_table' (line 94)
    kernel_table_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'kernel_table', False)
    # Obtaining the member 'append' of a type (line 94)
    append_323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), kernel_table_322, 'append')
    # Calling append(args, kwargs) (line 94)
    append_call_result_326 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), append_323, *[kernel_row_324], **kwargs_325)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'kernel_table' (line 95)
    kernel_table_327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 11), 'kernel_table')
    # Assigning a type to the variable 'stypy_return_type' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'stypy_return_type', kernel_table_327)
    
    # ################# End of 'create_kernel_table(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_kernel_table' in the type store
    # Getting the type of 'stypy_return_type' (line 85)
    stypy_return_type_328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_328)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_kernel_table'
    return stypy_return_type_328

# Assigning a type to the variable 'create_kernel_table' (line 85)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'create_kernel_table', create_kernel_table)

@norecursion
def train_adatron(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'train_adatron'
    module_type_store = module_type_store.open_function_context('train_adatron', 98, 0, False)
    
    # Passed parameters checking function
    train_adatron.stypy_localization = localization
    train_adatron.stypy_type_of_self = None
    train_adatron.stypy_type_store = module_type_store
    train_adatron.stypy_function_name = 'train_adatron'
    train_adatron.stypy_param_names_list = ['kernel_table', 'label_table', 'h', 'c']
    train_adatron.stypy_varargs_param_name = None
    train_adatron.stypy_kwargs_param_name = None
    train_adatron.stypy_call_defaults = defaults
    train_adatron.stypy_call_varargs = varargs
    train_adatron.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'train_adatron', ['kernel_table', 'label_table', 'h', 'c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'train_adatron', localization, ['kernel_table', 'label_table', 'h', 'c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'train_adatron(...)' code ##################

    
    # Assigning a Num to a Name (line 99):
    
    # Assigning a Num to a Name (line 99):
    float_329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 16), 'float')
    # Assigning a type to the variable 'tolerance' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'tolerance', float_329)
    
    # Assigning a ListComp to a Name (line 100):
    
    # Assigning a ListComp to a Name (line 100):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 100)
    # Processing the call arguments (line 100)
    
    # Call to len(...): (line 100)
    # Processing the call arguments (line 100)
    
    # Obtaining the type of the subscript
    int_339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 73), 'int')
    # Getting the type of 'label_table' (line 100)
    label_table_340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 61), 'label_table', False)
    # Obtaining the member '__getitem__' of a type (line 100)
    getitem___341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 61), label_table_340, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 100)
    subscript_call_result_342 = invoke(stypy.reporting.localization.Localization(__file__, 100, 61), getitem___341, int_339)
    
    # Processing the call keyword arguments (line 100)
    kwargs_343 = {}
    # Getting the type of 'len' (line 100)
    len_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 57), 'len', False)
    # Calling len(args, kwargs) (line 100)
    len_call_result_344 = invoke(stypy.reporting.localization.Localization(__file__, 100, 57), len_338, *[subscript_call_result_342], **kwargs_343)
    
    # Processing the call keyword arguments (line 100)
    kwargs_345 = {}
    # Getting the type of 'range' (line 100)
    range_337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 51), 'range', False)
    # Calling range(args, kwargs) (line 100)
    range_call_result_346 = invoke(stypy.reporting.localization.Localization(__file__, 100, 51), range_337, *[len_call_result_344], **kwargs_345)
    
    comprehension_347 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 14), range_call_result_346)
    # Assigning a type to the variable '_' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 14), '_', comprehension_347)
    
    # Obtaining an instance of the builtin type 'list' (line 100)
    list_330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 100)
    # Adding element type (line 100)
    float_331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 16), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 15), list_330, float_331)
    
    
    # Call to len(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'kernel_table' (line 100)
    kernel_table_333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 27), 'kernel_table', False)
    # Processing the call keyword arguments (line 100)
    kwargs_334 = {}
    # Getting the type of 'len' (line 100)
    len_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 23), 'len', False)
    # Calling len(args, kwargs) (line 100)
    len_call_result_335 = invoke(stypy.reporting.localization.Localization(__file__, 100, 23), len_332, *[kernel_table_333], **kwargs_334)
    
    # Applying the binary operator '*' (line 100)
    result_mul_336 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 15), '*', list_330, len_call_result_335)
    
    list_348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 14), list_348, result_mul_336)
    # Assigning a type to the variable 'alphas' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'alphas', list_348)
    
    # Assigning a ListComp to a Name (line 101):
    
    # Assigning a ListComp to a Name (line 101):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 101)
    # Processing the call arguments (line 101)
    
    # Call to len(...): (line 101)
    # Processing the call arguments (line 101)
    
    # Obtaining the type of the subscript
    int_358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 72), 'int')
    # Getting the type of 'label_table' (line 101)
    label_table_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 60), 'label_table', False)
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 60), label_table_359, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_361 = invoke(stypy.reporting.localization.Localization(__file__, 101, 60), getitem___360, int_358)
    
    # Processing the call keyword arguments (line 101)
    kwargs_362 = {}
    # Getting the type of 'len' (line 101)
    len_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 56), 'len', False)
    # Calling len(args, kwargs) (line 101)
    len_call_result_363 = invoke(stypy.reporting.localization.Localization(__file__, 101, 56), len_357, *[subscript_call_result_361], **kwargs_362)
    
    # Processing the call keyword arguments (line 101)
    kwargs_364 = {}
    # Getting the type of 'range' (line 101)
    range_356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 50), 'range', False)
    # Calling range(args, kwargs) (line 101)
    range_call_result_365 = invoke(stypy.reporting.localization.Localization(__file__, 101, 50), range_356, *[len_call_result_363], **kwargs_364)
    
    comprehension_366 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 13), range_call_result_365)
    # Assigning a type to the variable '_' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 13), '_', comprehension_366)
    
    # Obtaining an instance of the builtin type 'list' (line 101)
    list_349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 101)
    # Adding element type (line 101)
    float_350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 15), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 14), list_349, float_350)
    
    
    # Call to len(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'kernel_table' (line 101)
    kernel_table_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 26), 'kernel_table', False)
    # Processing the call keyword arguments (line 101)
    kwargs_353 = {}
    # Getting the type of 'len' (line 101)
    len_351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 22), 'len', False)
    # Calling len(args, kwargs) (line 101)
    len_call_result_354 = invoke(stypy.reporting.localization.Localization(__file__, 101, 22), len_351, *[kernel_table_352], **kwargs_353)
    
    # Applying the binary operator '*' (line 101)
    result_mul_355 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 14), '*', list_349, len_call_result_354)
    
    list_367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 13), list_367, result_mul_355)
    # Assigning a type to the variable 'betas' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'betas', list_367)
    
    # Assigning a BinOp to a Name (line 102):
    
    # Assigning a BinOp to a Name (line 102):
    
    # Obtaining an instance of the builtin type 'list' (line 102)
    list_368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 102)
    # Adding element type (line 102)
    float_369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 12), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 11), list_368, float_369)
    
    
    # Call to len(...): (line 102)
    # Processing the call arguments (line 102)
    
    # Obtaining the type of the subscript
    int_371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 35), 'int')
    # Getting the type of 'label_table' (line 102)
    label_table_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 23), 'label_table', False)
    # Obtaining the member '__getitem__' of a type (line 102)
    getitem___373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 23), label_table_372, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 102)
    subscript_call_result_374 = invoke(stypy.reporting.localization.Localization(__file__, 102, 23), getitem___373, int_371)
    
    # Processing the call keyword arguments (line 102)
    kwargs_375 = {}
    # Getting the type of 'len' (line 102)
    len_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 19), 'len', False)
    # Calling len(args, kwargs) (line 102)
    len_call_result_376 = invoke(stypy.reporting.localization.Localization(__file__, 102, 19), len_370, *[subscript_call_result_374], **kwargs_375)
    
    # Applying the binary operator '*' (line 102)
    result_mul_377 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 11), '*', list_368, len_call_result_376)
    
    # Assigning a type to the variable 'bias' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'bias', result_mul_377)
    
    # Assigning a BinOp to a Name (line 103):
    
    # Assigning a BinOp to a Name (line 103):
    
    # Obtaining an instance of the builtin type 'list' (line 103)
    list_378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 103)
    # Adding element type (line 103)
    float_379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 19), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 18), list_378, float_379)
    
    
    # Call to len(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'kernel_table' (line 103)
    kernel_table_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 30), 'kernel_table', False)
    # Processing the call keyword arguments (line 103)
    kwargs_382 = {}
    # Getting the type of 'len' (line 103)
    len_380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 26), 'len', False)
    # Calling len(args, kwargs) (line 103)
    len_call_result_383 = invoke(stypy.reporting.localization.Localization(__file__, 103, 26), len_380, *[kernel_table_381], **kwargs_382)
    
    # Applying the binary operator '*' (line 103)
    result_mul_384 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 18), '*', list_378, len_call_result_383)
    
    # Assigning a type to the variable 'labelalphas' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'labelalphas', result_mul_384)
    
    # Assigning a BinOp to a Name (line 104):
    
    # Assigning a BinOp to a Name (line 104):
    
    # Obtaining an instance of the builtin type 'list' (line 104)
    list_385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 104)
    # Adding element type (line 104)
    
    # Obtaining an instance of the builtin type 'tuple' (line 104)
    tuple_386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 104)
    # Adding element type (line 104)
    float_387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 24), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 24), tuple_386, float_387)
    # Adding element type (line 104)
    int_388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 24), tuple_386, int_388)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 22), list_385, tuple_386)
    
    
    # Call to len(...): (line 104)
    # Processing the call arguments (line 104)
    
    # Obtaining the type of the subscript
    int_390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 51), 'int')
    # Getting the type of 'label_table' (line 104)
    label_table_391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 39), 'label_table', False)
    # Obtaining the member '__getitem__' of a type (line 104)
    getitem___392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 39), label_table_391, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 104)
    subscript_call_result_393 = invoke(stypy.reporting.localization.Localization(__file__, 104, 39), getitem___392, int_390)
    
    # Processing the call keyword arguments (line 104)
    kwargs_394 = {}
    # Getting the type of 'len' (line 104)
    len_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 35), 'len', False)
    # Calling len(args, kwargs) (line 104)
    len_call_result_395 = invoke(stypy.reporting.localization.Localization(__file__, 104, 35), len_389, *[subscript_call_result_393], **kwargs_394)
    
    # Applying the binary operator '*' (line 104)
    result_mul_396 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 22), '*', list_385, len_call_result_395)
    
    # Assigning a type to the variable 'max_differences' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'max_differences', result_mul_396)
    
    
    # Call to range(...): (line 105)
    # Processing the call arguments (line 105)
    int_398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 27), 'int')
    
    # Call to len(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'kernel_table' (line 105)
    kernel_table_400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 34), 'kernel_table', False)
    # Processing the call keyword arguments (line 105)
    kwargs_401 = {}
    # Getting the type of 'len' (line 105)
    len_399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 30), 'len', False)
    # Calling len(args, kwargs) (line 105)
    len_call_result_402 = invoke(stypy.reporting.localization.Localization(__file__, 105, 30), len_399, *[kernel_table_400], **kwargs_401)
    
    # Applying the binary operator '*' (line 105)
    result_mul_403 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 27), '*', int_398, len_call_result_402)
    
    # Processing the call keyword arguments (line 105)
    kwargs_404 = {}
    # Getting the type of 'range' (line 105)
    range_397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 21), 'range', False)
    # Calling range(args, kwargs) (line 105)
    range_call_result_405 = invoke(stypy.reporting.localization.Localization(__file__, 105, 21), range_397, *[result_mul_403], **kwargs_404)
    
    # Testing the type of a for loop iterable (line 105)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 105, 4), range_call_result_405)
    # Getting the type of the for loop variable (line 105)
    for_loop_var_406 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 105, 4), range_call_result_405)
    # Assigning a type to the variable 'iteration' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'iteration', for_loop_var_406)
    # SSA begins for a for statement (line 105)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'iteration' (line 107)
    iteration_407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 11), 'iteration')
    int_408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 24), 'int')
    # Applying the binary operator '==' (line 107)
    result_eq_409 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 11), '==', iteration_407, int_408)
    
    # Testing the type of an if condition (line 107)
    if_condition_410 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 8), result_eq_409)
    # Assigning a type to the variable 'if_condition_410' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'if_condition_410', if_condition_410)
    # SSA begins for if statement (line 107)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 108)
    tuple_411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 108)
    # Adding element type (line 108)
    # Getting the type of 'alphas' (line 108)
    alphas_412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 19), 'alphas')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 19), tuple_411, alphas_412)
    # Adding element type (line 108)
    # Getting the type of 'bias' (line 108)
    bias_413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 27), 'bias')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 19), tuple_411, bias_413)
    
    # Assigning a type to the variable 'stypy_return_type' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'stypy_return_type', tuple_411)
    # SSA join for if statement (line 107)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 109)
    # Processing the call arguments (line 109)
    
    # Call to len(...): (line 109)
    # Processing the call arguments (line 109)
    
    # Obtaining the type of the subscript
    int_416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 43), 'int')
    # Getting the type of 'label_table' (line 109)
    label_table_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 31), 'label_table', False)
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 31), label_table_417, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_419 = invoke(stypy.reporting.localization.Localization(__file__, 109, 31), getitem___418, int_416)
    
    # Processing the call keyword arguments (line 109)
    kwargs_420 = {}
    # Getting the type of 'len' (line 109)
    len_415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'len', False)
    # Calling len(args, kwargs) (line 109)
    len_call_result_421 = invoke(stypy.reporting.localization.Localization(__file__, 109, 27), len_415, *[subscript_call_result_419], **kwargs_420)
    
    # Processing the call keyword arguments (line 109)
    kwargs_422 = {}
    # Getting the type of 'range' (line 109)
    range_414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 21), 'range', False)
    # Calling range(args, kwargs) (line 109)
    range_call_result_423 = invoke(stypy.reporting.localization.Localization(__file__, 109, 21), range_414, *[len_call_result_421], **kwargs_422)
    
    # Testing the type of a for loop iterable (line 109)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 109, 8), range_call_result_423)
    # Getting the type of the for loop variable (line 109)
    for_loop_var_424 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 109, 8), range_call_result_423)
    # Assigning a type to the variable 'klass' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'klass', for_loop_var_424)
    # SSA begins for a for statement (line 109)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Tuple to a Subscript (line 110):
    
    # Assigning a Tuple to a Subscript (line 110):
    
    # Obtaining an instance of the builtin type 'tuple' (line 110)
    tuple_425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 110)
    # Adding element type (line 110)
    float_426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 38), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 38), tuple_425, float_426)
    # Adding element type (line 110)
    int_427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 38), tuple_425, int_427)
    
    # Getting the type of 'max_differences' (line 110)
    max_differences_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'max_differences')
    # Getting the type of 'klass' (line 110)
    klass_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 28), 'klass')
    # Storing an element on a container (line 110)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 12), max_differences_428, (klass_429, tuple_425))
    
    
    # Call to range(...): (line 111)
    # Processing the call arguments (line 111)
    
    # Call to len(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'kernel_table' (line 111)
    kernel_table_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 34), 'kernel_table', False)
    # Processing the call keyword arguments (line 111)
    kwargs_433 = {}
    # Getting the type of 'len' (line 111)
    len_431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 30), 'len', False)
    # Calling len(args, kwargs) (line 111)
    len_call_result_434 = invoke(stypy.reporting.localization.Localization(__file__, 111, 30), len_431, *[kernel_table_432], **kwargs_433)
    
    # Processing the call keyword arguments (line 111)
    kwargs_435 = {}
    # Getting the type of 'range' (line 111)
    range_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'range', False)
    # Calling range(args, kwargs) (line 111)
    range_call_result_436 = invoke(stypy.reporting.localization.Localization(__file__, 111, 24), range_430, *[len_call_result_434], **kwargs_435)
    
    # Testing the type of a for loop iterable (line 111)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 111, 12), range_call_result_436)
    # Getting the type of the for loop variable (line 111)
    for_loop_var_437 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 111, 12), range_call_result_436)
    # Assigning a type to the variable 'elem' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'elem', for_loop_var_437)
    # SSA begins for a for statement (line 111)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Subscript (line 112):
    
    # Assigning a BinOp to a Subscript (line 112):
    
    # Obtaining the type of the subscript
    # Getting the type of 'klass' (line 112)
    klass_438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 54), 'klass')
    
    # Obtaining the type of the subscript
    # Getting the type of 'elem' (line 112)
    elem_439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 48), 'elem')
    # Getting the type of 'label_table' (line 112)
    label_table_440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 36), 'label_table')
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 36), label_table_440, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 112)
    subscript_call_result_442 = invoke(stypy.reporting.localization.Localization(__file__, 112, 36), getitem___441, elem_439)
    
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 36), subscript_call_result_442, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 112)
    subscript_call_result_444 = invoke(stypy.reporting.localization.Localization(__file__, 112, 36), getitem___443, klass_438)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'elem' (line 112)
    elem_445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 77), 'elem')
    
    # Obtaining the type of the subscript
    # Getting the type of 'klass' (line 112)
    klass_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 70), 'klass')
    # Getting the type of 'alphas' (line 112)
    alphas_447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 63), 'alphas')
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 63), alphas_447, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 112)
    subscript_call_result_449 = invoke(stypy.reporting.localization.Localization(__file__, 112, 63), getitem___448, klass_446)
    
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 63), subscript_call_result_449, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 112)
    subscript_call_result_451 = invoke(stypy.reporting.localization.Localization(__file__, 112, 63), getitem___450, elem_445)
    
    # Applying the binary operator '*' (line 112)
    result_mul_452 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 36), '*', subscript_call_result_444, subscript_call_result_451)
    
    # Getting the type of 'labelalphas' (line 112)
    labelalphas_453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'labelalphas')
    # Getting the type of 'elem' (line 112)
    elem_454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 28), 'elem')
    # Storing an element on a container (line 112)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 16), labelalphas_453, (elem_454, result_mul_452))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 113)
    # Processing the call arguments (line 113)
    
    # Call to len(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'kernel_table' (line 113)
    kernel_table_457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 41), 'kernel_table', False)
    # Processing the call keyword arguments (line 113)
    kwargs_458 = {}
    # Getting the type of 'len' (line 113)
    len_456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 37), 'len', False)
    # Calling len(args, kwargs) (line 113)
    len_call_result_459 = invoke(stypy.reporting.localization.Localization(__file__, 113, 37), len_456, *[kernel_table_457], **kwargs_458)
    
    # Processing the call keyword arguments (line 113)
    kwargs_460 = {}
    # Getting the type of 'range' (line 113)
    range_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 31), 'range', False)
    # Calling range(args, kwargs) (line 113)
    range_call_result_461 = invoke(stypy.reporting.localization.Localization(__file__, 113, 31), range_455, *[len_call_result_459], **kwargs_460)
    
    # Testing the type of a for loop iterable (line 113)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 113, 12), range_call_result_461)
    # Getting the type of the for loop variable (line 113)
    for_loop_var_462 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 113, 12), range_call_result_461)
    # Assigning a type to the variable 'col_counter' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'col_counter', for_loop_var_462)
    # SSA begins for a for statement (line 113)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Num to a Name (line 114):
    
    # Assigning a Num to a Name (line 114):
    float_463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 29), 'float')
    # Assigning a type to the variable 'prediction' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'prediction', float_463)
    
    
    # Call to range(...): (line 115)
    # Processing the call arguments (line 115)
    
    # Call to len(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'kernel_table' (line 115)
    kernel_table_466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 45), 'kernel_table', False)
    # Processing the call keyword arguments (line 115)
    kwargs_467 = {}
    # Getting the type of 'len' (line 115)
    len_465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 41), 'len', False)
    # Calling len(args, kwargs) (line 115)
    len_call_result_468 = invoke(stypy.reporting.localization.Localization(__file__, 115, 41), len_465, *[kernel_table_466], **kwargs_467)
    
    # Processing the call keyword arguments (line 115)
    kwargs_469 = {}
    # Getting the type of 'range' (line 115)
    range_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 35), 'range', False)
    # Calling range(args, kwargs) (line 115)
    range_call_result_470 = invoke(stypy.reporting.localization.Localization(__file__, 115, 35), range_464, *[len_call_result_468], **kwargs_469)
    
    # Testing the type of a for loop iterable (line 115)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 115, 16), range_call_result_470)
    # Getting the type of the for loop variable (line 115)
    for_loop_var_471 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 115, 16), range_call_result_470)
    # Assigning a type to the variable 'row_counter' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'row_counter', for_loop_var_471)
    # SSA begins for a for statement (line 115)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'prediction' (line 116)
    prediction_472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 20), 'prediction')
    
    # Obtaining the type of the subscript
    # Getting the type of 'row_counter' (line 116)
    row_counter_473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 60), 'row_counter')
    
    # Obtaining the type of the subscript
    # Getting the type of 'col_counter' (line 116)
    col_counter_474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 47), 'col_counter')
    # Getting the type of 'kernel_table' (line 116)
    kernel_table_475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 34), 'kernel_table')
    # Obtaining the member '__getitem__' of a type (line 116)
    getitem___476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 34), kernel_table_475, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 116)
    subscript_call_result_477 = invoke(stypy.reporting.localization.Localization(__file__, 116, 34), getitem___476, col_counter_474)
    
    # Obtaining the member '__getitem__' of a type (line 116)
    getitem___478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 34), subscript_call_result_477, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 116)
    subscript_call_result_479 = invoke(stypy.reporting.localization.Localization(__file__, 116, 34), getitem___478, row_counter_473)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'row_counter' (line 117)
    row_counter_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 45), 'row_counter')
    # Getting the type of 'labelalphas' (line 117)
    labelalphas_481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 33), 'labelalphas')
    # Obtaining the member '__getitem__' of a type (line 117)
    getitem___482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 33), labelalphas_481, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 117)
    subscript_call_result_483 = invoke(stypy.reporting.localization.Localization(__file__, 117, 33), getitem___482, row_counter_480)
    
    # Applying the binary operator '*' (line 116)
    result_mul_484 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 34), '*', subscript_call_result_479, subscript_call_result_483)
    
    # Applying the binary operator '+=' (line 116)
    result_iadd_485 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 20), '+=', prediction_472, result_mul_484)
    # Assigning a type to the variable 'prediction' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 20), 'prediction', result_iadd_485)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 118):
    
    # Assigning a BinOp to a Name (line 118):
    float_486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 20), 'float')
    # Getting the type of 'prediction' (line 118)
    prediction_487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 28), 'prediction')
    
    # Obtaining the type of the subscript
    # Getting the type of 'klass' (line 118)
    klass_488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 46), 'klass')
    # Getting the type of 'bias' (line 118)
    bias_489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 41), 'bias')
    # Obtaining the member '__getitem__' of a type (line 118)
    getitem___490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 41), bias_489, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 118)
    subscript_call_result_491 = invoke(stypy.reporting.localization.Localization(__file__, 118, 41), getitem___490, klass_488)
    
    # Applying the binary operator '+' (line 118)
    result_add_492 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 28), '+', prediction_487, subscript_call_result_491)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'klass' (line 118)
    klass_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 81), 'klass')
    
    # Obtaining the type of the subscript
    # Getting the type of 'col_counter' (line 118)
    col_counter_494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 68), 'col_counter')
    # Getting the type of 'label_table' (line 118)
    label_table_495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 56), 'label_table')
    # Obtaining the member '__getitem__' of a type (line 118)
    getitem___496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 56), label_table_495, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 118)
    subscript_call_result_497 = invoke(stypy.reporting.localization.Localization(__file__, 118, 56), getitem___496, col_counter_494)
    
    # Obtaining the member '__getitem__' of a type (line 118)
    getitem___498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 56), subscript_call_result_497, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 118)
    subscript_call_result_499 = invoke(stypy.reporting.localization.Localization(__file__, 118, 56), getitem___498, klass_493)
    
    # Applying the binary operator '*' (line 118)
    result_mul_500 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 27), '*', result_add_492, subscript_call_result_499)
    
    # Applying the binary operator '-' (line 118)
    result_sub_501 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 20), '-', float_486, result_mul_500)
    
    # Assigning a type to the variable 'g' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'g', result_sub_501)
    
    # Assigning a Call to a Subscript (line 119):
    
    # Assigning a Call to a Subscript (line 119):
    
    # Call to min(...): (line 119)
    # Processing the call arguments (line 119)
    
    # Call to max(...): (line 119)
    # Processing the call arguments (line 119)
    
    # Obtaining the type of the subscript
    # Getting the type of 'col_counter' (line 119)
    col_counter_504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 67), 'col_counter', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'klass' (line 119)
    klass_505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 60), 'klass', False)
    # Getting the type of 'alphas' (line 119)
    alphas_506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 53), 'alphas', False)
    # Obtaining the member '__getitem__' of a type (line 119)
    getitem___507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 53), alphas_506, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 119)
    subscript_call_result_508 = invoke(stypy.reporting.localization.Localization(__file__, 119, 53), getitem___507, klass_505)
    
    # Obtaining the member '__getitem__' of a type (line 119)
    getitem___509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 53), subscript_call_result_508, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 119)
    subscript_call_result_510 = invoke(stypy.reporting.localization.Localization(__file__, 119, 53), getitem___509, col_counter_504)
    
    # Getting the type of 'h' (line 119)
    h_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 82), 'h', False)
    # Getting the type of 'g' (line 119)
    g_512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 86), 'g', False)
    # Applying the binary operator '*' (line 119)
    result_mul_513 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 82), '*', h_511, g_512)
    
    # Applying the binary operator '+' (line 119)
    result_add_514 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 53), '+', subscript_call_result_510, result_mul_513)
    
    float_515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 90), 'float')
    # Processing the call keyword arguments (line 119)
    kwargs_516 = {}
    # Getting the type of 'max' (line 119)
    max_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 48), 'max', False)
    # Calling max(args, kwargs) (line 119)
    max_call_result_517 = invoke(stypy.reporting.localization.Localization(__file__, 119, 48), max_503, *[result_add_514, float_515], **kwargs_516)
    
    # Getting the type of 'c' (line 119)
    c_518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 96), 'c', False)
    # Processing the call keyword arguments (line 119)
    kwargs_519 = {}
    # Getting the type of 'min' (line 119)
    min_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 44), 'min', False)
    # Calling min(args, kwargs) (line 119)
    min_call_result_520 = invoke(stypy.reporting.localization.Localization(__file__, 119, 44), min_502, *[max_call_result_517, c_518], **kwargs_519)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'klass' (line 119)
    klass_521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 22), 'klass')
    # Getting the type of 'betas' (line 119)
    betas_522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'betas')
    # Obtaining the member '__getitem__' of a type (line 119)
    getitem___523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 16), betas_522, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 119)
    subscript_call_result_524 = invoke(stypy.reporting.localization.Localization(__file__, 119, 16), getitem___523, klass_521)
    
    # Getting the type of 'col_counter' (line 119)
    col_counter_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 29), 'col_counter')
    # Storing an element on a container (line 119)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 16), subscript_call_result_524, (col_counter_525, min_call_result_520))
    
    # Assigning a Call to a Name (line 120):
    
    # Assigning a Call to a Name (line 120):
    
    # Call to abs(...): (line 120)
    # Processing the call arguments (line 120)
    
    # Obtaining the type of the subscript
    # Getting the type of 'col_counter' (line 120)
    col_counter_527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 47), 'col_counter', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'klass' (line 120)
    klass_528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 40), 'klass', False)
    # Getting the type of 'alphas' (line 120)
    alphas_529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 33), 'alphas', False)
    # Obtaining the member '__getitem__' of a type (line 120)
    getitem___530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 33), alphas_529, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 120)
    subscript_call_result_531 = invoke(stypy.reporting.localization.Localization(__file__, 120, 33), getitem___530, klass_528)
    
    # Obtaining the member '__getitem__' of a type (line 120)
    getitem___532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 33), subscript_call_result_531, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 120)
    subscript_call_result_533 = invoke(stypy.reporting.localization.Localization(__file__, 120, 33), getitem___532, col_counter_527)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'col_counter' (line 120)
    col_counter_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 75), 'col_counter', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'klass' (line 120)
    klass_535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 68), 'klass', False)
    # Getting the type of 'betas' (line 120)
    betas_536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 62), 'betas', False)
    # Obtaining the member '__getitem__' of a type (line 120)
    getitem___537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 62), betas_536, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 120)
    subscript_call_result_538 = invoke(stypy.reporting.localization.Localization(__file__, 120, 62), getitem___537, klass_535)
    
    # Obtaining the member '__getitem__' of a type (line 120)
    getitem___539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 62), subscript_call_result_538, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 120)
    subscript_call_result_540 = invoke(stypy.reporting.localization.Localization(__file__, 120, 62), getitem___539, col_counter_534)
    
    # Applying the binary operator '-' (line 120)
    result_sub_541 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 33), '-', subscript_call_result_533, subscript_call_result_540)
    
    # Processing the call keyword arguments (line 120)
    kwargs_542 = {}
    # Getting the type of 'abs' (line 120)
    abs_526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 29), 'abs', False)
    # Calling abs(args, kwargs) (line 120)
    abs_call_result_543 = invoke(stypy.reporting.localization.Localization(__file__, 120, 29), abs_526, *[result_sub_541], **kwargs_542)
    
    # Assigning a type to the variable 'difference' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'difference', abs_call_result_543)
    
    
    # Getting the type of 'difference' (line 121)
    difference_544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 19), 'difference')
    
    # Obtaining the type of the subscript
    int_545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 55), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'klass' (line 121)
    klass_546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 48), 'klass')
    # Getting the type of 'max_differences' (line 121)
    max_differences_547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 32), 'max_differences')
    # Obtaining the member '__getitem__' of a type (line 121)
    getitem___548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 32), max_differences_547, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 121)
    subscript_call_result_549 = invoke(stypy.reporting.localization.Localization(__file__, 121, 32), getitem___548, klass_546)
    
    # Obtaining the member '__getitem__' of a type (line 121)
    getitem___550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 32), subscript_call_result_549, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 121)
    subscript_call_result_551 = invoke(stypy.reporting.localization.Localization(__file__, 121, 32), getitem___550, int_545)
    
    # Applying the binary operator '>' (line 121)
    result_gt_552 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 19), '>', difference_544, subscript_call_result_551)
    
    # Testing the type of an if condition (line 121)
    if_condition_553 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 16), result_gt_552)
    # Assigning a type to the variable 'if_condition_553' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'if_condition_553', if_condition_553)
    # SSA begins for if statement (line 121)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Subscript (line 122):
    
    # Assigning a Tuple to a Subscript (line 122):
    
    # Obtaining an instance of the builtin type 'tuple' (line 122)
    tuple_554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 122)
    # Adding element type (line 122)
    # Getting the type of 'difference' (line 122)
    difference_555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 46), 'difference')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 46), tuple_554, difference_555)
    # Adding element type (line 122)
    # Getting the type of 'col_counter' (line 122)
    col_counter_556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 58), 'col_counter')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 46), tuple_554, col_counter_556)
    
    # Getting the type of 'max_differences' (line 122)
    max_differences_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 20), 'max_differences')
    # Getting the type of 'klass' (line 122)
    klass_558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 36), 'klass')
    # Storing an element on a container (line 122)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 20), max_differences_557, (klass_558, tuple_554))
    # SSA join for if statement (line 121)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to all(...): (line 124)
    # Processing the call arguments (line 124)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'max_differences' (line 124)
    max_differences_566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 72), 'max_differences', False)
    comprehension_567 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 20), max_differences_566)
    # Assigning a type to the variable 'max_difference' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'max_difference', comprehension_567)
    
    
    # Obtaining the type of the subscript
    int_560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 35), 'int')
    # Getting the type of 'max_difference' (line 124)
    max_difference_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'max_difference', False)
    # Obtaining the member '__getitem__' of a type (line 124)
    getitem___562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 20), max_difference_561, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 124)
    subscript_call_result_563 = invoke(stypy.reporting.localization.Localization(__file__, 124, 20), getitem___562, int_560)
    
    # Getting the type of 'tolerance' (line 124)
    tolerance_564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 40), 'tolerance', False)
    # Applying the binary operator '<' (line 124)
    result_lt_565 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 20), '<', subscript_call_result_563, tolerance_564)
    
    list_568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 20), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 20), list_568, result_lt_565)
    # Processing the call keyword arguments (line 124)
    kwargs_569 = {}
    # Getting the type of 'all' (line 124)
    all_559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 15), 'all', False)
    # Calling all(args, kwargs) (line 124)
    all_call_result_570 = invoke(stypy.reporting.localization.Localization(__file__, 124, 15), all_559, *[list_568], **kwargs_569)
    
    # Testing the type of an if condition (line 124)
    if_condition_571 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 12), all_call_result_570)
    # Assigning a type to the variable 'if_condition_571' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'if_condition_571', if_condition_571)
    # SSA begins for if statement (line 124)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 125)
    tuple_572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 125)
    # Adding element type (line 125)
    # Getting the type of 'alphas' (line 125)
    alphas_573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 23), 'alphas')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 23), tuple_572, alphas_573)
    # Adding element type (line 125)
    # Getting the type of 'bias' (line 125)
    bias_574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 31), 'bias')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 23), tuple_572, bias_574)
    
    # Assigning a type to the variable 'stypy_return_type' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'stypy_return_type', tuple_572)
    # SSA branch for the else part of an if statement (line 124)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Subscript (line 127):
    
    # Assigning a Subscript to a Subscript (line 127):
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    int_575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 95), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'klass' (line 127)
    klass_576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 88), 'klass')
    # Getting the type of 'max_differences' (line 127)
    max_differences_577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 72), 'max_differences')
    # Obtaining the member '__getitem__' of a type (line 127)
    getitem___578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 72), max_differences_577, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
    subscript_call_result_579 = invoke(stypy.reporting.localization.Localization(__file__, 127, 72), getitem___578, klass_576)
    
    # Obtaining the member '__getitem__' of a type (line 127)
    getitem___580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 72), subscript_call_result_579, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
    subscript_call_result_581 = invoke(stypy.reporting.localization.Localization(__file__, 127, 72), getitem___580, int_575)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'klass' (line 127)
    klass_582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 65), 'klass')
    # Getting the type of 'betas' (line 127)
    betas_583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 59), 'betas')
    # Obtaining the member '__getitem__' of a type (line 127)
    getitem___584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 59), betas_583, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
    subscript_call_result_585 = invoke(stypy.reporting.localization.Localization(__file__, 127, 59), getitem___584, klass_582)
    
    # Obtaining the member '__getitem__' of a type (line 127)
    getitem___586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 59), subscript_call_result_585, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
    subscript_call_result_587 = invoke(stypy.reporting.localization.Localization(__file__, 127, 59), getitem___586, subscript_call_result_581)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'klass' (line 127)
    klass_588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'klass')
    # Getting the type of 'alphas' (line 127)
    alphas_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'alphas')
    # Obtaining the member '__getitem__' of a type (line 127)
    getitem___590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 16), alphas_589, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
    subscript_call_result_591 = invoke(stypy.reporting.localization.Localization(__file__, 127, 16), getitem___590, klass_588)
    
    
    # Obtaining the type of the subscript
    int_592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 53), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'klass' (line 127)
    klass_593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 46), 'klass')
    # Getting the type of 'max_differences' (line 127)
    max_differences_594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 30), 'max_differences')
    # Obtaining the member '__getitem__' of a type (line 127)
    getitem___595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 30), max_differences_594, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
    subscript_call_result_596 = invoke(stypy.reporting.localization.Localization(__file__, 127, 30), getitem___595, klass_593)
    
    # Obtaining the member '__getitem__' of a type (line 127)
    getitem___597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 30), subscript_call_result_596, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
    subscript_call_result_598 = invoke(stypy.reporting.localization.Localization(__file__, 127, 30), getitem___597, int_592)
    
    # Storing an element on a container (line 127)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 16), subscript_call_result_591, (subscript_call_result_598, subscript_call_result_587))
    
    # Assigning a Num to a Name (line 128):
    
    # Assigning a Num to a Name (line 128):
    float_599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 30), 'float')
    # Assigning a type to the variable 'element_sum' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'element_sum', float_599)
    
    
    # Call to range(...): (line 129)
    # Processing the call arguments (line 129)
    
    # Call to len(...): (line 129)
    # Processing the call arguments (line 129)
    # Getting the type of 'kernel_table' (line 129)
    kernel_table_602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 49), 'kernel_table', False)
    # Processing the call keyword arguments (line 129)
    kwargs_603 = {}
    # Getting the type of 'len' (line 129)
    len_601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 45), 'len', False)
    # Calling len(args, kwargs) (line 129)
    len_call_result_604 = invoke(stypy.reporting.localization.Localization(__file__, 129, 45), len_601, *[kernel_table_602], **kwargs_603)
    
    # Processing the call keyword arguments (line 129)
    kwargs_605 = {}
    # Getting the type of 'range' (line 129)
    range_600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 39), 'range', False)
    # Calling range(args, kwargs) (line 129)
    range_call_result_606 = invoke(stypy.reporting.localization.Localization(__file__, 129, 39), range_600, *[len_call_result_604], **kwargs_605)
    
    # Testing the type of a for loop iterable (line 129)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 129, 16), range_call_result_606)
    # Getting the type of the for loop variable (line 129)
    for_loop_var_607 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 129, 16), range_call_result_606)
    # Assigning a type to the variable 'element_counter' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'element_counter', for_loop_var_607)
    # SSA begins for a for statement (line 129)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'element_sum' (line 130)
    element_sum_608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 20), 'element_sum')
    
    # Obtaining the type of the subscript
    # Getting the type of 'klass' (line 130)
    klass_609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 64), 'klass')
    
    # Obtaining the type of the subscript
    # Getting the type of 'element_counter' (line 130)
    element_counter_610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 47), 'element_counter')
    # Getting the type of 'label_table' (line 130)
    label_table_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 35), 'label_table')
    # Obtaining the member '__getitem__' of a type (line 130)
    getitem___612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 35), label_table_611, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 130)
    subscript_call_result_613 = invoke(stypy.reporting.localization.Localization(__file__, 130, 35), getitem___612, element_counter_610)
    
    # Obtaining the member '__getitem__' of a type (line 130)
    getitem___614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 35), subscript_call_result_613, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 130)
    subscript_call_result_615 = invoke(stypy.reporting.localization.Localization(__file__, 130, 35), getitem___614, klass_609)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'element_counter' (line 130)
    element_counter_616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 87), 'element_counter')
    
    # Obtaining the type of the subscript
    # Getting the type of 'klass' (line 130)
    klass_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 80), 'klass')
    # Getting the type of 'alphas' (line 130)
    alphas_618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 73), 'alphas')
    # Obtaining the member '__getitem__' of a type (line 130)
    getitem___619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 73), alphas_618, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 130)
    subscript_call_result_620 = invoke(stypy.reporting.localization.Localization(__file__, 130, 73), getitem___619, klass_617)
    
    # Obtaining the member '__getitem__' of a type (line 130)
    getitem___621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 73), subscript_call_result_620, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 130)
    subscript_call_result_622 = invoke(stypy.reporting.localization.Localization(__file__, 130, 73), getitem___621, element_counter_616)
    
    # Applying the binary operator '*' (line 130)
    result_mul_623 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 35), '*', subscript_call_result_615, subscript_call_result_622)
    
    int_624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 106), 'int')
    # Applying the binary operator 'div' (line 130)
    result_div_625 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 104), 'div', result_mul_623, int_624)
    
    # Applying the binary operator '+=' (line 130)
    result_iadd_626 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 20), '+=', element_sum_608, result_div_625)
    # Assigning a type to the variable 'element_sum' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 20), 'element_sum', result_iadd_626)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Subscript (line 131):
    
    # Assigning a BinOp to a Subscript (line 131):
    
    # Obtaining the type of the subscript
    # Getting the type of 'klass' (line 131)
    klass_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 35), 'klass')
    # Getting the type of 'bias' (line 131)
    bias_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 30), 'bias')
    # Obtaining the member '__getitem__' of a type (line 131)
    getitem___629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 30), bias_628, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 131)
    subscript_call_result_630 = invoke(stypy.reporting.localization.Localization(__file__, 131, 30), getitem___629, klass_627)
    
    # Getting the type of 'element_sum' (line 131)
    element_sum_631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 44), 'element_sum')
    # Applying the binary operator '+' (line 131)
    result_add_632 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 30), '+', subscript_call_result_630, element_sum_631)
    
    # Getting the type of 'bias' (line 131)
    bias_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'bias')
    # Getting the type of 'klass' (line 131)
    klass_634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 21), 'klass')
    # Storing an element on a container (line 131)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 16), bias_633, (klass_634, result_add_632))
    # SSA join for if statement (line 124)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'train_adatron(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'train_adatron' in the type store
    # Getting the type of 'stypy_return_type' (line 98)
    stypy_return_type_635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_635)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'train_adatron'
    return stypy_return_type_635

# Assigning a type to the variable 'train_adatron' (line 98)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'train_adatron', train_adatron)

@norecursion
def calculate_error(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'calculate_error'
    module_type_store = module_type_store.open_function_context('calculate_error', 133, 0, False)
    
    # Passed parameters checking function
    calculate_error.stypy_localization = localization
    calculate_error.stypy_type_of_self = None
    calculate_error.stypy_type_store = module_type_store
    calculate_error.stypy_function_name = 'calculate_error'
    calculate_error.stypy_param_names_list = ['alphas', 'bias', 'kernel_table', 'label_table']
    calculate_error.stypy_varargs_param_name = None
    calculate_error.stypy_kwargs_param_name = None
    calculate_error.stypy_call_defaults = defaults
    calculate_error.stypy_call_varargs = varargs
    calculate_error.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'calculate_error', ['alphas', 'bias', 'kernel_table', 'label_table'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'calculate_error', localization, ['alphas', 'bias', 'kernel_table', 'label_table'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'calculate_error(...)' code ##################

    
    # Assigning a Num to a Name (line 134):
    
    # Assigning a Num to a Name (line 134):
    float_636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 17), 'float')
    # Assigning a type to the variable 'prediction' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'prediction', float_636)
    
    # Assigning a ListComp to a Name (line 135):
    
    # Assigning a ListComp to a Name (line 135):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 135)
    # Processing the call arguments (line 135)
    
    # Call to len(...): (line 135)
    # Processing the call arguments (line 135)
    
    # Obtaining the type of the subscript
    int_646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 78), 'int')
    # Getting the type of 'label_table' (line 135)
    label_table_647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 66), 'label_table', False)
    # Obtaining the member '__getitem__' of a type (line 135)
    getitem___648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 66), label_table_647, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 135)
    subscript_call_result_649 = invoke(stypy.reporting.localization.Localization(__file__, 135, 66), getitem___648, int_646)
    
    # Processing the call keyword arguments (line 135)
    kwargs_650 = {}
    # Getting the type of 'len' (line 135)
    len_645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 62), 'len', False)
    # Calling len(args, kwargs) (line 135)
    len_call_result_651 = invoke(stypy.reporting.localization.Localization(__file__, 135, 62), len_645, *[subscript_call_result_649], **kwargs_650)
    
    # Processing the call keyword arguments (line 135)
    kwargs_652 = {}
    # Getting the type of 'range' (line 135)
    range_644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 56), 'range', False)
    # Calling range(args, kwargs) (line 135)
    range_call_result_653 = invoke(stypy.reporting.localization.Localization(__file__, 135, 56), range_644, *[len_call_result_651], **kwargs_652)
    
    comprehension_654 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 19), range_call_result_653)
    # Assigning a type to the variable '_' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 19), '_', comprehension_654)
    
    # Obtaining an instance of the builtin type 'list' (line 135)
    list_637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 135)
    # Adding element type (line 135)
    float_638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 20), list_637, float_638)
    
    
    # Call to len(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'kernel_table' (line 135)
    kernel_table_640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 32), 'kernel_table', False)
    # Processing the call keyword arguments (line 135)
    kwargs_641 = {}
    # Getting the type of 'len' (line 135)
    len_639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 28), 'len', False)
    # Calling len(args, kwargs) (line 135)
    len_call_result_642 = invoke(stypy.reporting.localization.Localization(__file__, 135, 28), len_639, *[kernel_table_640], **kwargs_641)
    
    # Applying the binary operator '*' (line 135)
    result_mul_643 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 20), '*', list_637, len_call_result_642)
    
    list_655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 19), list_655, result_mul_643)
    # Assigning a type to the variable 'predictions' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'predictions', list_655)
    
    
    # Call to range(...): (line 136)
    # Processing the call arguments (line 136)
    
    # Call to len(...): (line 136)
    # Processing the call arguments (line 136)
    
    # Obtaining the type of the subscript
    int_658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 39), 'int')
    # Getting the type of 'label_table' (line 136)
    label_table_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 27), 'label_table', False)
    # Obtaining the member '__getitem__' of a type (line 136)
    getitem___660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 27), label_table_659, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 136)
    subscript_call_result_661 = invoke(stypy.reporting.localization.Localization(__file__, 136, 27), getitem___660, int_658)
    
    # Processing the call keyword arguments (line 136)
    kwargs_662 = {}
    # Getting the type of 'len' (line 136)
    len_657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 23), 'len', False)
    # Calling len(args, kwargs) (line 136)
    len_call_result_663 = invoke(stypy.reporting.localization.Localization(__file__, 136, 23), len_657, *[subscript_call_result_661], **kwargs_662)
    
    # Processing the call keyword arguments (line 136)
    kwargs_664 = {}
    # Getting the type of 'range' (line 136)
    range_656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 17), 'range', False)
    # Calling range(args, kwargs) (line 136)
    range_call_result_665 = invoke(stypy.reporting.localization.Localization(__file__, 136, 17), range_656, *[len_call_result_663], **kwargs_664)
    
    # Testing the type of a for loop iterable (line 136)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 136, 4), range_call_result_665)
    # Getting the type of the for loop variable (line 136)
    for_loop_var_666 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 136, 4), range_call_result_665)
    # Assigning a type to the variable 'klass' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'klass', for_loop_var_666)
    # SSA begins for a for statement (line 136)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to range(...): (line 137)
    # Processing the call arguments (line 137)
    
    # Call to len(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'kernel_table' (line 137)
    kernel_table_669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 37), 'kernel_table', False)
    # Processing the call keyword arguments (line 137)
    kwargs_670 = {}
    # Getting the type of 'len' (line 137)
    len_668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 33), 'len', False)
    # Calling len(args, kwargs) (line 137)
    len_call_result_671 = invoke(stypy.reporting.localization.Localization(__file__, 137, 33), len_668, *[kernel_table_669], **kwargs_670)
    
    # Processing the call keyword arguments (line 137)
    kwargs_672 = {}
    # Getting the type of 'range' (line 137)
    range_667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 27), 'range', False)
    # Calling range(args, kwargs) (line 137)
    range_call_result_673 = invoke(stypy.reporting.localization.Localization(__file__, 137, 27), range_667, *[len_call_result_671], **kwargs_672)
    
    # Testing the type of a for loop iterable (line 137)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 137, 8), range_call_result_673)
    # Getting the type of the for loop variable (line 137)
    for_loop_var_674 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 137, 8), range_call_result_673)
    # Assigning a type to the variable 'col_counter' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'col_counter', for_loop_var_674)
    # SSA begins for a for statement (line 137)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to range(...): (line 138)
    # Processing the call arguments (line 138)
    
    # Call to len(...): (line 138)
    # Processing the call arguments (line 138)
    # Getting the type of 'kernel_table' (line 138)
    kernel_table_677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 41), 'kernel_table', False)
    # Processing the call keyword arguments (line 138)
    kwargs_678 = {}
    # Getting the type of 'len' (line 138)
    len_676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 37), 'len', False)
    # Calling len(args, kwargs) (line 138)
    len_call_result_679 = invoke(stypy.reporting.localization.Localization(__file__, 138, 37), len_676, *[kernel_table_677], **kwargs_678)
    
    # Processing the call keyword arguments (line 138)
    kwargs_680 = {}
    # Getting the type of 'range' (line 138)
    range_675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 31), 'range', False)
    # Calling range(args, kwargs) (line 138)
    range_call_result_681 = invoke(stypy.reporting.localization.Localization(__file__, 138, 31), range_675, *[len_call_result_679], **kwargs_680)
    
    # Testing the type of a for loop iterable (line 138)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 138, 12), range_call_result_681)
    # Getting the type of the for loop variable (line 138)
    for_loop_var_682 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 138, 12), range_call_result_681)
    # Assigning a type to the variable 'row_counter' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'row_counter', for_loop_var_682)
    # SSA begins for a for statement (line 138)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'prediction' (line 139)
    prediction_683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 'prediction')
    
    # Obtaining the type of the subscript
    # Getting the type of 'row_counter' (line 139)
    row_counter_684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 56), 'row_counter')
    
    # Obtaining the type of the subscript
    # Getting the type of 'col_counter' (line 139)
    col_counter_685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 43), 'col_counter')
    # Getting the type of 'kernel_table' (line 139)
    kernel_table_686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 30), 'kernel_table')
    # Obtaining the member '__getitem__' of a type (line 139)
    getitem___687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 30), kernel_table_686, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 139)
    subscript_call_result_688 = invoke(stypy.reporting.localization.Localization(__file__, 139, 30), getitem___687, col_counter_685)
    
    # Obtaining the member '__getitem__' of a type (line 139)
    getitem___689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 30), subscript_call_result_688, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 139)
    subscript_call_result_690 = invoke(stypy.reporting.localization.Localization(__file__, 139, 30), getitem___689, row_counter_684)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'klass' (line 140)
    klass_691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 55), 'klass')
    
    # Obtaining the type of the subscript
    # Getting the type of 'row_counter' (line 140)
    row_counter_692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 42), 'row_counter')
    # Getting the type of 'label_table' (line 140)
    label_table_693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 30), 'label_table')
    # Obtaining the member '__getitem__' of a type (line 140)
    getitem___694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 30), label_table_693, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 140)
    subscript_call_result_695 = invoke(stypy.reporting.localization.Localization(__file__, 140, 30), getitem___694, row_counter_692)
    
    # Obtaining the member '__getitem__' of a type (line 140)
    getitem___696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 30), subscript_call_result_695, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 140)
    subscript_call_result_697 = invoke(stypy.reporting.localization.Localization(__file__, 140, 30), getitem___696, klass_691)
    
    # Applying the binary operator '*' (line 139)
    result_mul_698 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 30), '*', subscript_call_result_690, subscript_call_result_697)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'row_counter' (line 140)
    row_counter_699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 78), 'row_counter')
    
    # Obtaining the type of the subscript
    # Getting the type of 'klass' (line 140)
    klass_700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 71), 'klass')
    # Getting the type of 'alphas' (line 140)
    alphas_701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 64), 'alphas')
    # Obtaining the member '__getitem__' of a type (line 140)
    getitem___702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 64), alphas_701, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 140)
    subscript_call_result_703 = invoke(stypy.reporting.localization.Localization(__file__, 140, 64), getitem___702, klass_700)
    
    # Obtaining the member '__getitem__' of a type (line 140)
    getitem___704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 64), subscript_call_result_703, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 140)
    subscript_call_result_705 = invoke(stypy.reporting.localization.Localization(__file__, 140, 64), getitem___704, row_counter_699)
    
    # Applying the binary operator '*' (line 140)
    result_mul_706 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 62), '*', result_mul_698, subscript_call_result_705)
    
    # Applying the binary operator '+=' (line 139)
    result_iadd_707 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 16), '+=', prediction_683, result_mul_706)
    # Assigning a type to the variable 'prediction' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 'prediction', result_iadd_707)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Subscript (line 141):
    
    # Assigning a BinOp to a Subscript (line 141):
    # Getting the type of 'prediction' (line 141)
    prediction_708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 46), 'prediction')
    
    # Obtaining the type of the subscript
    # Getting the type of 'klass' (line 141)
    klass_709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 64), 'klass')
    # Getting the type of 'bias' (line 141)
    bias_710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 59), 'bias')
    # Obtaining the member '__getitem__' of a type (line 141)
    getitem___711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 59), bias_710, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 141)
    subscript_call_result_712 = invoke(stypy.reporting.localization.Localization(__file__, 141, 59), getitem___711, klass_709)
    
    # Applying the binary operator '+' (line 141)
    result_add_713 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 46), '+', prediction_708, subscript_call_result_712)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'klass' (line 141)
    klass_714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 24), 'klass')
    # Getting the type of 'predictions' (line 141)
    predictions_715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'predictions')
    # Obtaining the member '__getitem__' of a type (line 141)
    getitem___716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 12), predictions_715, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 141)
    subscript_call_result_717 = invoke(stypy.reporting.localization.Localization(__file__, 141, 12), getitem___716, klass_714)
    
    # Getting the type of 'col_counter' (line 141)
    col_counter_718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 31), 'col_counter')
    # Storing an element on a container (line 141)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 12), subscript_call_result_717, (col_counter_718, result_add_713))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 143)
    # Processing the call arguments (line 143)
    
    # Call to len(...): (line 143)
    # Processing the call arguments (line 143)
    # Getting the type of 'kernel_table' (line 143)
    kernel_table_721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 33), 'kernel_table', False)
    # Processing the call keyword arguments (line 143)
    kwargs_722 = {}
    # Getting the type of 'len' (line 143)
    len_720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 29), 'len', False)
    # Calling len(args, kwargs) (line 143)
    len_call_result_723 = invoke(stypy.reporting.localization.Localization(__file__, 143, 29), len_720, *[kernel_table_721], **kwargs_722)
    
    # Processing the call keyword arguments (line 143)
    kwargs_724 = {}
    # Getting the type of 'range' (line 143)
    range_719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 23), 'range', False)
    # Calling range(args, kwargs) (line 143)
    range_call_result_725 = invoke(stypy.reporting.localization.Localization(__file__, 143, 23), range_719, *[len_call_result_723], **kwargs_724)
    
    # Testing the type of a for loop iterable (line 143)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 143, 4), range_call_result_725)
    # Getting the type of the for loop variable (line 143)
    for_loop_var_726 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 143, 4), range_call_result_725)
    # Assigning a type to the variable 'col_counter' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'col_counter', for_loop_var_726)
    # SSA begins for a for statement (line 143)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a List to a Name (line 144):
    
    # Assigning a List to a Name (line 144):
    
    # Obtaining an instance of the builtin type 'list' (line 144)
    list_727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 144)
    
    # Assigning a type to the variable 'current_predictions' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'current_predictions', list_727)
    
    # Assigning a Num to a Name (line 145):
    
    # Assigning a Num to a Name (line 145):
    int_728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 16), 'int')
    # Assigning a type to the variable 'error' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'error', int_728)
    
    
    # Call to range(...): (line 146)
    # Processing the call arguments (line 146)
    
    # Call to len(...): (line 146)
    # Processing the call arguments (line 146)
    
    # Obtaining the type of the subscript
    int_731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 49), 'int')
    # Getting the type of 'label_table' (line 146)
    label_table_732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 37), 'label_table', False)
    # Obtaining the member '__getitem__' of a type (line 146)
    getitem___733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 37), label_table_732, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 146)
    subscript_call_result_734 = invoke(stypy.reporting.localization.Localization(__file__, 146, 37), getitem___733, int_731)
    
    # Processing the call keyword arguments (line 146)
    kwargs_735 = {}
    # Getting the type of 'len' (line 146)
    len_730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 33), 'len', False)
    # Calling len(args, kwargs) (line 146)
    len_call_result_736 = invoke(stypy.reporting.localization.Localization(__file__, 146, 33), len_730, *[subscript_call_result_734], **kwargs_735)
    
    # Processing the call keyword arguments (line 146)
    kwargs_737 = {}
    # Getting the type of 'range' (line 146)
    range_729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 27), 'range', False)
    # Calling range(args, kwargs) (line 146)
    range_call_result_738 = invoke(stypy.reporting.localization.Localization(__file__, 146, 27), range_729, *[len_call_result_736], **kwargs_737)
    
    # Testing the type of a for loop iterable (line 146)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 146, 8), range_call_result_738)
    # Getting the type of the for loop variable (line 146)
    for_loop_var_739 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 146, 8), range_call_result_738)
    # Assigning a type to the variable 'row_counter' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'row_counter', for_loop_var_739)
    # SSA begins for a for statement (line 146)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 147)
    # Processing the call arguments (line 147)
    
    # Obtaining the type of the subscript
    # Getting the type of 'col_counter' (line 147)
    col_counter_742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 64), 'col_counter', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'row_counter' (line 147)
    row_counter_743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 51), 'row_counter', False)
    # Getting the type of 'predictions' (line 147)
    predictions_744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 39), 'predictions', False)
    # Obtaining the member '__getitem__' of a type (line 147)
    getitem___745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 39), predictions_744, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 147)
    subscript_call_result_746 = invoke(stypy.reporting.localization.Localization(__file__, 147, 39), getitem___745, row_counter_743)
    
    # Obtaining the member '__getitem__' of a type (line 147)
    getitem___747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 39), subscript_call_result_746, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 147)
    subscript_call_result_748 = invoke(stypy.reporting.localization.Localization(__file__, 147, 39), getitem___747, col_counter_742)
    
    # Processing the call keyword arguments (line 147)
    kwargs_749 = {}
    # Getting the type of 'current_predictions' (line 147)
    current_predictions_740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'current_predictions', False)
    # Obtaining the member 'append' of a type (line 147)
    append_741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 12), current_predictions_740, 'append')
    # Calling append(args, kwargs) (line 147)
    append_call_result_750 = invoke(stypy.reporting.localization.Localization(__file__, 147, 12), append_741, *[subscript_call_result_748], **kwargs_749)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 149):
    
    # Assigning a Call to a Name (line 149):
    
    # Call to index(...): (line 149)
    # Processing the call arguments (line 149)
    
    # Call to max(...): (line 149)
    # Processing the call arguments (line 149)
    # Getting the type of 'current_predictions' (line 149)
    current_predictions_754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 56), 'current_predictions', False)
    # Processing the call keyword arguments (line 149)
    kwargs_755 = {}
    # Getting the type of 'max' (line 149)
    max_753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 52), 'max', False)
    # Calling max(args, kwargs) (line 149)
    max_call_result_756 = invoke(stypy.reporting.localization.Localization(__file__, 149, 52), max_753, *[current_predictions_754], **kwargs_755)
    
    # Processing the call keyword arguments (line 149)
    kwargs_757 = {}
    # Getting the type of 'current_predictions' (line 149)
    current_predictions_751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 26), 'current_predictions', False)
    # Obtaining the member 'index' of a type (line 149)
    index_752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 26), current_predictions_751, 'index')
    # Calling index(args, kwargs) (line 149)
    index_call_result_758 = invoke(stypy.reporting.localization.Localization(__file__, 149, 26), index_752, *[max_call_result_756], **kwargs_757)
    
    # Assigning a type to the variable 'predicted_class' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'predicted_class', index_call_result_758)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'predicted_class' (line 151)
    predicted_class_759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 36), 'predicted_class')
    
    # Obtaining the type of the subscript
    # Getting the type of 'col_counter' (line 151)
    col_counter_760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 23), 'col_counter')
    # Getting the type of 'label_table' (line 151)
    label_table_761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 11), 'label_table')
    # Obtaining the member '__getitem__' of a type (line 151)
    getitem___762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 11), label_table_761, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 151)
    subscript_call_result_763 = invoke(stypy.reporting.localization.Localization(__file__, 151, 11), getitem___762, col_counter_760)
    
    # Obtaining the member '__getitem__' of a type (line 151)
    getitem___764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 11), subscript_call_result_763, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 151)
    subscript_call_result_765 = invoke(stypy.reporting.localization.Localization(__file__, 151, 11), getitem___764, predicted_class_759)
    
    int_766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 55), 'int')
    # Applying the binary operator '<' (line 151)
    result_lt_767 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 11), '<', subscript_call_result_765, int_766)
    
    # Testing the type of an if condition (line 151)
    if_condition_768 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 8), result_lt_767)
    # Assigning a type to the variable 'if_condition_768' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'if_condition_768', if_condition_768)
    # SSA begins for if statement (line 151)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'error' (line 152)
    error_769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'error')
    int_770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 21), 'int')
    # Applying the binary operator '+=' (line 152)
    result_iadd_771 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 12), '+=', error_769, int_770)
    # Assigning a type to the variable 'error' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'error', result_iadd_771)
    
    # SSA join for if statement (line 151)
    module_type_store = module_type_store.join_ssa_context()
    
    float_772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 15), 'float')
    # Getting the type of 'error' (line 154)
    error_773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 21), 'error')
    # Applying the binary operator '*' (line 154)
    result_mul_774 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 15), '*', float_772, error_773)
    
    
    # Call to len(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of 'kernel_table' (line 154)
    kernel_table_776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 33), 'kernel_table', False)
    # Processing the call keyword arguments (line 154)
    kwargs_777 = {}
    # Getting the type of 'len' (line 154)
    len_775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 29), 'len', False)
    # Calling len(args, kwargs) (line 154)
    len_call_result_778 = invoke(stypy.reporting.localization.Localization(__file__, 154, 29), len_775, *[kernel_table_776], **kwargs_777)
    
    # Applying the binary operator 'div' (line 154)
    result_div_779 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 27), 'div', result_mul_774, len_call_result_778)
    
    # Assigning a type to the variable 'stypy_return_type' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'stypy_return_type', result_div_779)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'calculate_error(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'calculate_error' in the type store
    # Getting the type of 'stypy_return_type' (line 133)
    stypy_return_type_780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_780)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'calculate_error'
    return stypy_return_type_780

# Assigning a type to the variable 'calculate_error' (line 133)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 0), 'calculate_error', calculate_error)

@norecursion
def main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'main'
    module_type_store = module_type_store.open_function_context('main', 157, 0, False)
    
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

    
    
    # Obtaining an instance of the builtin type 'list' (line 158)
    list_781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 158)
    # Adding element type (line 158)
    
    # Obtaining an instance of the builtin type 'tuple' (line 158)
    tuple_782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 158)
    # Adding element type (line 158)
    
    # Call to Relative(...): (line 158)
    # Processing the call arguments (line 158)
    str_784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 37), 'str', 'testdata/c.txt')
    # Processing the call keyword arguments (line 158)
    kwargs_785 = {}
    # Getting the type of 'Relative' (line 158)
    Relative_783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 28), 'Relative', False)
    # Calling Relative(args, kwargs) (line 158)
    Relative_call_result_786 = invoke(stypy.reporting.localization.Localization(__file__, 158, 28), Relative_783, *[str_784], **kwargs_785)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 28), tuple_782, Relative_call_result_786)
    # Adding element type (line 158)
    # Getting the type of 'CYTOSOLIC' (line 158)
    CYTOSOLIC_787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 56), 'CYTOSOLIC')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 28), tuple_782, CYTOSOLIC_787)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 26), list_781, tuple_782)
    # Adding element type (line 158)
    
    # Obtaining an instance of the builtin type 'tuple' (line 158)
    tuple_788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 69), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 158)
    # Adding element type (line 158)
    
    # Call to Relative(...): (line 158)
    # Processing the call arguments (line 158)
    str_790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 78), 'str', 'testdata/e.txt')
    # Processing the call keyword arguments (line 158)
    kwargs_791 = {}
    # Getting the type of 'Relative' (line 158)
    Relative_789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 69), 'Relative', False)
    # Calling Relative(args, kwargs) (line 158)
    Relative_call_result_792 = invoke(stypy.reporting.localization.Localization(__file__, 158, 69), Relative_789, *[str_790], **kwargs_791)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 69), tuple_788, Relative_call_result_792)
    # Adding element type (line 158)
    # Getting the type of 'EXTRACELLULAR' (line 158)
    EXTRACELLULAR_793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 97), 'EXTRACELLULAR')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 69), tuple_788, EXTRACELLULAR_793)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 26), list_781, tuple_788)
    # Adding element type (line 158)
    
    # Obtaining an instance of the builtin type 'tuple' (line 158)
    tuple_794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 114), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 158)
    # Adding element type (line 158)
    
    # Call to Relative(...): (line 158)
    # Processing the call arguments (line 158)
    str_796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 123), 'str', 'testdata/n.txt')
    # Processing the call keyword arguments (line 158)
    kwargs_797 = {}
    # Getting the type of 'Relative' (line 158)
    Relative_795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 114), 'Relative', False)
    # Calling Relative(args, kwargs) (line 158)
    Relative_call_result_798 = invoke(stypy.reporting.localization.Localization(__file__, 158, 114), Relative_795, *[str_796], **kwargs_797)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 114), tuple_794, Relative_call_result_798)
    # Adding element type (line 158)
    # Getting the type of 'NUCLEAR' (line 158)
    NUCLEAR_799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 142), 'NUCLEAR')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 114), tuple_794, NUCLEAR_799)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 26), list_781, tuple_794)
    # Adding element type (line 158)
    
    # Obtaining an instance of the builtin type 'tuple' (line 158)
    tuple_800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 153), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 158)
    # Adding element type (line 158)
    
    # Call to Relative(...): (line 158)
    # Processing the call arguments (line 158)
    str_802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 162), 'str', 'testdata/m.txt')
    # Processing the call keyword arguments (line 158)
    kwargs_803 = {}
    # Getting the type of 'Relative' (line 158)
    Relative_801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 153), 'Relative', False)
    # Calling Relative(args, kwargs) (line 158)
    Relative_call_result_804 = invoke(stypy.reporting.localization.Localization(__file__, 158, 153), Relative_801, *[str_802], **kwargs_803)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 153), tuple_800, Relative_call_result_804)
    # Adding element type (line 158)
    # Getting the type of 'MITOCHONDRIAL' (line 158)
    MITOCHONDRIAL_805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 181), 'MITOCHONDRIAL')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 153), tuple_800, MITOCHONDRIAL_805)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 26), list_781, tuple_800)
    
    # Testing the type of a for loop iterable (line 158)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 158, 4), list_781)
    # Getting the type of the for loop variable (line 158)
    for_loop_var_806 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 158, 4), list_781)
    # Assigning a type to the variable 'filename' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'filename', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 4), for_loop_var_806))
    # Assigning a type to the variable 'type' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'type', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 4), for_loop_var_806))
    # SSA begins for a for statement (line 158)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to load_file(...): (line 159)
    # Processing the call arguments (line 159)
    # Getting the type of 'filename' (line 159)
    filename_808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 18), 'filename', False)
    # Getting the type of 'type' (line 159)
    type_809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 28), 'type', False)
    # Processing the call keyword arguments (line 159)
    kwargs_810 = {}
    # Getting the type of 'load_file' (line 159)
    load_file_807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'load_file', False)
    # Calling load_file(args, kwargs) (line 159)
    load_file_call_result_811 = invoke(stypy.reporting.localization.Localization(__file__, 159, 8), load_file_807, *[filename_808, type_809], **kwargs_810)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 161):
    
    # Assigning a Subscript to a Name (line 161):
    
    # Obtaining the type of the subscript
    int_812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 4), 'int')
    
    # Call to create_tables(...): (line 161)
    # Processing the call keyword arguments (line 161)
    kwargs_814 = {}
    # Getting the type of 'create_tables' (line 161)
    create_tables_813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 33), 'create_tables', False)
    # Calling create_tables(args, kwargs) (line 161)
    create_tables_call_result_815 = invoke(stypy.reporting.localization.Localization(__file__, 161, 33), create_tables_813, *[], **kwargs_814)
    
    # Obtaining the member '__getitem__' of a type (line 161)
    getitem___816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 4), create_tables_call_result_815, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 161)
    subscript_call_result_817 = invoke(stypy.reporting.localization.Localization(__file__, 161, 4), getitem___816, int_812)
    
    # Assigning a type to the variable 'tuple_var_assignment_6' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'tuple_var_assignment_6', subscript_call_result_817)
    
    # Assigning a Subscript to a Name (line 161):
    
    # Obtaining the type of the subscript
    int_818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 4), 'int')
    
    # Call to create_tables(...): (line 161)
    # Processing the call keyword arguments (line 161)
    kwargs_820 = {}
    # Getting the type of 'create_tables' (line 161)
    create_tables_819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 33), 'create_tables', False)
    # Calling create_tables(args, kwargs) (line 161)
    create_tables_call_result_821 = invoke(stypy.reporting.localization.Localization(__file__, 161, 33), create_tables_819, *[], **kwargs_820)
    
    # Obtaining the member '__getitem__' of a type (line 161)
    getitem___822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 4), create_tables_call_result_821, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 161)
    subscript_call_result_823 = invoke(stypy.reporting.localization.Localization(__file__, 161, 4), getitem___822, int_818)
    
    # Assigning a type to the variable 'tuple_var_assignment_7' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'tuple_var_assignment_7', subscript_call_result_823)
    
    # Assigning a Name to a Name (line 161):
    # Getting the type of 'tuple_var_assignment_6' (line 161)
    tuple_var_assignment_6_824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'tuple_var_assignment_6')
    # Assigning a type to the variable 'feature_table' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'feature_table', tuple_var_assignment_6_824)
    
    # Assigning a Name to a Name (line 161):
    # Getting the type of 'tuple_var_assignment_7' (line 161)
    tuple_var_assignment_7_825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'tuple_var_assignment_7')
    # Assigning a type to the variable 'label_table' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 19), 'label_table', tuple_var_assignment_7_825)
    
    # Assigning a Call to a Name (line 168):
    
    # Assigning a Call to a Name (line 168):
    
    # Call to create_kernel_table(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'feature_table' (line 168)
    feature_table_827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 39), 'feature_table', False)
    # Processing the call keyword arguments (line 168)
    kwargs_828 = {}
    # Getting the type of 'create_kernel_table' (line 168)
    create_kernel_table_826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 19), 'create_kernel_table', False)
    # Calling create_kernel_table(args, kwargs) (line 168)
    create_kernel_table_call_result_829 = invoke(stypy.reporting.localization.Localization(__file__, 168, 19), create_kernel_table_826, *[feature_table_827], **kwargs_828)
    
    # Assigning a type to the variable 'kernel_table' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'kernel_table', create_kernel_table_call_result_829)
    
    # Assigning a Call to a Tuple (line 170):
    
    # Assigning a Subscript to a Name (line 170):
    
    # Obtaining the type of the subscript
    int_830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 4), 'int')
    
    # Call to train_adatron(...): (line 170)
    # Processing the call arguments (line 170)
    # Getting the type of 'kernel_table' (line 170)
    kernel_table_832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 33), 'kernel_table', False)
    # Getting the type of 'label_table' (line 170)
    label_table_833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 47), 'label_table', False)
    float_834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 60), 'float')
    float_835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 65), 'float')
    # Processing the call keyword arguments (line 170)
    kwargs_836 = {}
    # Getting the type of 'train_adatron' (line 170)
    train_adatron_831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 19), 'train_adatron', False)
    # Calling train_adatron(args, kwargs) (line 170)
    train_adatron_call_result_837 = invoke(stypy.reporting.localization.Localization(__file__, 170, 19), train_adatron_831, *[kernel_table_832, label_table_833, float_834, float_835], **kwargs_836)
    
    # Obtaining the member '__getitem__' of a type (line 170)
    getitem___838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 4), train_adatron_call_result_837, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 170)
    subscript_call_result_839 = invoke(stypy.reporting.localization.Localization(__file__, 170, 4), getitem___838, int_830)
    
    # Assigning a type to the variable 'tuple_var_assignment_8' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'tuple_var_assignment_8', subscript_call_result_839)
    
    # Assigning a Subscript to a Name (line 170):
    
    # Obtaining the type of the subscript
    int_840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 4), 'int')
    
    # Call to train_adatron(...): (line 170)
    # Processing the call arguments (line 170)
    # Getting the type of 'kernel_table' (line 170)
    kernel_table_842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 33), 'kernel_table', False)
    # Getting the type of 'label_table' (line 170)
    label_table_843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 47), 'label_table', False)
    float_844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 60), 'float')
    float_845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 65), 'float')
    # Processing the call keyword arguments (line 170)
    kwargs_846 = {}
    # Getting the type of 'train_adatron' (line 170)
    train_adatron_841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 19), 'train_adatron', False)
    # Calling train_adatron(args, kwargs) (line 170)
    train_adatron_call_result_847 = invoke(stypy.reporting.localization.Localization(__file__, 170, 19), train_adatron_841, *[kernel_table_842, label_table_843, float_844, float_845], **kwargs_846)
    
    # Obtaining the member '__getitem__' of a type (line 170)
    getitem___848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 4), train_adatron_call_result_847, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 170)
    subscript_call_result_849 = invoke(stypy.reporting.localization.Localization(__file__, 170, 4), getitem___848, int_840)
    
    # Assigning a type to the variable 'tuple_var_assignment_9' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'tuple_var_assignment_9', subscript_call_result_849)
    
    # Assigning a Name to a Name (line 170):
    # Getting the type of 'tuple_var_assignment_8' (line 170)
    tuple_var_assignment_8_850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'tuple_var_assignment_8')
    # Assigning a type to the variable 'alphas' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'alphas', tuple_var_assignment_8_850)
    
    # Assigning a Name to a Name (line 170):
    # Getting the type of 'tuple_var_assignment_9' (line 170)
    tuple_var_assignment_9_851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'tuple_var_assignment_9')
    # Assigning a type to the variable 'bias' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'bias', tuple_var_assignment_9_851)
    
    # Call to calculate_error(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'alphas' (line 172)
    alphas_853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 20), 'alphas', False)
    # Getting the type of 'bias' (line 172)
    bias_854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 28), 'bias', False)
    # Getting the type of 'kernel_table' (line 172)
    kernel_table_855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 34), 'kernel_table', False)
    # Getting the type of 'label_table' (line 172)
    label_table_856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 48), 'label_table', False)
    # Processing the call keyword arguments (line 172)
    kwargs_857 = {}
    # Getting the type of 'calculate_error' (line 172)
    calculate_error_852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'calculate_error', False)
    # Calling calculate_error(args, kwargs) (line 172)
    calculate_error_call_result_858 = invoke(stypy.reporting.localization.Localization(__file__, 172, 4), calculate_error_852, *[alphas_853, bias_854, kernel_table_855, label_table_856], **kwargs_857)
    
    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 157)
    stypy_return_type_859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_859)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_859

# Assigning a type to the variable 'main' (line 157)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'main', main)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 175, 0, False)
    
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

    
    # Call to main(...): (line 176)
    # Processing the call keyword arguments (line 176)
    kwargs_861 = {}
    # Getting the type of 'main' (line 176)
    main_860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'main', False)
    # Calling main(args, kwargs) (line 176)
    main_call_result_862 = invoke(stypy.reporting.localization.Localization(__file__, 176, 4), main_860, *[], **kwargs_861)
    
    # Getting the type of 'True' (line 177)
    True_863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'stypy_return_type', True_863)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 175)
    stypy_return_type_864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_864)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_864

# Assigning a type to the variable 'run' (line 175)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 0), 'run', run)

# Call to run(...): (line 179)
# Processing the call keyword arguments (line 179)
kwargs_866 = {}
# Getting the type of 'run' (line 179)
run_865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 0), 'run', False)
# Calling run(args, kwargs) (line 179)
run_call_result_867 = invoke(stypy.reporting.localization.Localization(__file__, 179, 0), run_865, *[], **kwargs_866)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

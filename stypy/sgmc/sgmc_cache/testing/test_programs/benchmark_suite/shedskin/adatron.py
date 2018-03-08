
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
    file___19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 40), '__file__', False)
    # Processing the call keyword arguments (line 11)
    kwargs_20 = {}
    # Getting the type of 'os' (line 11)
    os_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 11)
    path_17 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 24), os_16, 'path')
    # Obtaining the member 'dirname' of a type (line 11)
    dirname_18 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 24), path_17, 'dirname')
    # Calling dirname(args, kwargs) (line 11)
    dirname_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 11, 24), dirname_18, *[file___19], **kwargs_20)
    
    # Getting the type of 'path' (line 11)
    path_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 51), 'path', False)
    # Processing the call keyword arguments (line 11)
    kwargs_23 = {}
    # Getting the type of 'os' (line 11)
    os_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 11)
    path_14 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 11), os_13, 'path')
    # Obtaining the member 'join' of a type (line 11)
    join_15 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 11), path_14, 'join')
    # Calling join(args, kwargs) (line 11)
    join_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 11, 11), join_15, *[dirname_call_result_21, path_22], **kwargs_23)
    
    # Assigning a type to the variable 'stypy_return_type' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'stypy_return_type', join_call_result_24)
    
    # ################# End of 'Relative(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Relative' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Relative'
    return stypy_return_type_25

# Assigning a type to the variable 'Relative' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'Relative', Relative)

# Assigning a Num to a Name (line 13):

# Assigning a Num to a Name (line 13):
int_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 12), 'int')
# Assigning a type to the variable 'CYTOSOLIC' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'CYTOSOLIC', int_26)

# Assigning a Num to a Name (line 14):

# Assigning a Num to a Name (line 14):
int_27 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 16), 'int')
# Assigning a type to the variable 'EXTRACELLULAR' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'EXTRACELLULAR', int_27)

# Assigning a Num to a Name (line 15):

# Assigning a Num to a Name (line 15):
int_28 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 10), 'int')
# Assigning a type to the variable 'NUCLEAR' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'NUCLEAR', int_28)

# Assigning a Num to a Name (line 16):

# Assigning a Num to a Name (line 16):
int_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 16), 'int')
# Assigning a type to the variable 'MITOCHONDRIAL' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'MITOCHONDRIAL', int_29)

# Assigning a Num to a Name (line 17):

# Assigning a Num to a Name (line 17):
int_30 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 8), 'int')
# Assigning a type to the variable 'BLIND' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'BLIND', int_30)

# Assigning a Num to a Name (line 19):

# Assigning a Num to a Name (line 19):
float_31 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 4), 'float')
# Assigning a type to the variable 'D' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'D', float_31)

# Assigning a Num to a Name (line 21):

# Assigning a Num to a Name (line 21):
int_32 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 9), 'int')
# Assigning a type to the variable 'LENGTH' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'LENGTH', int_32)

# Assigning a List to a Name (line 23):

# Assigning a List to a Name (line 23):

# Obtaining an instance of the builtin type 'list' (line 23)
list_33 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 11), 'list')
# Adding type elements to the builtin type 'list' instance (line 23)

# Assigning a type to the variable 'PROTEINS' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'PROTEINS', list_33)

# Assigning a Str to a Name (line 25):

# Assigning a Str to a Name (line 25):
str_34 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 13), 'str', 'ACDEFGHIKLMNPQRSTVWY')
# Assigning a type to the variable 'AMINOACIDS' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'AMINOACIDS', str_34)
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
        name_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 20), 'name')
        # Getting the type of 'self' (line 29)
        self_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self')
        # Setting the type of the member 'name' of a type (line 29)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_36, 'name', name_35)
        
        # Assigning a Name to a Attribute (line 30):
        
        # Assigning a Name to a Attribute (line 30):
        # Getting the type of 'mass' (line 30)
        mass_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 20), 'mass')
        # Getting the type of 'self' (line 30)
        self_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self')
        # Setting the type of the member 'mass' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_38, 'mass', mass_37)
        
        # Assigning a Name to a Attribute (line 31):
        
        # Assigning a Name to a Attribute (line 31):
        # Getting the type of 'isoelectric_point' (line 31)
        isoelectric_point_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 33), 'isoelectric_point')
        # Getting the type of 'self' (line 31)
        self_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self')
        # Setting the type of the member 'isoelectric_point' of a type (line 31)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_40, 'isoelectric_point', isoelectric_point_39)
        
        # Assigning a Name to a Attribute (line 32):
        
        # Assigning a Name to a Attribute (line 32):
        # Getting the type of 'size' (line 32)
        size_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 20), 'size')
        # Getting the type of 'self' (line 32)
        self_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'self')
        # Setting the type of the member 'size' of a type (line 32)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), self_42, 'size', size_41)
        
        # Assigning a Name to a Attribute (line 33):
        
        # Assigning a Name to a Attribute (line 33):
        # Getting the type of 'sequence' (line 33)
        sequence_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 24), 'sequence')
        # Getting the type of 'self' (line 33)
        self_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self')
        # Setting the type of the member 'sequence' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_44, 'sequence', sequence_43)
        
        # Assigning a Name to a Attribute (line 34):
        
        # Assigning a Name to a Attribute (line 34):
        # Getting the type of 'type' (line 34)
        type_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 20), 'type')
        # Getting the type of 'self' (line 34)
        self_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self')
        # Setting the type of the member 'type' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_46, 'type', type_45)
        
        # Call to extract_composition(...): (line 35)
        # Processing the call keyword arguments (line 35)
        kwargs_49 = {}
        # Getting the type of 'self' (line 35)
        self_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'self', False)
        # Obtaining the member 'extract_composition' of a type (line 35)
        extract_composition_48 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), self_47, 'extract_composition')
        # Calling extract_composition(args, kwargs) (line 35)
        extract_composition_call_result_50 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), extract_composition_48, *[], **kwargs_49)
        
        
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
        AMINOACIDS_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 57), 'AMINOACIDS', False)
        comprehension_56 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 39), AMINOACIDS_55)
        # Assigning a type to the variable 'x' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 39), 'x', comprehension_56)
        
        # Obtaining an instance of the builtin type 'tuple' (line 38)
        tuple_52 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 38)
        # Adding element type (line 38)
        # Getting the type of 'x' (line 38)
        x_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 40), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 40), tuple_52, x_53)
        # Adding element type (line 38)
        float_54 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 40), tuple_52, float_54)
        
        list_57 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 39), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 39), list_57, tuple_52)
        # Processing the call keyword arguments (line 38)
        kwargs_58 = {}
        # Getting the type of 'dict' (line 38)
        dict_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 33), 'dict', False)
        # Calling dict(args, kwargs) (line 38)
        dict_call_result_59 = invoke(stypy.reporting.localization.Localization(__file__, 38, 33), dict_51, *[list_57], **kwargs_58)
        
        # Getting the type of 'self' (line 38)
        self_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self')
        # Setting the type of the member 'local_composition' of a type (line 38)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_60, 'local_composition', dict_call_result_59)
        
        
        # Call to range(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'LENGTH' (line 39)
        LENGTH_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 29), 'LENGTH', False)
        # Processing the call keyword arguments (line 39)
        kwargs_63 = {}
        # Getting the type of 'range' (line 39)
        range_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 23), 'range', False)
        # Calling range(args, kwargs) (line 39)
        range_call_result_64 = invoke(stypy.reporting.localization.Localization(__file__, 39, 23), range_61, *[LENGTH_62], **kwargs_63)
        
        # Assigning a type to the variable 'range_call_result_64' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'range_call_result_64', range_call_result_64)
        # Testing if the for loop is going to be iterated (line 39)
        # Testing the type of a for loop iterable (line 39)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 39, 8), range_call_result_64)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 39, 8), range_call_result_64):
            # Getting the type of the for loop variable (line 39)
            for_loop_var_65 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 39, 8), range_call_result_64)
            # Assigning a type to the variable 'counter' (line 39)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'counter', for_loop_var_65)
            # SSA begins for a for statement (line 39)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'self' (line 40)
            self_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'self')
            # Obtaining the member 'local_composition' of a type (line 40)
            local_composition_67 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 12), self_66, 'local_composition')
            
            # Obtaining the type of the subscript
            
            # Obtaining the type of the subscript
            # Getting the type of 'counter' (line 40)
            counter_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 49), 'counter')
            # Getting the type of 'self' (line 40)
            self_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 35), 'self')
            # Obtaining the member 'sequence' of a type (line 40)
            sequence_70 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 35), self_69, 'sequence')
            # Obtaining the member '__getitem__' of a type (line 40)
            getitem___71 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 35), sequence_70, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 40)
            subscript_call_result_72 = invoke(stypy.reporting.localization.Localization(__file__, 40, 35), getitem___71, counter_68)
            
            # Getting the type of 'self' (line 40)
            self_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'self')
            # Obtaining the member 'local_composition' of a type (line 40)
            local_composition_74 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 12), self_73, 'local_composition')
            # Obtaining the member '__getitem__' of a type (line 40)
            getitem___75 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 12), local_composition_74, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 40)
            subscript_call_result_76 = invoke(stypy.reporting.localization.Localization(__file__, 40, 12), getitem___75, subscript_call_result_72)
            
            float_77 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 62), 'float')
            # Getting the type of 'LENGTH' (line 40)
            LENGTH_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 68), 'LENGTH')
            # Applying the binary operator 'div' (line 40)
            result_div_79 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 62), 'div', float_77, LENGTH_78)
            
            # Applying the binary operator '+=' (line 40)
            result_iadd_80 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 12), '+=', subscript_call_result_76, result_div_79)
            # Getting the type of 'self' (line 40)
            self_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'self')
            # Obtaining the member 'local_composition' of a type (line 40)
            local_composition_82 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 12), self_81, 'local_composition')
            
            # Obtaining the type of the subscript
            # Getting the type of 'counter' (line 40)
            counter_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 49), 'counter')
            # Getting the type of 'self' (line 40)
            self_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 35), 'self')
            # Obtaining the member 'sequence' of a type (line 40)
            sequence_85 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 35), self_84, 'sequence')
            # Obtaining the member '__getitem__' of a type (line 40)
            getitem___86 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 35), sequence_85, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 40)
            subscript_call_result_87 = invoke(stypy.reporting.localization.Localization(__file__, 40, 35), getitem___86, counter_83)
            
            # Storing an element on a container (line 40)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 12), local_composition_82, (subscript_call_result_87, result_iadd_80))
            
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
        AMINOACIDS_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 58), 'AMINOACIDS', False)
        comprehension_93 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 40), AMINOACIDS_92)
        # Assigning a type to the variable 'x' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 40), 'x', comprehension_93)
        
        # Obtaining an instance of the builtin type 'tuple' (line 41)
        tuple_89 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 41)
        # Adding element type (line 41)
        # Getting the type of 'x' (line 41)
        x_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 41), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 41), tuple_89, x_90)
        # Adding element type (line 41)
        float_91 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 41), tuple_89, float_91)
        
        list_94 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 40), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 40), list_94, tuple_89)
        # Processing the call keyword arguments (line 41)
        kwargs_95 = {}
        # Getting the type of 'dict' (line 41)
        dict_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 34), 'dict', False)
        # Calling dict(args, kwargs) (line 41)
        dict_call_result_96 = invoke(stypy.reporting.localization.Localization(__file__, 41, 34), dict_88, *[list_94], **kwargs_95)
        
        # Getting the type of 'self' (line 41)
        self_97 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'self')
        # Setting the type of the member 'global_composition' of a type (line 41)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), self_97, 'global_composition', dict_call_result_96)
        
        # Getting the type of 'self' (line 42)
        self_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 25), 'self')
        # Obtaining the member 'sequence' of a type (line 42)
        sequence_99 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 25), self_98, 'sequence')
        # Assigning a type to the variable 'sequence_99' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'sequence_99', sequence_99)
        # Testing if the for loop is going to be iterated (line 42)
        # Testing the type of a for loop iterable (line 42)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 42, 8), sequence_99)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 42, 8), sequence_99):
            # Getting the type of the for loop variable (line 42)
            for_loop_var_100 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 42, 8), sequence_99)
            # Assigning a type to the variable 'aminoacid' (line 42)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'aminoacid', for_loop_var_100)
            # SSA begins for a for statement (line 42)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'self' (line 43)
            self_101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'self')
            # Obtaining the member 'global_composition' of a type (line 43)
            global_composition_102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 12), self_101, 'global_composition')
            
            # Obtaining the type of the subscript
            # Getting the type of 'aminoacid' (line 43)
            aminoacid_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 36), 'aminoacid')
            # Getting the type of 'self' (line 43)
            self_104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'self')
            # Obtaining the member 'global_composition' of a type (line 43)
            global_composition_105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 12), self_104, 'global_composition')
            # Obtaining the member '__getitem__' of a type (line 43)
            getitem___106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 12), global_composition_105, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 43)
            subscript_call_result_107 = invoke(stypy.reporting.localization.Localization(__file__, 43, 12), getitem___106, aminoacid_103)
            
            float_108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 50), 'float')
            
            # Call to len(...): (line 43)
            # Processing the call arguments (line 43)
            # Getting the type of 'self' (line 43)
            self_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 60), 'self', False)
            # Obtaining the member 'sequence' of a type (line 43)
            sequence_111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 60), self_110, 'sequence')
            # Processing the call keyword arguments (line 43)
            kwargs_112 = {}
            # Getting the type of 'len' (line 43)
            len_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 56), 'len', False)
            # Calling len(args, kwargs) (line 43)
            len_call_result_113 = invoke(stypy.reporting.localization.Localization(__file__, 43, 56), len_109, *[sequence_111], **kwargs_112)
            
            # Applying the binary operator 'div' (line 43)
            result_div_114 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 50), 'div', float_108, len_call_result_113)
            
            # Applying the binary operator '+=' (line 43)
            result_iadd_115 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 12), '+=', subscript_call_result_107, result_div_114)
            # Getting the type of 'self' (line 43)
            self_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'self')
            # Obtaining the member 'global_composition' of a type (line 43)
            global_composition_117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 12), self_116, 'global_composition')
            # Getting the type of 'aminoacid' (line 43)
            aminoacid_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 36), 'aminoacid')
            # Storing an element on a container (line 43)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 12), global_composition_117, (aminoacid_118, result_iadd_115))
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'extract_composition(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'extract_composition' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_119)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'extract_composition'
        return stypy_return_type_119


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
        list_120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 46)
        
        # Assigning a type to the variable 'vector' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'vector', list_120)
        
        
        # Call to sorted(...): (line 47)
        # Processing the call arguments (line 47)
        
        # Call to items(...): (line 47)
        # Processing the call keyword arguments (line 47)
        kwargs_125 = {}
        # Getting the type of 'self' (line 47)
        self_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 33), 'self', False)
        # Obtaining the member 'local_composition' of a type (line 47)
        local_composition_123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 33), self_122, 'local_composition')
        # Obtaining the member 'items' of a type (line 47)
        items_124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 33), local_composition_123, 'items')
        # Calling items(args, kwargs) (line 47)
        items_call_result_126 = invoke(stypy.reporting.localization.Localization(__file__, 47, 33), items_124, *[], **kwargs_125)
        
        # Processing the call keyword arguments (line 47)
        kwargs_127 = {}
        # Getting the type of 'sorted' (line 47)
        sorted_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 26), 'sorted', False)
        # Calling sorted(args, kwargs) (line 47)
        sorted_call_result_128 = invoke(stypy.reporting.localization.Localization(__file__, 47, 26), sorted_121, *[items_call_result_126], **kwargs_127)
        
        # Assigning a type to the variable 'sorted_call_result_128' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'sorted_call_result_128', sorted_call_result_128)
        # Testing if the for loop is going to be iterated (line 47)
        # Testing the type of a for loop iterable (line 47)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 47, 8), sorted_call_result_128)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 47, 8), sorted_call_result_128):
            # Getting the type of the for loop variable (line 47)
            for_loop_var_129 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 47, 8), sorted_call_result_128)
            # Assigning a type to the variable 'key' (line 47)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 8), for_loop_var_129, 2, 0))
            # Assigning a type to the variable 'value' (line 47)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 8), for_loop_var_129, 2, 1))
            # SSA begins for a for statement (line 47)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to append(...): (line 48)
            # Processing the call arguments (line 48)
            # Getting the type of 'value' (line 48)
            value_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 26), 'value', False)
            # Processing the call keyword arguments (line 48)
            kwargs_133 = {}
            # Getting the type of 'vector' (line 48)
            vector_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'vector', False)
            # Obtaining the member 'append' of a type (line 48)
            append_131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 12), vector_130, 'append')
            # Calling append(args, kwargs) (line 48)
            append_call_result_134 = invoke(stypy.reporting.localization.Localization(__file__, 48, 12), append_131, *[value_132], **kwargs_133)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to sorted(...): (line 49)
        # Processing the call arguments (line 49)
        
        # Call to keys(...): (line 49)
        # Processing the call keyword arguments (line 49)
        kwargs_139 = {}
        # Getting the type of 'self' (line 49)
        self_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 26), 'self', False)
        # Obtaining the member 'global_composition' of a type (line 49)
        global_composition_137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 26), self_136, 'global_composition')
        # Obtaining the member 'keys' of a type (line 49)
        keys_138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 26), global_composition_137, 'keys')
        # Calling keys(args, kwargs) (line 49)
        keys_call_result_140 = invoke(stypy.reporting.localization.Localization(__file__, 49, 26), keys_138, *[], **kwargs_139)
        
        # Processing the call keyword arguments (line 49)
        kwargs_141 = {}
        # Getting the type of 'sorted' (line 49)
        sorted_135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 'sorted', False)
        # Calling sorted(args, kwargs) (line 49)
        sorted_call_result_142 = invoke(stypy.reporting.localization.Localization(__file__, 49, 19), sorted_135, *[keys_call_result_140], **kwargs_141)
        
        # Assigning a type to the variable 'sorted_call_result_142' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'sorted_call_result_142', sorted_call_result_142)
        # Testing if the for loop is going to be iterated (line 49)
        # Testing the type of a for loop iterable (line 49)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 49, 8), sorted_call_result_142)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 49, 8), sorted_call_result_142):
            # Getting the type of the for loop variable (line 49)
            for_loop_var_143 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 49, 8), sorted_call_result_142)
            # Assigning a type to the variable 'key' (line 49)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'key', for_loop_var_143)
            # SSA begins for a for statement (line 49)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to append(...): (line 50)
            # Processing the call arguments (line 50)
            # Getting the type of 'value' (line 50)
            value_146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 26), 'value', False)
            # Processing the call keyword arguments (line 50)
            kwargs_147 = {}
            # Getting the type of 'vector' (line 50)
            vector_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'vector', False)
            # Obtaining the member 'append' of a type (line 50)
            append_145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), vector_144, 'append')
            # Calling append(args, kwargs) (line 50)
            append_call_result_148 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), append_145, *[value_146], **kwargs_147)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'vector' (line 51)
        vector_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'vector')
        # Assigning a type to the variable 'stypy_return_type' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', vector_149)
        
        # ################# End of 'create_vector(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_vector' in the type store
        # Getting the type of 'stypy_return_type' (line 45)
        stypy_return_type_150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_150)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_vector'
        return stypy_return_type_150


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
    filename_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 20), 'filename', False)
    # Processing the call keyword arguments (line 56)
    kwargs_153 = {}
    # Getting the type of 'open' (line 56)
    open_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 15), 'open', False)
    # Calling open(args, kwargs) (line 56)
    open_call_result_154 = invoke(stypy.reporting.localization.Localization(__file__, 56, 15), open_151, *[filename_152], **kwargs_153)
    
    # Assigning a type to the variable 'protfile' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'protfile', open_call_result_154)
    
    # Getting the type of 'protfile' (line 57)
    protfile_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'protfile')
    # Assigning a type to the variable 'protfile_155' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'protfile_155', protfile_155)
    # Testing if the for loop is going to be iterated (line 57)
    # Testing the type of a for loop iterable (line 57)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 57, 4), protfile_155)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 57, 4), protfile_155):
        # Getting the type of the for loop variable (line 57)
        for_loop_var_156 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 57, 4), protfile_155)
        # Assigning a type to the variable 'line' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'line', for_loop_var_156)
        # SSA begins for a for statement (line 57)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to startswith(...): (line 58)
        # Processing the call arguments (line 58)
        str_159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 27), 'str', 'name')
        # Processing the call keyword arguments (line 58)
        kwargs_160 = {}
        # Getting the type of 'line' (line 58)
        line_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 11), 'line', False)
        # Obtaining the member 'startswith' of a type (line 58)
        startswith_158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 11), line_157, 'startswith')
        # Calling startswith(args, kwargs) (line 58)
        startswith_call_result_161 = invoke(stypy.reporting.localization.Localization(__file__, 58, 11), startswith_158, *[str_159], **kwargs_160)
        
        # Testing if the type of an if condition is none (line 58)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 58, 8), startswith_call_result_161):
            pass
        else:
            
            # Testing the type of an if condition (line 58)
            if_condition_162 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 8), startswith_call_result_161)
            # Assigning a type to the variable 'if_condition_162' (line 58)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'if_condition_162', if_condition_162)
            # SSA begins for if statement (line 58)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 58)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Tuple (line 60):
        
        # Assigning a Call to a Name:
        
        # Call to split(...): (line 60)
        # Processing the call arguments (line 60)
        str_168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 75), 'str', '\t')
        # Processing the call keyword arguments (line 60)
        kwargs_169 = {}
        
        # Call to strip(...): (line 60)
        # Processing the call keyword arguments (line 60)
        kwargs_165 = {}
        # Getting the type of 'line' (line 60)
        line_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 56), 'line', False)
        # Obtaining the member 'strip' of a type (line 60)
        strip_164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 56), line_163, 'strip')
        # Calling strip(args, kwargs) (line 60)
        strip_call_result_166 = invoke(stypy.reporting.localization.Localization(__file__, 60, 56), strip_164, *[], **kwargs_165)
        
        # Obtaining the member 'split' of a type (line 60)
        split_167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 56), strip_call_result_166, 'split')
        # Calling split(args, kwargs) (line 60)
        split_call_result_170 = invoke(stypy.reporting.localization.Localization(__file__, 60, 56), split_167, *[str_168], **kwargs_169)
        
        # Assigning a type to the variable 'call_assignment_1' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_1', split_call_result_170)
        
        # Assigning a Call to a Name (line 60):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 8), 'int')
        # Processing the call keyword arguments
        kwargs_174 = {}
        # Getting the type of 'call_assignment_1' (line 60)
        call_assignment_1_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_1', False)
        # Obtaining the member '__getitem__' of a type (line 60)
        getitem___172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), call_assignment_1_171, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_175 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___172, *[int_173], **kwargs_174)
        
        # Assigning a type to the variable 'call_assignment_2' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_2', getitem___call_result_175)
        
        # Assigning a Name to a Name (line 60):
        # Getting the type of 'call_assignment_2' (line 60)
        call_assignment_2_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_2')
        # Assigning a type to the variable 'name' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'name', call_assignment_2_176)
        
        # Assigning a Call to a Name (line 60):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 8), 'int')
        # Processing the call keyword arguments
        kwargs_180 = {}
        # Getting the type of 'call_assignment_1' (line 60)
        call_assignment_1_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_1', False)
        # Obtaining the member '__getitem__' of a type (line 60)
        getitem___178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), call_assignment_1_177, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_181 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___178, *[int_179], **kwargs_180)
        
        # Assigning a type to the variable 'call_assignment_3' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_3', getitem___call_result_181)
        
        # Assigning a Name to a Name (line 60):
        # Getting the type of 'call_assignment_3' (line 60)
        call_assignment_3_182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_3')
        # Assigning a type to the variable 'mass' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 14), 'mass', call_assignment_3_182)
        
        # Assigning a Call to a Name (line 60):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 8), 'int')
        # Processing the call keyword arguments
        kwargs_186 = {}
        # Getting the type of 'call_assignment_1' (line 60)
        call_assignment_1_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_1', False)
        # Obtaining the member '__getitem__' of a type (line 60)
        getitem___184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), call_assignment_1_183, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_187 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___184, *[int_185], **kwargs_186)
        
        # Assigning a type to the variable 'call_assignment_4' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_4', getitem___call_result_187)
        
        # Assigning a Name to a Name (line 60):
        # Getting the type of 'call_assignment_4' (line 60)
        call_assignment_4_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_4')
        # Assigning a type to the variable 'isoelectric_point' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'isoelectric_point', call_assignment_4_188)
        
        # Assigning a Call to a Name (line 60):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 8), 'int')
        # Processing the call keyword arguments
        kwargs_192 = {}
        # Getting the type of 'call_assignment_1' (line 60)
        call_assignment_1_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_1', False)
        # Obtaining the member '__getitem__' of a type (line 60)
        getitem___190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), call_assignment_1_189, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_193 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___190, *[int_191], **kwargs_192)
        
        # Assigning a type to the variable 'call_assignment_5' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_5', getitem___call_result_193)
        
        # Assigning a Name to a Name (line 60):
        # Getting the type of 'call_assignment_5' (line 60)
        call_assignment_5_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_5')
        # Assigning a type to the variable 'size' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 39), 'size', call_assignment_5_194)
        
        # Assigning a Call to a Name (line 60):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 8), 'int')
        # Processing the call keyword arguments
        kwargs_198 = {}
        # Getting the type of 'call_assignment_1' (line 60)
        call_assignment_1_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_1', False)
        # Obtaining the member '__getitem__' of a type (line 60)
        getitem___196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), call_assignment_1_195, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_199 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___196, *[int_197], **kwargs_198)
        
        # Assigning a type to the variable 'call_assignment_6' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_6', getitem___call_result_199)
        
        # Assigning a Name to a Name (line 60):
        # Getting the type of 'call_assignment_6' (line 60)
        call_assignment_6_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_6')
        # Assigning a type to the variable 'sequence' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 45), 'sequence', call_assignment_6_200)
        
        # Assigning a Call to a Name (line 61):
        
        # Assigning a Call to a Name (line 61):
        
        # Call to Protein(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'name' (line 61)
        name_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 26), 'name', False)
        # Getting the type of 'mass' (line 61)
        mass_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 32), 'mass', False)
        # Getting the type of 'isoelectric_point' (line 61)
        isoelectric_point_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 38), 'isoelectric_point', False)
        # Getting the type of 'size' (line 61)
        size_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 57), 'size', False)
        # Getting the type of 'sequence' (line 61)
        sequence_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 63), 'sequence', False)
        # Getting the type of 'type' (line 61)
        type_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 73), 'type', False)
        # Processing the call keyword arguments (line 61)
        kwargs_208 = {}
        # Getting the type of 'Protein' (line 61)
        Protein_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 18), 'Protein', False)
        # Calling Protein(args, kwargs) (line 61)
        Protein_call_result_209 = invoke(stypy.reporting.localization.Localization(__file__, 61, 18), Protein_201, *[name_202, mass_203, isoelectric_point_204, size_205, sequence_206, type_207], **kwargs_208)
        
        # Assigning a type to the variable 'protein' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'protein', Protein_call_result_209)
        
        # Call to append(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'protein' (line 62)
        protein_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 24), 'protein', False)
        # Processing the call keyword arguments (line 62)
        kwargs_213 = {}
        # Getting the type of 'PROTEINS' (line 62)
        PROTEINS_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'PROTEINS', False)
        # Obtaining the member 'append' of a type (line 62)
        append_211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), PROTEINS_210, 'append')
        # Calling append(args, kwargs) (line 62)
        append_call_result_214 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), append_211, *[protein_212], **kwargs_213)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Call to close(...): (line 63)
    # Processing the call keyword arguments (line 63)
    kwargs_217 = {}
    # Getting the type of 'protfile' (line 63)
    protfile_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'protfile', False)
    # Obtaining the member 'close' of a type (line 63)
    close_216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 4), protfile_215, 'close')
    # Calling close(args, kwargs) (line 63)
    close_call_result_218 = invoke(stypy.reporting.localization.Localization(__file__, 63, 4), close_216, *[], **kwargs_217)
    
    
    # ################# End of 'load_file(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'load_file' in the type store
    # Getting the type of 'stypy_return_type' (line 54)
    stypy_return_type_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_219)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'load_file'
    return stypy_return_type_219

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

    str_220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 4), 'str', 'Create the feature and label tables.')
    
    # Assigning a List to a Name (line 68):
    
    # Assigning a List to a Name (line 68):
    
    # Obtaining an instance of the builtin type 'list' (line 68)
    list_221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 68)
    
    # Assigning a type to the variable 'feature_table' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'feature_table', list_221)
    
    # Assigning a List to a Name (line 69):
    
    # Assigning a List to a Name (line 69):
    
    # Obtaining an instance of the builtin type 'list' (line 69)
    list_222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 69)
    
    # Assigning a type to the variable 'label_table' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'label_table', list_222)
    
    # Getting the type of 'PROTEINS' (line 71)
    PROTEINS_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'PROTEINS')
    # Assigning a type to the variable 'PROTEINS_223' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'PROTEINS_223', PROTEINS_223)
    # Testing if the for loop is going to be iterated (line 71)
    # Testing the type of a for loop iterable (line 71)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 71, 4), PROTEINS_223)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 71, 4), PROTEINS_223):
        # Getting the type of the for loop variable (line 71)
        for_loop_var_224 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 71, 4), PROTEINS_223)
        # Assigning a type to the variable 'protein' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'protein', for_loop_var_224)
        # SSA begins for a for statement (line 71)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 72)
        # Processing the call arguments (line 72)
        
        # Call to create_vector(...): (line 72)
        # Processing the call keyword arguments (line 72)
        kwargs_229 = {}
        # Getting the type of 'protein' (line 72)
        protein_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 29), 'protein', False)
        # Obtaining the member 'create_vector' of a type (line 72)
        create_vector_228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 29), protein_227, 'create_vector')
        # Calling create_vector(args, kwargs) (line 72)
        create_vector_call_result_230 = invoke(stypy.reporting.localization.Localization(__file__, 72, 29), create_vector_228, *[], **kwargs_229)
        
        # Processing the call keyword arguments (line 72)
        kwargs_231 = {}
        # Getting the type of 'feature_table' (line 72)
        feature_table_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'feature_table', False)
        # Obtaining the member 'append' of a type (line 72)
        append_226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), feature_table_225, 'append')
        # Calling append(args, kwargs) (line 72)
        append_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), append_226, *[create_vector_call_result_230], **kwargs_231)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Getting the type of 'PROTEINS' (line 74)
    PROTEINS_233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 19), 'PROTEINS')
    # Assigning a type to the variable 'PROTEINS_233' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'PROTEINS_233', PROTEINS_233)
    # Testing if the for loop is going to be iterated (line 74)
    # Testing the type of a for loop iterable (line 74)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 74, 4), PROTEINS_233)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 74, 4), PROTEINS_233):
        # Getting the type of the for loop variable (line 74)
        for_loop_var_234 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 74, 4), PROTEINS_233)
        # Assigning a type to the variable 'protein' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'protein', for_loop_var_234)
        # SSA begins for a for statement (line 74)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'protein' (line 75)
        protein_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 11), 'protein')
        # Obtaining the member 'type' of a type (line 75)
        type_236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 11), protein_235, 'type')
        # Getting the type of 'BLIND' (line 75)
        BLIND_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 27), 'BLIND')
        # Applying the binary operator '==' (line 75)
        result_eq_238 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 11), '==', type_236, BLIND_237)
        
        # Testing if the type of an if condition is none (line 75)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 8), result_eq_238):
            pass
        else:
            
            # Testing the type of an if condition (line 75)
            if_condition_239 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 8), result_eq_238)
            # Assigning a type to the variable 'if_condition_239' (line 75)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'if_condition_239', if_condition_239)
            # SSA begins for if statement (line 75)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 75)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 77):
        
        # Assigning a BinOp to a Name (line 77):
        
        # Obtaining an instance of the builtin type 'list' (line 77)
        list_240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 77)
        # Adding element type (line 77)
        int_241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 17), list_240, int_241)
        
        int_242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 24), 'int')
        # Applying the binary operator '*' (line 77)
        result_mul_243 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 17), '*', list_240, int_242)
        
        # Assigning a type to the variable 'labels' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'labels', result_mul_243)
        
        # Getting the type of 'labels' (line 79)
        labels_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'labels')
        
        # Obtaining the type of the subscript
        # Getting the type of 'protein' (line 79)
        protein_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 15), 'protein')
        # Obtaining the member 'type' of a type (line 79)
        type_246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 15), protein_245, 'type')
        # Getting the type of 'labels' (line 79)
        labels_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'labels')
        # Obtaining the member '__getitem__' of a type (line 79)
        getitem___248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), labels_247, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 79)
        subscript_call_result_249 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), getitem___248, type_246)
        
        int_250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 32), 'int')
        # Applying the binary operator '*=' (line 79)
        result_imul_251 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 8), '*=', subscript_call_result_249, int_250)
        # Getting the type of 'labels' (line 79)
        labels_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'labels')
        # Getting the type of 'protein' (line 79)
        protein_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 15), 'protein')
        # Obtaining the member 'type' of a type (line 79)
        type_254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 15), protein_253, 'type')
        # Storing an element on a container (line 79)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 8), labels_252, (type_254, result_imul_251))
        
        
        # Call to append(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'labels' (line 80)
        labels_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 27), 'labels', False)
        # Processing the call keyword arguments (line 80)
        kwargs_258 = {}
        # Getting the type of 'label_table' (line 80)
        label_table_255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'label_table', False)
        # Obtaining the member 'append' of a type (line 80)
        append_256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), label_table_255, 'append')
        # Calling append(args, kwargs) (line 80)
        append_call_result_259 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), append_256, *[labels_257], **kwargs_258)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Obtaining an instance of the builtin type 'tuple' (line 82)
    tuple_260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 82)
    # Adding element type (line 82)
    # Getting the type of 'feature_table' (line 82)
    feature_table_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 11), 'feature_table')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 11), tuple_260, feature_table_261)
    # Adding element type (line 82)
    # Getting the type of 'label_table' (line 82)
    label_table_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'label_table')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 11), tuple_260, label_table_262)
    
    # Assigning a type to the variable 'stypy_return_type' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type', tuple_260)
    
    # ################# End of 'create_tables(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_tables' in the type store
    # Getting the type of 'stypy_return_type' (line 66)
    stypy_return_type_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_263)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_tables'
    return stypy_return_type_263

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
    list_264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 86)
    
    # Assigning a type to the variable 'kernel_table' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'kernel_table', list_264)
    
    # Getting the type of 'feature_table' (line 87)
    feature_table_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'feature_table')
    # Assigning a type to the variable 'feature_table_265' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'feature_table_265', feature_table_265)
    # Testing if the for loop is going to be iterated (line 87)
    # Testing the type of a for loop iterable (line 87)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 87, 4), feature_table_265)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 87, 4), feature_table_265):
        # Getting the type of the for loop variable (line 87)
        for_loop_var_266 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 87, 4), feature_table_265)
        # Assigning a type to the variable 'row' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'row', for_loop_var_266)
        # SSA begins for a for statement (line 87)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a List to a Name (line 88):
        
        # Assigning a List to a Name (line 88):
        
        # Obtaining an instance of the builtin type 'list' (line 88)
        list_267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 88)
        
        # Assigning a type to the variable 'kernel_row' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'kernel_row', list_267)
        
        # Getting the type of 'feature_table' (line 89)
        feature_table_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 25), 'feature_table')
        # Assigning a type to the variable 'feature_table_268' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'feature_table_268', feature_table_268)
        # Testing if the for loop is going to be iterated (line 89)
        # Testing the type of a for loop iterable (line 89)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 89, 8), feature_table_268)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 89, 8), feature_table_268):
            # Getting the type of the for loop variable (line 89)
            for_loop_var_269 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 89, 8), feature_table_268)
            # Assigning a type to the variable 'candidate' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'candidate', for_loop_var_269)
            # SSA begins for a for statement (line 89)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Num to a Name (line 90):
            
            # Assigning a Num to a Name (line 90):
            float_270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 25), 'float')
            # Assigning a type to the variable 'difference' (line 90)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'difference', float_270)
            
            
            # Call to range(...): (line 91)
            # Processing the call arguments (line 91)
            
            # Call to len(...): (line 91)
            # Processing the call arguments (line 91)
            # Getting the type of 'row' (line 91)
            row_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 37), 'row', False)
            # Processing the call keyword arguments (line 91)
            kwargs_274 = {}
            # Getting the type of 'len' (line 91)
            len_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 33), 'len', False)
            # Calling len(args, kwargs) (line 91)
            len_call_result_275 = invoke(stypy.reporting.localization.Localization(__file__, 91, 33), len_272, *[row_273], **kwargs_274)
            
            # Processing the call keyword arguments (line 91)
            kwargs_276 = {}
            # Getting the type of 'range' (line 91)
            range_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 27), 'range', False)
            # Calling range(args, kwargs) (line 91)
            range_call_result_277 = invoke(stypy.reporting.localization.Localization(__file__, 91, 27), range_271, *[len_call_result_275], **kwargs_276)
            
            # Assigning a type to the variable 'range_call_result_277' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'range_call_result_277', range_call_result_277)
            # Testing if the for loop is going to be iterated (line 91)
            # Testing the type of a for loop iterable (line 91)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 91, 12), range_call_result_277)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 91, 12), range_call_result_277):
                # Getting the type of the for loop variable (line 91)
                for_loop_var_278 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 91, 12), range_call_result_277)
                # Assigning a type to the variable 'counter' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'counter', for_loop_var_278)
                # SSA begins for a for statement (line 91)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'difference' (line 92)
                difference_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'difference')
                
                # Obtaining the type of the subscript
                # Getting the type of 'counter' (line 92)
                counter_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 35), 'counter')
                # Getting the type of 'row' (line 92)
                row_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 31), 'row')
                # Obtaining the member '__getitem__' of a type (line 92)
                getitem___282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 31), row_281, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 92)
                subscript_call_result_283 = invoke(stypy.reporting.localization.Localization(__file__, 92, 31), getitem___282, counter_280)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'counter' (line 92)
                counter_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 56), 'counter')
                # Getting the type of 'candidate' (line 92)
                candidate_285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 46), 'candidate')
                # Obtaining the member '__getitem__' of a type (line 92)
                getitem___286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 46), candidate_285, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 92)
                subscript_call_result_287 = invoke(stypy.reporting.localization.Localization(__file__, 92, 46), getitem___286, counter_284)
                
                # Applying the binary operator '-' (line 92)
                result_sub_288 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 31), '-', subscript_call_result_283, subscript_call_result_287)
                
                int_289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 69), 'int')
                # Applying the binary operator '**' (line 92)
                result_pow_290 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 30), '**', result_sub_288, int_289)
                
                # Applying the binary operator '+=' (line 92)
                result_iadd_291 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 16), '+=', difference_279, result_pow_290)
                # Assigning a type to the variable 'difference' (line 92)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'difference', result_iadd_291)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Call to append(...): (line 93)
            # Processing the call arguments (line 93)
            
            # Call to exp(...): (line 93)
            # Processing the call arguments (line 93)
            
            # Getting the type of 'D' (line 93)
            D_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 35), 'D', False)
            # Applying the 'usub' unary operator (line 93)
            result___neg___296 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 34), 'usub', D_295)
            
            # Getting the type of 'difference' (line 93)
            difference_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 37), 'difference', False)
            # Applying the binary operator '*' (line 93)
            result_mul_298 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 34), '*', result___neg___296, difference_297)
            
            # Processing the call keyword arguments (line 93)
            kwargs_299 = {}
            # Getting the type of 'exp' (line 93)
            exp_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 30), 'exp', False)
            # Calling exp(args, kwargs) (line 93)
            exp_call_result_300 = invoke(stypy.reporting.localization.Localization(__file__, 93, 30), exp_294, *[result_mul_298], **kwargs_299)
            
            # Processing the call keyword arguments (line 93)
            kwargs_301 = {}
            # Getting the type of 'kernel_row' (line 93)
            kernel_row_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'kernel_row', False)
            # Obtaining the member 'append' of a type (line 93)
            append_293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 12), kernel_row_292, 'append')
            # Calling append(args, kwargs) (line 93)
            append_call_result_302 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), append_293, *[exp_call_result_300], **kwargs_301)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to append(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'kernel_row' (line 94)
        kernel_row_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 28), 'kernel_row', False)
        # Processing the call keyword arguments (line 94)
        kwargs_306 = {}
        # Getting the type of 'kernel_table' (line 94)
        kernel_table_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'kernel_table', False)
        # Obtaining the member 'append' of a type (line 94)
        append_304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), kernel_table_303, 'append')
        # Calling append(args, kwargs) (line 94)
        append_call_result_307 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), append_304, *[kernel_row_305], **kwargs_306)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'kernel_table' (line 95)
    kernel_table_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 11), 'kernel_table')
    # Assigning a type to the variable 'stypy_return_type' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'stypy_return_type', kernel_table_308)
    
    # ################# End of 'create_kernel_table(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_kernel_table' in the type store
    # Getting the type of 'stypy_return_type' (line 85)
    stypy_return_type_309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_309)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_kernel_table'
    return stypy_return_type_309

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
    float_310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 16), 'float')
    # Assigning a type to the variable 'tolerance' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'tolerance', float_310)
    
    # Assigning a ListComp to a Name (line 100):
    
    # Assigning a ListComp to a Name (line 100):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 100)
    # Processing the call arguments (line 100)
    
    # Call to len(...): (line 100)
    # Processing the call arguments (line 100)
    
    # Obtaining the type of the subscript
    int_320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 73), 'int')
    # Getting the type of 'label_table' (line 100)
    label_table_321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 61), 'label_table', False)
    # Obtaining the member '__getitem__' of a type (line 100)
    getitem___322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 61), label_table_321, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 100)
    subscript_call_result_323 = invoke(stypy.reporting.localization.Localization(__file__, 100, 61), getitem___322, int_320)
    
    # Processing the call keyword arguments (line 100)
    kwargs_324 = {}
    # Getting the type of 'len' (line 100)
    len_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 57), 'len', False)
    # Calling len(args, kwargs) (line 100)
    len_call_result_325 = invoke(stypy.reporting.localization.Localization(__file__, 100, 57), len_319, *[subscript_call_result_323], **kwargs_324)
    
    # Processing the call keyword arguments (line 100)
    kwargs_326 = {}
    # Getting the type of 'range' (line 100)
    range_318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 51), 'range', False)
    # Calling range(args, kwargs) (line 100)
    range_call_result_327 = invoke(stypy.reporting.localization.Localization(__file__, 100, 51), range_318, *[len_call_result_325], **kwargs_326)
    
    comprehension_328 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 14), range_call_result_327)
    # Assigning a type to the variable '_' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 14), '_', comprehension_328)
    
    # Obtaining an instance of the builtin type 'list' (line 100)
    list_311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 100)
    # Adding element type (line 100)
    float_312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 16), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 15), list_311, float_312)
    
    
    # Call to len(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'kernel_table' (line 100)
    kernel_table_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 27), 'kernel_table', False)
    # Processing the call keyword arguments (line 100)
    kwargs_315 = {}
    # Getting the type of 'len' (line 100)
    len_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 23), 'len', False)
    # Calling len(args, kwargs) (line 100)
    len_call_result_316 = invoke(stypy.reporting.localization.Localization(__file__, 100, 23), len_313, *[kernel_table_314], **kwargs_315)
    
    # Applying the binary operator '*' (line 100)
    result_mul_317 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 15), '*', list_311, len_call_result_316)
    
    list_329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 14), list_329, result_mul_317)
    # Assigning a type to the variable 'alphas' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'alphas', list_329)
    
    # Assigning a ListComp to a Name (line 101):
    
    # Assigning a ListComp to a Name (line 101):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 101)
    # Processing the call arguments (line 101)
    
    # Call to len(...): (line 101)
    # Processing the call arguments (line 101)
    
    # Obtaining the type of the subscript
    int_339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 72), 'int')
    # Getting the type of 'label_table' (line 101)
    label_table_340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 60), 'label_table', False)
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 60), label_table_340, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_342 = invoke(stypy.reporting.localization.Localization(__file__, 101, 60), getitem___341, int_339)
    
    # Processing the call keyword arguments (line 101)
    kwargs_343 = {}
    # Getting the type of 'len' (line 101)
    len_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 56), 'len', False)
    # Calling len(args, kwargs) (line 101)
    len_call_result_344 = invoke(stypy.reporting.localization.Localization(__file__, 101, 56), len_338, *[subscript_call_result_342], **kwargs_343)
    
    # Processing the call keyword arguments (line 101)
    kwargs_345 = {}
    # Getting the type of 'range' (line 101)
    range_337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 50), 'range', False)
    # Calling range(args, kwargs) (line 101)
    range_call_result_346 = invoke(stypy.reporting.localization.Localization(__file__, 101, 50), range_337, *[len_call_result_344], **kwargs_345)
    
    comprehension_347 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 13), range_call_result_346)
    # Assigning a type to the variable '_' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 13), '_', comprehension_347)
    
    # Obtaining an instance of the builtin type 'list' (line 101)
    list_330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 101)
    # Adding element type (line 101)
    float_331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 15), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 14), list_330, float_331)
    
    
    # Call to len(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'kernel_table' (line 101)
    kernel_table_333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 26), 'kernel_table', False)
    # Processing the call keyword arguments (line 101)
    kwargs_334 = {}
    # Getting the type of 'len' (line 101)
    len_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 22), 'len', False)
    # Calling len(args, kwargs) (line 101)
    len_call_result_335 = invoke(stypy.reporting.localization.Localization(__file__, 101, 22), len_332, *[kernel_table_333], **kwargs_334)
    
    # Applying the binary operator '*' (line 101)
    result_mul_336 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 14), '*', list_330, len_call_result_335)
    
    list_348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 13), list_348, result_mul_336)
    # Assigning a type to the variable 'betas' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'betas', list_348)
    
    # Assigning a BinOp to a Name (line 102):
    
    # Assigning a BinOp to a Name (line 102):
    
    # Obtaining an instance of the builtin type 'list' (line 102)
    list_349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 102)
    # Adding element type (line 102)
    float_350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 12), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 11), list_349, float_350)
    
    
    # Call to len(...): (line 102)
    # Processing the call arguments (line 102)
    
    # Obtaining the type of the subscript
    int_352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 35), 'int')
    # Getting the type of 'label_table' (line 102)
    label_table_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 23), 'label_table', False)
    # Obtaining the member '__getitem__' of a type (line 102)
    getitem___354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 23), label_table_353, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 102)
    subscript_call_result_355 = invoke(stypy.reporting.localization.Localization(__file__, 102, 23), getitem___354, int_352)
    
    # Processing the call keyword arguments (line 102)
    kwargs_356 = {}
    # Getting the type of 'len' (line 102)
    len_351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 19), 'len', False)
    # Calling len(args, kwargs) (line 102)
    len_call_result_357 = invoke(stypy.reporting.localization.Localization(__file__, 102, 19), len_351, *[subscript_call_result_355], **kwargs_356)
    
    # Applying the binary operator '*' (line 102)
    result_mul_358 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 11), '*', list_349, len_call_result_357)
    
    # Assigning a type to the variable 'bias' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'bias', result_mul_358)
    
    # Assigning a BinOp to a Name (line 103):
    
    # Assigning a BinOp to a Name (line 103):
    
    # Obtaining an instance of the builtin type 'list' (line 103)
    list_359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 103)
    # Adding element type (line 103)
    float_360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 19), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 18), list_359, float_360)
    
    
    # Call to len(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'kernel_table' (line 103)
    kernel_table_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 30), 'kernel_table', False)
    # Processing the call keyword arguments (line 103)
    kwargs_363 = {}
    # Getting the type of 'len' (line 103)
    len_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 26), 'len', False)
    # Calling len(args, kwargs) (line 103)
    len_call_result_364 = invoke(stypy.reporting.localization.Localization(__file__, 103, 26), len_361, *[kernel_table_362], **kwargs_363)
    
    # Applying the binary operator '*' (line 103)
    result_mul_365 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 18), '*', list_359, len_call_result_364)
    
    # Assigning a type to the variable 'labelalphas' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'labelalphas', result_mul_365)
    
    # Assigning a BinOp to a Name (line 104):
    
    # Assigning a BinOp to a Name (line 104):
    
    # Obtaining an instance of the builtin type 'list' (line 104)
    list_366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 104)
    # Adding element type (line 104)
    
    # Obtaining an instance of the builtin type 'tuple' (line 104)
    tuple_367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 104)
    # Adding element type (line 104)
    float_368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 24), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 24), tuple_367, float_368)
    # Adding element type (line 104)
    int_369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 24), tuple_367, int_369)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 22), list_366, tuple_367)
    
    
    # Call to len(...): (line 104)
    # Processing the call arguments (line 104)
    
    # Obtaining the type of the subscript
    int_371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 51), 'int')
    # Getting the type of 'label_table' (line 104)
    label_table_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 39), 'label_table', False)
    # Obtaining the member '__getitem__' of a type (line 104)
    getitem___373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 39), label_table_372, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 104)
    subscript_call_result_374 = invoke(stypy.reporting.localization.Localization(__file__, 104, 39), getitem___373, int_371)
    
    # Processing the call keyword arguments (line 104)
    kwargs_375 = {}
    # Getting the type of 'len' (line 104)
    len_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 35), 'len', False)
    # Calling len(args, kwargs) (line 104)
    len_call_result_376 = invoke(stypy.reporting.localization.Localization(__file__, 104, 35), len_370, *[subscript_call_result_374], **kwargs_375)
    
    # Applying the binary operator '*' (line 104)
    result_mul_377 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 22), '*', list_366, len_call_result_376)
    
    # Assigning a type to the variable 'max_differences' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'max_differences', result_mul_377)
    
    
    # Call to range(...): (line 105)
    # Processing the call arguments (line 105)
    int_379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 27), 'int')
    
    # Call to len(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'kernel_table' (line 105)
    kernel_table_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 34), 'kernel_table', False)
    # Processing the call keyword arguments (line 105)
    kwargs_382 = {}
    # Getting the type of 'len' (line 105)
    len_380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 30), 'len', False)
    # Calling len(args, kwargs) (line 105)
    len_call_result_383 = invoke(stypy.reporting.localization.Localization(__file__, 105, 30), len_380, *[kernel_table_381], **kwargs_382)
    
    # Applying the binary operator '*' (line 105)
    result_mul_384 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 27), '*', int_379, len_call_result_383)
    
    # Processing the call keyword arguments (line 105)
    kwargs_385 = {}
    # Getting the type of 'range' (line 105)
    range_378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 21), 'range', False)
    # Calling range(args, kwargs) (line 105)
    range_call_result_386 = invoke(stypy.reporting.localization.Localization(__file__, 105, 21), range_378, *[result_mul_384], **kwargs_385)
    
    # Assigning a type to the variable 'range_call_result_386' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'range_call_result_386', range_call_result_386)
    # Testing if the for loop is going to be iterated (line 105)
    # Testing the type of a for loop iterable (line 105)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 105, 4), range_call_result_386)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 105, 4), range_call_result_386):
        # Getting the type of the for loop variable (line 105)
        for_loop_var_387 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 105, 4), range_call_result_386)
        # Assigning a type to the variable 'iteration' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'iteration', for_loop_var_387)
        # SSA begins for a for statement (line 105)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'iteration' (line 107)
        iteration_388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 11), 'iteration')
        int_389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 24), 'int')
        # Applying the binary operator '==' (line 107)
        result_eq_390 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 11), '==', iteration_388, int_389)
        
        # Testing if the type of an if condition is none (line 107)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 107, 8), result_eq_390):
            pass
        else:
            
            # Testing the type of an if condition (line 107)
            if_condition_391 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 8), result_eq_390)
            # Assigning a type to the variable 'if_condition_391' (line 107)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'if_condition_391', if_condition_391)
            # SSA begins for if statement (line 107)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining an instance of the builtin type 'tuple' (line 108)
            tuple_392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 108)
            # Adding element type (line 108)
            # Getting the type of 'alphas' (line 108)
            alphas_393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 19), 'alphas')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 19), tuple_392, alphas_393)
            # Adding element type (line 108)
            # Getting the type of 'bias' (line 108)
            bias_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 27), 'bias')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 19), tuple_392, bias_394)
            
            # Assigning a type to the variable 'stypy_return_type' (line 108)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'stypy_return_type', tuple_392)
            # SSA join for if statement (line 107)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to range(...): (line 109)
        # Processing the call arguments (line 109)
        
        # Call to len(...): (line 109)
        # Processing the call arguments (line 109)
        
        # Obtaining the type of the subscript
        int_397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 43), 'int')
        # Getting the type of 'label_table' (line 109)
        label_table_398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 31), 'label_table', False)
        # Obtaining the member '__getitem__' of a type (line 109)
        getitem___399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 31), label_table_398, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 109)
        subscript_call_result_400 = invoke(stypy.reporting.localization.Localization(__file__, 109, 31), getitem___399, int_397)
        
        # Processing the call keyword arguments (line 109)
        kwargs_401 = {}
        # Getting the type of 'len' (line 109)
        len_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'len', False)
        # Calling len(args, kwargs) (line 109)
        len_call_result_402 = invoke(stypy.reporting.localization.Localization(__file__, 109, 27), len_396, *[subscript_call_result_400], **kwargs_401)
        
        # Processing the call keyword arguments (line 109)
        kwargs_403 = {}
        # Getting the type of 'range' (line 109)
        range_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 21), 'range', False)
        # Calling range(args, kwargs) (line 109)
        range_call_result_404 = invoke(stypy.reporting.localization.Localization(__file__, 109, 21), range_395, *[len_call_result_402], **kwargs_403)
        
        # Assigning a type to the variable 'range_call_result_404' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'range_call_result_404', range_call_result_404)
        # Testing if the for loop is going to be iterated (line 109)
        # Testing the type of a for loop iterable (line 109)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 109, 8), range_call_result_404)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 109, 8), range_call_result_404):
            # Getting the type of the for loop variable (line 109)
            for_loop_var_405 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 109, 8), range_call_result_404)
            # Assigning a type to the variable 'klass' (line 109)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'klass', for_loop_var_405)
            # SSA begins for a for statement (line 109)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Tuple to a Subscript (line 110):
            
            # Assigning a Tuple to a Subscript (line 110):
            
            # Obtaining an instance of the builtin type 'tuple' (line 110)
            tuple_406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 38), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 110)
            # Adding element type (line 110)
            float_407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 38), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 38), tuple_406, float_407)
            # Adding element type (line 110)
            int_408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 43), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 38), tuple_406, int_408)
            
            # Getting the type of 'max_differences' (line 110)
            max_differences_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'max_differences')
            # Getting the type of 'klass' (line 110)
            klass_410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 28), 'klass')
            # Storing an element on a container (line 110)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 12), max_differences_409, (klass_410, tuple_406))
            
            
            # Call to range(...): (line 111)
            # Processing the call arguments (line 111)
            
            # Call to len(...): (line 111)
            # Processing the call arguments (line 111)
            # Getting the type of 'kernel_table' (line 111)
            kernel_table_413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 34), 'kernel_table', False)
            # Processing the call keyword arguments (line 111)
            kwargs_414 = {}
            # Getting the type of 'len' (line 111)
            len_412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 30), 'len', False)
            # Calling len(args, kwargs) (line 111)
            len_call_result_415 = invoke(stypy.reporting.localization.Localization(__file__, 111, 30), len_412, *[kernel_table_413], **kwargs_414)
            
            # Processing the call keyword arguments (line 111)
            kwargs_416 = {}
            # Getting the type of 'range' (line 111)
            range_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'range', False)
            # Calling range(args, kwargs) (line 111)
            range_call_result_417 = invoke(stypy.reporting.localization.Localization(__file__, 111, 24), range_411, *[len_call_result_415], **kwargs_416)
            
            # Assigning a type to the variable 'range_call_result_417' (line 111)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'range_call_result_417', range_call_result_417)
            # Testing if the for loop is going to be iterated (line 111)
            # Testing the type of a for loop iterable (line 111)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 111, 12), range_call_result_417)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 111, 12), range_call_result_417):
                # Getting the type of the for loop variable (line 111)
                for_loop_var_418 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 111, 12), range_call_result_417)
                # Assigning a type to the variable 'elem' (line 111)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'elem', for_loop_var_418)
                # SSA begins for a for statement (line 111)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a BinOp to a Subscript (line 112):
                
                # Assigning a BinOp to a Subscript (line 112):
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 112)
                klass_419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 54), 'klass')
                
                # Obtaining the type of the subscript
                # Getting the type of 'elem' (line 112)
                elem_420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 48), 'elem')
                # Getting the type of 'label_table' (line 112)
                label_table_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 36), 'label_table')
                # Obtaining the member '__getitem__' of a type (line 112)
                getitem___422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 36), label_table_421, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 112)
                subscript_call_result_423 = invoke(stypy.reporting.localization.Localization(__file__, 112, 36), getitem___422, elem_420)
                
                # Obtaining the member '__getitem__' of a type (line 112)
                getitem___424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 36), subscript_call_result_423, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 112)
                subscript_call_result_425 = invoke(stypy.reporting.localization.Localization(__file__, 112, 36), getitem___424, klass_419)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'elem' (line 112)
                elem_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 77), 'elem')
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 112)
                klass_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 70), 'klass')
                # Getting the type of 'alphas' (line 112)
                alphas_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 63), 'alphas')
                # Obtaining the member '__getitem__' of a type (line 112)
                getitem___429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 63), alphas_428, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 112)
                subscript_call_result_430 = invoke(stypy.reporting.localization.Localization(__file__, 112, 63), getitem___429, klass_427)
                
                # Obtaining the member '__getitem__' of a type (line 112)
                getitem___431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 63), subscript_call_result_430, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 112)
                subscript_call_result_432 = invoke(stypy.reporting.localization.Localization(__file__, 112, 63), getitem___431, elem_426)
                
                # Applying the binary operator '*' (line 112)
                result_mul_433 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 36), '*', subscript_call_result_425, subscript_call_result_432)
                
                # Getting the type of 'labelalphas' (line 112)
                labelalphas_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'labelalphas')
                # Getting the type of 'elem' (line 112)
                elem_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 28), 'elem')
                # Storing an element on a container (line 112)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 16), labelalphas_434, (elem_435, result_mul_433))
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            
            # Call to range(...): (line 113)
            # Processing the call arguments (line 113)
            
            # Call to len(...): (line 113)
            # Processing the call arguments (line 113)
            # Getting the type of 'kernel_table' (line 113)
            kernel_table_438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 41), 'kernel_table', False)
            # Processing the call keyword arguments (line 113)
            kwargs_439 = {}
            # Getting the type of 'len' (line 113)
            len_437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 37), 'len', False)
            # Calling len(args, kwargs) (line 113)
            len_call_result_440 = invoke(stypy.reporting.localization.Localization(__file__, 113, 37), len_437, *[kernel_table_438], **kwargs_439)
            
            # Processing the call keyword arguments (line 113)
            kwargs_441 = {}
            # Getting the type of 'range' (line 113)
            range_436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 31), 'range', False)
            # Calling range(args, kwargs) (line 113)
            range_call_result_442 = invoke(stypy.reporting.localization.Localization(__file__, 113, 31), range_436, *[len_call_result_440], **kwargs_441)
            
            # Assigning a type to the variable 'range_call_result_442' (line 113)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'range_call_result_442', range_call_result_442)
            # Testing if the for loop is going to be iterated (line 113)
            # Testing the type of a for loop iterable (line 113)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 113, 12), range_call_result_442)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 113, 12), range_call_result_442):
                # Getting the type of the for loop variable (line 113)
                for_loop_var_443 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 113, 12), range_call_result_442)
                # Assigning a type to the variable 'col_counter' (line 113)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'col_counter', for_loop_var_443)
                # SSA begins for a for statement (line 113)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Num to a Name (line 114):
                
                # Assigning a Num to a Name (line 114):
                float_444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 29), 'float')
                # Assigning a type to the variable 'prediction' (line 114)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'prediction', float_444)
                
                
                # Call to range(...): (line 115)
                # Processing the call arguments (line 115)
                
                # Call to len(...): (line 115)
                # Processing the call arguments (line 115)
                # Getting the type of 'kernel_table' (line 115)
                kernel_table_447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 45), 'kernel_table', False)
                # Processing the call keyword arguments (line 115)
                kwargs_448 = {}
                # Getting the type of 'len' (line 115)
                len_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 41), 'len', False)
                # Calling len(args, kwargs) (line 115)
                len_call_result_449 = invoke(stypy.reporting.localization.Localization(__file__, 115, 41), len_446, *[kernel_table_447], **kwargs_448)
                
                # Processing the call keyword arguments (line 115)
                kwargs_450 = {}
                # Getting the type of 'range' (line 115)
                range_445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 35), 'range', False)
                # Calling range(args, kwargs) (line 115)
                range_call_result_451 = invoke(stypy.reporting.localization.Localization(__file__, 115, 35), range_445, *[len_call_result_449], **kwargs_450)
                
                # Assigning a type to the variable 'range_call_result_451' (line 115)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'range_call_result_451', range_call_result_451)
                # Testing if the for loop is going to be iterated (line 115)
                # Testing the type of a for loop iterable (line 115)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 115, 16), range_call_result_451)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 115, 16), range_call_result_451):
                    # Getting the type of the for loop variable (line 115)
                    for_loop_var_452 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 115, 16), range_call_result_451)
                    # Assigning a type to the variable 'row_counter' (line 115)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'row_counter', for_loop_var_452)
                    # SSA begins for a for statement (line 115)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Getting the type of 'prediction' (line 116)
                    prediction_453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 20), 'prediction')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'row_counter' (line 116)
                    row_counter_454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 60), 'row_counter')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'col_counter' (line 116)
                    col_counter_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 47), 'col_counter')
                    # Getting the type of 'kernel_table' (line 116)
                    kernel_table_456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 34), 'kernel_table')
                    # Obtaining the member '__getitem__' of a type (line 116)
                    getitem___457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 34), kernel_table_456, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 116)
                    subscript_call_result_458 = invoke(stypy.reporting.localization.Localization(__file__, 116, 34), getitem___457, col_counter_455)
                    
                    # Obtaining the member '__getitem__' of a type (line 116)
                    getitem___459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 34), subscript_call_result_458, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 116)
                    subscript_call_result_460 = invoke(stypy.reporting.localization.Localization(__file__, 116, 34), getitem___459, row_counter_454)
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'row_counter' (line 117)
                    row_counter_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 45), 'row_counter')
                    # Getting the type of 'labelalphas' (line 117)
                    labelalphas_462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 33), 'labelalphas')
                    # Obtaining the member '__getitem__' of a type (line 117)
                    getitem___463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 33), labelalphas_462, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 117)
                    subscript_call_result_464 = invoke(stypy.reporting.localization.Localization(__file__, 117, 33), getitem___463, row_counter_461)
                    
                    # Applying the binary operator '*' (line 116)
                    result_mul_465 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 34), '*', subscript_call_result_460, subscript_call_result_464)
                    
                    # Applying the binary operator '+=' (line 116)
                    result_iadd_466 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 20), '+=', prediction_453, result_mul_465)
                    # Assigning a type to the variable 'prediction' (line 116)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 20), 'prediction', result_iadd_466)
                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Assigning a BinOp to a Name (line 118):
                
                # Assigning a BinOp to a Name (line 118):
                float_467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 20), 'float')
                # Getting the type of 'prediction' (line 118)
                prediction_468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 28), 'prediction')
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 118)
                klass_469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 46), 'klass')
                # Getting the type of 'bias' (line 118)
                bias_470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 41), 'bias')
                # Obtaining the member '__getitem__' of a type (line 118)
                getitem___471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 41), bias_470, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 118)
                subscript_call_result_472 = invoke(stypy.reporting.localization.Localization(__file__, 118, 41), getitem___471, klass_469)
                
                # Applying the binary operator '+' (line 118)
                result_add_473 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 28), '+', prediction_468, subscript_call_result_472)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 118)
                klass_474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 81), 'klass')
                
                # Obtaining the type of the subscript
                # Getting the type of 'col_counter' (line 118)
                col_counter_475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 68), 'col_counter')
                # Getting the type of 'label_table' (line 118)
                label_table_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 56), 'label_table')
                # Obtaining the member '__getitem__' of a type (line 118)
                getitem___477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 56), label_table_476, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 118)
                subscript_call_result_478 = invoke(stypy.reporting.localization.Localization(__file__, 118, 56), getitem___477, col_counter_475)
                
                # Obtaining the member '__getitem__' of a type (line 118)
                getitem___479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 56), subscript_call_result_478, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 118)
                subscript_call_result_480 = invoke(stypy.reporting.localization.Localization(__file__, 118, 56), getitem___479, klass_474)
                
                # Applying the binary operator '*' (line 118)
                result_mul_481 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 27), '*', result_add_473, subscript_call_result_480)
                
                # Applying the binary operator '-' (line 118)
                result_sub_482 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 20), '-', float_467, result_mul_481)
                
                # Assigning a type to the variable 'g' (line 118)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'g', result_sub_482)
                
                # Assigning a Call to a Subscript (line 119):
                
                # Assigning a Call to a Subscript (line 119):
                
                # Call to min(...): (line 119)
                # Processing the call arguments (line 119)
                
                # Call to max(...): (line 119)
                # Processing the call arguments (line 119)
                
                # Obtaining the type of the subscript
                # Getting the type of 'col_counter' (line 119)
                col_counter_485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 67), 'col_counter', False)
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 119)
                klass_486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 60), 'klass', False)
                # Getting the type of 'alphas' (line 119)
                alphas_487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 53), 'alphas', False)
                # Obtaining the member '__getitem__' of a type (line 119)
                getitem___488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 53), alphas_487, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 119)
                subscript_call_result_489 = invoke(stypy.reporting.localization.Localization(__file__, 119, 53), getitem___488, klass_486)
                
                # Obtaining the member '__getitem__' of a type (line 119)
                getitem___490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 53), subscript_call_result_489, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 119)
                subscript_call_result_491 = invoke(stypy.reporting.localization.Localization(__file__, 119, 53), getitem___490, col_counter_485)
                
                # Getting the type of 'h' (line 119)
                h_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 82), 'h', False)
                # Getting the type of 'g' (line 119)
                g_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 86), 'g', False)
                # Applying the binary operator '*' (line 119)
                result_mul_494 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 82), '*', h_492, g_493)
                
                # Applying the binary operator '+' (line 119)
                result_add_495 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 53), '+', subscript_call_result_491, result_mul_494)
                
                float_496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 90), 'float')
                # Processing the call keyword arguments (line 119)
                kwargs_497 = {}
                # Getting the type of 'max' (line 119)
                max_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 48), 'max', False)
                # Calling max(args, kwargs) (line 119)
                max_call_result_498 = invoke(stypy.reporting.localization.Localization(__file__, 119, 48), max_484, *[result_add_495, float_496], **kwargs_497)
                
                # Getting the type of 'c' (line 119)
                c_499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 96), 'c', False)
                # Processing the call keyword arguments (line 119)
                kwargs_500 = {}
                # Getting the type of 'min' (line 119)
                min_483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 44), 'min', False)
                # Calling min(args, kwargs) (line 119)
                min_call_result_501 = invoke(stypy.reporting.localization.Localization(__file__, 119, 44), min_483, *[max_call_result_498, c_499], **kwargs_500)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 119)
                klass_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 22), 'klass')
                # Getting the type of 'betas' (line 119)
                betas_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'betas')
                # Obtaining the member '__getitem__' of a type (line 119)
                getitem___504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 16), betas_503, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 119)
                subscript_call_result_505 = invoke(stypy.reporting.localization.Localization(__file__, 119, 16), getitem___504, klass_502)
                
                # Getting the type of 'col_counter' (line 119)
                col_counter_506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 29), 'col_counter')
                # Storing an element on a container (line 119)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 16), subscript_call_result_505, (col_counter_506, min_call_result_501))
                
                # Assigning a Call to a Name (line 120):
                
                # Assigning a Call to a Name (line 120):
                
                # Call to abs(...): (line 120)
                # Processing the call arguments (line 120)
                
                # Obtaining the type of the subscript
                # Getting the type of 'col_counter' (line 120)
                col_counter_508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 47), 'col_counter', False)
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 120)
                klass_509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 40), 'klass', False)
                # Getting the type of 'alphas' (line 120)
                alphas_510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 33), 'alphas', False)
                # Obtaining the member '__getitem__' of a type (line 120)
                getitem___511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 33), alphas_510, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 120)
                subscript_call_result_512 = invoke(stypy.reporting.localization.Localization(__file__, 120, 33), getitem___511, klass_509)
                
                # Obtaining the member '__getitem__' of a type (line 120)
                getitem___513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 33), subscript_call_result_512, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 120)
                subscript_call_result_514 = invoke(stypy.reporting.localization.Localization(__file__, 120, 33), getitem___513, col_counter_508)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'col_counter' (line 120)
                col_counter_515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 75), 'col_counter', False)
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 120)
                klass_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 68), 'klass', False)
                # Getting the type of 'betas' (line 120)
                betas_517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 62), 'betas', False)
                # Obtaining the member '__getitem__' of a type (line 120)
                getitem___518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 62), betas_517, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 120)
                subscript_call_result_519 = invoke(stypy.reporting.localization.Localization(__file__, 120, 62), getitem___518, klass_516)
                
                # Obtaining the member '__getitem__' of a type (line 120)
                getitem___520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 62), subscript_call_result_519, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 120)
                subscript_call_result_521 = invoke(stypy.reporting.localization.Localization(__file__, 120, 62), getitem___520, col_counter_515)
                
                # Applying the binary operator '-' (line 120)
                result_sub_522 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 33), '-', subscript_call_result_514, subscript_call_result_521)
                
                # Processing the call keyword arguments (line 120)
                kwargs_523 = {}
                # Getting the type of 'abs' (line 120)
                abs_507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 29), 'abs', False)
                # Calling abs(args, kwargs) (line 120)
                abs_call_result_524 = invoke(stypy.reporting.localization.Localization(__file__, 120, 29), abs_507, *[result_sub_522], **kwargs_523)
                
                # Assigning a type to the variable 'difference' (line 120)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'difference', abs_call_result_524)
                
                # Getting the type of 'difference' (line 121)
                difference_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 19), 'difference')
                
                # Obtaining the type of the subscript
                int_526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 55), 'int')
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 121)
                klass_527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 48), 'klass')
                # Getting the type of 'max_differences' (line 121)
                max_differences_528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 32), 'max_differences')
                # Obtaining the member '__getitem__' of a type (line 121)
                getitem___529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 32), max_differences_528, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 121)
                subscript_call_result_530 = invoke(stypy.reporting.localization.Localization(__file__, 121, 32), getitem___529, klass_527)
                
                # Obtaining the member '__getitem__' of a type (line 121)
                getitem___531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 32), subscript_call_result_530, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 121)
                subscript_call_result_532 = invoke(stypy.reporting.localization.Localization(__file__, 121, 32), getitem___531, int_526)
                
                # Applying the binary operator '>' (line 121)
                result_gt_533 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 19), '>', difference_525, subscript_call_result_532)
                
                # Testing if the type of an if condition is none (line 121)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 121, 16), result_gt_533):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 121)
                    if_condition_534 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 16), result_gt_533)
                    # Assigning a type to the variable 'if_condition_534' (line 121)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'if_condition_534', if_condition_534)
                    # SSA begins for if statement (line 121)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Tuple to a Subscript (line 122):
                    
                    # Assigning a Tuple to a Subscript (line 122):
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 122)
                    tuple_535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 46), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 122)
                    # Adding element type (line 122)
                    # Getting the type of 'difference' (line 122)
                    difference_536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 46), 'difference')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 46), tuple_535, difference_536)
                    # Adding element type (line 122)
                    # Getting the type of 'col_counter' (line 122)
                    col_counter_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 58), 'col_counter')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 46), tuple_535, col_counter_537)
                    
                    # Getting the type of 'max_differences' (line 122)
                    max_differences_538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 20), 'max_differences')
                    # Getting the type of 'klass' (line 122)
                    klass_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 36), 'klass')
                    # Storing an element on a container (line 122)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 20), max_differences_538, (klass_539, tuple_535))
                    # SSA join for if statement (line 121)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Call to all(...): (line 124)
            # Processing the call arguments (line 124)
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'max_differences' (line 124)
            max_differences_547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 72), 'max_differences', False)
            comprehension_548 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 20), max_differences_547)
            # Assigning a type to the variable 'max_difference' (line 124)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'max_difference', comprehension_548)
            
            
            # Obtaining the type of the subscript
            int_541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 35), 'int')
            # Getting the type of 'max_difference' (line 124)
            max_difference_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'max_difference', False)
            # Obtaining the member '__getitem__' of a type (line 124)
            getitem___543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 20), max_difference_542, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 124)
            subscript_call_result_544 = invoke(stypy.reporting.localization.Localization(__file__, 124, 20), getitem___543, int_541)
            
            # Getting the type of 'tolerance' (line 124)
            tolerance_545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 40), 'tolerance', False)
            # Applying the binary operator '<' (line 124)
            result_lt_546 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 20), '<', subscript_call_result_544, tolerance_545)
            
            list_549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 20), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 20), list_549, result_lt_546)
            # Processing the call keyword arguments (line 124)
            kwargs_550 = {}
            # Getting the type of 'all' (line 124)
            all_540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 15), 'all', False)
            # Calling all(args, kwargs) (line 124)
            all_call_result_551 = invoke(stypy.reporting.localization.Localization(__file__, 124, 15), all_540, *[list_549], **kwargs_550)
            
            # Testing if the type of an if condition is none (line 124)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 124, 12), all_call_result_551):
                
                # Assigning a Subscript to a Subscript (line 127):
                
                # Assigning a Subscript to a Subscript (line 127):
                
                # Obtaining the type of the subscript
                
                # Obtaining the type of the subscript
                int_556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 95), 'int')
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 127)
                klass_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 88), 'klass')
                # Getting the type of 'max_differences' (line 127)
                max_differences_558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 72), 'max_differences')
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 72), max_differences_558, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_560 = invoke(stypy.reporting.localization.Localization(__file__, 127, 72), getitem___559, klass_557)
                
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 72), subscript_call_result_560, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_562 = invoke(stypy.reporting.localization.Localization(__file__, 127, 72), getitem___561, int_556)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 127)
                klass_563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 65), 'klass')
                # Getting the type of 'betas' (line 127)
                betas_564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 59), 'betas')
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 59), betas_564, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_566 = invoke(stypy.reporting.localization.Localization(__file__, 127, 59), getitem___565, klass_563)
                
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 59), subscript_call_result_566, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_568 = invoke(stypy.reporting.localization.Localization(__file__, 127, 59), getitem___567, subscript_call_result_562)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 127)
                klass_569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'klass')
                # Getting the type of 'alphas' (line 127)
                alphas_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'alphas')
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 16), alphas_570, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_572 = invoke(stypy.reporting.localization.Localization(__file__, 127, 16), getitem___571, klass_569)
                
                
                # Obtaining the type of the subscript
                int_573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 53), 'int')
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 127)
                klass_574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 46), 'klass')
                # Getting the type of 'max_differences' (line 127)
                max_differences_575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 30), 'max_differences')
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 30), max_differences_575, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_577 = invoke(stypy.reporting.localization.Localization(__file__, 127, 30), getitem___576, klass_574)
                
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 30), subscript_call_result_577, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_579 = invoke(stypy.reporting.localization.Localization(__file__, 127, 30), getitem___578, int_573)
                
                # Storing an element on a container (line 127)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 16), subscript_call_result_572, (subscript_call_result_579, subscript_call_result_568))
                
                # Assigning a Num to a Name (line 128):
                
                # Assigning a Num to a Name (line 128):
                float_580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 30), 'float')
                # Assigning a type to the variable 'element_sum' (line 128)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'element_sum', float_580)
                
                
                # Call to range(...): (line 129)
                # Processing the call arguments (line 129)
                
                # Call to len(...): (line 129)
                # Processing the call arguments (line 129)
                # Getting the type of 'kernel_table' (line 129)
                kernel_table_583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 49), 'kernel_table', False)
                # Processing the call keyword arguments (line 129)
                kwargs_584 = {}
                # Getting the type of 'len' (line 129)
                len_582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 45), 'len', False)
                # Calling len(args, kwargs) (line 129)
                len_call_result_585 = invoke(stypy.reporting.localization.Localization(__file__, 129, 45), len_582, *[kernel_table_583], **kwargs_584)
                
                # Processing the call keyword arguments (line 129)
                kwargs_586 = {}
                # Getting the type of 'range' (line 129)
                range_581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 39), 'range', False)
                # Calling range(args, kwargs) (line 129)
                range_call_result_587 = invoke(stypy.reporting.localization.Localization(__file__, 129, 39), range_581, *[len_call_result_585], **kwargs_586)
                
                # Assigning a type to the variable 'range_call_result_587' (line 129)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'range_call_result_587', range_call_result_587)
                # Testing if the for loop is going to be iterated (line 129)
                # Testing the type of a for loop iterable (line 129)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 129, 16), range_call_result_587)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 129, 16), range_call_result_587):
                    # Getting the type of the for loop variable (line 129)
                    for_loop_var_588 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 129, 16), range_call_result_587)
                    # Assigning a type to the variable 'element_counter' (line 129)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'element_counter', for_loop_var_588)
                    # SSA begins for a for statement (line 129)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Getting the type of 'element_sum' (line 130)
                    element_sum_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 20), 'element_sum')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'klass' (line 130)
                    klass_590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 64), 'klass')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'element_counter' (line 130)
                    element_counter_591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 47), 'element_counter')
                    # Getting the type of 'label_table' (line 130)
                    label_table_592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 35), 'label_table')
                    # Obtaining the member '__getitem__' of a type (line 130)
                    getitem___593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 35), label_table_592, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                    subscript_call_result_594 = invoke(stypy.reporting.localization.Localization(__file__, 130, 35), getitem___593, element_counter_591)
                    
                    # Obtaining the member '__getitem__' of a type (line 130)
                    getitem___595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 35), subscript_call_result_594, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                    subscript_call_result_596 = invoke(stypy.reporting.localization.Localization(__file__, 130, 35), getitem___595, klass_590)
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'element_counter' (line 130)
                    element_counter_597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 87), 'element_counter')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'klass' (line 130)
                    klass_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 80), 'klass')
                    # Getting the type of 'alphas' (line 130)
                    alphas_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 73), 'alphas')
                    # Obtaining the member '__getitem__' of a type (line 130)
                    getitem___600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 73), alphas_599, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                    subscript_call_result_601 = invoke(stypy.reporting.localization.Localization(__file__, 130, 73), getitem___600, klass_598)
                    
                    # Obtaining the member '__getitem__' of a type (line 130)
                    getitem___602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 73), subscript_call_result_601, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                    subscript_call_result_603 = invoke(stypy.reporting.localization.Localization(__file__, 130, 73), getitem___602, element_counter_597)
                    
                    # Applying the binary operator '*' (line 130)
                    result_mul_604 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 35), '*', subscript_call_result_596, subscript_call_result_603)
                    
                    int_605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 106), 'int')
                    # Applying the binary operator 'div' (line 130)
                    result_div_606 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 104), 'div', result_mul_604, int_605)
                    
                    # Applying the binary operator '+=' (line 130)
                    result_iadd_607 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 20), '+=', element_sum_589, result_div_606)
                    # Assigning a type to the variable 'element_sum' (line 130)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 20), 'element_sum', result_iadd_607)
                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Assigning a BinOp to a Subscript (line 131):
                
                # Assigning a BinOp to a Subscript (line 131):
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 131)
                klass_608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 35), 'klass')
                # Getting the type of 'bias' (line 131)
                bias_609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 30), 'bias')
                # Obtaining the member '__getitem__' of a type (line 131)
                getitem___610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 30), bias_609, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 131)
                subscript_call_result_611 = invoke(stypy.reporting.localization.Localization(__file__, 131, 30), getitem___610, klass_608)
                
                # Getting the type of 'element_sum' (line 131)
                element_sum_612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 44), 'element_sum')
                # Applying the binary operator '+' (line 131)
                result_add_613 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 30), '+', subscript_call_result_611, element_sum_612)
                
                # Getting the type of 'bias' (line 131)
                bias_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'bias')
                # Getting the type of 'klass' (line 131)
                klass_615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 21), 'klass')
                # Storing an element on a container (line 131)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 16), bias_614, (klass_615, result_add_613))
            else:
                
                # Testing the type of an if condition (line 124)
                if_condition_552 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 12), all_call_result_551)
                # Assigning a type to the variable 'if_condition_552' (line 124)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'if_condition_552', if_condition_552)
                # SSA begins for if statement (line 124)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Obtaining an instance of the builtin type 'tuple' (line 125)
                tuple_553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 23), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 125)
                # Adding element type (line 125)
                # Getting the type of 'alphas' (line 125)
                alphas_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 23), 'alphas')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 23), tuple_553, alphas_554)
                # Adding element type (line 125)
                # Getting the type of 'bias' (line 125)
                bias_555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 31), 'bias')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 23), tuple_553, bias_555)
                
                # Assigning a type to the variable 'stypy_return_type' (line 125)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'stypy_return_type', tuple_553)
                # SSA branch for the else part of an if statement (line 124)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Subscript to a Subscript (line 127):
                
                # Assigning a Subscript to a Subscript (line 127):
                
                # Obtaining the type of the subscript
                
                # Obtaining the type of the subscript
                int_556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 95), 'int')
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 127)
                klass_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 88), 'klass')
                # Getting the type of 'max_differences' (line 127)
                max_differences_558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 72), 'max_differences')
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 72), max_differences_558, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_560 = invoke(stypy.reporting.localization.Localization(__file__, 127, 72), getitem___559, klass_557)
                
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 72), subscript_call_result_560, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_562 = invoke(stypy.reporting.localization.Localization(__file__, 127, 72), getitem___561, int_556)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 127)
                klass_563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 65), 'klass')
                # Getting the type of 'betas' (line 127)
                betas_564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 59), 'betas')
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 59), betas_564, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_566 = invoke(stypy.reporting.localization.Localization(__file__, 127, 59), getitem___565, klass_563)
                
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 59), subscript_call_result_566, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_568 = invoke(stypy.reporting.localization.Localization(__file__, 127, 59), getitem___567, subscript_call_result_562)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 127)
                klass_569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'klass')
                # Getting the type of 'alphas' (line 127)
                alphas_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'alphas')
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 16), alphas_570, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_572 = invoke(stypy.reporting.localization.Localization(__file__, 127, 16), getitem___571, klass_569)
                
                
                # Obtaining the type of the subscript
                int_573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 53), 'int')
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 127)
                klass_574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 46), 'klass')
                # Getting the type of 'max_differences' (line 127)
                max_differences_575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 30), 'max_differences')
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 30), max_differences_575, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_577 = invoke(stypy.reporting.localization.Localization(__file__, 127, 30), getitem___576, klass_574)
                
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 30), subscript_call_result_577, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_579 = invoke(stypy.reporting.localization.Localization(__file__, 127, 30), getitem___578, int_573)
                
                # Storing an element on a container (line 127)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 16), subscript_call_result_572, (subscript_call_result_579, subscript_call_result_568))
                
                # Assigning a Num to a Name (line 128):
                
                # Assigning a Num to a Name (line 128):
                float_580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 30), 'float')
                # Assigning a type to the variable 'element_sum' (line 128)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'element_sum', float_580)
                
                
                # Call to range(...): (line 129)
                # Processing the call arguments (line 129)
                
                # Call to len(...): (line 129)
                # Processing the call arguments (line 129)
                # Getting the type of 'kernel_table' (line 129)
                kernel_table_583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 49), 'kernel_table', False)
                # Processing the call keyword arguments (line 129)
                kwargs_584 = {}
                # Getting the type of 'len' (line 129)
                len_582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 45), 'len', False)
                # Calling len(args, kwargs) (line 129)
                len_call_result_585 = invoke(stypy.reporting.localization.Localization(__file__, 129, 45), len_582, *[kernel_table_583], **kwargs_584)
                
                # Processing the call keyword arguments (line 129)
                kwargs_586 = {}
                # Getting the type of 'range' (line 129)
                range_581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 39), 'range', False)
                # Calling range(args, kwargs) (line 129)
                range_call_result_587 = invoke(stypy.reporting.localization.Localization(__file__, 129, 39), range_581, *[len_call_result_585], **kwargs_586)
                
                # Assigning a type to the variable 'range_call_result_587' (line 129)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'range_call_result_587', range_call_result_587)
                # Testing if the for loop is going to be iterated (line 129)
                # Testing the type of a for loop iterable (line 129)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 129, 16), range_call_result_587)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 129, 16), range_call_result_587):
                    # Getting the type of the for loop variable (line 129)
                    for_loop_var_588 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 129, 16), range_call_result_587)
                    # Assigning a type to the variable 'element_counter' (line 129)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'element_counter', for_loop_var_588)
                    # SSA begins for a for statement (line 129)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Getting the type of 'element_sum' (line 130)
                    element_sum_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 20), 'element_sum')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'klass' (line 130)
                    klass_590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 64), 'klass')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'element_counter' (line 130)
                    element_counter_591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 47), 'element_counter')
                    # Getting the type of 'label_table' (line 130)
                    label_table_592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 35), 'label_table')
                    # Obtaining the member '__getitem__' of a type (line 130)
                    getitem___593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 35), label_table_592, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                    subscript_call_result_594 = invoke(stypy.reporting.localization.Localization(__file__, 130, 35), getitem___593, element_counter_591)
                    
                    # Obtaining the member '__getitem__' of a type (line 130)
                    getitem___595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 35), subscript_call_result_594, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                    subscript_call_result_596 = invoke(stypy.reporting.localization.Localization(__file__, 130, 35), getitem___595, klass_590)
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'element_counter' (line 130)
                    element_counter_597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 87), 'element_counter')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'klass' (line 130)
                    klass_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 80), 'klass')
                    # Getting the type of 'alphas' (line 130)
                    alphas_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 73), 'alphas')
                    # Obtaining the member '__getitem__' of a type (line 130)
                    getitem___600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 73), alphas_599, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                    subscript_call_result_601 = invoke(stypy.reporting.localization.Localization(__file__, 130, 73), getitem___600, klass_598)
                    
                    # Obtaining the member '__getitem__' of a type (line 130)
                    getitem___602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 73), subscript_call_result_601, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                    subscript_call_result_603 = invoke(stypy.reporting.localization.Localization(__file__, 130, 73), getitem___602, element_counter_597)
                    
                    # Applying the binary operator '*' (line 130)
                    result_mul_604 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 35), '*', subscript_call_result_596, subscript_call_result_603)
                    
                    int_605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 106), 'int')
                    # Applying the binary operator 'div' (line 130)
                    result_div_606 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 104), 'div', result_mul_604, int_605)
                    
                    # Applying the binary operator '+=' (line 130)
                    result_iadd_607 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 20), '+=', element_sum_589, result_div_606)
                    # Assigning a type to the variable 'element_sum' (line 130)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 20), 'element_sum', result_iadd_607)
                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Assigning a BinOp to a Subscript (line 131):
                
                # Assigning a BinOp to a Subscript (line 131):
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 131)
                klass_608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 35), 'klass')
                # Getting the type of 'bias' (line 131)
                bias_609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 30), 'bias')
                # Obtaining the member '__getitem__' of a type (line 131)
                getitem___610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 30), bias_609, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 131)
                subscript_call_result_611 = invoke(stypy.reporting.localization.Localization(__file__, 131, 30), getitem___610, klass_608)
                
                # Getting the type of 'element_sum' (line 131)
                element_sum_612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 44), 'element_sum')
                # Applying the binary operator '+' (line 131)
                result_add_613 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 30), '+', subscript_call_result_611, element_sum_612)
                
                # Getting the type of 'bias' (line 131)
                bias_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'bias')
                # Getting the type of 'klass' (line 131)
                klass_615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 21), 'klass')
                # Storing an element on a container (line 131)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 16), bias_614, (klass_615, result_add_613))
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
    stypy_return_type_616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_616)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'train_adatron'
    return stypy_return_type_616

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
    float_617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 17), 'float')
    # Assigning a type to the variable 'prediction' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'prediction', float_617)
    
    # Assigning a ListComp to a Name (line 135):
    
    # Assigning a ListComp to a Name (line 135):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 135)
    # Processing the call arguments (line 135)
    
    # Call to len(...): (line 135)
    # Processing the call arguments (line 135)
    
    # Obtaining the type of the subscript
    int_627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 78), 'int')
    # Getting the type of 'label_table' (line 135)
    label_table_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 66), 'label_table', False)
    # Obtaining the member '__getitem__' of a type (line 135)
    getitem___629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 66), label_table_628, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 135)
    subscript_call_result_630 = invoke(stypy.reporting.localization.Localization(__file__, 135, 66), getitem___629, int_627)
    
    # Processing the call keyword arguments (line 135)
    kwargs_631 = {}
    # Getting the type of 'len' (line 135)
    len_626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 62), 'len', False)
    # Calling len(args, kwargs) (line 135)
    len_call_result_632 = invoke(stypy.reporting.localization.Localization(__file__, 135, 62), len_626, *[subscript_call_result_630], **kwargs_631)
    
    # Processing the call keyword arguments (line 135)
    kwargs_633 = {}
    # Getting the type of 'range' (line 135)
    range_625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 56), 'range', False)
    # Calling range(args, kwargs) (line 135)
    range_call_result_634 = invoke(stypy.reporting.localization.Localization(__file__, 135, 56), range_625, *[len_call_result_632], **kwargs_633)
    
    comprehension_635 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 19), range_call_result_634)
    # Assigning a type to the variable '_' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 19), '_', comprehension_635)
    
    # Obtaining an instance of the builtin type 'list' (line 135)
    list_618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 135)
    # Adding element type (line 135)
    float_619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 20), list_618, float_619)
    
    
    # Call to len(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'kernel_table' (line 135)
    kernel_table_621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 32), 'kernel_table', False)
    # Processing the call keyword arguments (line 135)
    kwargs_622 = {}
    # Getting the type of 'len' (line 135)
    len_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 28), 'len', False)
    # Calling len(args, kwargs) (line 135)
    len_call_result_623 = invoke(stypy.reporting.localization.Localization(__file__, 135, 28), len_620, *[kernel_table_621], **kwargs_622)
    
    # Applying the binary operator '*' (line 135)
    result_mul_624 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 20), '*', list_618, len_call_result_623)
    
    list_636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 19), list_636, result_mul_624)
    # Assigning a type to the variable 'predictions' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'predictions', list_636)
    
    
    # Call to range(...): (line 136)
    # Processing the call arguments (line 136)
    
    # Call to len(...): (line 136)
    # Processing the call arguments (line 136)
    
    # Obtaining the type of the subscript
    int_639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 39), 'int')
    # Getting the type of 'label_table' (line 136)
    label_table_640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 27), 'label_table', False)
    # Obtaining the member '__getitem__' of a type (line 136)
    getitem___641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 27), label_table_640, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 136)
    subscript_call_result_642 = invoke(stypy.reporting.localization.Localization(__file__, 136, 27), getitem___641, int_639)
    
    # Processing the call keyword arguments (line 136)
    kwargs_643 = {}
    # Getting the type of 'len' (line 136)
    len_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 23), 'len', False)
    # Calling len(args, kwargs) (line 136)
    len_call_result_644 = invoke(stypy.reporting.localization.Localization(__file__, 136, 23), len_638, *[subscript_call_result_642], **kwargs_643)
    
    # Processing the call keyword arguments (line 136)
    kwargs_645 = {}
    # Getting the type of 'range' (line 136)
    range_637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 17), 'range', False)
    # Calling range(args, kwargs) (line 136)
    range_call_result_646 = invoke(stypy.reporting.localization.Localization(__file__, 136, 17), range_637, *[len_call_result_644], **kwargs_645)
    
    # Assigning a type to the variable 'range_call_result_646' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'range_call_result_646', range_call_result_646)
    # Testing if the for loop is going to be iterated (line 136)
    # Testing the type of a for loop iterable (line 136)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 136, 4), range_call_result_646)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 136, 4), range_call_result_646):
        # Getting the type of the for loop variable (line 136)
        for_loop_var_647 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 136, 4), range_call_result_646)
        # Assigning a type to the variable 'klass' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'klass', for_loop_var_647)
        # SSA begins for a for statement (line 136)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 137)
        # Processing the call arguments (line 137)
        
        # Call to len(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'kernel_table' (line 137)
        kernel_table_650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 37), 'kernel_table', False)
        # Processing the call keyword arguments (line 137)
        kwargs_651 = {}
        # Getting the type of 'len' (line 137)
        len_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 33), 'len', False)
        # Calling len(args, kwargs) (line 137)
        len_call_result_652 = invoke(stypy.reporting.localization.Localization(__file__, 137, 33), len_649, *[kernel_table_650], **kwargs_651)
        
        # Processing the call keyword arguments (line 137)
        kwargs_653 = {}
        # Getting the type of 'range' (line 137)
        range_648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 27), 'range', False)
        # Calling range(args, kwargs) (line 137)
        range_call_result_654 = invoke(stypy.reporting.localization.Localization(__file__, 137, 27), range_648, *[len_call_result_652], **kwargs_653)
        
        # Assigning a type to the variable 'range_call_result_654' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'range_call_result_654', range_call_result_654)
        # Testing if the for loop is going to be iterated (line 137)
        # Testing the type of a for loop iterable (line 137)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 137, 8), range_call_result_654)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 137, 8), range_call_result_654):
            # Getting the type of the for loop variable (line 137)
            for_loop_var_655 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 137, 8), range_call_result_654)
            # Assigning a type to the variable 'col_counter' (line 137)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'col_counter', for_loop_var_655)
            # SSA begins for a for statement (line 137)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to range(...): (line 138)
            # Processing the call arguments (line 138)
            
            # Call to len(...): (line 138)
            # Processing the call arguments (line 138)
            # Getting the type of 'kernel_table' (line 138)
            kernel_table_658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 41), 'kernel_table', False)
            # Processing the call keyword arguments (line 138)
            kwargs_659 = {}
            # Getting the type of 'len' (line 138)
            len_657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 37), 'len', False)
            # Calling len(args, kwargs) (line 138)
            len_call_result_660 = invoke(stypy.reporting.localization.Localization(__file__, 138, 37), len_657, *[kernel_table_658], **kwargs_659)
            
            # Processing the call keyword arguments (line 138)
            kwargs_661 = {}
            # Getting the type of 'range' (line 138)
            range_656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 31), 'range', False)
            # Calling range(args, kwargs) (line 138)
            range_call_result_662 = invoke(stypy.reporting.localization.Localization(__file__, 138, 31), range_656, *[len_call_result_660], **kwargs_661)
            
            # Assigning a type to the variable 'range_call_result_662' (line 138)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'range_call_result_662', range_call_result_662)
            # Testing if the for loop is going to be iterated (line 138)
            # Testing the type of a for loop iterable (line 138)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 138, 12), range_call_result_662)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 138, 12), range_call_result_662):
                # Getting the type of the for loop variable (line 138)
                for_loop_var_663 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 138, 12), range_call_result_662)
                # Assigning a type to the variable 'row_counter' (line 138)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'row_counter', for_loop_var_663)
                # SSA begins for a for statement (line 138)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'prediction' (line 139)
                prediction_664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 'prediction')
                
                # Obtaining the type of the subscript
                # Getting the type of 'row_counter' (line 139)
                row_counter_665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 56), 'row_counter')
                
                # Obtaining the type of the subscript
                # Getting the type of 'col_counter' (line 139)
                col_counter_666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 43), 'col_counter')
                # Getting the type of 'kernel_table' (line 139)
                kernel_table_667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 30), 'kernel_table')
                # Obtaining the member '__getitem__' of a type (line 139)
                getitem___668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 30), kernel_table_667, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 139)
                subscript_call_result_669 = invoke(stypy.reporting.localization.Localization(__file__, 139, 30), getitem___668, col_counter_666)
                
                # Obtaining the member '__getitem__' of a type (line 139)
                getitem___670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 30), subscript_call_result_669, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 139)
                subscript_call_result_671 = invoke(stypy.reporting.localization.Localization(__file__, 139, 30), getitem___670, row_counter_665)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 140)
                klass_672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 55), 'klass')
                
                # Obtaining the type of the subscript
                # Getting the type of 'row_counter' (line 140)
                row_counter_673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 42), 'row_counter')
                # Getting the type of 'label_table' (line 140)
                label_table_674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 30), 'label_table')
                # Obtaining the member '__getitem__' of a type (line 140)
                getitem___675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 30), label_table_674, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 140)
                subscript_call_result_676 = invoke(stypy.reporting.localization.Localization(__file__, 140, 30), getitem___675, row_counter_673)
                
                # Obtaining the member '__getitem__' of a type (line 140)
                getitem___677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 30), subscript_call_result_676, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 140)
                subscript_call_result_678 = invoke(stypy.reporting.localization.Localization(__file__, 140, 30), getitem___677, klass_672)
                
                # Applying the binary operator '*' (line 139)
                result_mul_679 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 30), '*', subscript_call_result_671, subscript_call_result_678)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'row_counter' (line 140)
                row_counter_680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 78), 'row_counter')
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 140)
                klass_681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 71), 'klass')
                # Getting the type of 'alphas' (line 140)
                alphas_682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 64), 'alphas')
                # Obtaining the member '__getitem__' of a type (line 140)
                getitem___683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 64), alphas_682, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 140)
                subscript_call_result_684 = invoke(stypy.reporting.localization.Localization(__file__, 140, 64), getitem___683, klass_681)
                
                # Obtaining the member '__getitem__' of a type (line 140)
                getitem___685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 64), subscript_call_result_684, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 140)
                subscript_call_result_686 = invoke(stypy.reporting.localization.Localization(__file__, 140, 64), getitem___685, row_counter_680)
                
                # Applying the binary operator '*' (line 140)
                result_mul_687 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 62), '*', result_mul_679, subscript_call_result_686)
                
                # Applying the binary operator '+=' (line 139)
                result_iadd_688 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 16), '+=', prediction_664, result_mul_687)
                # Assigning a type to the variable 'prediction' (line 139)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 'prediction', result_iadd_688)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a BinOp to a Subscript (line 141):
            
            # Assigning a BinOp to a Subscript (line 141):
            # Getting the type of 'prediction' (line 141)
            prediction_689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 46), 'prediction')
            
            # Obtaining the type of the subscript
            # Getting the type of 'klass' (line 141)
            klass_690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 64), 'klass')
            # Getting the type of 'bias' (line 141)
            bias_691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 59), 'bias')
            # Obtaining the member '__getitem__' of a type (line 141)
            getitem___692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 59), bias_691, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 141)
            subscript_call_result_693 = invoke(stypy.reporting.localization.Localization(__file__, 141, 59), getitem___692, klass_690)
            
            # Applying the binary operator '+' (line 141)
            result_add_694 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 46), '+', prediction_689, subscript_call_result_693)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'klass' (line 141)
            klass_695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 24), 'klass')
            # Getting the type of 'predictions' (line 141)
            predictions_696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'predictions')
            # Obtaining the member '__getitem__' of a type (line 141)
            getitem___697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 12), predictions_696, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 141)
            subscript_call_result_698 = invoke(stypy.reporting.localization.Localization(__file__, 141, 12), getitem___697, klass_695)
            
            # Getting the type of 'col_counter' (line 141)
            col_counter_699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 31), 'col_counter')
            # Storing an element on a container (line 141)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 12), subscript_call_result_698, (col_counter_699, result_add_694))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    
    # Call to range(...): (line 143)
    # Processing the call arguments (line 143)
    
    # Call to len(...): (line 143)
    # Processing the call arguments (line 143)
    # Getting the type of 'kernel_table' (line 143)
    kernel_table_702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 33), 'kernel_table', False)
    # Processing the call keyword arguments (line 143)
    kwargs_703 = {}
    # Getting the type of 'len' (line 143)
    len_701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 29), 'len', False)
    # Calling len(args, kwargs) (line 143)
    len_call_result_704 = invoke(stypy.reporting.localization.Localization(__file__, 143, 29), len_701, *[kernel_table_702], **kwargs_703)
    
    # Processing the call keyword arguments (line 143)
    kwargs_705 = {}
    # Getting the type of 'range' (line 143)
    range_700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 23), 'range', False)
    # Calling range(args, kwargs) (line 143)
    range_call_result_706 = invoke(stypy.reporting.localization.Localization(__file__, 143, 23), range_700, *[len_call_result_704], **kwargs_705)
    
    # Assigning a type to the variable 'range_call_result_706' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'range_call_result_706', range_call_result_706)
    # Testing if the for loop is going to be iterated (line 143)
    # Testing the type of a for loop iterable (line 143)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 143, 4), range_call_result_706)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 143, 4), range_call_result_706):
        # Getting the type of the for loop variable (line 143)
        for_loop_var_707 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 143, 4), range_call_result_706)
        # Assigning a type to the variable 'col_counter' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'col_counter', for_loop_var_707)
        # SSA begins for a for statement (line 143)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a List to a Name (line 144):
        
        # Assigning a List to a Name (line 144):
        
        # Obtaining an instance of the builtin type 'list' (line 144)
        list_708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 144)
        
        # Assigning a type to the variable 'current_predictions' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'current_predictions', list_708)
        
        # Assigning a Num to a Name (line 145):
        
        # Assigning a Num to a Name (line 145):
        int_709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 16), 'int')
        # Assigning a type to the variable 'error' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'error', int_709)
        
        
        # Call to range(...): (line 146)
        # Processing the call arguments (line 146)
        
        # Call to len(...): (line 146)
        # Processing the call arguments (line 146)
        
        # Obtaining the type of the subscript
        int_712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 49), 'int')
        # Getting the type of 'label_table' (line 146)
        label_table_713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 37), 'label_table', False)
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 37), label_table_713, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_715 = invoke(stypy.reporting.localization.Localization(__file__, 146, 37), getitem___714, int_712)
        
        # Processing the call keyword arguments (line 146)
        kwargs_716 = {}
        # Getting the type of 'len' (line 146)
        len_711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 33), 'len', False)
        # Calling len(args, kwargs) (line 146)
        len_call_result_717 = invoke(stypy.reporting.localization.Localization(__file__, 146, 33), len_711, *[subscript_call_result_715], **kwargs_716)
        
        # Processing the call keyword arguments (line 146)
        kwargs_718 = {}
        # Getting the type of 'range' (line 146)
        range_710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 27), 'range', False)
        # Calling range(args, kwargs) (line 146)
        range_call_result_719 = invoke(stypy.reporting.localization.Localization(__file__, 146, 27), range_710, *[len_call_result_717], **kwargs_718)
        
        # Assigning a type to the variable 'range_call_result_719' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'range_call_result_719', range_call_result_719)
        # Testing if the for loop is going to be iterated (line 146)
        # Testing the type of a for loop iterable (line 146)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 146, 8), range_call_result_719)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 146, 8), range_call_result_719):
            # Getting the type of the for loop variable (line 146)
            for_loop_var_720 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 146, 8), range_call_result_719)
            # Assigning a type to the variable 'row_counter' (line 146)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'row_counter', for_loop_var_720)
            # SSA begins for a for statement (line 146)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to append(...): (line 147)
            # Processing the call arguments (line 147)
            
            # Obtaining the type of the subscript
            # Getting the type of 'col_counter' (line 147)
            col_counter_723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 64), 'col_counter', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row_counter' (line 147)
            row_counter_724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 51), 'row_counter', False)
            # Getting the type of 'predictions' (line 147)
            predictions_725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 39), 'predictions', False)
            # Obtaining the member '__getitem__' of a type (line 147)
            getitem___726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 39), predictions_725, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 147)
            subscript_call_result_727 = invoke(stypy.reporting.localization.Localization(__file__, 147, 39), getitem___726, row_counter_724)
            
            # Obtaining the member '__getitem__' of a type (line 147)
            getitem___728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 39), subscript_call_result_727, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 147)
            subscript_call_result_729 = invoke(stypy.reporting.localization.Localization(__file__, 147, 39), getitem___728, col_counter_723)
            
            # Processing the call keyword arguments (line 147)
            kwargs_730 = {}
            # Getting the type of 'current_predictions' (line 147)
            current_predictions_721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'current_predictions', False)
            # Obtaining the member 'append' of a type (line 147)
            append_722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 12), current_predictions_721, 'append')
            # Calling append(args, kwargs) (line 147)
            append_call_result_731 = invoke(stypy.reporting.localization.Localization(__file__, 147, 12), append_722, *[subscript_call_result_729], **kwargs_730)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Name (line 149):
        
        # Assigning a Call to a Name (line 149):
        
        # Call to index(...): (line 149)
        # Processing the call arguments (line 149)
        
        # Call to max(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'current_predictions' (line 149)
        current_predictions_735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 56), 'current_predictions', False)
        # Processing the call keyword arguments (line 149)
        kwargs_736 = {}
        # Getting the type of 'max' (line 149)
        max_734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 52), 'max', False)
        # Calling max(args, kwargs) (line 149)
        max_call_result_737 = invoke(stypy.reporting.localization.Localization(__file__, 149, 52), max_734, *[current_predictions_735], **kwargs_736)
        
        # Processing the call keyword arguments (line 149)
        kwargs_738 = {}
        # Getting the type of 'current_predictions' (line 149)
        current_predictions_732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 26), 'current_predictions', False)
        # Obtaining the member 'index' of a type (line 149)
        index_733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 26), current_predictions_732, 'index')
        # Calling index(args, kwargs) (line 149)
        index_call_result_739 = invoke(stypy.reporting.localization.Localization(__file__, 149, 26), index_733, *[max_call_result_737], **kwargs_738)
        
        # Assigning a type to the variable 'predicted_class' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'predicted_class', index_call_result_739)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'predicted_class' (line 151)
        predicted_class_740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 36), 'predicted_class')
        
        # Obtaining the type of the subscript
        # Getting the type of 'col_counter' (line 151)
        col_counter_741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 23), 'col_counter')
        # Getting the type of 'label_table' (line 151)
        label_table_742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 11), 'label_table')
        # Obtaining the member '__getitem__' of a type (line 151)
        getitem___743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 11), label_table_742, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 151)
        subscript_call_result_744 = invoke(stypy.reporting.localization.Localization(__file__, 151, 11), getitem___743, col_counter_741)
        
        # Obtaining the member '__getitem__' of a type (line 151)
        getitem___745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 11), subscript_call_result_744, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 151)
        subscript_call_result_746 = invoke(stypy.reporting.localization.Localization(__file__, 151, 11), getitem___745, predicted_class_740)
        
        int_747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 55), 'int')
        # Applying the binary operator '<' (line 151)
        result_lt_748 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 11), '<', subscript_call_result_746, int_747)
        
        # Testing if the type of an if condition is none (line 151)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 151, 8), result_lt_748):
            pass
        else:
            
            # Testing the type of an if condition (line 151)
            if_condition_749 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 8), result_lt_748)
            # Assigning a type to the variable 'if_condition_749' (line 151)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'if_condition_749', if_condition_749)
            # SSA begins for if statement (line 151)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'error' (line 152)
            error_750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'error')
            int_751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 21), 'int')
            # Applying the binary operator '+=' (line 152)
            result_iadd_752 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 12), '+=', error_750, int_751)
            # Assigning a type to the variable 'error' (line 152)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'error', result_iadd_752)
            
            # SSA join for if statement (line 151)
            module_type_store = module_type_store.join_ssa_context()
            

        float_753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 15), 'float')
        # Getting the type of 'error' (line 154)
        error_754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 21), 'error')
        # Applying the binary operator '*' (line 154)
        result_mul_755 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 15), '*', float_753, error_754)
        
        
        # Call to len(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'kernel_table' (line 154)
        kernel_table_757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 33), 'kernel_table', False)
        # Processing the call keyword arguments (line 154)
        kwargs_758 = {}
        # Getting the type of 'len' (line 154)
        len_756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 29), 'len', False)
        # Calling len(args, kwargs) (line 154)
        len_call_result_759 = invoke(stypy.reporting.localization.Localization(__file__, 154, 29), len_756, *[kernel_table_757], **kwargs_758)
        
        # Applying the binary operator 'div' (line 154)
        result_div_760 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 27), 'div', result_mul_755, len_call_result_759)
        
        # Assigning a type to the variable 'stypy_return_type' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'stypy_return_type', result_div_760)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'calculate_error(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'calculate_error' in the type store
    # Getting the type of 'stypy_return_type' (line 133)
    stypy_return_type_761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_761)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'calculate_error'
    return stypy_return_type_761

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
    list_762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 158)
    # Adding element type (line 158)
    
    # Obtaining an instance of the builtin type 'tuple' (line 158)
    tuple_763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 158)
    # Adding element type (line 158)
    
    # Call to Relative(...): (line 158)
    # Processing the call arguments (line 158)
    str_765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 37), 'str', 'testdata/c.txt')
    # Processing the call keyword arguments (line 158)
    kwargs_766 = {}
    # Getting the type of 'Relative' (line 158)
    Relative_764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 28), 'Relative', False)
    # Calling Relative(args, kwargs) (line 158)
    Relative_call_result_767 = invoke(stypy.reporting.localization.Localization(__file__, 158, 28), Relative_764, *[str_765], **kwargs_766)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 28), tuple_763, Relative_call_result_767)
    # Adding element type (line 158)
    # Getting the type of 'CYTOSOLIC' (line 158)
    CYTOSOLIC_768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 56), 'CYTOSOLIC')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 28), tuple_763, CYTOSOLIC_768)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 26), list_762, tuple_763)
    # Adding element type (line 158)
    
    # Obtaining an instance of the builtin type 'tuple' (line 158)
    tuple_769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 69), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 158)
    # Adding element type (line 158)
    
    # Call to Relative(...): (line 158)
    # Processing the call arguments (line 158)
    str_771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 78), 'str', 'testdata/e.txt')
    # Processing the call keyword arguments (line 158)
    kwargs_772 = {}
    # Getting the type of 'Relative' (line 158)
    Relative_770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 69), 'Relative', False)
    # Calling Relative(args, kwargs) (line 158)
    Relative_call_result_773 = invoke(stypy.reporting.localization.Localization(__file__, 158, 69), Relative_770, *[str_771], **kwargs_772)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 69), tuple_769, Relative_call_result_773)
    # Adding element type (line 158)
    # Getting the type of 'EXTRACELLULAR' (line 158)
    EXTRACELLULAR_774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 97), 'EXTRACELLULAR')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 69), tuple_769, EXTRACELLULAR_774)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 26), list_762, tuple_769)
    # Adding element type (line 158)
    
    # Obtaining an instance of the builtin type 'tuple' (line 158)
    tuple_775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 114), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 158)
    # Adding element type (line 158)
    
    # Call to Relative(...): (line 158)
    # Processing the call arguments (line 158)
    str_777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 123), 'str', 'testdata/n.txt')
    # Processing the call keyword arguments (line 158)
    kwargs_778 = {}
    # Getting the type of 'Relative' (line 158)
    Relative_776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 114), 'Relative', False)
    # Calling Relative(args, kwargs) (line 158)
    Relative_call_result_779 = invoke(stypy.reporting.localization.Localization(__file__, 158, 114), Relative_776, *[str_777], **kwargs_778)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 114), tuple_775, Relative_call_result_779)
    # Adding element type (line 158)
    # Getting the type of 'NUCLEAR' (line 158)
    NUCLEAR_780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 142), 'NUCLEAR')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 114), tuple_775, NUCLEAR_780)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 26), list_762, tuple_775)
    # Adding element type (line 158)
    
    # Obtaining an instance of the builtin type 'tuple' (line 158)
    tuple_781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 153), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 158)
    # Adding element type (line 158)
    
    # Call to Relative(...): (line 158)
    # Processing the call arguments (line 158)
    str_783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 162), 'str', 'testdata/m.txt')
    # Processing the call keyword arguments (line 158)
    kwargs_784 = {}
    # Getting the type of 'Relative' (line 158)
    Relative_782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 153), 'Relative', False)
    # Calling Relative(args, kwargs) (line 158)
    Relative_call_result_785 = invoke(stypy.reporting.localization.Localization(__file__, 158, 153), Relative_782, *[str_783], **kwargs_784)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 153), tuple_781, Relative_call_result_785)
    # Adding element type (line 158)
    # Getting the type of 'MITOCHONDRIAL' (line 158)
    MITOCHONDRIAL_786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 181), 'MITOCHONDRIAL')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 153), tuple_781, MITOCHONDRIAL_786)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 26), list_762, tuple_781)
    
    # Assigning a type to the variable 'list_762' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'list_762', list_762)
    # Testing if the for loop is going to be iterated (line 158)
    # Testing the type of a for loop iterable (line 158)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 158, 4), list_762)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 158, 4), list_762):
        # Getting the type of the for loop variable (line 158)
        for_loop_var_787 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 158, 4), list_762)
        # Assigning a type to the variable 'filename' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'filename', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 4), for_loop_var_787, 2, 0))
        # Assigning a type to the variable 'type' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'type', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 4), for_loop_var_787, 2, 1))
        # SSA begins for a for statement (line 158)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to load_file(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'filename' (line 159)
        filename_789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 18), 'filename', False)
        # Getting the type of 'type' (line 159)
        type_790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 28), 'type', False)
        # Processing the call keyword arguments (line 159)
        kwargs_791 = {}
        # Getting the type of 'load_file' (line 159)
        load_file_788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'load_file', False)
        # Calling load_file(args, kwargs) (line 159)
        load_file_call_result_792 = invoke(stypy.reporting.localization.Localization(__file__, 159, 8), load_file_788, *[filename_789, type_790], **kwargs_791)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a Call to a Tuple (line 161):
    
    # Assigning a Call to a Name:
    
    # Call to create_tables(...): (line 161)
    # Processing the call keyword arguments (line 161)
    kwargs_794 = {}
    # Getting the type of 'create_tables' (line 161)
    create_tables_793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 33), 'create_tables', False)
    # Calling create_tables(args, kwargs) (line 161)
    create_tables_call_result_795 = invoke(stypy.reporting.localization.Localization(__file__, 161, 33), create_tables_793, *[], **kwargs_794)
    
    # Assigning a type to the variable 'call_assignment_7' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'call_assignment_7', create_tables_call_result_795)
    
    # Assigning a Call to a Name (line 161):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 4), 'int')
    # Processing the call keyword arguments
    kwargs_799 = {}
    # Getting the type of 'call_assignment_7' (line 161)
    call_assignment_7_796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'call_assignment_7', False)
    # Obtaining the member '__getitem__' of a type (line 161)
    getitem___797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 4), call_assignment_7_796, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_800 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___797, *[int_798], **kwargs_799)
    
    # Assigning a type to the variable 'call_assignment_8' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'call_assignment_8', getitem___call_result_800)
    
    # Assigning a Name to a Name (line 161):
    # Getting the type of 'call_assignment_8' (line 161)
    call_assignment_8_801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'call_assignment_8')
    # Assigning a type to the variable 'feature_table' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'feature_table', call_assignment_8_801)
    
    # Assigning a Call to a Name (line 161):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 4), 'int')
    # Processing the call keyword arguments
    kwargs_805 = {}
    # Getting the type of 'call_assignment_7' (line 161)
    call_assignment_7_802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'call_assignment_7', False)
    # Obtaining the member '__getitem__' of a type (line 161)
    getitem___803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 4), call_assignment_7_802, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_806 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___803, *[int_804], **kwargs_805)
    
    # Assigning a type to the variable 'call_assignment_9' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'call_assignment_9', getitem___call_result_806)
    
    # Assigning a Name to a Name (line 161):
    # Getting the type of 'call_assignment_9' (line 161)
    call_assignment_9_807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'call_assignment_9')
    # Assigning a type to the variable 'label_table' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 19), 'label_table', call_assignment_9_807)
    
    # Assigning a Call to a Name (line 168):
    
    # Assigning a Call to a Name (line 168):
    
    # Call to create_kernel_table(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'feature_table' (line 168)
    feature_table_809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 39), 'feature_table', False)
    # Processing the call keyword arguments (line 168)
    kwargs_810 = {}
    # Getting the type of 'create_kernel_table' (line 168)
    create_kernel_table_808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 19), 'create_kernel_table', False)
    # Calling create_kernel_table(args, kwargs) (line 168)
    create_kernel_table_call_result_811 = invoke(stypy.reporting.localization.Localization(__file__, 168, 19), create_kernel_table_808, *[feature_table_809], **kwargs_810)
    
    # Assigning a type to the variable 'kernel_table' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'kernel_table', create_kernel_table_call_result_811)
    
    # Assigning a Call to a Tuple (line 170):
    
    # Assigning a Call to a Name:
    
    # Call to train_adatron(...): (line 170)
    # Processing the call arguments (line 170)
    # Getting the type of 'kernel_table' (line 170)
    kernel_table_813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 33), 'kernel_table', False)
    # Getting the type of 'label_table' (line 170)
    label_table_814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 47), 'label_table', False)
    float_815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 60), 'float')
    float_816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 65), 'float')
    # Processing the call keyword arguments (line 170)
    kwargs_817 = {}
    # Getting the type of 'train_adatron' (line 170)
    train_adatron_812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 19), 'train_adatron', False)
    # Calling train_adatron(args, kwargs) (line 170)
    train_adatron_call_result_818 = invoke(stypy.reporting.localization.Localization(__file__, 170, 19), train_adatron_812, *[kernel_table_813, label_table_814, float_815, float_816], **kwargs_817)
    
    # Assigning a type to the variable 'call_assignment_10' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'call_assignment_10', train_adatron_call_result_818)
    
    # Assigning a Call to a Name (line 170):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 4), 'int')
    # Processing the call keyword arguments
    kwargs_822 = {}
    # Getting the type of 'call_assignment_10' (line 170)
    call_assignment_10_819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'call_assignment_10', False)
    # Obtaining the member '__getitem__' of a type (line 170)
    getitem___820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 4), call_assignment_10_819, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_823 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___820, *[int_821], **kwargs_822)
    
    # Assigning a type to the variable 'call_assignment_11' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'call_assignment_11', getitem___call_result_823)
    
    # Assigning a Name to a Name (line 170):
    # Getting the type of 'call_assignment_11' (line 170)
    call_assignment_11_824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'call_assignment_11')
    # Assigning a type to the variable 'alphas' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'alphas', call_assignment_11_824)
    
    # Assigning a Call to a Name (line 170):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 4), 'int')
    # Processing the call keyword arguments
    kwargs_828 = {}
    # Getting the type of 'call_assignment_10' (line 170)
    call_assignment_10_825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'call_assignment_10', False)
    # Obtaining the member '__getitem__' of a type (line 170)
    getitem___826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 4), call_assignment_10_825, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_829 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___826, *[int_827], **kwargs_828)
    
    # Assigning a type to the variable 'call_assignment_12' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'call_assignment_12', getitem___call_result_829)
    
    # Assigning a Name to a Name (line 170):
    # Getting the type of 'call_assignment_12' (line 170)
    call_assignment_12_830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'call_assignment_12')
    # Assigning a type to the variable 'bias' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'bias', call_assignment_12_830)
    
    # Call to calculate_error(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'alphas' (line 172)
    alphas_832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 20), 'alphas', False)
    # Getting the type of 'bias' (line 172)
    bias_833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 28), 'bias', False)
    # Getting the type of 'kernel_table' (line 172)
    kernel_table_834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 34), 'kernel_table', False)
    # Getting the type of 'label_table' (line 172)
    label_table_835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 48), 'label_table', False)
    # Processing the call keyword arguments (line 172)
    kwargs_836 = {}
    # Getting the type of 'calculate_error' (line 172)
    calculate_error_831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'calculate_error', False)
    # Calling calculate_error(args, kwargs) (line 172)
    calculate_error_call_result_837 = invoke(stypy.reporting.localization.Localization(__file__, 172, 4), calculate_error_831, *[alphas_832, bias_833, kernel_table_834, label_table_835], **kwargs_836)
    
    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 157)
    stypy_return_type_838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_838)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_838

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
    kwargs_840 = {}
    # Getting the type of 'main' (line 176)
    main_839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'main', False)
    # Calling main(args, kwargs) (line 176)
    main_call_result_841 = invoke(stypy.reporting.localization.Localization(__file__, 176, 4), main_839, *[], **kwargs_840)
    
    # Getting the type of 'True' (line 177)
    True_842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'stypy_return_type', True_842)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 175)
    stypy_return_type_843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_843)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_843

# Assigning a type to the variable 'run' (line 175)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 0), 'run', run)

# Call to run(...): (line 179)
# Processing the call keyword arguments (line 179)
kwargs_845 = {}
# Getting the type of 'run' (line 179)
run_844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 0), 'run', False)
# Calling run(args, kwargs) (line 179)
run_call_result_846 = invoke(stypy.reporting.localization.Localization(__file__, 179, 0), run_844, *[], **kwargs_845)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

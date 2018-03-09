
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
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_1' (line 60)
        call_assignment_1_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_1', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_172 = stypy_get_value_from_tuple(call_assignment_1_171, 5, 0)
        
        # Assigning a type to the variable 'call_assignment_2' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_2', stypy_get_value_from_tuple_call_result_172)
        
        # Assigning a Name to a Name (line 60):
        # Getting the type of 'call_assignment_2' (line 60)
        call_assignment_2_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_2')
        # Assigning a type to the variable 'name' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'name', call_assignment_2_173)
        
        # Assigning a Call to a Name (line 60):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_1' (line 60)
        call_assignment_1_174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_1', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_175 = stypy_get_value_from_tuple(call_assignment_1_174, 5, 1)
        
        # Assigning a type to the variable 'call_assignment_3' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_3', stypy_get_value_from_tuple_call_result_175)
        
        # Assigning a Name to a Name (line 60):
        # Getting the type of 'call_assignment_3' (line 60)
        call_assignment_3_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_3')
        # Assigning a type to the variable 'mass' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 14), 'mass', call_assignment_3_176)
        
        # Assigning a Call to a Name (line 60):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_1' (line 60)
        call_assignment_1_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_1', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_178 = stypy_get_value_from_tuple(call_assignment_1_177, 5, 2)
        
        # Assigning a type to the variable 'call_assignment_4' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_4', stypy_get_value_from_tuple_call_result_178)
        
        # Assigning a Name to a Name (line 60):
        # Getting the type of 'call_assignment_4' (line 60)
        call_assignment_4_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_4')
        # Assigning a type to the variable 'isoelectric_point' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'isoelectric_point', call_assignment_4_179)
        
        # Assigning a Call to a Name (line 60):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_1' (line 60)
        call_assignment_1_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_1', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_181 = stypy_get_value_from_tuple(call_assignment_1_180, 5, 3)
        
        # Assigning a type to the variable 'call_assignment_5' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_5', stypy_get_value_from_tuple_call_result_181)
        
        # Assigning a Name to a Name (line 60):
        # Getting the type of 'call_assignment_5' (line 60)
        call_assignment_5_182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_5')
        # Assigning a type to the variable 'size' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 39), 'size', call_assignment_5_182)
        
        # Assigning a Call to a Name (line 60):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_1' (line 60)
        call_assignment_1_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_1', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_184 = stypy_get_value_from_tuple(call_assignment_1_183, 5, 4)
        
        # Assigning a type to the variable 'call_assignment_6' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_6', stypy_get_value_from_tuple_call_result_184)
        
        # Assigning a Name to a Name (line 60):
        # Getting the type of 'call_assignment_6' (line 60)
        call_assignment_6_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'call_assignment_6')
        # Assigning a type to the variable 'sequence' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 45), 'sequence', call_assignment_6_185)
        
        # Assigning a Call to a Name (line 61):
        
        # Assigning a Call to a Name (line 61):
        
        # Call to Protein(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'name' (line 61)
        name_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 26), 'name', False)
        # Getting the type of 'mass' (line 61)
        mass_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 32), 'mass', False)
        # Getting the type of 'isoelectric_point' (line 61)
        isoelectric_point_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 38), 'isoelectric_point', False)
        # Getting the type of 'size' (line 61)
        size_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 57), 'size', False)
        # Getting the type of 'sequence' (line 61)
        sequence_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 63), 'sequence', False)
        # Getting the type of 'type' (line 61)
        type_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 73), 'type', False)
        # Processing the call keyword arguments (line 61)
        kwargs_193 = {}
        # Getting the type of 'Protein' (line 61)
        Protein_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 18), 'Protein', False)
        # Calling Protein(args, kwargs) (line 61)
        Protein_call_result_194 = invoke(stypy.reporting.localization.Localization(__file__, 61, 18), Protein_186, *[name_187, mass_188, isoelectric_point_189, size_190, sequence_191, type_192], **kwargs_193)
        
        # Assigning a type to the variable 'protein' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'protein', Protein_call_result_194)
        
        # Call to append(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'protein' (line 62)
        protein_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 24), 'protein', False)
        # Processing the call keyword arguments (line 62)
        kwargs_198 = {}
        # Getting the type of 'PROTEINS' (line 62)
        PROTEINS_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'PROTEINS', False)
        # Obtaining the member 'append' of a type (line 62)
        append_196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), PROTEINS_195, 'append')
        # Calling append(args, kwargs) (line 62)
        append_call_result_199 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), append_196, *[protein_197], **kwargs_198)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Call to close(...): (line 63)
    # Processing the call keyword arguments (line 63)
    kwargs_202 = {}
    # Getting the type of 'protfile' (line 63)
    protfile_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'protfile', False)
    # Obtaining the member 'close' of a type (line 63)
    close_201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 4), protfile_200, 'close')
    # Calling close(args, kwargs) (line 63)
    close_call_result_203 = invoke(stypy.reporting.localization.Localization(__file__, 63, 4), close_201, *[], **kwargs_202)
    
    
    # ################# End of 'load_file(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'load_file' in the type store
    # Getting the type of 'stypy_return_type' (line 54)
    stypy_return_type_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_204)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'load_file'
    return stypy_return_type_204

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

    str_205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 4), 'str', 'Create the feature and label tables.')
    
    # Assigning a List to a Name (line 68):
    
    # Assigning a List to a Name (line 68):
    
    # Obtaining an instance of the builtin type 'list' (line 68)
    list_206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 68)
    
    # Assigning a type to the variable 'feature_table' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'feature_table', list_206)
    
    # Assigning a List to a Name (line 69):
    
    # Assigning a List to a Name (line 69):
    
    # Obtaining an instance of the builtin type 'list' (line 69)
    list_207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 69)
    
    # Assigning a type to the variable 'label_table' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'label_table', list_207)
    
    # Getting the type of 'PROTEINS' (line 71)
    PROTEINS_208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'PROTEINS')
    # Assigning a type to the variable 'PROTEINS_208' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'PROTEINS_208', PROTEINS_208)
    # Testing if the for loop is going to be iterated (line 71)
    # Testing the type of a for loop iterable (line 71)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 71, 4), PROTEINS_208)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 71, 4), PROTEINS_208):
        # Getting the type of the for loop variable (line 71)
        for_loop_var_209 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 71, 4), PROTEINS_208)
        # Assigning a type to the variable 'protein' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'protein', for_loop_var_209)
        # SSA begins for a for statement (line 71)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 72)
        # Processing the call arguments (line 72)
        
        # Call to create_vector(...): (line 72)
        # Processing the call keyword arguments (line 72)
        kwargs_214 = {}
        # Getting the type of 'protein' (line 72)
        protein_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 29), 'protein', False)
        # Obtaining the member 'create_vector' of a type (line 72)
        create_vector_213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 29), protein_212, 'create_vector')
        # Calling create_vector(args, kwargs) (line 72)
        create_vector_call_result_215 = invoke(stypy.reporting.localization.Localization(__file__, 72, 29), create_vector_213, *[], **kwargs_214)
        
        # Processing the call keyword arguments (line 72)
        kwargs_216 = {}
        # Getting the type of 'feature_table' (line 72)
        feature_table_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'feature_table', False)
        # Obtaining the member 'append' of a type (line 72)
        append_211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), feature_table_210, 'append')
        # Calling append(args, kwargs) (line 72)
        append_call_result_217 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), append_211, *[create_vector_call_result_215], **kwargs_216)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Getting the type of 'PROTEINS' (line 74)
    PROTEINS_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 19), 'PROTEINS')
    # Assigning a type to the variable 'PROTEINS_218' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'PROTEINS_218', PROTEINS_218)
    # Testing if the for loop is going to be iterated (line 74)
    # Testing the type of a for loop iterable (line 74)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 74, 4), PROTEINS_218)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 74, 4), PROTEINS_218):
        # Getting the type of the for loop variable (line 74)
        for_loop_var_219 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 74, 4), PROTEINS_218)
        # Assigning a type to the variable 'protein' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'protein', for_loop_var_219)
        # SSA begins for a for statement (line 74)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'protein' (line 75)
        protein_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 11), 'protein')
        # Obtaining the member 'type' of a type (line 75)
        type_221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 11), protein_220, 'type')
        # Getting the type of 'BLIND' (line 75)
        BLIND_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 27), 'BLIND')
        # Applying the binary operator '==' (line 75)
        result_eq_223 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 11), '==', type_221, BLIND_222)
        
        # Testing if the type of an if condition is none (line 75)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 8), result_eq_223):
            pass
        else:
            
            # Testing the type of an if condition (line 75)
            if_condition_224 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 8), result_eq_223)
            # Assigning a type to the variable 'if_condition_224' (line 75)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'if_condition_224', if_condition_224)
            # SSA begins for if statement (line 75)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 75)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 77):
        
        # Assigning a BinOp to a Name (line 77):
        
        # Obtaining an instance of the builtin type 'list' (line 77)
        list_225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 77)
        # Adding element type (line 77)
        int_226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 17), list_225, int_226)
        
        int_227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 24), 'int')
        # Applying the binary operator '*' (line 77)
        result_mul_228 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 17), '*', list_225, int_227)
        
        # Assigning a type to the variable 'labels' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'labels', result_mul_228)
        
        # Getting the type of 'labels' (line 79)
        labels_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'labels')
        
        # Obtaining the type of the subscript
        # Getting the type of 'protein' (line 79)
        protein_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 15), 'protein')
        # Obtaining the member 'type' of a type (line 79)
        type_231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 15), protein_230, 'type')
        # Getting the type of 'labels' (line 79)
        labels_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'labels')
        # Obtaining the member '__getitem__' of a type (line 79)
        getitem___233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), labels_232, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 79)
        subscript_call_result_234 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), getitem___233, type_231)
        
        int_235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 32), 'int')
        # Applying the binary operator '*=' (line 79)
        result_imul_236 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 8), '*=', subscript_call_result_234, int_235)
        # Getting the type of 'labels' (line 79)
        labels_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'labels')
        # Getting the type of 'protein' (line 79)
        protein_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 15), 'protein')
        # Obtaining the member 'type' of a type (line 79)
        type_239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 15), protein_238, 'type')
        # Storing an element on a container (line 79)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 8), labels_237, (type_239, result_imul_236))
        
        
        # Call to append(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'labels' (line 80)
        labels_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 27), 'labels', False)
        # Processing the call keyword arguments (line 80)
        kwargs_243 = {}
        # Getting the type of 'label_table' (line 80)
        label_table_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'label_table', False)
        # Obtaining the member 'append' of a type (line 80)
        append_241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), label_table_240, 'append')
        # Calling append(args, kwargs) (line 80)
        append_call_result_244 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), append_241, *[labels_242], **kwargs_243)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Obtaining an instance of the builtin type 'tuple' (line 82)
    tuple_245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 82)
    # Adding element type (line 82)
    # Getting the type of 'feature_table' (line 82)
    feature_table_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 11), 'feature_table')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 11), tuple_245, feature_table_246)
    # Adding element type (line 82)
    # Getting the type of 'label_table' (line 82)
    label_table_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'label_table')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 11), tuple_245, label_table_247)
    
    # Assigning a type to the variable 'stypy_return_type' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type', tuple_245)
    
    # ################# End of 'create_tables(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_tables' in the type store
    # Getting the type of 'stypy_return_type' (line 66)
    stypy_return_type_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_248)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_tables'
    return stypy_return_type_248

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
    list_249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 86)
    
    # Assigning a type to the variable 'kernel_table' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'kernel_table', list_249)
    
    # Getting the type of 'feature_table' (line 87)
    feature_table_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'feature_table')
    # Assigning a type to the variable 'feature_table_250' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'feature_table_250', feature_table_250)
    # Testing if the for loop is going to be iterated (line 87)
    # Testing the type of a for loop iterable (line 87)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 87, 4), feature_table_250)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 87, 4), feature_table_250):
        # Getting the type of the for loop variable (line 87)
        for_loop_var_251 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 87, 4), feature_table_250)
        # Assigning a type to the variable 'row' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'row', for_loop_var_251)
        # SSA begins for a for statement (line 87)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a List to a Name (line 88):
        
        # Assigning a List to a Name (line 88):
        
        # Obtaining an instance of the builtin type 'list' (line 88)
        list_252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 88)
        
        # Assigning a type to the variable 'kernel_row' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'kernel_row', list_252)
        
        # Getting the type of 'feature_table' (line 89)
        feature_table_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 25), 'feature_table')
        # Assigning a type to the variable 'feature_table_253' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'feature_table_253', feature_table_253)
        # Testing if the for loop is going to be iterated (line 89)
        # Testing the type of a for loop iterable (line 89)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 89, 8), feature_table_253)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 89, 8), feature_table_253):
            # Getting the type of the for loop variable (line 89)
            for_loop_var_254 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 89, 8), feature_table_253)
            # Assigning a type to the variable 'candidate' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'candidate', for_loop_var_254)
            # SSA begins for a for statement (line 89)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Num to a Name (line 90):
            
            # Assigning a Num to a Name (line 90):
            float_255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 25), 'float')
            # Assigning a type to the variable 'difference' (line 90)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'difference', float_255)
            
            
            # Call to range(...): (line 91)
            # Processing the call arguments (line 91)
            
            # Call to len(...): (line 91)
            # Processing the call arguments (line 91)
            # Getting the type of 'row' (line 91)
            row_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 37), 'row', False)
            # Processing the call keyword arguments (line 91)
            kwargs_259 = {}
            # Getting the type of 'len' (line 91)
            len_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 33), 'len', False)
            # Calling len(args, kwargs) (line 91)
            len_call_result_260 = invoke(stypy.reporting.localization.Localization(__file__, 91, 33), len_257, *[row_258], **kwargs_259)
            
            # Processing the call keyword arguments (line 91)
            kwargs_261 = {}
            # Getting the type of 'range' (line 91)
            range_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 27), 'range', False)
            # Calling range(args, kwargs) (line 91)
            range_call_result_262 = invoke(stypy.reporting.localization.Localization(__file__, 91, 27), range_256, *[len_call_result_260], **kwargs_261)
            
            # Assigning a type to the variable 'range_call_result_262' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'range_call_result_262', range_call_result_262)
            # Testing if the for loop is going to be iterated (line 91)
            # Testing the type of a for loop iterable (line 91)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 91, 12), range_call_result_262)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 91, 12), range_call_result_262):
                # Getting the type of the for loop variable (line 91)
                for_loop_var_263 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 91, 12), range_call_result_262)
                # Assigning a type to the variable 'counter' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'counter', for_loop_var_263)
                # SSA begins for a for statement (line 91)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'difference' (line 92)
                difference_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'difference')
                
                # Obtaining the type of the subscript
                # Getting the type of 'counter' (line 92)
                counter_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 35), 'counter')
                # Getting the type of 'row' (line 92)
                row_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 31), 'row')
                # Obtaining the member '__getitem__' of a type (line 92)
                getitem___267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 31), row_266, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 92)
                subscript_call_result_268 = invoke(stypy.reporting.localization.Localization(__file__, 92, 31), getitem___267, counter_265)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'counter' (line 92)
                counter_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 56), 'counter')
                # Getting the type of 'candidate' (line 92)
                candidate_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 46), 'candidate')
                # Obtaining the member '__getitem__' of a type (line 92)
                getitem___271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 46), candidate_270, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 92)
                subscript_call_result_272 = invoke(stypy.reporting.localization.Localization(__file__, 92, 46), getitem___271, counter_269)
                
                # Applying the binary operator '-' (line 92)
                result_sub_273 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 31), '-', subscript_call_result_268, subscript_call_result_272)
                
                int_274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 69), 'int')
                # Applying the binary operator '**' (line 92)
                result_pow_275 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 30), '**', result_sub_273, int_274)
                
                # Applying the binary operator '+=' (line 92)
                result_iadd_276 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 16), '+=', difference_264, result_pow_275)
                # Assigning a type to the variable 'difference' (line 92)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'difference', result_iadd_276)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Call to append(...): (line 93)
            # Processing the call arguments (line 93)
            
            # Call to exp(...): (line 93)
            # Processing the call arguments (line 93)
            
            # Getting the type of 'D' (line 93)
            D_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 35), 'D', False)
            # Applying the 'usub' unary operator (line 93)
            result___neg___281 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 34), 'usub', D_280)
            
            # Getting the type of 'difference' (line 93)
            difference_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 37), 'difference', False)
            # Applying the binary operator '*' (line 93)
            result_mul_283 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 34), '*', result___neg___281, difference_282)
            
            # Processing the call keyword arguments (line 93)
            kwargs_284 = {}
            # Getting the type of 'exp' (line 93)
            exp_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 30), 'exp', False)
            # Calling exp(args, kwargs) (line 93)
            exp_call_result_285 = invoke(stypy.reporting.localization.Localization(__file__, 93, 30), exp_279, *[result_mul_283], **kwargs_284)
            
            # Processing the call keyword arguments (line 93)
            kwargs_286 = {}
            # Getting the type of 'kernel_row' (line 93)
            kernel_row_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'kernel_row', False)
            # Obtaining the member 'append' of a type (line 93)
            append_278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 12), kernel_row_277, 'append')
            # Calling append(args, kwargs) (line 93)
            append_call_result_287 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), append_278, *[exp_call_result_285], **kwargs_286)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to append(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'kernel_row' (line 94)
        kernel_row_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 28), 'kernel_row', False)
        # Processing the call keyword arguments (line 94)
        kwargs_291 = {}
        # Getting the type of 'kernel_table' (line 94)
        kernel_table_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'kernel_table', False)
        # Obtaining the member 'append' of a type (line 94)
        append_289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), kernel_table_288, 'append')
        # Calling append(args, kwargs) (line 94)
        append_call_result_292 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), append_289, *[kernel_row_290], **kwargs_291)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'kernel_table' (line 95)
    kernel_table_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 11), 'kernel_table')
    # Assigning a type to the variable 'stypy_return_type' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'stypy_return_type', kernel_table_293)
    
    # ################# End of 'create_kernel_table(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_kernel_table' in the type store
    # Getting the type of 'stypy_return_type' (line 85)
    stypy_return_type_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_294)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_kernel_table'
    return stypy_return_type_294

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
    float_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 16), 'float')
    # Assigning a type to the variable 'tolerance' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'tolerance', float_295)
    
    # Assigning a ListComp to a Name (line 100):
    
    # Assigning a ListComp to a Name (line 100):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 100)
    # Processing the call arguments (line 100)
    
    # Call to len(...): (line 100)
    # Processing the call arguments (line 100)
    
    # Obtaining the type of the subscript
    int_305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 73), 'int')
    # Getting the type of 'label_table' (line 100)
    label_table_306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 61), 'label_table', False)
    # Obtaining the member '__getitem__' of a type (line 100)
    getitem___307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 61), label_table_306, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 100)
    subscript_call_result_308 = invoke(stypy.reporting.localization.Localization(__file__, 100, 61), getitem___307, int_305)
    
    # Processing the call keyword arguments (line 100)
    kwargs_309 = {}
    # Getting the type of 'len' (line 100)
    len_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 57), 'len', False)
    # Calling len(args, kwargs) (line 100)
    len_call_result_310 = invoke(stypy.reporting.localization.Localization(__file__, 100, 57), len_304, *[subscript_call_result_308], **kwargs_309)
    
    # Processing the call keyword arguments (line 100)
    kwargs_311 = {}
    # Getting the type of 'range' (line 100)
    range_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 51), 'range', False)
    # Calling range(args, kwargs) (line 100)
    range_call_result_312 = invoke(stypy.reporting.localization.Localization(__file__, 100, 51), range_303, *[len_call_result_310], **kwargs_311)
    
    comprehension_313 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 14), range_call_result_312)
    # Assigning a type to the variable '_' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 14), '_', comprehension_313)
    
    # Obtaining an instance of the builtin type 'list' (line 100)
    list_296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 100)
    # Adding element type (line 100)
    float_297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 16), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 15), list_296, float_297)
    
    
    # Call to len(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'kernel_table' (line 100)
    kernel_table_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 27), 'kernel_table', False)
    # Processing the call keyword arguments (line 100)
    kwargs_300 = {}
    # Getting the type of 'len' (line 100)
    len_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 23), 'len', False)
    # Calling len(args, kwargs) (line 100)
    len_call_result_301 = invoke(stypy.reporting.localization.Localization(__file__, 100, 23), len_298, *[kernel_table_299], **kwargs_300)
    
    # Applying the binary operator '*' (line 100)
    result_mul_302 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 15), '*', list_296, len_call_result_301)
    
    list_314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 14), list_314, result_mul_302)
    # Assigning a type to the variable 'alphas' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'alphas', list_314)
    
    # Assigning a ListComp to a Name (line 101):
    
    # Assigning a ListComp to a Name (line 101):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 101)
    # Processing the call arguments (line 101)
    
    # Call to len(...): (line 101)
    # Processing the call arguments (line 101)
    
    # Obtaining the type of the subscript
    int_324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 72), 'int')
    # Getting the type of 'label_table' (line 101)
    label_table_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 60), 'label_table', False)
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 60), label_table_325, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_327 = invoke(stypy.reporting.localization.Localization(__file__, 101, 60), getitem___326, int_324)
    
    # Processing the call keyword arguments (line 101)
    kwargs_328 = {}
    # Getting the type of 'len' (line 101)
    len_323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 56), 'len', False)
    # Calling len(args, kwargs) (line 101)
    len_call_result_329 = invoke(stypy.reporting.localization.Localization(__file__, 101, 56), len_323, *[subscript_call_result_327], **kwargs_328)
    
    # Processing the call keyword arguments (line 101)
    kwargs_330 = {}
    # Getting the type of 'range' (line 101)
    range_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 50), 'range', False)
    # Calling range(args, kwargs) (line 101)
    range_call_result_331 = invoke(stypy.reporting.localization.Localization(__file__, 101, 50), range_322, *[len_call_result_329], **kwargs_330)
    
    comprehension_332 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 13), range_call_result_331)
    # Assigning a type to the variable '_' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 13), '_', comprehension_332)
    
    # Obtaining an instance of the builtin type 'list' (line 101)
    list_315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 101)
    # Adding element type (line 101)
    float_316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 15), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 14), list_315, float_316)
    
    
    # Call to len(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'kernel_table' (line 101)
    kernel_table_318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 26), 'kernel_table', False)
    # Processing the call keyword arguments (line 101)
    kwargs_319 = {}
    # Getting the type of 'len' (line 101)
    len_317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 22), 'len', False)
    # Calling len(args, kwargs) (line 101)
    len_call_result_320 = invoke(stypy.reporting.localization.Localization(__file__, 101, 22), len_317, *[kernel_table_318], **kwargs_319)
    
    # Applying the binary operator '*' (line 101)
    result_mul_321 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 14), '*', list_315, len_call_result_320)
    
    list_333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 13), list_333, result_mul_321)
    # Assigning a type to the variable 'betas' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'betas', list_333)
    
    # Assigning a BinOp to a Name (line 102):
    
    # Assigning a BinOp to a Name (line 102):
    
    # Obtaining an instance of the builtin type 'list' (line 102)
    list_334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 102)
    # Adding element type (line 102)
    float_335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 12), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 11), list_334, float_335)
    
    
    # Call to len(...): (line 102)
    # Processing the call arguments (line 102)
    
    # Obtaining the type of the subscript
    int_337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 35), 'int')
    # Getting the type of 'label_table' (line 102)
    label_table_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 23), 'label_table', False)
    # Obtaining the member '__getitem__' of a type (line 102)
    getitem___339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 23), label_table_338, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 102)
    subscript_call_result_340 = invoke(stypy.reporting.localization.Localization(__file__, 102, 23), getitem___339, int_337)
    
    # Processing the call keyword arguments (line 102)
    kwargs_341 = {}
    # Getting the type of 'len' (line 102)
    len_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 19), 'len', False)
    # Calling len(args, kwargs) (line 102)
    len_call_result_342 = invoke(stypy.reporting.localization.Localization(__file__, 102, 19), len_336, *[subscript_call_result_340], **kwargs_341)
    
    # Applying the binary operator '*' (line 102)
    result_mul_343 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 11), '*', list_334, len_call_result_342)
    
    # Assigning a type to the variable 'bias' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'bias', result_mul_343)
    
    # Assigning a BinOp to a Name (line 103):
    
    # Assigning a BinOp to a Name (line 103):
    
    # Obtaining an instance of the builtin type 'list' (line 103)
    list_344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 103)
    # Adding element type (line 103)
    float_345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 19), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 18), list_344, float_345)
    
    
    # Call to len(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'kernel_table' (line 103)
    kernel_table_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 30), 'kernel_table', False)
    # Processing the call keyword arguments (line 103)
    kwargs_348 = {}
    # Getting the type of 'len' (line 103)
    len_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 26), 'len', False)
    # Calling len(args, kwargs) (line 103)
    len_call_result_349 = invoke(stypy.reporting.localization.Localization(__file__, 103, 26), len_346, *[kernel_table_347], **kwargs_348)
    
    # Applying the binary operator '*' (line 103)
    result_mul_350 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 18), '*', list_344, len_call_result_349)
    
    # Assigning a type to the variable 'labelalphas' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'labelalphas', result_mul_350)
    
    # Assigning a BinOp to a Name (line 104):
    
    # Assigning a BinOp to a Name (line 104):
    
    # Obtaining an instance of the builtin type 'list' (line 104)
    list_351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 104)
    # Adding element type (line 104)
    
    # Obtaining an instance of the builtin type 'tuple' (line 104)
    tuple_352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 104)
    # Adding element type (line 104)
    float_353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 24), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 24), tuple_352, float_353)
    # Adding element type (line 104)
    int_354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 24), tuple_352, int_354)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 22), list_351, tuple_352)
    
    
    # Call to len(...): (line 104)
    # Processing the call arguments (line 104)
    
    # Obtaining the type of the subscript
    int_356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 51), 'int')
    # Getting the type of 'label_table' (line 104)
    label_table_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 39), 'label_table', False)
    # Obtaining the member '__getitem__' of a type (line 104)
    getitem___358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 39), label_table_357, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 104)
    subscript_call_result_359 = invoke(stypy.reporting.localization.Localization(__file__, 104, 39), getitem___358, int_356)
    
    # Processing the call keyword arguments (line 104)
    kwargs_360 = {}
    # Getting the type of 'len' (line 104)
    len_355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 35), 'len', False)
    # Calling len(args, kwargs) (line 104)
    len_call_result_361 = invoke(stypy.reporting.localization.Localization(__file__, 104, 35), len_355, *[subscript_call_result_359], **kwargs_360)
    
    # Applying the binary operator '*' (line 104)
    result_mul_362 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 22), '*', list_351, len_call_result_361)
    
    # Assigning a type to the variable 'max_differences' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'max_differences', result_mul_362)
    
    
    # Call to range(...): (line 105)
    # Processing the call arguments (line 105)
    int_364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 27), 'int')
    
    # Call to len(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'kernel_table' (line 105)
    kernel_table_366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 34), 'kernel_table', False)
    # Processing the call keyword arguments (line 105)
    kwargs_367 = {}
    # Getting the type of 'len' (line 105)
    len_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 30), 'len', False)
    # Calling len(args, kwargs) (line 105)
    len_call_result_368 = invoke(stypy.reporting.localization.Localization(__file__, 105, 30), len_365, *[kernel_table_366], **kwargs_367)
    
    # Applying the binary operator '*' (line 105)
    result_mul_369 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 27), '*', int_364, len_call_result_368)
    
    # Processing the call keyword arguments (line 105)
    kwargs_370 = {}
    # Getting the type of 'range' (line 105)
    range_363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 21), 'range', False)
    # Calling range(args, kwargs) (line 105)
    range_call_result_371 = invoke(stypy.reporting.localization.Localization(__file__, 105, 21), range_363, *[result_mul_369], **kwargs_370)
    
    # Assigning a type to the variable 'range_call_result_371' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'range_call_result_371', range_call_result_371)
    # Testing if the for loop is going to be iterated (line 105)
    # Testing the type of a for loop iterable (line 105)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 105, 4), range_call_result_371)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 105, 4), range_call_result_371):
        # Getting the type of the for loop variable (line 105)
        for_loop_var_372 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 105, 4), range_call_result_371)
        # Assigning a type to the variable 'iteration' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'iteration', for_loop_var_372)
        # SSA begins for a for statement (line 105)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'iteration' (line 107)
        iteration_373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 11), 'iteration')
        int_374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 24), 'int')
        # Applying the binary operator '==' (line 107)
        result_eq_375 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 11), '==', iteration_373, int_374)
        
        # Testing if the type of an if condition is none (line 107)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 107, 8), result_eq_375):
            pass
        else:
            
            # Testing the type of an if condition (line 107)
            if_condition_376 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 8), result_eq_375)
            # Assigning a type to the variable 'if_condition_376' (line 107)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'if_condition_376', if_condition_376)
            # SSA begins for if statement (line 107)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining an instance of the builtin type 'tuple' (line 108)
            tuple_377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 108)
            # Adding element type (line 108)
            # Getting the type of 'alphas' (line 108)
            alphas_378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 19), 'alphas')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 19), tuple_377, alphas_378)
            # Adding element type (line 108)
            # Getting the type of 'bias' (line 108)
            bias_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 27), 'bias')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 19), tuple_377, bias_379)
            
            # Assigning a type to the variable 'stypy_return_type' (line 108)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'stypy_return_type', tuple_377)
            # SSA join for if statement (line 107)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to range(...): (line 109)
        # Processing the call arguments (line 109)
        
        # Call to len(...): (line 109)
        # Processing the call arguments (line 109)
        
        # Obtaining the type of the subscript
        int_382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 43), 'int')
        # Getting the type of 'label_table' (line 109)
        label_table_383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 31), 'label_table', False)
        # Obtaining the member '__getitem__' of a type (line 109)
        getitem___384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 31), label_table_383, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 109)
        subscript_call_result_385 = invoke(stypy.reporting.localization.Localization(__file__, 109, 31), getitem___384, int_382)
        
        # Processing the call keyword arguments (line 109)
        kwargs_386 = {}
        # Getting the type of 'len' (line 109)
        len_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'len', False)
        # Calling len(args, kwargs) (line 109)
        len_call_result_387 = invoke(stypy.reporting.localization.Localization(__file__, 109, 27), len_381, *[subscript_call_result_385], **kwargs_386)
        
        # Processing the call keyword arguments (line 109)
        kwargs_388 = {}
        # Getting the type of 'range' (line 109)
        range_380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 21), 'range', False)
        # Calling range(args, kwargs) (line 109)
        range_call_result_389 = invoke(stypy.reporting.localization.Localization(__file__, 109, 21), range_380, *[len_call_result_387], **kwargs_388)
        
        # Assigning a type to the variable 'range_call_result_389' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'range_call_result_389', range_call_result_389)
        # Testing if the for loop is going to be iterated (line 109)
        # Testing the type of a for loop iterable (line 109)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 109, 8), range_call_result_389)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 109, 8), range_call_result_389):
            # Getting the type of the for loop variable (line 109)
            for_loop_var_390 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 109, 8), range_call_result_389)
            # Assigning a type to the variable 'klass' (line 109)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'klass', for_loop_var_390)
            # SSA begins for a for statement (line 109)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Tuple to a Subscript (line 110):
            
            # Assigning a Tuple to a Subscript (line 110):
            
            # Obtaining an instance of the builtin type 'tuple' (line 110)
            tuple_391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 38), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 110)
            # Adding element type (line 110)
            float_392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 38), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 38), tuple_391, float_392)
            # Adding element type (line 110)
            int_393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 43), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 38), tuple_391, int_393)
            
            # Getting the type of 'max_differences' (line 110)
            max_differences_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'max_differences')
            # Getting the type of 'klass' (line 110)
            klass_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 28), 'klass')
            # Storing an element on a container (line 110)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 12), max_differences_394, (klass_395, tuple_391))
            
            
            # Call to range(...): (line 111)
            # Processing the call arguments (line 111)
            
            # Call to len(...): (line 111)
            # Processing the call arguments (line 111)
            # Getting the type of 'kernel_table' (line 111)
            kernel_table_398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 34), 'kernel_table', False)
            # Processing the call keyword arguments (line 111)
            kwargs_399 = {}
            # Getting the type of 'len' (line 111)
            len_397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 30), 'len', False)
            # Calling len(args, kwargs) (line 111)
            len_call_result_400 = invoke(stypy.reporting.localization.Localization(__file__, 111, 30), len_397, *[kernel_table_398], **kwargs_399)
            
            # Processing the call keyword arguments (line 111)
            kwargs_401 = {}
            # Getting the type of 'range' (line 111)
            range_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'range', False)
            # Calling range(args, kwargs) (line 111)
            range_call_result_402 = invoke(stypy.reporting.localization.Localization(__file__, 111, 24), range_396, *[len_call_result_400], **kwargs_401)
            
            # Assigning a type to the variable 'range_call_result_402' (line 111)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'range_call_result_402', range_call_result_402)
            # Testing if the for loop is going to be iterated (line 111)
            # Testing the type of a for loop iterable (line 111)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 111, 12), range_call_result_402)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 111, 12), range_call_result_402):
                # Getting the type of the for loop variable (line 111)
                for_loop_var_403 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 111, 12), range_call_result_402)
                # Assigning a type to the variable 'elem' (line 111)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'elem', for_loop_var_403)
                # SSA begins for a for statement (line 111)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a BinOp to a Subscript (line 112):
                
                # Assigning a BinOp to a Subscript (line 112):
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 112)
                klass_404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 54), 'klass')
                
                # Obtaining the type of the subscript
                # Getting the type of 'elem' (line 112)
                elem_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 48), 'elem')
                # Getting the type of 'label_table' (line 112)
                label_table_406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 36), 'label_table')
                # Obtaining the member '__getitem__' of a type (line 112)
                getitem___407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 36), label_table_406, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 112)
                subscript_call_result_408 = invoke(stypy.reporting.localization.Localization(__file__, 112, 36), getitem___407, elem_405)
                
                # Obtaining the member '__getitem__' of a type (line 112)
                getitem___409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 36), subscript_call_result_408, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 112)
                subscript_call_result_410 = invoke(stypy.reporting.localization.Localization(__file__, 112, 36), getitem___409, klass_404)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'elem' (line 112)
                elem_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 77), 'elem')
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 112)
                klass_412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 70), 'klass')
                # Getting the type of 'alphas' (line 112)
                alphas_413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 63), 'alphas')
                # Obtaining the member '__getitem__' of a type (line 112)
                getitem___414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 63), alphas_413, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 112)
                subscript_call_result_415 = invoke(stypy.reporting.localization.Localization(__file__, 112, 63), getitem___414, klass_412)
                
                # Obtaining the member '__getitem__' of a type (line 112)
                getitem___416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 63), subscript_call_result_415, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 112)
                subscript_call_result_417 = invoke(stypy.reporting.localization.Localization(__file__, 112, 63), getitem___416, elem_411)
                
                # Applying the binary operator '*' (line 112)
                result_mul_418 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 36), '*', subscript_call_result_410, subscript_call_result_417)
                
                # Getting the type of 'labelalphas' (line 112)
                labelalphas_419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'labelalphas')
                # Getting the type of 'elem' (line 112)
                elem_420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 28), 'elem')
                # Storing an element on a container (line 112)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 16), labelalphas_419, (elem_420, result_mul_418))
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            
            # Call to range(...): (line 113)
            # Processing the call arguments (line 113)
            
            # Call to len(...): (line 113)
            # Processing the call arguments (line 113)
            # Getting the type of 'kernel_table' (line 113)
            kernel_table_423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 41), 'kernel_table', False)
            # Processing the call keyword arguments (line 113)
            kwargs_424 = {}
            # Getting the type of 'len' (line 113)
            len_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 37), 'len', False)
            # Calling len(args, kwargs) (line 113)
            len_call_result_425 = invoke(stypy.reporting.localization.Localization(__file__, 113, 37), len_422, *[kernel_table_423], **kwargs_424)
            
            # Processing the call keyword arguments (line 113)
            kwargs_426 = {}
            # Getting the type of 'range' (line 113)
            range_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 31), 'range', False)
            # Calling range(args, kwargs) (line 113)
            range_call_result_427 = invoke(stypy.reporting.localization.Localization(__file__, 113, 31), range_421, *[len_call_result_425], **kwargs_426)
            
            # Assigning a type to the variable 'range_call_result_427' (line 113)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'range_call_result_427', range_call_result_427)
            # Testing if the for loop is going to be iterated (line 113)
            # Testing the type of a for loop iterable (line 113)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 113, 12), range_call_result_427)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 113, 12), range_call_result_427):
                # Getting the type of the for loop variable (line 113)
                for_loop_var_428 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 113, 12), range_call_result_427)
                # Assigning a type to the variable 'col_counter' (line 113)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'col_counter', for_loop_var_428)
                # SSA begins for a for statement (line 113)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Num to a Name (line 114):
                
                # Assigning a Num to a Name (line 114):
                float_429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 29), 'float')
                # Assigning a type to the variable 'prediction' (line 114)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'prediction', float_429)
                
                
                # Call to range(...): (line 115)
                # Processing the call arguments (line 115)
                
                # Call to len(...): (line 115)
                # Processing the call arguments (line 115)
                # Getting the type of 'kernel_table' (line 115)
                kernel_table_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 45), 'kernel_table', False)
                # Processing the call keyword arguments (line 115)
                kwargs_433 = {}
                # Getting the type of 'len' (line 115)
                len_431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 41), 'len', False)
                # Calling len(args, kwargs) (line 115)
                len_call_result_434 = invoke(stypy.reporting.localization.Localization(__file__, 115, 41), len_431, *[kernel_table_432], **kwargs_433)
                
                # Processing the call keyword arguments (line 115)
                kwargs_435 = {}
                # Getting the type of 'range' (line 115)
                range_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 35), 'range', False)
                # Calling range(args, kwargs) (line 115)
                range_call_result_436 = invoke(stypy.reporting.localization.Localization(__file__, 115, 35), range_430, *[len_call_result_434], **kwargs_435)
                
                # Assigning a type to the variable 'range_call_result_436' (line 115)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'range_call_result_436', range_call_result_436)
                # Testing if the for loop is going to be iterated (line 115)
                # Testing the type of a for loop iterable (line 115)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 115, 16), range_call_result_436)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 115, 16), range_call_result_436):
                    # Getting the type of the for loop variable (line 115)
                    for_loop_var_437 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 115, 16), range_call_result_436)
                    # Assigning a type to the variable 'row_counter' (line 115)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'row_counter', for_loop_var_437)
                    # SSA begins for a for statement (line 115)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Getting the type of 'prediction' (line 116)
                    prediction_438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 20), 'prediction')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'row_counter' (line 116)
                    row_counter_439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 60), 'row_counter')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'col_counter' (line 116)
                    col_counter_440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 47), 'col_counter')
                    # Getting the type of 'kernel_table' (line 116)
                    kernel_table_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 34), 'kernel_table')
                    # Obtaining the member '__getitem__' of a type (line 116)
                    getitem___442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 34), kernel_table_441, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 116)
                    subscript_call_result_443 = invoke(stypy.reporting.localization.Localization(__file__, 116, 34), getitem___442, col_counter_440)
                    
                    # Obtaining the member '__getitem__' of a type (line 116)
                    getitem___444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 34), subscript_call_result_443, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 116)
                    subscript_call_result_445 = invoke(stypy.reporting.localization.Localization(__file__, 116, 34), getitem___444, row_counter_439)
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'row_counter' (line 117)
                    row_counter_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 45), 'row_counter')
                    # Getting the type of 'labelalphas' (line 117)
                    labelalphas_447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 33), 'labelalphas')
                    # Obtaining the member '__getitem__' of a type (line 117)
                    getitem___448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 33), labelalphas_447, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 117)
                    subscript_call_result_449 = invoke(stypy.reporting.localization.Localization(__file__, 117, 33), getitem___448, row_counter_446)
                    
                    # Applying the binary operator '*' (line 116)
                    result_mul_450 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 34), '*', subscript_call_result_445, subscript_call_result_449)
                    
                    # Applying the binary operator '+=' (line 116)
                    result_iadd_451 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 20), '+=', prediction_438, result_mul_450)
                    # Assigning a type to the variable 'prediction' (line 116)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 20), 'prediction', result_iadd_451)
                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Assigning a BinOp to a Name (line 118):
                
                # Assigning a BinOp to a Name (line 118):
                float_452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 20), 'float')
                # Getting the type of 'prediction' (line 118)
                prediction_453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 28), 'prediction')
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 118)
                klass_454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 46), 'klass')
                # Getting the type of 'bias' (line 118)
                bias_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 41), 'bias')
                # Obtaining the member '__getitem__' of a type (line 118)
                getitem___456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 41), bias_455, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 118)
                subscript_call_result_457 = invoke(stypy.reporting.localization.Localization(__file__, 118, 41), getitem___456, klass_454)
                
                # Applying the binary operator '+' (line 118)
                result_add_458 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 28), '+', prediction_453, subscript_call_result_457)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 118)
                klass_459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 81), 'klass')
                
                # Obtaining the type of the subscript
                # Getting the type of 'col_counter' (line 118)
                col_counter_460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 68), 'col_counter')
                # Getting the type of 'label_table' (line 118)
                label_table_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 56), 'label_table')
                # Obtaining the member '__getitem__' of a type (line 118)
                getitem___462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 56), label_table_461, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 118)
                subscript_call_result_463 = invoke(stypy.reporting.localization.Localization(__file__, 118, 56), getitem___462, col_counter_460)
                
                # Obtaining the member '__getitem__' of a type (line 118)
                getitem___464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 56), subscript_call_result_463, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 118)
                subscript_call_result_465 = invoke(stypy.reporting.localization.Localization(__file__, 118, 56), getitem___464, klass_459)
                
                # Applying the binary operator '*' (line 118)
                result_mul_466 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 27), '*', result_add_458, subscript_call_result_465)
                
                # Applying the binary operator '-' (line 118)
                result_sub_467 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 20), '-', float_452, result_mul_466)
                
                # Assigning a type to the variable 'g' (line 118)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'g', result_sub_467)
                
                # Assigning a Call to a Subscript (line 119):
                
                # Assigning a Call to a Subscript (line 119):
                
                # Call to min(...): (line 119)
                # Processing the call arguments (line 119)
                
                # Call to max(...): (line 119)
                # Processing the call arguments (line 119)
                
                # Obtaining the type of the subscript
                # Getting the type of 'col_counter' (line 119)
                col_counter_470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 67), 'col_counter', False)
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 119)
                klass_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 60), 'klass', False)
                # Getting the type of 'alphas' (line 119)
                alphas_472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 53), 'alphas', False)
                # Obtaining the member '__getitem__' of a type (line 119)
                getitem___473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 53), alphas_472, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 119)
                subscript_call_result_474 = invoke(stypy.reporting.localization.Localization(__file__, 119, 53), getitem___473, klass_471)
                
                # Obtaining the member '__getitem__' of a type (line 119)
                getitem___475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 53), subscript_call_result_474, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 119)
                subscript_call_result_476 = invoke(stypy.reporting.localization.Localization(__file__, 119, 53), getitem___475, col_counter_470)
                
                # Getting the type of 'h' (line 119)
                h_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 82), 'h', False)
                # Getting the type of 'g' (line 119)
                g_478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 86), 'g', False)
                # Applying the binary operator '*' (line 119)
                result_mul_479 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 82), '*', h_477, g_478)
                
                # Applying the binary operator '+' (line 119)
                result_add_480 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 53), '+', subscript_call_result_476, result_mul_479)
                
                float_481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 90), 'float')
                # Processing the call keyword arguments (line 119)
                kwargs_482 = {}
                # Getting the type of 'max' (line 119)
                max_469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 48), 'max', False)
                # Calling max(args, kwargs) (line 119)
                max_call_result_483 = invoke(stypy.reporting.localization.Localization(__file__, 119, 48), max_469, *[result_add_480, float_481], **kwargs_482)
                
                # Getting the type of 'c' (line 119)
                c_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 96), 'c', False)
                # Processing the call keyword arguments (line 119)
                kwargs_485 = {}
                # Getting the type of 'min' (line 119)
                min_468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 44), 'min', False)
                # Calling min(args, kwargs) (line 119)
                min_call_result_486 = invoke(stypy.reporting.localization.Localization(__file__, 119, 44), min_468, *[max_call_result_483, c_484], **kwargs_485)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 119)
                klass_487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 22), 'klass')
                # Getting the type of 'betas' (line 119)
                betas_488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'betas')
                # Obtaining the member '__getitem__' of a type (line 119)
                getitem___489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 16), betas_488, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 119)
                subscript_call_result_490 = invoke(stypy.reporting.localization.Localization(__file__, 119, 16), getitem___489, klass_487)
                
                # Getting the type of 'col_counter' (line 119)
                col_counter_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 29), 'col_counter')
                # Storing an element on a container (line 119)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 16), subscript_call_result_490, (col_counter_491, min_call_result_486))
                
                # Assigning a Call to a Name (line 120):
                
                # Assigning a Call to a Name (line 120):
                
                # Call to abs(...): (line 120)
                # Processing the call arguments (line 120)
                
                # Obtaining the type of the subscript
                # Getting the type of 'col_counter' (line 120)
                col_counter_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 47), 'col_counter', False)
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 120)
                klass_494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 40), 'klass', False)
                # Getting the type of 'alphas' (line 120)
                alphas_495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 33), 'alphas', False)
                # Obtaining the member '__getitem__' of a type (line 120)
                getitem___496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 33), alphas_495, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 120)
                subscript_call_result_497 = invoke(stypy.reporting.localization.Localization(__file__, 120, 33), getitem___496, klass_494)
                
                # Obtaining the member '__getitem__' of a type (line 120)
                getitem___498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 33), subscript_call_result_497, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 120)
                subscript_call_result_499 = invoke(stypy.reporting.localization.Localization(__file__, 120, 33), getitem___498, col_counter_493)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'col_counter' (line 120)
                col_counter_500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 75), 'col_counter', False)
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 120)
                klass_501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 68), 'klass', False)
                # Getting the type of 'betas' (line 120)
                betas_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 62), 'betas', False)
                # Obtaining the member '__getitem__' of a type (line 120)
                getitem___503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 62), betas_502, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 120)
                subscript_call_result_504 = invoke(stypy.reporting.localization.Localization(__file__, 120, 62), getitem___503, klass_501)
                
                # Obtaining the member '__getitem__' of a type (line 120)
                getitem___505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 62), subscript_call_result_504, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 120)
                subscript_call_result_506 = invoke(stypy.reporting.localization.Localization(__file__, 120, 62), getitem___505, col_counter_500)
                
                # Applying the binary operator '-' (line 120)
                result_sub_507 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 33), '-', subscript_call_result_499, subscript_call_result_506)
                
                # Processing the call keyword arguments (line 120)
                kwargs_508 = {}
                # Getting the type of 'abs' (line 120)
                abs_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 29), 'abs', False)
                # Calling abs(args, kwargs) (line 120)
                abs_call_result_509 = invoke(stypy.reporting.localization.Localization(__file__, 120, 29), abs_492, *[result_sub_507], **kwargs_508)
                
                # Assigning a type to the variable 'difference' (line 120)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'difference', abs_call_result_509)
                
                # Getting the type of 'difference' (line 121)
                difference_510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 19), 'difference')
                
                # Obtaining the type of the subscript
                int_511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 55), 'int')
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 121)
                klass_512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 48), 'klass')
                # Getting the type of 'max_differences' (line 121)
                max_differences_513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 32), 'max_differences')
                # Obtaining the member '__getitem__' of a type (line 121)
                getitem___514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 32), max_differences_513, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 121)
                subscript_call_result_515 = invoke(stypy.reporting.localization.Localization(__file__, 121, 32), getitem___514, klass_512)
                
                # Obtaining the member '__getitem__' of a type (line 121)
                getitem___516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 32), subscript_call_result_515, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 121)
                subscript_call_result_517 = invoke(stypy.reporting.localization.Localization(__file__, 121, 32), getitem___516, int_511)
                
                # Applying the binary operator '>' (line 121)
                result_gt_518 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 19), '>', difference_510, subscript_call_result_517)
                
                # Testing if the type of an if condition is none (line 121)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 121, 16), result_gt_518):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 121)
                    if_condition_519 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 16), result_gt_518)
                    # Assigning a type to the variable 'if_condition_519' (line 121)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'if_condition_519', if_condition_519)
                    # SSA begins for if statement (line 121)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Tuple to a Subscript (line 122):
                    
                    # Assigning a Tuple to a Subscript (line 122):
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 122)
                    tuple_520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 46), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 122)
                    # Adding element type (line 122)
                    # Getting the type of 'difference' (line 122)
                    difference_521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 46), 'difference')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 46), tuple_520, difference_521)
                    # Adding element type (line 122)
                    # Getting the type of 'col_counter' (line 122)
                    col_counter_522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 58), 'col_counter')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 46), tuple_520, col_counter_522)
                    
                    # Getting the type of 'max_differences' (line 122)
                    max_differences_523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 20), 'max_differences')
                    # Getting the type of 'klass' (line 122)
                    klass_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 36), 'klass')
                    # Storing an element on a container (line 122)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 20), max_differences_523, (klass_524, tuple_520))
                    # SSA join for if statement (line 121)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Call to all(...): (line 124)
            # Processing the call arguments (line 124)
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'max_differences' (line 124)
            max_differences_532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 72), 'max_differences', False)
            comprehension_533 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 20), max_differences_532)
            # Assigning a type to the variable 'max_difference' (line 124)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'max_difference', comprehension_533)
            
            
            # Obtaining the type of the subscript
            int_526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 35), 'int')
            # Getting the type of 'max_difference' (line 124)
            max_difference_527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'max_difference', False)
            # Obtaining the member '__getitem__' of a type (line 124)
            getitem___528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 20), max_difference_527, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 124)
            subscript_call_result_529 = invoke(stypy.reporting.localization.Localization(__file__, 124, 20), getitem___528, int_526)
            
            # Getting the type of 'tolerance' (line 124)
            tolerance_530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 40), 'tolerance', False)
            # Applying the binary operator '<' (line 124)
            result_lt_531 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 20), '<', subscript_call_result_529, tolerance_530)
            
            list_534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 20), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 20), list_534, result_lt_531)
            # Processing the call keyword arguments (line 124)
            kwargs_535 = {}
            # Getting the type of 'all' (line 124)
            all_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 15), 'all', False)
            # Calling all(args, kwargs) (line 124)
            all_call_result_536 = invoke(stypy.reporting.localization.Localization(__file__, 124, 15), all_525, *[list_534], **kwargs_535)
            
            # Testing if the type of an if condition is none (line 124)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 124, 12), all_call_result_536):
                
                # Assigning a Subscript to a Subscript (line 127):
                
                # Assigning a Subscript to a Subscript (line 127):
                
                # Obtaining the type of the subscript
                
                # Obtaining the type of the subscript
                int_541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 95), 'int')
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 127)
                klass_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 88), 'klass')
                # Getting the type of 'max_differences' (line 127)
                max_differences_543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 72), 'max_differences')
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 72), max_differences_543, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_545 = invoke(stypy.reporting.localization.Localization(__file__, 127, 72), getitem___544, klass_542)
                
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 72), subscript_call_result_545, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_547 = invoke(stypy.reporting.localization.Localization(__file__, 127, 72), getitem___546, int_541)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 127)
                klass_548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 65), 'klass')
                # Getting the type of 'betas' (line 127)
                betas_549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 59), 'betas')
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 59), betas_549, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_551 = invoke(stypy.reporting.localization.Localization(__file__, 127, 59), getitem___550, klass_548)
                
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 59), subscript_call_result_551, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_553 = invoke(stypy.reporting.localization.Localization(__file__, 127, 59), getitem___552, subscript_call_result_547)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 127)
                klass_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'klass')
                # Getting the type of 'alphas' (line 127)
                alphas_555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'alphas')
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 16), alphas_555, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_557 = invoke(stypy.reporting.localization.Localization(__file__, 127, 16), getitem___556, klass_554)
                
                
                # Obtaining the type of the subscript
                int_558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 53), 'int')
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 127)
                klass_559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 46), 'klass')
                # Getting the type of 'max_differences' (line 127)
                max_differences_560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 30), 'max_differences')
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 30), max_differences_560, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_562 = invoke(stypy.reporting.localization.Localization(__file__, 127, 30), getitem___561, klass_559)
                
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 30), subscript_call_result_562, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_564 = invoke(stypy.reporting.localization.Localization(__file__, 127, 30), getitem___563, int_558)
                
                # Storing an element on a container (line 127)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 16), subscript_call_result_557, (subscript_call_result_564, subscript_call_result_553))
                
                # Assigning a Num to a Name (line 128):
                
                # Assigning a Num to a Name (line 128):
                float_565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 30), 'float')
                # Assigning a type to the variable 'element_sum' (line 128)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'element_sum', float_565)
                
                
                # Call to range(...): (line 129)
                # Processing the call arguments (line 129)
                
                # Call to len(...): (line 129)
                # Processing the call arguments (line 129)
                # Getting the type of 'kernel_table' (line 129)
                kernel_table_568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 49), 'kernel_table', False)
                # Processing the call keyword arguments (line 129)
                kwargs_569 = {}
                # Getting the type of 'len' (line 129)
                len_567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 45), 'len', False)
                # Calling len(args, kwargs) (line 129)
                len_call_result_570 = invoke(stypy.reporting.localization.Localization(__file__, 129, 45), len_567, *[kernel_table_568], **kwargs_569)
                
                # Processing the call keyword arguments (line 129)
                kwargs_571 = {}
                # Getting the type of 'range' (line 129)
                range_566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 39), 'range', False)
                # Calling range(args, kwargs) (line 129)
                range_call_result_572 = invoke(stypy.reporting.localization.Localization(__file__, 129, 39), range_566, *[len_call_result_570], **kwargs_571)
                
                # Assigning a type to the variable 'range_call_result_572' (line 129)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'range_call_result_572', range_call_result_572)
                # Testing if the for loop is going to be iterated (line 129)
                # Testing the type of a for loop iterable (line 129)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 129, 16), range_call_result_572)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 129, 16), range_call_result_572):
                    # Getting the type of the for loop variable (line 129)
                    for_loop_var_573 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 129, 16), range_call_result_572)
                    # Assigning a type to the variable 'element_counter' (line 129)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'element_counter', for_loop_var_573)
                    # SSA begins for a for statement (line 129)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Getting the type of 'element_sum' (line 130)
                    element_sum_574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 20), 'element_sum')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'klass' (line 130)
                    klass_575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 64), 'klass')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'element_counter' (line 130)
                    element_counter_576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 47), 'element_counter')
                    # Getting the type of 'label_table' (line 130)
                    label_table_577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 35), 'label_table')
                    # Obtaining the member '__getitem__' of a type (line 130)
                    getitem___578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 35), label_table_577, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                    subscript_call_result_579 = invoke(stypy.reporting.localization.Localization(__file__, 130, 35), getitem___578, element_counter_576)
                    
                    # Obtaining the member '__getitem__' of a type (line 130)
                    getitem___580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 35), subscript_call_result_579, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                    subscript_call_result_581 = invoke(stypy.reporting.localization.Localization(__file__, 130, 35), getitem___580, klass_575)
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'element_counter' (line 130)
                    element_counter_582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 87), 'element_counter')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'klass' (line 130)
                    klass_583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 80), 'klass')
                    # Getting the type of 'alphas' (line 130)
                    alphas_584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 73), 'alphas')
                    # Obtaining the member '__getitem__' of a type (line 130)
                    getitem___585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 73), alphas_584, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                    subscript_call_result_586 = invoke(stypy.reporting.localization.Localization(__file__, 130, 73), getitem___585, klass_583)
                    
                    # Obtaining the member '__getitem__' of a type (line 130)
                    getitem___587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 73), subscript_call_result_586, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                    subscript_call_result_588 = invoke(stypy.reporting.localization.Localization(__file__, 130, 73), getitem___587, element_counter_582)
                    
                    # Applying the binary operator '*' (line 130)
                    result_mul_589 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 35), '*', subscript_call_result_581, subscript_call_result_588)
                    
                    int_590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 106), 'int')
                    # Applying the binary operator 'div' (line 130)
                    result_div_591 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 104), 'div', result_mul_589, int_590)
                    
                    # Applying the binary operator '+=' (line 130)
                    result_iadd_592 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 20), '+=', element_sum_574, result_div_591)
                    # Assigning a type to the variable 'element_sum' (line 130)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 20), 'element_sum', result_iadd_592)
                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Assigning a BinOp to a Subscript (line 131):
                
                # Assigning a BinOp to a Subscript (line 131):
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 131)
                klass_593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 35), 'klass')
                # Getting the type of 'bias' (line 131)
                bias_594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 30), 'bias')
                # Obtaining the member '__getitem__' of a type (line 131)
                getitem___595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 30), bias_594, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 131)
                subscript_call_result_596 = invoke(stypy.reporting.localization.Localization(__file__, 131, 30), getitem___595, klass_593)
                
                # Getting the type of 'element_sum' (line 131)
                element_sum_597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 44), 'element_sum')
                # Applying the binary operator '+' (line 131)
                result_add_598 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 30), '+', subscript_call_result_596, element_sum_597)
                
                # Getting the type of 'bias' (line 131)
                bias_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'bias')
                # Getting the type of 'klass' (line 131)
                klass_600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 21), 'klass')
                # Storing an element on a container (line 131)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 16), bias_599, (klass_600, result_add_598))
            else:
                
                # Testing the type of an if condition (line 124)
                if_condition_537 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 12), all_call_result_536)
                # Assigning a type to the variable 'if_condition_537' (line 124)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'if_condition_537', if_condition_537)
                # SSA begins for if statement (line 124)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Obtaining an instance of the builtin type 'tuple' (line 125)
                tuple_538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 23), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 125)
                # Adding element type (line 125)
                # Getting the type of 'alphas' (line 125)
                alphas_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 23), 'alphas')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 23), tuple_538, alphas_539)
                # Adding element type (line 125)
                # Getting the type of 'bias' (line 125)
                bias_540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 31), 'bias')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 23), tuple_538, bias_540)
                
                # Assigning a type to the variable 'stypy_return_type' (line 125)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'stypy_return_type', tuple_538)
                # SSA branch for the else part of an if statement (line 124)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Subscript to a Subscript (line 127):
                
                # Assigning a Subscript to a Subscript (line 127):
                
                # Obtaining the type of the subscript
                
                # Obtaining the type of the subscript
                int_541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 95), 'int')
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 127)
                klass_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 88), 'klass')
                # Getting the type of 'max_differences' (line 127)
                max_differences_543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 72), 'max_differences')
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 72), max_differences_543, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_545 = invoke(stypy.reporting.localization.Localization(__file__, 127, 72), getitem___544, klass_542)
                
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 72), subscript_call_result_545, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_547 = invoke(stypy.reporting.localization.Localization(__file__, 127, 72), getitem___546, int_541)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 127)
                klass_548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 65), 'klass')
                # Getting the type of 'betas' (line 127)
                betas_549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 59), 'betas')
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 59), betas_549, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_551 = invoke(stypy.reporting.localization.Localization(__file__, 127, 59), getitem___550, klass_548)
                
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 59), subscript_call_result_551, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_553 = invoke(stypy.reporting.localization.Localization(__file__, 127, 59), getitem___552, subscript_call_result_547)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 127)
                klass_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'klass')
                # Getting the type of 'alphas' (line 127)
                alphas_555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'alphas')
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 16), alphas_555, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_557 = invoke(stypy.reporting.localization.Localization(__file__, 127, 16), getitem___556, klass_554)
                
                
                # Obtaining the type of the subscript
                int_558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 53), 'int')
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 127)
                klass_559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 46), 'klass')
                # Getting the type of 'max_differences' (line 127)
                max_differences_560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 30), 'max_differences')
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 30), max_differences_560, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_562 = invoke(stypy.reporting.localization.Localization(__file__, 127, 30), getitem___561, klass_559)
                
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 30), subscript_call_result_562, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_564 = invoke(stypy.reporting.localization.Localization(__file__, 127, 30), getitem___563, int_558)
                
                # Storing an element on a container (line 127)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 16), subscript_call_result_557, (subscript_call_result_564, subscript_call_result_553))
                
                # Assigning a Num to a Name (line 128):
                
                # Assigning a Num to a Name (line 128):
                float_565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 30), 'float')
                # Assigning a type to the variable 'element_sum' (line 128)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'element_sum', float_565)
                
                
                # Call to range(...): (line 129)
                # Processing the call arguments (line 129)
                
                # Call to len(...): (line 129)
                # Processing the call arguments (line 129)
                # Getting the type of 'kernel_table' (line 129)
                kernel_table_568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 49), 'kernel_table', False)
                # Processing the call keyword arguments (line 129)
                kwargs_569 = {}
                # Getting the type of 'len' (line 129)
                len_567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 45), 'len', False)
                # Calling len(args, kwargs) (line 129)
                len_call_result_570 = invoke(stypy.reporting.localization.Localization(__file__, 129, 45), len_567, *[kernel_table_568], **kwargs_569)
                
                # Processing the call keyword arguments (line 129)
                kwargs_571 = {}
                # Getting the type of 'range' (line 129)
                range_566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 39), 'range', False)
                # Calling range(args, kwargs) (line 129)
                range_call_result_572 = invoke(stypy.reporting.localization.Localization(__file__, 129, 39), range_566, *[len_call_result_570], **kwargs_571)
                
                # Assigning a type to the variable 'range_call_result_572' (line 129)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'range_call_result_572', range_call_result_572)
                # Testing if the for loop is going to be iterated (line 129)
                # Testing the type of a for loop iterable (line 129)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 129, 16), range_call_result_572)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 129, 16), range_call_result_572):
                    # Getting the type of the for loop variable (line 129)
                    for_loop_var_573 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 129, 16), range_call_result_572)
                    # Assigning a type to the variable 'element_counter' (line 129)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'element_counter', for_loop_var_573)
                    # SSA begins for a for statement (line 129)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Getting the type of 'element_sum' (line 130)
                    element_sum_574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 20), 'element_sum')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'klass' (line 130)
                    klass_575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 64), 'klass')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'element_counter' (line 130)
                    element_counter_576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 47), 'element_counter')
                    # Getting the type of 'label_table' (line 130)
                    label_table_577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 35), 'label_table')
                    # Obtaining the member '__getitem__' of a type (line 130)
                    getitem___578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 35), label_table_577, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                    subscript_call_result_579 = invoke(stypy.reporting.localization.Localization(__file__, 130, 35), getitem___578, element_counter_576)
                    
                    # Obtaining the member '__getitem__' of a type (line 130)
                    getitem___580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 35), subscript_call_result_579, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                    subscript_call_result_581 = invoke(stypy.reporting.localization.Localization(__file__, 130, 35), getitem___580, klass_575)
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'element_counter' (line 130)
                    element_counter_582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 87), 'element_counter')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'klass' (line 130)
                    klass_583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 80), 'klass')
                    # Getting the type of 'alphas' (line 130)
                    alphas_584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 73), 'alphas')
                    # Obtaining the member '__getitem__' of a type (line 130)
                    getitem___585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 73), alphas_584, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                    subscript_call_result_586 = invoke(stypy.reporting.localization.Localization(__file__, 130, 73), getitem___585, klass_583)
                    
                    # Obtaining the member '__getitem__' of a type (line 130)
                    getitem___587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 73), subscript_call_result_586, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 130)
                    subscript_call_result_588 = invoke(stypy.reporting.localization.Localization(__file__, 130, 73), getitem___587, element_counter_582)
                    
                    # Applying the binary operator '*' (line 130)
                    result_mul_589 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 35), '*', subscript_call_result_581, subscript_call_result_588)
                    
                    int_590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 106), 'int')
                    # Applying the binary operator 'div' (line 130)
                    result_div_591 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 104), 'div', result_mul_589, int_590)
                    
                    # Applying the binary operator '+=' (line 130)
                    result_iadd_592 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 20), '+=', element_sum_574, result_div_591)
                    # Assigning a type to the variable 'element_sum' (line 130)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 20), 'element_sum', result_iadd_592)
                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Assigning a BinOp to a Subscript (line 131):
                
                # Assigning a BinOp to a Subscript (line 131):
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 131)
                klass_593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 35), 'klass')
                # Getting the type of 'bias' (line 131)
                bias_594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 30), 'bias')
                # Obtaining the member '__getitem__' of a type (line 131)
                getitem___595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 30), bias_594, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 131)
                subscript_call_result_596 = invoke(stypy.reporting.localization.Localization(__file__, 131, 30), getitem___595, klass_593)
                
                # Getting the type of 'element_sum' (line 131)
                element_sum_597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 44), 'element_sum')
                # Applying the binary operator '+' (line 131)
                result_add_598 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 30), '+', subscript_call_result_596, element_sum_597)
                
                # Getting the type of 'bias' (line 131)
                bias_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'bias')
                # Getting the type of 'klass' (line 131)
                klass_600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 21), 'klass')
                # Storing an element on a container (line 131)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 16), bias_599, (klass_600, result_add_598))
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
    stypy_return_type_601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_601)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'train_adatron'
    return stypy_return_type_601

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
    float_602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 17), 'float')
    # Assigning a type to the variable 'prediction' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'prediction', float_602)
    
    # Assigning a ListComp to a Name (line 135):
    
    # Assigning a ListComp to a Name (line 135):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 135)
    # Processing the call arguments (line 135)
    
    # Call to len(...): (line 135)
    # Processing the call arguments (line 135)
    
    # Obtaining the type of the subscript
    int_612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 78), 'int')
    # Getting the type of 'label_table' (line 135)
    label_table_613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 66), 'label_table', False)
    # Obtaining the member '__getitem__' of a type (line 135)
    getitem___614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 66), label_table_613, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 135)
    subscript_call_result_615 = invoke(stypy.reporting.localization.Localization(__file__, 135, 66), getitem___614, int_612)
    
    # Processing the call keyword arguments (line 135)
    kwargs_616 = {}
    # Getting the type of 'len' (line 135)
    len_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 62), 'len', False)
    # Calling len(args, kwargs) (line 135)
    len_call_result_617 = invoke(stypy.reporting.localization.Localization(__file__, 135, 62), len_611, *[subscript_call_result_615], **kwargs_616)
    
    # Processing the call keyword arguments (line 135)
    kwargs_618 = {}
    # Getting the type of 'range' (line 135)
    range_610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 56), 'range', False)
    # Calling range(args, kwargs) (line 135)
    range_call_result_619 = invoke(stypy.reporting.localization.Localization(__file__, 135, 56), range_610, *[len_call_result_617], **kwargs_618)
    
    comprehension_620 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 19), range_call_result_619)
    # Assigning a type to the variable '_' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 19), '_', comprehension_620)
    
    # Obtaining an instance of the builtin type 'list' (line 135)
    list_603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 135)
    # Adding element type (line 135)
    float_604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 20), list_603, float_604)
    
    
    # Call to len(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'kernel_table' (line 135)
    kernel_table_606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 32), 'kernel_table', False)
    # Processing the call keyword arguments (line 135)
    kwargs_607 = {}
    # Getting the type of 'len' (line 135)
    len_605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 28), 'len', False)
    # Calling len(args, kwargs) (line 135)
    len_call_result_608 = invoke(stypy.reporting.localization.Localization(__file__, 135, 28), len_605, *[kernel_table_606], **kwargs_607)
    
    # Applying the binary operator '*' (line 135)
    result_mul_609 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 20), '*', list_603, len_call_result_608)
    
    list_621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 19), list_621, result_mul_609)
    # Assigning a type to the variable 'predictions' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'predictions', list_621)
    
    
    # Call to range(...): (line 136)
    # Processing the call arguments (line 136)
    
    # Call to len(...): (line 136)
    # Processing the call arguments (line 136)
    
    # Obtaining the type of the subscript
    int_624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 39), 'int')
    # Getting the type of 'label_table' (line 136)
    label_table_625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 27), 'label_table', False)
    # Obtaining the member '__getitem__' of a type (line 136)
    getitem___626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 27), label_table_625, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 136)
    subscript_call_result_627 = invoke(stypy.reporting.localization.Localization(__file__, 136, 27), getitem___626, int_624)
    
    # Processing the call keyword arguments (line 136)
    kwargs_628 = {}
    # Getting the type of 'len' (line 136)
    len_623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 23), 'len', False)
    # Calling len(args, kwargs) (line 136)
    len_call_result_629 = invoke(stypy.reporting.localization.Localization(__file__, 136, 23), len_623, *[subscript_call_result_627], **kwargs_628)
    
    # Processing the call keyword arguments (line 136)
    kwargs_630 = {}
    # Getting the type of 'range' (line 136)
    range_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 17), 'range', False)
    # Calling range(args, kwargs) (line 136)
    range_call_result_631 = invoke(stypy.reporting.localization.Localization(__file__, 136, 17), range_622, *[len_call_result_629], **kwargs_630)
    
    # Assigning a type to the variable 'range_call_result_631' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'range_call_result_631', range_call_result_631)
    # Testing if the for loop is going to be iterated (line 136)
    # Testing the type of a for loop iterable (line 136)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 136, 4), range_call_result_631)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 136, 4), range_call_result_631):
        # Getting the type of the for loop variable (line 136)
        for_loop_var_632 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 136, 4), range_call_result_631)
        # Assigning a type to the variable 'klass' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'klass', for_loop_var_632)
        # SSA begins for a for statement (line 136)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 137)
        # Processing the call arguments (line 137)
        
        # Call to len(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'kernel_table' (line 137)
        kernel_table_635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 37), 'kernel_table', False)
        # Processing the call keyword arguments (line 137)
        kwargs_636 = {}
        # Getting the type of 'len' (line 137)
        len_634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 33), 'len', False)
        # Calling len(args, kwargs) (line 137)
        len_call_result_637 = invoke(stypy.reporting.localization.Localization(__file__, 137, 33), len_634, *[kernel_table_635], **kwargs_636)
        
        # Processing the call keyword arguments (line 137)
        kwargs_638 = {}
        # Getting the type of 'range' (line 137)
        range_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 27), 'range', False)
        # Calling range(args, kwargs) (line 137)
        range_call_result_639 = invoke(stypy.reporting.localization.Localization(__file__, 137, 27), range_633, *[len_call_result_637], **kwargs_638)
        
        # Assigning a type to the variable 'range_call_result_639' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'range_call_result_639', range_call_result_639)
        # Testing if the for loop is going to be iterated (line 137)
        # Testing the type of a for loop iterable (line 137)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 137, 8), range_call_result_639)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 137, 8), range_call_result_639):
            # Getting the type of the for loop variable (line 137)
            for_loop_var_640 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 137, 8), range_call_result_639)
            # Assigning a type to the variable 'col_counter' (line 137)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'col_counter', for_loop_var_640)
            # SSA begins for a for statement (line 137)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to range(...): (line 138)
            # Processing the call arguments (line 138)
            
            # Call to len(...): (line 138)
            # Processing the call arguments (line 138)
            # Getting the type of 'kernel_table' (line 138)
            kernel_table_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 41), 'kernel_table', False)
            # Processing the call keyword arguments (line 138)
            kwargs_644 = {}
            # Getting the type of 'len' (line 138)
            len_642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 37), 'len', False)
            # Calling len(args, kwargs) (line 138)
            len_call_result_645 = invoke(stypy.reporting.localization.Localization(__file__, 138, 37), len_642, *[kernel_table_643], **kwargs_644)
            
            # Processing the call keyword arguments (line 138)
            kwargs_646 = {}
            # Getting the type of 'range' (line 138)
            range_641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 31), 'range', False)
            # Calling range(args, kwargs) (line 138)
            range_call_result_647 = invoke(stypy.reporting.localization.Localization(__file__, 138, 31), range_641, *[len_call_result_645], **kwargs_646)
            
            # Assigning a type to the variable 'range_call_result_647' (line 138)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'range_call_result_647', range_call_result_647)
            # Testing if the for loop is going to be iterated (line 138)
            # Testing the type of a for loop iterable (line 138)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 138, 12), range_call_result_647)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 138, 12), range_call_result_647):
                # Getting the type of the for loop variable (line 138)
                for_loop_var_648 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 138, 12), range_call_result_647)
                # Assigning a type to the variable 'row_counter' (line 138)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'row_counter', for_loop_var_648)
                # SSA begins for a for statement (line 138)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'prediction' (line 139)
                prediction_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 'prediction')
                
                # Obtaining the type of the subscript
                # Getting the type of 'row_counter' (line 139)
                row_counter_650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 56), 'row_counter')
                
                # Obtaining the type of the subscript
                # Getting the type of 'col_counter' (line 139)
                col_counter_651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 43), 'col_counter')
                # Getting the type of 'kernel_table' (line 139)
                kernel_table_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 30), 'kernel_table')
                # Obtaining the member '__getitem__' of a type (line 139)
                getitem___653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 30), kernel_table_652, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 139)
                subscript_call_result_654 = invoke(stypy.reporting.localization.Localization(__file__, 139, 30), getitem___653, col_counter_651)
                
                # Obtaining the member '__getitem__' of a type (line 139)
                getitem___655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 30), subscript_call_result_654, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 139)
                subscript_call_result_656 = invoke(stypy.reporting.localization.Localization(__file__, 139, 30), getitem___655, row_counter_650)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 140)
                klass_657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 55), 'klass')
                
                # Obtaining the type of the subscript
                # Getting the type of 'row_counter' (line 140)
                row_counter_658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 42), 'row_counter')
                # Getting the type of 'label_table' (line 140)
                label_table_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 30), 'label_table')
                # Obtaining the member '__getitem__' of a type (line 140)
                getitem___660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 30), label_table_659, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 140)
                subscript_call_result_661 = invoke(stypy.reporting.localization.Localization(__file__, 140, 30), getitem___660, row_counter_658)
                
                # Obtaining the member '__getitem__' of a type (line 140)
                getitem___662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 30), subscript_call_result_661, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 140)
                subscript_call_result_663 = invoke(stypy.reporting.localization.Localization(__file__, 140, 30), getitem___662, klass_657)
                
                # Applying the binary operator '*' (line 139)
                result_mul_664 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 30), '*', subscript_call_result_656, subscript_call_result_663)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'row_counter' (line 140)
                row_counter_665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 78), 'row_counter')
                
                # Obtaining the type of the subscript
                # Getting the type of 'klass' (line 140)
                klass_666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 71), 'klass')
                # Getting the type of 'alphas' (line 140)
                alphas_667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 64), 'alphas')
                # Obtaining the member '__getitem__' of a type (line 140)
                getitem___668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 64), alphas_667, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 140)
                subscript_call_result_669 = invoke(stypy.reporting.localization.Localization(__file__, 140, 64), getitem___668, klass_666)
                
                # Obtaining the member '__getitem__' of a type (line 140)
                getitem___670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 64), subscript_call_result_669, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 140)
                subscript_call_result_671 = invoke(stypy.reporting.localization.Localization(__file__, 140, 64), getitem___670, row_counter_665)
                
                # Applying the binary operator '*' (line 140)
                result_mul_672 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 62), '*', result_mul_664, subscript_call_result_671)
                
                # Applying the binary operator '+=' (line 139)
                result_iadd_673 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 16), '+=', prediction_649, result_mul_672)
                # Assigning a type to the variable 'prediction' (line 139)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 'prediction', result_iadd_673)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a BinOp to a Subscript (line 141):
            
            # Assigning a BinOp to a Subscript (line 141):
            # Getting the type of 'prediction' (line 141)
            prediction_674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 46), 'prediction')
            
            # Obtaining the type of the subscript
            # Getting the type of 'klass' (line 141)
            klass_675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 64), 'klass')
            # Getting the type of 'bias' (line 141)
            bias_676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 59), 'bias')
            # Obtaining the member '__getitem__' of a type (line 141)
            getitem___677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 59), bias_676, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 141)
            subscript_call_result_678 = invoke(stypy.reporting.localization.Localization(__file__, 141, 59), getitem___677, klass_675)
            
            # Applying the binary operator '+' (line 141)
            result_add_679 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 46), '+', prediction_674, subscript_call_result_678)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'klass' (line 141)
            klass_680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 24), 'klass')
            # Getting the type of 'predictions' (line 141)
            predictions_681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'predictions')
            # Obtaining the member '__getitem__' of a type (line 141)
            getitem___682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 12), predictions_681, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 141)
            subscript_call_result_683 = invoke(stypy.reporting.localization.Localization(__file__, 141, 12), getitem___682, klass_680)
            
            # Getting the type of 'col_counter' (line 141)
            col_counter_684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 31), 'col_counter')
            # Storing an element on a container (line 141)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 12), subscript_call_result_683, (col_counter_684, result_add_679))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    
    # Call to range(...): (line 143)
    # Processing the call arguments (line 143)
    
    # Call to len(...): (line 143)
    # Processing the call arguments (line 143)
    # Getting the type of 'kernel_table' (line 143)
    kernel_table_687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 33), 'kernel_table', False)
    # Processing the call keyword arguments (line 143)
    kwargs_688 = {}
    # Getting the type of 'len' (line 143)
    len_686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 29), 'len', False)
    # Calling len(args, kwargs) (line 143)
    len_call_result_689 = invoke(stypy.reporting.localization.Localization(__file__, 143, 29), len_686, *[kernel_table_687], **kwargs_688)
    
    # Processing the call keyword arguments (line 143)
    kwargs_690 = {}
    # Getting the type of 'range' (line 143)
    range_685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 23), 'range', False)
    # Calling range(args, kwargs) (line 143)
    range_call_result_691 = invoke(stypy.reporting.localization.Localization(__file__, 143, 23), range_685, *[len_call_result_689], **kwargs_690)
    
    # Assigning a type to the variable 'range_call_result_691' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'range_call_result_691', range_call_result_691)
    # Testing if the for loop is going to be iterated (line 143)
    # Testing the type of a for loop iterable (line 143)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 143, 4), range_call_result_691)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 143, 4), range_call_result_691):
        # Getting the type of the for loop variable (line 143)
        for_loop_var_692 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 143, 4), range_call_result_691)
        # Assigning a type to the variable 'col_counter' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'col_counter', for_loop_var_692)
        # SSA begins for a for statement (line 143)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a List to a Name (line 144):
        
        # Assigning a List to a Name (line 144):
        
        # Obtaining an instance of the builtin type 'list' (line 144)
        list_693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 144)
        
        # Assigning a type to the variable 'current_predictions' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'current_predictions', list_693)
        
        # Assigning a Num to a Name (line 145):
        
        # Assigning a Num to a Name (line 145):
        int_694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 16), 'int')
        # Assigning a type to the variable 'error' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'error', int_694)
        
        
        # Call to range(...): (line 146)
        # Processing the call arguments (line 146)
        
        # Call to len(...): (line 146)
        # Processing the call arguments (line 146)
        
        # Obtaining the type of the subscript
        int_697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 49), 'int')
        # Getting the type of 'label_table' (line 146)
        label_table_698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 37), 'label_table', False)
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 37), label_table_698, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_700 = invoke(stypy.reporting.localization.Localization(__file__, 146, 37), getitem___699, int_697)
        
        # Processing the call keyword arguments (line 146)
        kwargs_701 = {}
        # Getting the type of 'len' (line 146)
        len_696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 33), 'len', False)
        # Calling len(args, kwargs) (line 146)
        len_call_result_702 = invoke(stypy.reporting.localization.Localization(__file__, 146, 33), len_696, *[subscript_call_result_700], **kwargs_701)
        
        # Processing the call keyword arguments (line 146)
        kwargs_703 = {}
        # Getting the type of 'range' (line 146)
        range_695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 27), 'range', False)
        # Calling range(args, kwargs) (line 146)
        range_call_result_704 = invoke(stypy.reporting.localization.Localization(__file__, 146, 27), range_695, *[len_call_result_702], **kwargs_703)
        
        # Assigning a type to the variable 'range_call_result_704' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'range_call_result_704', range_call_result_704)
        # Testing if the for loop is going to be iterated (line 146)
        # Testing the type of a for loop iterable (line 146)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 146, 8), range_call_result_704)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 146, 8), range_call_result_704):
            # Getting the type of the for loop variable (line 146)
            for_loop_var_705 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 146, 8), range_call_result_704)
            # Assigning a type to the variable 'row_counter' (line 146)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'row_counter', for_loop_var_705)
            # SSA begins for a for statement (line 146)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to append(...): (line 147)
            # Processing the call arguments (line 147)
            
            # Obtaining the type of the subscript
            # Getting the type of 'col_counter' (line 147)
            col_counter_708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 64), 'col_counter', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row_counter' (line 147)
            row_counter_709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 51), 'row_counter', False)
            # Getting the type of 'predictions' (line 147)
            predictions_710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 39), 'predictions', False)
            # Obtaining the member '__getitem__' of a type (line 147)
            getitem___711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 39), predictions_710, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 147)
            subscript_call_result_712 = invoke(stypy.reporting.localization.Localization(__file__, 147, 39), getitem___711, row_counter_709)
            
            # Obtaining the member '__getitem__' of a type (line 147)
            getitem___713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 39), subscript_call_result_712, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 147)
            subscript_call_result_714 = invoke(stypy.reporting.localization.Localization(__file__, 147, 39), getitem___713, col_counter_708)
            
            # Processing the call keyword arguments (line 147)
            kwargs_715 = {}
            # Getting the type of 'current_predictions' (line 147)
            current_predictions_706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'current_predictions', False)
            # Obtaining the member 'append' of a type (line 147)
            append_707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 12), current_predictions_706, 'append')
            # Calling append(args, kwargs) (line 147)
            append_call_result_716 = invoke(stypy.reporting.localization.Localization(__file__, 147, 12), append_707, *[subscript_call_result_714], **kwargs_715)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Name (line 149):
        
        # Assigning a Call to a Name (line 149):
        
        # Call to index(...): (line 149)
        # Processing the call arguments (line 149)
        
        # Call to max(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'current_predictions' (line 149)
        current_predictions_720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 56), 'current_predictions', False)
        # Processing the call keyword arguments (line 149)
        kwargs_721 = {}
        # Getting the type of 'max' (line 149)
        max_719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 52), 'max', False)
        # Calling max(args, kwargs) (line 149)
        max_call_result_722 = invoke(stypy.reporting.localization.Localization(__file__, 149, 52), max_719, *[current_predictions_720], **kwargs_721)
        
        # Processing the call keyword arguments (line 149)
        kwargs_723 = {}
        # Getting the type of 'current_predictions' (line 149)
        current_predictions_717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 26), 'current_predictions', False)
        # Obtaining the member 'index' of a type (line 149)
        index_718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 26), current_predictions_717, 'index')
        # Calling index(args, kwargs) (line 149)
        index_call_result_724 = invoke(stypy.reporting.localization.Localization(__file__, 149, 26), index_718, *[max_call_result_722], **kwargs_723)
        
        # Assigning a type to the variable 'predicted_class' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'predicted_class', index_call_result_724)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'predicted_class' (line 151)
        predicted_class_725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 36), 'predicted_class')
        
        # Obtaining the type of the subscript
        # Getting the type of 'col_counter' (line 151)
        col_counter_726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 23), 'col_counter')
        # Getting the type of 'label_table' (line 151)
        label_table_727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 11), 'label_table')
        # Obtaining the member '__getitem__' of a type (line 151)
        getitem___728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 11), label_table_727, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 151)
        subscript_call_result_729 = invoke(stypy.reporting.localization.Localization(__file__, 151, 11), getitem___728, col_counter_726)
        
        # Obtaining the member '__getitem__' of a type (line 151)
        getitem___730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 11), subscript_call_result_729, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 151)
        subscript_call_result_731 = invoke(stypy.reporting.localization.Localization(__file__, 151, 11), getitem___730, predicted_class_725)
        
        int_732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 55), 'int')
        # Applying the binary operator '<' (line 151)
        result_lt_733 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 11), '<', subscript_call_result_731, int_732)
        
        # Testing if the type of an if condition is none (line 151)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 151, 8), result_lt_733):
            pass
        else:
            
            # Testing the type of an if condition (line 151)
            if_condition_734 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 8), result_lt_733)
            # Assigning a type to the variable 'if_condition_734' (line 151)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'if_condition_734', if_condition_734)
            # SSA begins for if statement (line 151)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'error' (line 152)
            error_735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'error')
            int_736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 21), 'int')
            # Applying the binary operator '+=' (line 152)
            result_iadd_737 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 12), '+=', error_735, int_736)
            # Assigning a type to the variable 'error' (line 152)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'error', result_iadd_737)
            
            # SSA join for if statement (line 151)
            module_type_store = module_type_store.join_ssa_context()
            

        float_738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 15), 'float')
        # Getting the type of 'error' (line 154)
        error_739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 21), 'error')
        # Applying the binary operator '*' (line 154)
        result_mul_740 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 15), '*', float_738, error_739)
        
        
        # Call to len(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'kernel_table' (line 154)
        kernel_table_742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 33), 'kernel_table', False)
        # Processing the call keyword arguments (line 154)
        kwargs_743 = {}
        # Getting the type of 'len' (line 154)
        len_741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 29), 'len', False)
        # Calling len(args, kwargs) (line 154)
        len_call_result_744 = invoke(stypy.reporting.localization.Localization(__file__, 154, 29), len_741, *[kernel_table_742], **kwargs_743)
        
        # Applying the binary operator 'div' (line 154)
        result_div_745 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 27), 'div', result_mul_740, len_call_result_744)
        
        # Assigning a type to the variable 'stypy_return_type' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'stypy_return_type', result_div_745)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'calculate_error(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'calculate_error' in the type store
    # Getting the type of 'stypy_return_type' (line 133)
    stypy_return_type_746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_746)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'calculate_error'
    return stypy_return_type_746

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
    list_747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 158)
    # Adding element type (line 158)
    
    # Obtaining an instance of the builtin type 'tuple' (line 158)
    tuple_748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 158)
    # Adding element type (line 158)
    
    # Call to Relative(...): (line 158)
    # Processing the call arguments (line 158)
    str_750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 37), 'str', 'testdata/c.txt')
    # Processing the call keyword arguments (line 158)
    kwargs_751 = {}
    # Getting the type of 'Relative' (line 158)
    Relative_749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 28), 'Relative', False)
    # Calling Relative(args, kwargs) (line 158)
    Relative_call_result_752 = invoke(stypy.reporting.localization.Localization(__file__, 158, 28), Relative_749, *[str_750], **kwargs_751)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 28), tuple_748, Relative_call_result_752)
    # Adding element type (line 158)
    # Getting the type of 'CYTOSOLIC' (line 158)
    CYTOSOLIC_753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 56), 'CYTOSOLIC')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 28), tuple_748, CYTOSOLIC_753)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 26), list_747, tuple_748)
    # Adding element type (line 158)
    
    # Obtaining an instance of the builtin type 'tuple' (line 158)
    tuple_754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 69), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 158)
    # Adding element type (line 158)
    
    # Call to Relative(...): (line 158)
    # Processing the call arguments (line 158)
    str_756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 78), 'str', 'testdata/e.txt')
    # Processing the call keyword arguments (line 158)
    kwargs_757 = {}
    # Getting the type of 'Relative' (line 158)
    Relative_755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 69), 'Relative', False)
    # Calling Relative(args, kwargs) (line 158)
    Relative_call_result_758 = invoke(stypy.reporting.localization.Localization(__file__, 158, 69), Relative_755, *[str_756], **kwargs_757)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 69), tuple_754, Relative_call_result_758)
    # Adding element type (line 158)
    # Getting the type of 'EXTRACELLULAR' (line 158)
    EXTRACELLULAR_759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 97), 'EXTRACELLULAR')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 69), tuple_754, EXTRACELLULAR_759)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 26), list_747, tuple_754)
    # Adding element type (line 158)
    
    # Obtaining an instance of the builtin type 'tuple' (line 158)
    tuple_760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 114), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 158)
    # Adding element type (line 158)
    
    # Call to Relative(...): (line 158)
    # Processing the call arguments (line 158)
    str_762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 123), 'str', 'testdata/n.txt')
    # Processing the call keyword arguments (line 158)
    kwargs_763 = {}
    # Getting the type of 'Relative' (line 158)
    Relative_761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 114), 'Relative', False)
    # Calling Relative(args, kwargs) (line 158)
    Relative_call_result_764 = invoke(stypy.reporting.localization.Localization(__file__, 158, 114), Relative_761, *[str_762], **kwargs_763)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 114), tuple_760, Relative_call_result_764)
    # Adding element type (line 158)
    # Getting the type of 'NUCLEAR' (line 158)
    NUCLEAR_765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 142), 'NUCLEAR')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 114), tuple_760, NUCLEAR_765)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 26), list_747, tuple_760)
    # Adding element type (line 158)
    
    # Obtaining an instance of the builtin type 'tuple' (line 158)
    tuple_766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 153), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 158)
    # Adding element type (line 158)
    
    # Call to Relative(...): (line 158)
    # Processing the call arguments (line 158)
    str_768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 162), 'str', 'testdata/m.txt')
    # Processing the call keyword arguments (line 158)
    kwargs_769 = {}
    # Getting the type of 'Relative' (line 158)
    Relative_767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 153), 'Relative', False)
    # Calling Relative(args, kwargs) (line 158)
    Relative_call_result_770 = invoke(stypy.reporting.localization.Localization(__file__, 158, 153), Relative_767, *[str_768], **kwargs_769)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 153), tuple_766, Relative_call_result_770)
    # Adding element type (line 158)
    # Getting the type of 'MITOCHONDRIAL' (line 158)
    MITOCHONDRIAL_771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 181), 'MITOCHONDRIAL')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 153), tuple_766, MITOCHONDRIAL_771)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 26), list_747, tuple_766)
    
    # Assigning a type to the variable 'list_747' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'list_747', list_747)
    # Testing if the for loop is going to be iterated (line 158)
    # Testing the type of a for loop iterable (line 158)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 158, 4), list_747)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 158, 4), list_747):
        # Getting the type of the for loop variable (line 158)
        for_loop_var_772 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 158, 4), list_747)
        # Assigning a type to the variable 'filename' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'filename', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 4), for_loop_var_772, 2, 0))
        # Assigning a type to the variable 'type' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'type', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 4), for_loop_var_772, 2, 1))
        # SSA begins for a for statement (line 158)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to load_file(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'filename' (line 159)
        filename_774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 18), 'filename', False)
        # Getting the type of 'type' (line 159)
        type_775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 28), 'type', False)
        # Processing the call keyword arguments (line 159)
        kwargs_776 = {}
        # Getting the type of 'load_file' (line 159)
        load_file_773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'load_file', False)
        # Calling load_file(args, kwargs) (line 159)
        load_file_call_result_777 = invoke(stypy.reporting.localization.Localization(__file__, 159, 8), load_file_773, *[filename_774, type_775], **kwargs_776)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a Call to a Tuple (line 161):
    
    # Assigning a Call to a Name:
    
    # Call to create_tables(...): (line 161)
    # Processing the call keyword arguments (line 161)
    kwargs_779 = {}
    # Getting the type of 'create_tables' (line 161)
    create_tables_778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 33), 'create_tables', False)
    # Calling create_tables(args, kwargs) (line 161)
    create_tables_call_result_780 = invoke(stypy.reporting.localization.Localization(__file__, 161, 33), create_tables_778, *[], **kwargs_779)
    
    # Assigning a type to the variable 'call_assignment_7' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'call_assignment_7', create_tables_call_result_780)
    
    # Assigning a Call to a Name (line 161):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_7' (line 161)
    call_assignment_7_781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'call_assignment_7', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_782 = stypy_get_value_from_tuple(call_assignment_7_781, 2, 0)
    
    # Assigning a type to the variable 'call_assignment_8' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'call_assignment_8', stypy_get_value_from_tuple_call_result_782)
    
    # Assigning a Name to a Name (line 161):
    # Getting the type of 'call_assignment_8' (line 161)
    call_assignment_8_783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'call_assignment_8')
    # Assigning a type to the variable 'feature_table' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'feature_table', call_assignment_8_783)
    
    # Assigning a Call to a Name (line 161):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_7' (line 161)
    call_assignment_7_784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'call_assignment_7', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_785 = stypy_get_value_from_tuple(call_assignment_7_784, 2, 1)
    
    # Assigning a type to the variable 'call_assignment_9' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'call_assignment_9', stypy_get_value_from_tuple_call_result_785)
    
    # Assigning a Name to a Name (line 161):
    # Getting the type of 'call_assignment_9' (line 161)
    call_assignment_9_786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'call_assignment_9')
    # Assigning a type to the variable 'label_table' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 19), 'label_table', call_assignment_9_786)
    
    # Assigning a Call to a Name (line 168):
    
    # Assigning a Call to a Name (line 168):
    
    # Call to create_kernel_table(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'feature_table' (line 168)
    feature_table_788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 39), 'feature_table', False)
    # Processing the call keyword arguments (line 168)
    kwargs_789 = {}
    # Getting the type of 'create_kernel_table' (line 168)
    create_kernel_table_787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 19), 'create_kernel_table', False)
    # Calling create_kernel_table(args, kwargs) (line 168)
    create_kernel_table_call_result_790 = invoke(stypy.reporting.localization.Localization(__file__, 168, 19), create_kernel_table_787, *[feature_table_788], **kwargs_789)
    
    # Assigning a type to the variable 'kernel_table' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'kernel_table', create_kernel_table_call_result_790)
    
    # Assigning a Call to a Tuple (line 170):
    
    # Assigning a Call to a Name:
    
    # Call to train_adatron(...): (line 170)
    # Processing the call arguments (line 170)
    # Getting the type of 'kernel_table' (line 170)
    kernel_table_792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 33), 'kernel_table', False)
    # Getting the type of 'label_table' (line 170)
    label_table_793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 47), 'label_table', False)
    float_794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 60), 'float')
    float_795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 65), 'float')
    # Processing the call keyword arguments (line 170)
    kwargs_796 = {}
    # Getting the type of 'train_adatron' (line 170)
    train_adatron_791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 19), 'train_adatron', False)
    # Calling train_adatron(args, kwargs) (line 170)
    train_adatron_call_result_797 = invoke(stypy.reporting.localization.Localization(__file__, 170, 19), train_adatron_791, *[kernel_table_792, label_table_793, float_794, float_795], **kwargs_796)
    
    # Assigning a type to the variable 'call_assignment_10' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'call_assignment_10', train_adatron_call_result_797)
    
    # Assigning a Call to a Name (line 170):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_10' (line 170)
    call_assignment_10_798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'call_assignment_10', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_799 = stypy_get_value_from_tuple(call_assignment_10_798, 2, 0)
    
    # Assigning a type to the variable 'call_assignment_11' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'call_assignment_11', stypy_get_value_from_tuple_call_result_799)
    
    # Assigning a Name to a Name (line 170):
    # Getting the type of 'call_assignment_11' (line 170)
    call_assignment_11_800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'call_assignment_11')
    # Assigning a type to the variable 'alphas' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'alphas', call_assignment_11_800)
    
    # Assigning a Call to a Name (line 170):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_10' (line 170)
    call_assignment_10_801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'call_assignment_10', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_802 = stypy_get_value_from_tuple(call_assignment_10_801, 2, 1)
    
    # Assigning a type to the variable 'call_assignment_12' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'call_assignment_12', stypy_get_value_from_tuple_call_result_802)
    
    # Assigning a Name to a Name (line 170):
    # Getting the type of 'call_assignment_12' (line 170)
    call_assignment_12_803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'call_assignment_12')
    # Assigning a type to the variable 'bias' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'bias', call_assignment_12_803)
    
    # Call to calculate_error(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'alphas' (line 172)
    alphas_805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 20), 'alphas', False)
    # Getting the type of 'bias' (line 172)
    bias_806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 28), 'bias', False)
    # Getting the type of 'kernel_table' (line 172)
    kernel_table_807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 34), 'kernel_table', False)
    # Getting the type of 'label_table' (line 172)
    label_table_808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 48), 'label_table', False)
    # Processing the call keyword arguments (line 172)
    kwargs_809 = {}
    # Getting the type of 'calculate_error' (line 172)
    calculate_error_804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'calculate_error', False)
    # Calling calculate_error(args, kwargs) (line 172)
    calculate_error_call_result_810 = invoke(stypy.reporting.localization.Localization(__file__, 172, 4), calculate_error_804, *[alphas_805, bias_806, kernel_table_807, label_table_808], **kwargs_809)
    
    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 157)
    stypy_return_type_811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_811)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_811

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
    kwargs_813 = {}
    # Getting the type of 'main' (line 176)
    main_812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'main', False)
    # Calling main(args, kwargs) (line 176)
    main_call_result_814 = invoke(stypy.reporting.localization.Localization(__file__, 176, 4), main_812, *[], **kwargs_813)
    
    # Getting the type of 'True' (line 177)
    True_815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'stypy_return_type', True_815)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 175)
    stypy_return_type_816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_816)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_816

# Assigning a type to the variable 'run' (line 175)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 0), 'run', run)

# Call to run(...): (line 179)
# Processing the call keyword arguments (line 179)
kwargs_818 = {}
# Getting the type of 'run' (line 179)
run_817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 0), 'run', False)
# Calling run(args, kwargs) (line 179)
run_call_result_819 = invoke(stypy.reporting.localization.Localization(__file__, 179, 0), run_817, *[], **kwargs_818)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

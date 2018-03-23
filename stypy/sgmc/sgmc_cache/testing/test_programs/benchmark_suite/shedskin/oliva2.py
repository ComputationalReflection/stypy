
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ### see output oliva.pgm
2: 
3: '''
4: Models for the simulations of the color pattern on the shells of mollusks
5: see also: Meinhardt,H. and Klingler,M. (1987) J. theor. Biol 126, 63-69
6: see also: H.Meinhardt: "Algorithmic beauty of sea shells"
7: (Springer Verlag) (c) H.Meinhardt, Tubingen
8: 
9: This is a short version of a program for the simulations of the color
10: patterns on tropical sea shells, here # Oliva porphyria# .
11: An autocatalytic activator a(i) leads to a burst-like activation
12: that is regulated back by the action of an inhibitor b(i). The life
13: time of the inhibitor is regulated via a hormone c, that is
14: homogeneously distributed along the growing edge. Whenever the number
15: of activated cells cells become too small, active cells # ain activated
16: until backwards waves are triggered
17: ------------------
18: 
19: Translated by Bearophile from QBASIC to SSPython (for ShedSkin), V. 1.1, Feb 17 2006.
20: This program requires Python and ShedSkin (http://shedskin.sourceforge.net).
21: '''
22: 
23: from random import random, randint
24: import os
25: 
26: 
27: def Relative(path):
28:     return os.path.join(os.path.dirname(__file__), path)
29: 
30: 
31: class SavePGMlines:
32:     '''SavePGMlines(matrix, filename): class, saves a PGM, 256 greys for each pixel, line by line.
33:     Values can be int or float in [0, 255].'''
34: 
35:     def __init__(self, filename, ny):
36:         self.out_file = file(Relative(filename), 'w')
37:         self.ny = ny
38:         self.nx = 0  # Unknown
39:         self.written_lines_count = 0  # lines written count
40: 
41:     def saverow(self, row):
42:         if self.written_lines_count:
43:             assert len(row) == self.nx
44:             assert self.written_lines_count < self.ny
45:         else:
46:             self.nx = len(row)
47:             # PPM header
48:         ##            print >>self.out_file, "P2"
49:         ##            print >>self.out_file, "# Image created with ppmlib."
50:         ##            print >>self.out_file, self.nx, self.ny
51:         ##            print >>self.out_file, "256"
52: 
53:         out_line = [""] * self.nx
54:         for i in xrange(self.nx):
55:             ipixel = int(round(row[i]))
56:             if ipixel >= 255:
57:                 out_line[i] = "255"
58:             elif ipixel <= 0:
59:                 out_line[i] = "0"
60:             else:
61:                 out_line[i] = str(ipixel)
62: 
63:         ##        print >>self.out_file, " ".join(out_line)
64:         self.written_lines_count += 1
65:         if self.written_lines_count == self.ny:
66:             self.out_file.close()
67: 
68: 
69: def oliva(nx=600,  # Length of the computed screen matrix (number of cells)
70:           ny=500,  # Height of the computed screen matrix
71:           kp=12,  # number of iterations between the displays ( = lines on the screen)
72:           da=0.015,  # Diffusion of the activator
73:           ra=0.1,  # Decay rate of the inhibitor
74:           ba=0.1,  # Basic production of the activator
75:           sa=0.25,  # Saturation of the autocatalysis
76:           db=0.0,  # Diffusion of the inhibitor (example = 0.0)
77:           rb=0.014,  # Decay rate of the inhibitor
78:           sb=0.1,  # Michaelis-Menten constant of the inhibition
79:           rc=0.1,  # Decay rate of the hormone
80:           out_file_name="oliva.pgm"):
81:     outPGM = SavePGMlines(out_file_name, ny)
82:     # ----------- Initial condition  --------------------------
83:     image_matrix = []  # image_matrix will become an array[ny][nx] of float
84:     c = 0.5  # Hormone-concentration, homogeneous in all cells
85:     a = [0.0] * (nx + 1)  # Activator, general initial concentration
86:     b = [0.1] * (nx + 1)  # Inhibitor, general initial concentration
87:     # z = fluctuation of the autocatalysis
88:     # z = [uniform(ra*0.96, ra*1.04) for i in xrange(nx)] # Not possible with SS yet
89:     z = [ra * (0.96 + 0.08 * random()) for i in xrange(nx)]
90: 
91:     # Seed the initially active cells, not too much close to each other
92:     # Example distribution: *              *         *                                        *
93:     i = 10
94:     for j in xrange(20):
95:         a[i] = 1.0
96:         i += randint(10, 60)
97:         if i >= nx:
98:             break
99: 
100:     # These constant factors are used again and again, therefore, they are calculated
101:     # only once at the begin of the calculation
102:     dac = 1.0 - ra - 2.0 * da
103:     dbc = 1.0 - rb - 2.0 * db
104:     dbcc = dbc
105: 
106:     for row in xrange(ny):
107:         # Begin of the iteration
108:         for niter in xrange(kp):
109:             # -------- Boundary impermeable
110:             a1 = a[0]  # a1 = concentration  of the actual cell. Since this
111:             # concentration is identical, no diffusion through the border.
112:             b1 = b[0]
113:             a[nx] = a[nx - 1]  # Concentration in a virtual right cell
114:             b[nx] = b[nx - 1]
115:             bsa = 0.0  # This will carry the sum of all activations of all cells
116: 
117:             # ---------- Reactions  ------
118:             for i in xrange(nx):  # i = actual cell, kx = right cell
119:                 af = a[i]  # local activator concentration
120:                 bf = b[i]  # local inhibitor concentration
121:                 aq = z[i] * af * af / (1.0 + sa * af * af)  # Saturating autocatalysis
122: 
123:                 # Calculation of the new activator and inhibitor concentration in cell i:
124:                 a[i] = af * dac + da * (a1 + a[i + 1]) + aq / (sb + bf)
125:                 # 1/BF => Action of the inhibitor
126:                 b[i] = bf * dbcc + db * (b1 + b[i + 1]) + aq  # new inhibitor conc.
127:                 bsa += rc * af  # Hormone production -> Sum of activations
128:                 a1 = af  # actual concentration will be concentration in left cell
129:                 b1 = bf  # in the concentration change of the next cell
130: 
131:             # New hormone concentration. 1/kx=normalization on total number of cells
132:             c = c * (1.0 - rc) + bsa / nx
133:             rbb = rb / c  # rbb => Effective life time of the inhibitor
134: 
135:             # Change in a cell by diffusion and decay. Must be recomputed since
136:             # lifetime of the inhibitor changes.
137:             dbcc = 1.0 - 2.0 * db - rbb
138: 
139:         # ----------- Plot-Save -------------
140:         outPGM.saverow([255 * a[ix] for ix in xrange(nx)])
141: 
142: 
143: def run():
144:     oliva()
145:     return True
146: 
147: 
148: run()
149: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, (-1)), 'str', '\nModels for the simulations of the color pattern on the shells of mollusks\nsee also: Meinhardt,H. and Klingler,M. (1987) J. theor. Biol 126, 63-69\nsee also: H.Meinhardt: "Algorithmic beauty of sea shells"\n(Springer Verlag) (c) H.Meinhardt, Tubingen\n\nThis is a short version of a program for the simulations of the color\npatterns on tropical sea shells, here # Oliva porphyria# .\nAn autocatalytic activator a(i) leads to a burst-like activation\nthat is regulated back by the action of an inhibitor b(i). The life\ntime of the inhibitor is regulated via a hormone c, that is\nhomogeneously distributed along the growing edge. Whenever the number\nof activated cells cells become too small, active cells # ain activated\nuntil backwards waves are triggered\n------------------\n\nTranslated by Bearophile from QBASIC to SSPython (for ShedSkin), V. 1.1, Feb 17 2006.\nThis program requires Python and ShedSkin (http://shedskin.sourceforge.net).\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from random import random, randint' statement (line 23)
try:
    from random import random, randint

except:
    random = UndefinedType
    randint = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'random', None, module_type_store, ['random', 'randint'], [random, randint])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'import os' statement (line 24)
import os

import_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'os', os, module_type_store)


@norecursion
def Relative(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Relative'
    module_type_store = module_type_store.open_function_context('Relative', 27, 0, False)
    
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

    
    # Call to join(...): (line 28)
    # Processing the call arguments (line 28)
    
    # Call to dirname(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of '__file__' (line 28)
    file___8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 40), '__file__', False)
    # Processing the call keyword arguments (line 28)
    kwargs_9 = {}
    # Getting the type of 'os' (line 28)
    os_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 28)
    path_6 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 24), os_5, 'path')
    # Obtaining the member 'dirname' of a type (line 28)
    dirname_7 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 24), path_6, 'dirname')
    # Calling dirname(args, kwargs) (line 28)
    dirname_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 28, 24), dirname_7, *[file___8], **kwargs_9)
    
    # Getting the type of 'path' (line 28)
    path_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 51), 'path', False)
    # Processing the call keyword arguments (line 28)
    kwargs_12 = {}
    # Getting the type of 'os' (line 28)
    os_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 28)
    path_3 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 11), os_2, 'path')
    # Obtaining the member 'join' of a type (line 28)
    join_4 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 11), path_3, 'join')
    # Calling join(args, kwargs) (line 28)
    join_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 28, 11), join_4, *[dirname_call_result_10, path_11], **kwargs_12)
    
    # Assigning a type to the variable 'stypy_return_type' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type', join_call_result_13)
    
    # ################# End of 'Relative(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Relative' in the type store
    # Getting the type of 'stypy_return_type' (line 27)
    stypy_return_type_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Relative'
    return stypy_return_type_14

# Assigning a type to the variable 'Relative' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'Relative', Relative)
# Declaration of the 'SavePGMlines' class

class SavePGMlines:
    str_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, (-1)), 'str', 'SavePGMlines(matrix, filename): class, saves a PGM, 256 greys for each pixel, line by line.\n    Values can be int or float in [0, 255].')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 35, 4, False)
        # Assigning a type to the variable 'self' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SavePGMlines.__init__', ['filename', 'ny'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['filename', 'ny'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 36):
        
        # Call to file(...): (line 36)
        # Processing the call arguments (line 36)
        
        # Call to Relative(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'filename' (line 36)
        filename_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 38), 'filename', False)
        # Processing the call keyword arguments (line 36)
        kwargs_19 = {}
        # Getting the type of 'Relative' (line 36)
        Relative_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 29), 'Relative', False)
        # Calling Relative(args, kwargs) (line 36)
        Relative_call_result_20 = invoke(stypy.reporting.localization.Localization(__file__, 36, 29), Relative_17, *[filename_18], **kwargs_19)
        
        str_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 49), 'str', 'w')
        # Processing the call keyword arguments (line 36)
        kwargs_22 = {}
        # Getting the type of 'file' (line 36)
        file_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 24), 'file', False)
        # Calling file(args, kwargs) (line 36)
        file_call_result_23 = invoke(stypy.reporting.localization.Localization(__file__, 36, 24), file_16, *[Relative_call_result_20, str_21], **kwargs_22)
        
        # Getting the type of 'self' (line 36)
        self_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'self')
        # Setting the type of the member 'out_file' of a type (line 36)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), self_24, 'out_file', file_call_result_23)
        
        # Assigning a Name to a Attribute (line 37):
        # Getting the type of 'ny' (line 37)
        ny_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 18), 'ny')
        # Getting the type of 'self' (line 37)
        self_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'self')
        # Setting the type of the member 'ny' of a type (line 37)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), self_26, 'ny', ny_25)
        
        # Assigning a Num to a Attribute (line 38):
        int_27 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 18), 'int')
        # Getting the type of 'self' (line 38)
        self_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self')
        # Setting the type of the member 'nx' of a type (line 38)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_28, 'nx', int_27)
        
        # Assigning a Num to a Attribute (line 39):
        int_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 35), 'int')
        # Getting the type of 'self' (line 39)
        self_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'self')
        # Setting the type of the member 'written_lines_count' of a type (line 39)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), self_30, 'written_lines_count', int_29)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def saverow(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'saverow'
        module_type_store = module_type_store.open_function_context('saverow', 41, 4, False)
        # Assigning a type to the variable 'self' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SavePGMlines.saverow.__dict__.__setitem__('stypy_localization', localization)
        SavePGMlines.saverow.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SavePGMlines.saverow.__dict__.__setitem__('stypy_type_store', module_type_store)
        SavePGMlines.saverow.__dict__.__setitem__('stypy_function_name', 'SavePGMlines.saverow')
        SavePGMlines.saverow.__dict__.__setitem__('stypy_param_names_list', ['row'])
        SavePGMlines.saverow.__dict__.__setitem__('stypy_varargs_param_name', None)
        SavePGMlines.saverow.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SavePGMlines.saverow.__dict__.__setitem__('stypy_call_defaults', defaults)
        SavePGMlines.saverow.__dict__.__setitem__('stypy_call_varargs', varargs)
        SavePGMlines.saverow.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SavePGMlines.saverow.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SavePGMlines.saverow', ['row'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'saverow', localization, ['row'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'saverow(...)' code ##################

        # Getting the type of 'self' (line 42)
        self_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'self')
        # Obtaining the member 'written_lines_count' of a type (line 42)
        written_lines_count_32 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 11), self_31, 'written_lines_count')
        # Testing if the type of an if condition is none (line 42)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 42, 8), written_lines_count_32):
            
            # Assigning a Call to a Attribute (line 46):
            
            # Call to len(...): (line 46)
            # Processing the call arguments (line 46)
            # Getting the type of 'row' (line 46)
            row_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 26), 'row', False)
            # Processing the call keyword arguments (line 46)
            kwargs_50 = {}
            # Getting the type of 'len' (line 46)
            len_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'len', False)
            # Calling len(args, kwargs) (line 46)
            len_call_result_51 = invoke(stypy.reporting.localization.Localization(__file__, 46, 22), len_48, *[row_49], **kwargs_50)
            
            # Getting the type of 'self' (line 46)
            self_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'self')
            # Setting the type of the member 'nx' of a type (line 46)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 12), self_52, 'nx', len_call_result_51)
        else:
            
            # Testing the type of an if condition (line 42)
            if_condition_33 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 8), written_lines_count_32)
            # Assigning a type to the variable 'if_condition_33' (line 42)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'if_condition_33', if_condition_33)
            # SSA begins for if statement (line 42)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Evaluating assert statement condition
            
            
            # Call to len(...): (line 43)
            # Processing the call arguments (line 43)
            # Getting the type of 'row' (line 43)
            row_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 23), 'row', False)
            # Processing the call keyword arguments (line 43)
            kwargs_36 = {}
            # Getting the type of 'len' (line 43)
            len_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 19), 'len', False)
            # Calling len(args, kwargs) (line 43)
            len_call_result_37 = invoke(stypy.reporting.localization.Localization(__file__, 43, 19), len_34, *[row_35], **kwargs_36)
            
            # Getting the type of 'self' (line 43)
            self_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 31), 'self')
            # Obtaining the member 'nx' of a type (line 43)
            nx_39 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 31), self_38, 'nx')
            # Applying the binary operator '==' (line 43)
            result_eq_40 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 19), '==', len_call_result_37, nx_39)
            
            assert_41 = result_eq_40
            # Assigning a type to the variable 'assert_41' (line 43)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'assert_41', result_eq_40)
            # Evaluating assert statement condition
            
            # Getting the type of 'self' (line 44)
            self_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 19), 'self')
            # Obtaining the member 'written_lines_count' of a type (line 44)
            written_lines_count_43 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 19), self_42, 'written_lines_count')
            # Getting the type of 'self' (line 44)
            self_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 46), 'self')
            # Obtaining the member 'ny' of a type (line 44)
            ny_45 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 46), self_44, 'ny')
            # Applying the binary operator '<' (line 44)
            result_lt_46 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 19), '<', written_lines_count_43, ny_45)
            
            assert_47 = result_lt_46
            # Assigning a type to the variable 'assert_47' (line 44)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'assert_47', result_lt_46)
            # SSA branch for the else part of an if statement (line 42)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Attribute (line 46):
            
            # Call to len(...): (line 46)
            # Processing the call arguments (line 46)
            # Getting the type of 'row' (line 46)
            row_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 26), 'row', False)
            # Processing the call keyword arguments (line 46)
            kwargs_50 = {}
            # Getting the type of 'len' (line 46)
            len_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'len', False)
            # Calling len(args, kwargs) (line 46)
            len_call_result_51 = invoke(stypy.reporting.localization.Localization(__file__, 46, 22), len_48, *[row_49], **kwargs_50)
            
            # Getting the type of 'self' (line 46)
            self_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'self')
            # Setting the type of the member 'nx' of a type (line 46)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 12), self_52, 'nx', len_call_result_51)
            # SSA join for if statement (line 42)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 53):
        
        # Obtaining an instance of the builtin type 'list' (line 53)
        list_53 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 53)
        # Adding element type (line 53)
        str_54 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 20), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 19), list_53, str_54)
        
        # Getting the type of 'self' (line 53)
        self_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 26), 'self')
        # Obtaining the member 'nx' of a type (line 53)
        nx_56 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 26), self_55, 'nx')
        # Applying the binary operator '*' (line 53)
        result_mul_57 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 19), '*', list_53, nx_56)
        
        # Assigning a type to the variable 'out_line' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'out_line', result_mul_57)
        
        
        # Call to xrange(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'self' (line 54)
        self_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'self', False)
        # Obtaining the member 'nx' of a type (line 54)
        nx_60 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 24), self_59, 'nx')
        # Processing the call keyword arguments (line 54)
        kwargs_61 = {}
        # Getting the type of 'xrange' (line 54)
        xrange_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 54)
        xrange_call_result_62 = invoke(stypy.reporting.localization.Localization(__file__, 54, 17), xrange_58, *[nx_60], **kwargs_61)
        
        # Assigning a type to the variable 'xrange_call_result_62' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'xrange_call_result_62', xrange_call_result_62)
        # Testing if the for loop is going to be iterated (line 54)
        # Testing the type of a for loop iterable (line 54)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 54, 8), xrange_call_result_62)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 54, 8), xrange_call_result_62):
            # Getting the type of the for loop variable (line 54)
            for_loop_var_63 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 54, 8), xrange_call_result_62)
            # Assigning a type to the variable 'i' (line 54)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'i', for_loop_var_63)
            # SSA begins for a for statement (line 54)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 55):
            
            # Call to int(...): (line 55)
            # Processing the call arguments (line 55)
            
            # Call to round(...): (line 55)
            # Processing the call arguments (line 55)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 55)
            i_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 35), 'i', False)
            # Getting the type of 'row' (line 55)
            row_67 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 31), 'row', False)
            # Obtaining the member '__getitem__' of a type (line 55)
            getitem___68 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 31), row_67, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 55)
            subscript_call_result_69 = invoke(stypy.reporting.localization.Localization(__file__, 55, 31), getitem___68, i_66)
            
            # Processing the call keyword arguments (line 55)
            kwargs_70 = {}
            # Getting the type of 'round' (line 55)
            round_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 25), 'round', False)
            # Calling round(args, kwargs) (line 55)
            round_call_result_71 = invoke(stypy.reporting.localization.Localization(__file__, 55, 25), round_65, *[subscript_call_result_69], **kwargs_70)
            
            # Processing the call keyword arguments (line 55)
            kwargs_72 = {}
            # Getting the type of 'int' (line 55)
            int_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 21), 'int', False)
            # Calling int(args, kwargs) (line 55)
            int_call_result_73 = invoke(stypy.reporting.localization.Localization(__file__, 55, 21), int_64, *[round_call_result_71], **kwargs_72)
            
            # Assigning a type to the variable 'ipixel' (line 55)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'ipixel', int_call_result_73)
            
            # Getting the type of 'ipixel' (line 56)
            ipixel_74 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 15), 'ipixel')
            int_75 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 25), 'int')
            # Applying the binary operator '>=' (line 56)
            result_ge_76 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 15), '>=', ipixel_74, int_75)
            
            # Testing if the type of an if condition is none (line 56)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 56, 12), result_ge_76):
                
                # Getting the type of 'ipixel' (line 58)
                ipixel_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 17), 'ipixel')
                int_82 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 27), 'int')
                # Applying the binary operator '<=' (line 58)
                result_le_83 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 17), '<=', ipixel_81, int_82)
                
                # Testing if the type of an if condition is none (line 58)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 58, 17), result_le_83):
                    
                    # Assigning a Call to a Subscript (line 61):
                    
                    # Call to str(...): (line 61)
                    # Processing the call arguments (line 61)
                    # Getting the type of 'ipixel' (line 61)
                    ipixel_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 34), 'ipixel', False)
                    # Processing the call keyword arguments (line 61)
                    kwargs_90 = {}
                    # Getting the type of 'str' (line 61)
                    str_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 30), 'str', False)
                    # Calling str(args, kwargs) (line 61)
                    str_call_result_91 = invoke(stypy.reporting.localization.Localization(__file__, 61, 30), str_88, *[ipixel_89], **kwargs_90)
                    
                    # Getting the type of 'out_line' (line 61)
                    out_line_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'out_line')
                    # Getting the type of 'i' (line 61)
                    i_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 25), 'i')
                    # Storing an element on a container (line 61)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 16), out_line_92, (i_93, str_call_result_91))
                else:
                    
                    # Testing the type of an if condition (line 58)
                    if_condition_84 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 17), result_le_83)
                    # Assigning a type to the variable 'if_condition_84' (line 58)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 17), 'if_condition_84', if_condition_84)
                    # SSA begins for if statement (line 58)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Str to a Subscript (line 59):
                    str_85 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 30), 'str', '0')
                    # Getting the type of 'out_line' (line 59)
                    out_line_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'out_line')
                    # Getting the type of 'i' (line 59)
                    i_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 25), 'i')
                    # Storing an element on a container (line 59)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 16), out_line_86, (i_87, str_85))
                    # SSA branch for the else part of an if statement (line 58)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Call to a Subscript (line 61):
                    
                    # Call to str(...): (line 61)
                    # Processing the call arguments (line 61)
                    # Getting the type of 'ipixel' (line 61)
                    ipixel_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 34), 'ipixel', False)
                    # Processing the call keyword arguments (line 61)
                    kwargs_90 = {}
                    # Getting the type of 'str' (line 61)
                    str_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 30), 'str', False)
                    # Calling str(args, kwargs) (line 61)
                    str_call_result_91 = invoke(stypy.reporting.localization.Localization(__file__, 61, 30), str_88, *[ipixel_89], **kwargs_90)
                    
                    # Getting the type of 'out_line' (line 61)
                    out_line_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'out_line')
                    # Getting the type of 'i' (line 61)
                    i_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 25), 'i')
                    # Storing an element on a container (line 61)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 16), out_line_92, (i_93, str_call_result_91))
                    # SSA join for if statement (line 58)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 56)
                if_condition_77 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 56, 12), result_ge_76)
                # Assigning a type to the variable 'if_condition_77' (line 56)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'if_condition_77', if_condition_77)
                # SSA begins for if statement (line 56)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Str to a Subscript (line 57):
                str_78 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 30), 'str', '255')
                # Getting the type of 'out_line' (line 57)
                out_line_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'out_line')
                # Getting the type of 'i' (line 57)
                i_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'i')
                # Storing an element on a container (line 57)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 16), out_line_79, (i_80, str_78))
                # SSA branch for the else part of an if statement (line 56)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'ipixel' (line 58)
                ipixel_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 17), 'ipixel')
                int_82 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 27), 'int')
                # Applying the binary operator '<=' (line 58)
                result_le_83 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 17), '<=', ipixel_81, int_82)
                
                # Testing if the type of an if condition is none (line 58)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 58, 17), result_le_83):
                    
                    # Assigning a Call to a Subscript (line 61):
                    
                    # Call to str(...): (line 61)
                    # Processing the call arguments (line 61)
                    # Getting the type of 'ipixel' (line 61)
                    ipixel_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 34), 'ipixel', False)
                    # Processing the call keyword arguments (line 61)
                    kwargs_90 = {}
                    # Getting the type of 'str' (line 61)
                    str_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 30), 'str', False)
                    # Calling str(args, kwargs) (line 61)
                    str_call_result_91 = invoke(stypy.reporting.localization.Localization(__file__, 61, 30), str_88, *[ipixel_89], **kwargs_90)
                    
                    # Getting the type of 'out_line' (line 61)
                    out_line_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'out_line')
                    # Getting the type of 'i' (line 61)
                    i_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 25), 'i')
                    # Storing an element on a container (line 61)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 16), out_line_92, (i_93, str_call_result_91))
                else:
                    
                    # Testing the type of an if condition (line 58)
                    if_condition_84 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 17), result_le_83)
                    # Assigning a type to the variable 'if_condition_84' (line 58)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 17), 'if_condition_84', if_condition_84)
                    # SSA begins for if statement (line 58)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Str to a Subscript (line 59):
                    str_85 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 30), 'str', '0')
                    # Getting the type of 'out_line' (line 59)
                    out_line_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'out_line')
                    # Getting the type of 'i' (line 59)
                    i_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 25), 'i')
                    # Storing an element on a container (line 59)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 16), out_line_86, (i_87, str_85))
                    # SSA branch for the else part of an if statement (line 58)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Call to a Subscript (line 61):
                    
                    # Call to str(...): (line 61)
                    # Processing the call arguments (line 61)
                    # Getting the type of 'ipixel' (line 61)
                    ipixel_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 34), 'ipixel', False)
                    # Processing the call keyword arguments (line 61)
                    kwargs_90 = {}
                    # Getting the type of 'str' (line 61)
                    str_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 30), 'str', False)
                    # Calling str(args, kwargs) (line 61)
                    str_call_result_91 = invoke(stypy.reporting.localization.Localization(__file__, 61, 30), str_88, *[ipixel_89], **kwargs_90)
                    
                    # Getting the type of 'out_line' (line 61)
                    out_line_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'out_line')
                    # Getting the type of 'i' (line 61)
                    i_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 25), 'i')
                    # Storing an element on a container (line 61)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 16), out_line_92, (i_93, str_call_result_91))
                    # SSA join for if statement (line 58)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 56)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 'self' (line 64)
        self_94 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self')
        # Obtaining the member 'written_lines_count' of a type (line 64)
        written_lines_count_95 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_94, 'written_lines_count')
        int_96 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 36), 'int')
        # Applying the binary operator '+=' (line 64)
        result_iadd_97 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 8), '+=', written_lines_count_95, int_96)
        # Getting the type of 'self' (line 64)
        self_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self')
        # Setting the type of the member 'written_lines_count' of a type (line 64)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_98, 'written_lines_count', result_iadd_97)
        
        
        # Getting the type of 'self' (line 65)
        self_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 11), 'self')
        # Obtaining the member 'written_lines_count' of a type (line 65)
        written_lines_count_100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 11), self_99, 'written_lines_count')
        # Getting the type of 'self' (line 65)
        self_101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 39), 'self')
        # Obtaining the member 'ny' of a type (line 65)
        ny_102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 39), self_101, 'ny')
        # Applying the binary operator '==' (line 65)
        result_eq_103 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 11), '==', written_lines_count_100, ny_102)
        
        # Testing if the type of an if condition is none (line 65)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 65, 8), result_eq_103):
            pass
        else:
            
            # Testing the type of an if condition (line 65)
            if_condition_104 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 65, 8), result_eq_103)
            # Assigning a type to the variable 'if_condition_104' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'if_condition_104', if_condition_104)
            # SSA begins for if statement (line 65)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to close(...): (line 66)
            # Processing the call keyword arguments (line 66)
            kwargs_108 = {}
            # Getting the type of 'self' (line 66)
            self_105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'self', False)
            # Obtaining the member 'out_file' of a type (line 66)
            out_file_106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), self_105, 'out_file')
            # Obtaining the member 'close' of a type (line 66)
            close_107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), out_file_106, 'close')
            # Calling close(args, kwargs) (line 66)
            close_call_result_109 = invoke(stypy.reporting.localization.Localization(__file__, 66, 12), close_107, *[], **kwargs_108)
            
            # SSA join for if statement (line 65)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'saverow(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'saverow' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_110)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'saverow'
        return stypy_return_type_110


# Assigning a type to the variable 'SavePGMlines' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'SavePGMlines', SavePGMlines)

@norecursion
def oliva(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 13), 'int')
    int_112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 13), 'int')
    int_113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 13), 'int')
    float_114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 13), 'float')
    float_115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 13), 'float')
    float_116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 13), 'float')
    float_117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 13), 'float')
    float_118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 13), 'float')
    float_119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 13), 'float')
    float_120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 13), 'float')
    float_121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 13), 'float')
    str_122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 24), 'str', 'oliva.pgm')
    defaults = [int_111, int_112, int_113, float_114, float_115, float_116, float_117, float_118, float_119, float_120, float_121, str_122]
    # Create a new context for function 'oliva'
    module_type_store = module_type_store.open_function_context('oliva', 69, 0, False)
    
    # Passed parameters checking function
    oliva.stypy_localization = localization
    oliva.stypy_type_of_self = None
    oliva.stypy_type_store = module_type_store
    oliva.stypy_function_name = 'oliva'
    oliva.stypy_param_names_list = ['nx', 'ny', 'kp', 'da', 'ra', 'ba', 'sa', 'db', 'rb', 'sb', 'rc', 'out_file_name']
    oliva.stypy_varargs_param_name = None
    oliva.stypy_kwargs_param_name = None
    oliva.stypy_call_defaults = defaults
    oliva.stypy_call_varargs = varargs
    oliva.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'oliva', ['nx', 'ny', 'kp', 'da', 'ra', 'ba', 'sa', 'db', 'rb', 'sb', 'rc', 'out_file_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'oliva', localization, ['nx', 'ny', 'kp', 'da', 'ra', 'ba', 'sa', 'db', 'rb', 'sb', 'rc', 'out_file_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'oliva(...)' code ##################

    
    # Assigning a Call to a Name (line 81):
    
    # Call to SavePGMlines(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'out_file_name' (line 81)
    out_file_name_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 26), 'out_file_name', False)
    # Getting the type of 'ny' (line 81)
    ny_125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 41), 'ny', False)
    # Processing the call keyword arguments (line 81)
    kwargs_126 = {}
    # Getting the type of 'SavePGMlines' (line 81)
    SavePGMlines_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 13), 'SavePGMlines', False)
    # Calling SavePGMlines(args, kwargs) (line 81)
    SavePGMlines_call_result_127 = invoke(stypy.reporting.localization.Localization(__file__, 81, 13), SavePGMlines_123, *[out_file_name_124, ny_125], **kwargs_126)
    
    # Assigning a type to the variable 'outPGM' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'outPGM', SavePGMlines_call_result_127)
    
    # Assigning a List to a Name (line 83):
    
    # Obtaining an instance of the builtin type 'list' (line 83)
    list_128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 83)
    
    # Assigning a type to the variable 'image_matrix' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'image_matrix', list_128)
    
    # Assigning a Num to a Name (line 84):
    float_129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 8), 'float')
    # Assigning a type to the variable 'c' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'c', float_129)
    
    # Assigning a BinOp to a Name (line 85):
    
    # Obtaining an instance of the builtin type 'list' (line 85)
    list_130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 85)
    # Adding element type (line 85)
    float_131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 8), list_130, float_131)
    
    # Getting the type of 'nx' (line 85)
    nx_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 17), 'nx')
    int_133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 22), 'int')
    # Applying the binary operator '+' (line 85)
    result_add_134 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 17), '+', nx_132, int_133)
    
    # Applying the binary operator '*' (line 85)
    result_mul_135 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 8), '*', list_130, result_add_134)
    
    # Assigning a type to the variable 'a' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'a', result_mul_135)
    
    # Assigning a BinOp to a Name (line 86):
    
    # Obtaining an instance of the builtin type 'list' (line 86)
    list_136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 86)
    # Adding element type (line 86)
    float_137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 8), list_136, float_137)
    
    # Getting the type of 'nx' (line 86)
    nx_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 17), 'nx')
    int_139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 22), 'int')
    # Applying the binary operator '+' (line 86)
    result_add_140 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 17), '+', nx_138, int_139)
    
    # Applying the binary operator '*' (line 86)
    result_mul_141 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 8), '*', list_136, result_add_140)
    
    # Assigning a type to the variable 'b' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'b', result_mul_141)
    
    # Assigning a ListComp to a Name (line 89):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to xrange(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'nx' (line 89)
    nx_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 55), 'nx', False)
    # Processing the call keyword arguments (line 89)
    kwargs_153 = {}
    # Getting the type of 'xrange' (line 89)
    xrange_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 48), 'xrange', False)
    # Calling xrange(args, kwargs) (line 89)
    xrange_call_result_154 = invoke(stypy.reporting.localization.Localization(__file__, 89, 48), xrange_151, *[nx_152], **kwargs_153)
    
    comprehension_155 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 9), xrange_call_result_154)
    # Assigning a type to the variable 'i' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 9), 'i', comprehension_155)
    # Getting the type of 'ra' (line 89)
    ra_142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 9), 'ra')
    float_143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 15), 'float')
    float_144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 22), 'float')
    
    # Call to random(...): (line 89)
    # Processing the call keyword arguments (line 89)
    kwargs_146 = {}
    # Getting the type of 'random' (line 89)
    random_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 29), 'random', False)
    # Calling random(args, kwargs) (line 89)
    random_call_result_147 = invoke(stypy.reporting.localization.Localization(__file__, 89, 29), random_145, *[], **kwargs_146)
    
    # Applying the binary operator '*' (line 89)
    result_mul_148 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 22), '*', float_144, random_call_result_147)
    
    # Applying the binary operator '+' (line 89)
    result_add_149 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 15), '+', float_143, result_mul_148)
    
    # Applying the binary operator '*' (line 89)
    result_mul_150 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 9), '*', ra_142, result_add_149)
    
    list_156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 9), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 9), list_156, result_mul_150)
    # Assigning a type to the variable 'z' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'z', list_156)
    
    # Assigning a Num to a Name (line 93):
    int_157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 8), 'int')
    # Assigning a type to the variable 'i' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'i', int_157)
    
    
    # Call to xrange(...): (line 94)
    # Processing the call arguments (line 94)
    int_159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 20), 'int')
    # Processing the call keyword arguments (line 94)
    kwargs_160 = {}
    # Getting the type of 'xrange' (line 94)
    xrange_158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 94)
    xrange_call_result_161 = invoke(stypy.reporting.localization.Localization(__file__, 94, 13), xrange_158, *[int_159], **kwargs_160)
    
    # Assigning a type to the variable 'xrange_call_result_161' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'xrange_call_result_161', xrange_call_result_161)
    # Testing if the for loop is going to be iterated (line 94)
    # Testing the type of a for loop iterable (line 94)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 94, 4), xrange_call_result_161)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 94, 4), xrange_call_result_161):
        # Getting the type of the for loop variable (line 94)
        for_loop_var_162 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 94, 4), xrange_call_result_161)
        # Assigning a type to the variable 'j' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'j', for_loop_var_162)
        # SSA begins for a for statement (line 94)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Num to a Subscript (line 95):
        float_163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 15), 'float')
        # Getting the type of 'a' (line 95)
        a_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'a')
        # Getting the type of 'i' (line 95)
        i_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 10), 'i')
        # Storing an element on a container (line 95)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 8), a_164, (i_165, float_163))
        
        # Getting the type of 'i' (line 96)
        i_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'i')
        
        # Call to randint(...): (line 96)
        # Processing the call arguments (line 96)
        int_168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 21), 'int')
        int_169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 25), 'int')
        # Processing the call keyword arguments (line 96)
        kwargs_170 = {}
        # Getting the type of 'randint' (line 96)
        randint_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 13), 'randint', False)
        # Calling randint(args, kwargs) (line 96)
        randint_call_result_171 = invoke(stypy.reporting.localization.Localization(__file__, 96, 13), randint_167, *[int_168, int_169], **kwargs_170)
        
        # Applying the binary operator '+=' (line 96)
        result_iadd_172 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 8), '+=', i_166, randint_call_result_171)
        # Assigning a type to the variable 'i' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'i', result_iadd_172)
        
        
        # Getting the type of 'i' (line 97)
        i_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 'i')
        # Getting the type of 'nx' (line 97)
        nx_174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 16), 'nx')
        # Applying the binary operator '>=' (line 97)
        result_ge_175 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 11), '>=', i_173, nx_174)
        
        # Testing if the type of an if condition is none (line 97)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 97, 8), result_ge_175):
            pass
        else:
            
            # Testing the type of an if condition (line 97)
            if_condition_176 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 8), result_ge_175)
            # Assigning a type to the variable 'if_condition_176' (line 97)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'if_condition_176', if_condition_176)
            # SSA begins for if statement (line 97)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 97)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a BinOp to a Name (line 102):
    float_177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 10), 'float')
    # Getting the type of 'ra' (line 102)
    ra_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'ra')
    # Applying the binary operator '-' (line 102)
    result_sub_179 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 10), '-', float_177, ra_178)
    
    float_180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 21), 'float')
    # Getting the type of 'da' (line 102)
    da_181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 27), 'da')
    # Applying the binary operator '*' (line 102)
    result_mul_182 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 21), '*', float_180, da_181)
    
    # Applying the binary operator '-' (line 102)
    result_sub_183 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 19), '-', result_sub_179, result_mul_182)
    
    # Assigning a type to the variable 'dac' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'dac', result_sub_183)
    
    # Assigning a BinOp to a Name (line 103):
    float_184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 10), 'float')
    # Getting the type of 'rb' (line 103)
    rb_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'rb')
    # Applying the binary operator '-' (line 103)
    result_sub_186 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 10), '-', float_184, rb_185)
    
    float_187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 21), 'float')
    # Getting the type of 'db' (line 103)
    db_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'db')
    # Applying the binary operator '*' (line 103)
    result_mul_189 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 21), '*', float_187, db_188)
    
    # Applying the binary operator '-' (line 103)
    result_sub_190 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 19), '-', result_sub_186, result_mul_189)
    
    # Assigning a type to the variable 'dbc' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'dbc', result_sub_190)
    
    # Assigning a Name to a Name (line 104):
    # Getting the type of 'dbc' (line 104)
    dbc_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 11), 'dbc')
    # Assigning a type to the variable 'dbcc' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'dbcc', dbc_191)
    
    
    # Call to xrange(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'ny' (line 106)
    ny_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 22), 'ny', False)
    # Processing the call keyword arguments (line 106)
    kwargs_194 = {}
    # Getting the type of 'xrange' (line 106)
    xrange_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 15), 'xrange', False)
    # Calling xrange(args, kwargs) (line 106)
    xrange_call_result_195 = invoke(stypy.reporting.localization.Localization(__file__, 106, 15), xrange_192, *[ny_193], **kwargs_194)
    
    # Assigning a type to the variable 'xrange_call_result_195' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'xrange_call_result_195', xrange_call_result_195)
    # Testing if the for loop is going to be iterated (line 106)
    # Testing the type of a for loop iterable (line 106)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 106, 4), xrange_call_result_195)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 106, 4), xrange_call_result_195):
        # Getting the type of the for loop variable (line 106)
        for_loop_var_196 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 106, 4), xrange_call_result_195)
        # Assigning a type to the variable 'row' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'row', for_loop_var_196)
        # SSA begins for a for statement (line 106)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to xrange(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'kp' (line 108)
        kp_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 28), 'kp', False)
        # Processing the call keyword arguments (line 108)
        kwargs_199 = {}
        # Getting the type of 'xrange' (line 108)
        xrange_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 21), 'xrange', False)
        # Calling xrange(args, kwargs) (line 108)
        xrange_call_result_200 = invoke(stypy.reporting.localization.Localization(__file__, 108, 21), xrange_197, *[kp_198], **kwargs_199)
        
        # Assigning a type to the variable 'xrange_call_result_200' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'xrange_call_result_200', xrange_call_result_200)
        # Testing if the for loop is going to be iterated (line 108)
        # Testing the type of a for loop iterable (line 108)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 108, 8), xrange_call_result_200)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 108, 8), xrange_call_result_200):
            # Getting the type of the for loop variable (line 108)
            for_loop_var_201 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 108, 8), xrange_call_result_200)
            # Assigning a type to the variable 'niter' (line 108)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'niter', for_loop_var_201)
            # SSA begins for a for statement (line 108)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Name (line 110):
            
            # Obtaining the type of the subscript
            int_202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 19), 'int')
            # Getting the type of 'a' (line 110)
            a_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 17), 'a')
            # Obtaining the member '__getitem__' of a type (line 110)
            getitem___204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 17), a_203, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 110)
            subscript_call_result_205 = invoke(stypy.reporting.localization.Localization(__file__, 110, 17), getitem___204, int_202)
            
            # Assigning a type to the variable 'a1' (line 110)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'a1', subscript_call_result_205)
            
            # Assigning a Subscript to a Name (line 112):
            
            # Obtaining the type of the subscript
            int_206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 19), 'int')
            # Getting the type of 'b' (line 112)
            b_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 17), 'b')
            # Obtaining the member '__getitem__' of a type (line 112)
            getitem___208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 17), b_207, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 112)
            subscript_call_result_209 = invoke(stypy.reporting.localization.Localization(__file__, 112, 17), getitem___208, int_206)
            
            # Assigning a type to the variable 'b1' (line 112)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'b1', subscript_call_result_209)
            
            # Assigning a Subscript to a Subscript (line 113):
            
            # Obtaining the type of the subscript
            # Getting the type of 'nx' (line 113)
            nx_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 22), 'nx')
            int_211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 27), 'int')
            # Applying the binary operator '-' (line 113)
            result_sub_212 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 22), '-', nx_210, int_211)
            
            # Getting the type of 'a' (line 113)
            a_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'a')
            # Obtaining the member '__getitem__' of a type (line 113)
            getitem___214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 20), a_213, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 113)
            subscript_call_result_215 = invoke(stypy.reporting.localization.Localization(__file__, 113, 20), getitem___214, result_sub_212)
            
            # Getting the type of 'a' (line 113)
            a_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'a')
            # Getting the type of 'nx' (line 113)
            nx_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 14), 'nx')
            # Storing an element on a container (line 113)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 12), a_216, (nx_217, subscript_call_result_215))
            
            # Assigning a Subscript to a Subscript (line 114):
            
            # Obtaining the type of the subscript
            # Getting the type of 'nx' (line 114)
            nx_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 22), 'nx')
            int_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 27), 'int')
            # Applying the binary operator '-' (line 114)
            result_sub_220 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 22), '-', nx_218, int_219)
            
            # Getting the type of 'b' (line 114)
            b_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 20), 'b')
            # Obtaining the member '__getitem__' of a type (line 114)
            getitem___222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 20), b_221, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 114)
            subscript_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 114, 20), getitem___222, result_sub_220)
            
            # Getting the type of 'b' (line 114)
            b_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'b')
            # Getting the type of 'nx' (line 114)
            nx_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 14), 'nx')
            # Storing an element on a container (line 114)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 12), b_224, (nx_225, subscript_call_result_223))
            
            # Assigning a Num to a Name (line 115):
            float_226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 18), 'float')
            # Assigning a type to the variable 'bsa' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'bsa', float_226)
            
            
            # Call to xrange(...): (line 118)
            # Processing the call arguments (line 118)
            # Getting the type of 'nx' (line 118)
            nx_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 28), 'nx', False)
            # Processing the call keyword arguments (line 118)
            kwargs_229 = {}
            # Getting the type of 'xrange' (line 118)
            xrange_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 21), 'xrange', False)
            # Calling xrange(args, kwargs) (line 118)
            xrange_call_result_230 = invoke(stypy.reporting.localization.Localization(__file__, 118, 21), xrange_227, *[nx_228], **kwargs_229)
            
            # Assigning a type to the variable 'xrange_call_result_230' (line 118)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'xrange_call_result_230', xrange_call_result_230)
            # Testing if the for loop is going to be iterated (line 118)
            # Testing the type of a for loop iterable (line 118)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 118, 12), xrange_call_result_230)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 118, 12), xrange_call_result_230):
                # Getting the type of the for loop variable (line 118)
                for_loop_var_231 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 118, 12), xrange_call_result_230)
                # Assigning a type to the variable 'i' (line 118)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'i', for_loop_var_231)
                # SSA begins for a for statement (line 118)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Subscript to a Name (line 119):
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 119)
                i_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 23), 'i')
                # Getting the type of 'a' (line 119)
                a_233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 21), 'a')
                # Obtaining the member '__getitem__' of a type (line 119)
                getitem___234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 21), a_233, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 119)
                subscript_call_result_235 = invoke(stypy.reporting.localization.Localization(__file__, 119, 21), getitem___234, i_232)
                
                # Assigning a type to the variable 'af' (line 119)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'af', subscript_call_result_235)
                
                # Assigning a Subscript to a Name (line 120):
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 120)
                i_236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 23), 'i')
                # Getting the type of 'b' (line 120)
                b_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 21), 'b')
                # Obtaining the member '__getitem__' of a type (line 120)
                getitem___238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 21), b_237, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 120)
                subscript_call_result_239 = invoke(stypy.reporting.localization.Localization(__file__, 120, 21), getitem___238, i_236)
                
                # Assigning a type to the variable 'bf' (line 120)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'bf', subscript_call_result_239)
                
                # Assigning a BinOp to a Name (line 121):
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 121)
                i_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'i')
                # Getting the type of 'z' (line 121)
                z_241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 21), 'z')
                # Obtaining the member '__getitem__' of a type (line 121)
                getitem___242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 21), z_241, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 121)
                subscript_call_result_243 = invoke(stypy.reporting.localization.Localization(__file__, 121, 21), getitem___242, i_240)
                
                # Getting the type of 'af' (line 121)
                af_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 28), 'af')
                # Applying the binary operator '*' (line 121)
                result_mul_245 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 21), '*', subscript_call_result_243, af_244)
                
                # Getting the type of 'af' (line 121)
                af_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 33), 'af')
                # Applying the binary operator '*' (line 121)
                result_mul_247 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 31), '*', result_mul_245, af_246)
                
                float_248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 39), 'float')
                # Getting the type of 'sa' (line 121)
                sa_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 45), 'sa')
                # Getting the type of 'af' (line 121)
                af_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 50), 'af')
                # Applying the binary operator '*' (line 121)
                result_mul_251 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 45), '*', sa_249, af_250)
                
                # Getting the type of 'af' (line 121)
                af_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 55), 'af')
                # Applying the binary operator '*' (line 121)
                result_mul_253 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 53), '*', result_mul_251, af_252)
                
                # Applying the binary operator '+' (line 121)
                result_add_254 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 39), '+', float_248, result_mul_253)
                
                # Applying the binary operator 'div' (line 121)
                result_div_255 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 36), 'div', result_mul_247, result_add_254)
                
                # Assigning a type to the variable 'aq' (line 121)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'aq', result_div_255)
                
                # Assigning a BinOp to a Subscript (line 124):
                # Getting the type of 'af' (line 124)
                af_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 23), 'af')
                # Getting the type of 'dac' (line 124)
                dac_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 28), 'dac')
                # Applying the binary operator '*' (line 124)
                result_mul_258 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 23), '*', af_256, dac_257)
                
                # Getting the type of 'da' (line 124)
                da_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 34), 'da')
                # Getting the type of 'a1' (line 124)
                a1_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 40), 'a1')
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 124)
                i_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 47), 'i')
                int_262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 51), 'int')
                # Applying the binary operator '+' (line 124)
                result_add_263 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 47), '+', i_261, int_262)
                
                # Getting the type of 'a' (line 124)
                a_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 45), 'a')
                # Obtaining the member '__getitem__' of a type (line 124)
                getitem___265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 45), a_264, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 124)
                subscript_call_result_266 = invoke(stypy.reporting.localization.Localization(__file__, 124, 45), getitem___265, result_add_263)
                
                # Applying the binary operator '+' (line 124)
                result_add_267 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 40), '+', a1_260, subscript_call_result_266)
                
                # Applying the binary operator '*' (line 124)
                result_mul_268 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 34), '*', da_259, result_add_267)
                
                # Applying the binary operator '+' (line 124)
                result_add_269 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 23), '+', result_mul_258, result_mul_268)
                
                # Getting the type of 'aq' (line 124)
                aq_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 57), 'aq')
                # Getting the type of 'sb' (line 124)
                sb_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 63), 'sb')
                # Getting the type of 'bf' (line 124)
                bf_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 68), 'bf')
                # Applying the binary operator '+' (line 124)
                result_add_273 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 63), '+', sb_271, bf_272)
                
                # Applying the binary operator 'div' (line 124)
                result_div_274 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 57), 'div', aq_270, result_add_273)
                
                # Applying the binary operator '+' (line 124)
                result_add_275 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 55), '+', result_add_269, result_div_274)
                
                # Getting the type of 'a' (line 124)
                a_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'a')
                # Getting the type of 'i' (line 124)
                i_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 18), 'i')
                # Storing an element on a container (line 124)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 16), a_276, (i_277, result_add_275))
                
                # Assigning a BinOp to a Subscript (line 126):
                # Getting the type of 'bf' (line 126)
                bf_278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 23), 'bf')
                # Getting the type of 'dbcc' (line 126)
                dbcc_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 28), 'dbcc')
                # Applying the binary operator '*' (line 126)
                result_mul_280 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 23), '*', bf_278, dbcc_279)
                
                # Getting the type of 'db' (line 126)
                db_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 35), 'db')
                # Getting the type of 'b1' (line 126)
                b1_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 41), 'b1')
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 126)
                i_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 48), 'i')
                int_284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 52), 'int')
                # Applying the binary operator '+' (line 126)
                result_add_285 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 48), '+', i_283, int_284)
                
                # Getting the type of 'b' (line 126)
                b_286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 46), 'b')
                # Obtaining the member '__getitem__' of a type (line 126)
                getitem___287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 46), b_286, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 126)
                subscript_call_result_288 = invoke(stypy.reporting.localization.Localization(__file__, 126, 46), getitem___287, result_add_285)
                
                # Applying the binary operator '+' (line 126)
                result_add_289 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 41), '+', b1_282, subscript_call_result_288)
                
                # Applying the binary operator '*' (line 126)
                result_mul_290 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 35), '*', db_281, result_add_289)
                
                # Applying the binary operator '+' (line 126)
                result_add_291 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 23), '+', result_mul_280, result_mul_290)
                
                # Getting the type of 'aq' (line 126)
                aq_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 58), 'aq')
                # Applying the binary operator '+' (line 126)
                result_add_293 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 56), '+', result_add_291, aq_292)
                
                # Getting the type of 'b' (line 126)
                b_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'b')
                # Getting the type of 'i' (line 126)
                i_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 18), 'i')
                # Storing an element on a container (line 126)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 16), b_294, (i_295, result_add_293))
                
                # Getting the type of 'bsa' (line 127)
                bsa_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'bsa')
                # Getting the type of 'rc' (line 127)
                rc_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'rc')
                # Getting the type of 'af' (line 127)
                af_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 28), 'af')
                # Applying the binary operator '*' (line 127)
                result_mul_299 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 23), '*', rc_297, af_298)
                
                # Applying the binary operator '+=' (line 127)
                result_iadd_300 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 16), '+=', bsa_296, result_mul_299)
                # Assigning a type to the variable 'bsa' (line 127)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'bsa', result_iadd_300)
                
                
                # Assigning a Name to a Name (line 128):
                # Getting the type of 'af' (line 128)
                af_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 21), 'af')
                # Assigning a type to the variable 'a1' (line 128)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'a1', af_301)
                
                # Assigning a Name to a Name (line 129):
                # Getting the type of 'bf' (line 129)
                bf_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 21), 'bf')
                # Assigning a type to the variable 'b1' (line 129)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'b1', bf_302)
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a BinOp to a Name (line 132):
            # Getting the type of 'c' (line 132)
            c_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'c')
            float_304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 21), 'float')
            # Getting the type of 'rc' (line 132)
            rc_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 27), 'rc')
            # Applying the binary operator '-' (line 132)
            result_sub_306 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 21), '-', float_304, rc_305)
            
            # Applying the binary operator '*' (line 132)
            result_mul_307 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 16), '*', c_303, result_sub_306)
            
            # Getting the type of 'bsa' (line 132)
            bsa_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 33), 'bsa')
            # Getting the type of 'nx' (line 132)
            nx_309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 39), 'nx')
            # Applying the binary operator 'div' (line 132)
            result_div_310 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 33), 'div', bsa_308, nx_309)
            
            # Applying the binary operator '+' (line 132)
            result_add_311 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 16), '+', result_mul_307, result_div_310)
            
            # Assigning a type to the variable 'c' (line 132)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'c', result_add_311)
            
            # Assigning a BinOp to a Name (line 133):
            # Getting the type of 'rb' (line 133)
            rb_312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 18), 'rb')
            # Getting the type of 'c' (line 133)
            c_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 23), 'c')
            # Applying the binary operator 'div' (line 133)
            result_div_314 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 18), 'div', rb_312, c_313)
            
            # Assigning a type to the variable 'rbb' (line 133)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'rbb', result_div_314)
            
            # Assigning a BinOp to a Name (line 137):
            float_315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 19), 'float')
            float_316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 25), 'float')
            # Getting the type of 'db' (line 137)
            db_317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 31), 'db')
            # Applying the binary operator '*' (line 137)
            result_mul_318 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 25), '*', float_316, db_317)
            
            # Applying the binary operator '-' (line 137)
            result_sub_319 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 19), '-', float_315, result_mul_318)
            
            # Getting the type of 'rbb' (line 137)
            rbb_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 36), 'rbb')
            # Applying the binary operator '-' (line 137)
            result_sub_321 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 34), '-', result_sub_319, rbb_320)
            
            # Assigning a type to the variable 'dbcc' (line 137)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'dbcc', result_sub_321)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to saverow(...): (line 140)
        # Processing the call arguments (line 140)
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to xrange(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'nx' (line 140)
        nx_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 53), 'nx', False)
        # Processing the call keyword arguments (line 140)
        kwargs_332 = {}
        # Getting the type of 'xrange' (line 140)
        xrange_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 46), 'xrange', False)
        # Calling xrange(args, kwargs) (line 140)
        xrange_call_result_333 = invoke(stypy.reporting.localization.Localization(__file__, 140, 46), xrange_330, *[nx_331], **kwargs_332)
        
        comprehension_334 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 24), xrange_call_result_333)
        # Assigning a type to the variable 'ix' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 24), 'ix', comprehension_334)
        int_324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 24), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'ix' (line 140)
        ix_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 32), 'ix', False)
        # Getting the type of 'a' (line 140)
        a_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 30), 'a', False)
        # Obtaining the member '__getitem__' of a type (line 140)
        getitem___327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 30), a_326, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 140)
        subscript_call_result_328 = invoke(stypy.reporting.localization.Localization(__file__, 140, 30), getitem___327, ix_325)
        
        # Applying the binary operator '*' (line 140)
        result_mul_329 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 24), '*', int_324, subscript_call_result_328)
        
        list_335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 24), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 24), list_335, result_mul_329)
        # Processing the call keyword arguments (line 140)
        kwargs_336 = {}
        # Getting the type of 'outPGM' (line 140)
        outPGM_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'outPGM', False)
        # Obtaining the member 'saverow' of a type (line 140)
        saverow_323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), outPGM_322, 'saverow')
        # Calling saverow(args, kwargs) (line 140)
        saverow_call_result_337 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), saverow_323, *[list_335], **kwargs_336)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'oliva(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'oliva' in the type store
    # Getting the type of 'stypy_return_type' (line 69)
    stypy_return_type_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_338)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'oliva'
    return stypy_return_type_338

# Assigning a type to the variable 'oliva' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'oliva', oliva)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 143, 0, False)
    
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

    
    # Call to oliva(...): (line 144)
    # Processing the call keyword arguments (line 144)
    kwargs_340 = {}
    # Getting the type of 'oliva' (line 144)
    oliva_339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'oliva', False)
    # Calling oliva(args, kwargs) (line 144)
    oliva_call_result_341 = invoke(stypy.reporting.localization.Localization(__file__, 144, 4), oliva_339, *[], **kwargs_340)
    
    # Getting the type of 'True' (line 145)
    True_342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'stypy_return_type', True_342)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 143)
    stypy_return_type_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_343)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_343

# Assigning a type to the variable 'run' (line 143)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 0), 'run', run)

# Call to run(...): (line 148)
# Processing the call keyword arguments (line 148)
kwargs_345 = {}
# Getting the type of 'run' (line 148)
run_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 0), 'run', False)
# Calling run(args, kwargs) (line 148)
run_call_result_346 = invoke(stypy.reporting.localization.Localization(__file__, 148, 0), run_344, *[], **kwargs_345)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

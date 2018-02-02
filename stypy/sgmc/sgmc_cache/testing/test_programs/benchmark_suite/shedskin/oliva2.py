
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
            row_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 26), 'row', False)
            # Processing the call keyword arguments (line 46)
            kwargs_48 = {}
            # Getting the type of 'len' (line 46)
            len_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'len', False)
            # Calling len(args, kwargs) (line 46)
            len_call_result_49 = invoke(stypy.reporting.localization.Localization(__file__, 46, 22), len_46, *[row_47], **kwargs_48)
            
            # Getting the type of 'self' (line 46)
            self_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'self')
            # Setting the type of the member 'nx' of a type (line 46)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 12), self_50, 'nx', len_call_result_49)
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
            
            # Evaluating assert statement condition
            
            # Getting the type of 'self' (line 44)
            self_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 19), 'self')
            # Obtaining the member 'written_lines_count' of a type (line 44)
            written_lines_count_42 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 19), self_41, 'written_lines_count')
            # Getting the type of 'self' (line 44)
            self_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 46), 'self')
            # Obtaining the member 'ny' of a type (line 44)
            ny_44 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 46), self_43, 'ny')
            # Applying the binary operator '<' (line 44)
            result_lt_45 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 19), '<', written_lines_count_42, ny_44)
            
            # SSA branch for the else part of an if statement (line 42)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Attribute (line 46):
            
            # Call to len(...): (line 46)
            # Processing the call arguments (line 46)
            # Getting the type of 'row' (line 46)
            row_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 26), 'row', False)
            # Processing the call keyword arguments (line 46)
            kwargs_48 = {}
            # Getting the type of 'len' (line 46)
            len_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'len', False)
            # Calling len(args, kwargs) (line 46)
            len_call_result_49 = invoke(stypy.reporting.localization.Localization(__file__, 46, 22), len_46, *[row_47], **kwargs_48)
            
            # Getting the type of 'self' (line 46)
            self_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'self')
            # Setting the type of the member 'nx' of a type (line 46)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 12), self_50, 'nx', len_call_result_49)
            # SSA join for if statement (line 42)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 53):
        
        # Obtaining an instance of the builtin type 'list' (line 53)
        list_51 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 53)
        # Adding element type (line 53)
        str_52 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 20), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 19), list_51, str_52)
        
        # Getting the type of 'self' (line 53)
        self_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 26), 'self')
        # Obtaining the member 'nx' of a type (line 53)
        nx_54 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 26), self_53, 'nx')
        # Applying the binary operator '*' (line 53)
        result_mul_55 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 19), '*', list_51, nx_54)
        
        # Assigning a type to the variable 'out_line' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'out_line', result_mul_55)
        
        
        # Call to xrange(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'self' (line 54)
        self_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'self', False)
        # Obtaining the member 'nx' of a type (line 54)
        nx_58 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 24), self_57, 'nx')
        # Processing the call keyword arguments (line 54)
        kwargs_59 = {}
        # Getting the type of 'xrange' (line 54)
        xrange_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 54)
        xrange_call_result_60 = invoke(stypy.reporting.localization.Localization(__file__, 54, 17), xrange_56, *[nx_58], **kwargs_59)
        
        # Testing if the for loop is going to be iterated (line 54)
        # Testing the type of a for loop iterable (line 54)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 54, 8), xrange_call_result_60)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 54, 8), xrange_call_result_60):
            # Getting the type of the for loop variable (line 54)
            for_loop_var_61 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 54, 8), xrange_call_result_60)
            # Assigning a type to the variable 'i' (line 54)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'i', for_loop_var_61)
            # SSA begins for a for statement (line 54)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 55):
            
            # Call to int(...): (line 55)
            # Processing the call arguments (line 55)
            
            # Call to round(...): (line 55)
            # Processing the call arguments (line 55)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 55)
            i_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 35), 'i', False)
            # Getting the type of 'row' (line 55)
            row_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 31), 'row', False)
            # Obtaining the member '__getitem__' of a type (line 55)
            getitem___66 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 31), row_65, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 55)
            subscript_call_result_67 = invoke(stypy.reporting.localization.Localization(__file__, 55, 31), getitem___66, i_64)
            
            # Processing the call keyword arguments (line 55)
            kwargs_68 = {}
            # Getting the type of 'round' (line 55)
            round_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 25), 'round', False)
            # Calling round(args, kwargs) (line 55)
            round_call_result_69 = invoke(stypy.reporting.localization.Localization(__file__, 55, 25), round_63, *[subscript_call_result_67], **kwargs_68)
            
            # Processing the call keyword arguments (line 55)
            kwargs_70 = {}
            # Getting the type of 'int' (line 55)
            int_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 21), 'int', False)
            # Calling int(args, kwargs) (line 55)
            int_call_result_71 = invoke(stypy.reporting.localization.Localization(__file__, 55, 21), int_62, *[round_call_result_69], **kwargs_70)
            
            # Assigning a type to the variable 'ipixel' (line 55)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'ipixel', int_call_result_71)
            
            # Getting the type of 'ipixel' (line 56)
            ipixel_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 15), 'ipixel')
            int_73 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 25), 'int')
            # Applying the binary operator '>=' (line 56)
            result_ge_74 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 15), '>=', ipixel_72, int_73)
            
            # Testing if the type of an if condition is none (line 56)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 56, 12), result_ge_74):
                
                # Getting the type of 'ipixel' (line 58)
                ipixel_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 17), 'ipixel')
                int_80 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 27), 'int')
                # Applying the binary operator '<=' (line 58)
                result_le_81 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 17), '<=', ipixel_79, int_80)
                
                # Testing if the type of an if condition is none (line 58)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 58, 17), result_le_81):
                    
                    # Assigning a Call to a Subscript (line 61):
                    
                    # Call to str(...): (line 61)
                    # Processing the call arguments (line 61)
                    # Getting the type of 'ipixel' (line 61)
                    ipixel_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 34), 'ipixel', False)
                    # Processing the call keyword arguments (line 61)
                    kwargs_88 = {}
                    # Getting the type of 'str' (line 61)
                    str_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 30), 'str', False)
                    # Calling str(args, kwargs) (line 61)
                    str_call_result_89 = invoke(stypy.reporting.localization.Localization(__file__, 61, 30), str_86, *[ipixel_87], **kwargs_88)
                    
                    # Getting the type of 'out_line' (line 61)
                    out_line_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'out_line')
                    # Getting the type of 'i' (line 61)
                    i_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 25), 'i')
                    # Storing an element on a container (line 61)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 16), out_line_90, (i_91, str_call_result_89))
                else:
                    
                    # Testing the type of an if condition (line 58)
                    if_condition_82 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 17), result_le_81)
                    # Assigning a type to the variable 'if_condition_82' (line 58)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 17), 'if_condition_82', if_condition_82)
                    # SSA begins for if statement (line 58)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Str to a Subscript (line 59):
                    str_83 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 30), 'str', '0')
                    # Getting the type of 'out_line' (line 59)
                    out_line_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'out_line')
                    # Getting the type of 'i' (line 59)
                    i_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 25), 'i')
                    # Storing an element on a container (line 59)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 16), out_line_84, (i_85, str_83))
                    # SSA branch for the else part of an if statement (line 58)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Call to a Subscript (line 61):
                    
                    # Call to str(...): (line 61)
                    # Processing the call arguments (line 61)
                    # Getting the type of 'ipixel' (line 61)
                    ipixel_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 34), 'ipixel', False)
                    # Processing the call keyword arguments (line 61)
                    kwargs_88 = {}
                    # Getting the type of 'str' (line 61)
                    str_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 30), 'str', False)
                    # Calling str(args, kwargs) (line 61)
                    str_call_result_89 = invoke(stypy.reporting.localization.Localization(__file__, 61, 30), str_86, *[ipixel_87], **kwargs_88)
                    
                    # Getting the type of 'out_line' (line 61)
                    out_line_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'out_line')
                    # Getting the type of 'i' (line 61)
                    i_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 25), 'i')
                    # Storing an element on a container (line 61)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 16), out_line_90, (i_91, str_call_result_89))
                    # SSA join for if statement (line 58)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 56)
                if_condition_75 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 56, 12), result_ge_74)
                # Assigning a type to the variable 'if_condition_75' (line 56)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'if_condition_75', if_condition_75)
                # SSA begins for if statement (line 56)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Str to a Subscript (line 57):
                str_76 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 30), 'str', '255')
                # Getting the type of 'out_line' (line 57)
                out_line_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'out_line')
                # Getting the type of 'i' (line 57)
                i_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'i')
                # Storing an element on a container (line 57)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 16), out_line_77, (i_78, str_76))
                # SSA branch for the else part of an if statement (line 56)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'ipixel' (line 58)
                ipixel_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 17), 'ipixel')
                int_80 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 27), 'int')
                # Applying the binary operator '<=' (line 58)
                result_le_81 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 17), '<=', ipixel_79, int_80)
                
                # Testing if the type of an if condition is none (line 58)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 58, 17), result_le_81):
                    
                    # Assigning a Call to a Subscript (line 61):
                    
                    # Call to str(...): (line 61)
                    # Processing the call arguments (line 61)
                    # Getting the type of 'ipixel' (line 61)
                    ipixel_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 34), 'ipixel', False)
                    # Processing the call keyword arguments (line 61)
                    kwargs_88 = {}
                    # Getting the type of 'str' (line 61)
                    str_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 30), 'str', False)
                    # Calling str(args, kwargs) (line 61)
                    str_call_result_89 = invoke(stypy.reporting.localization.Localization(__file__, 61, 30), str_86, *[ipixel_87], **kwargs_88)
                    
                    # Getting the type of 'out_line' (line 61)
                    out_line_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'out_line')
                    # Getting the type of 'i' (line 61)
                    i_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 25), 'i')
                    # Storing an element on a container (line 61)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 16), out_line_90, (i_91, str_call_result_89))
                else:
                    
                    # Testing the type of an if condition (line 58)
                    if_condition_82 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 17), result_le_81)
                    # Assigning a type to the variable 'if_condition_82' (line 58)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 17), 'if_condition_82', if_condition_82)
                    # SSA begins for if statement (line 58)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Str to a Subscript (line 59):
                    str_83 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 30), 'str', '0')
                    # Getting the type of 'out_line' (line 59)
                    out_line_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'out_line')
                    # Getting the type of 'i' (line 59)
                    i_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 25), 'i')
                    # Storing an element on a container (line 59)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 16), out_line_84, (i_85, str_83))
                    # SSA branch for the else part of an if statement (line 58)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Call to a Subscript (line 61):
                    
                    # Call to str(...): (line 61)
                    # Processing the call arguments (line 61)
                    # Getting the type of 'ipixel' (line 61)
                    ipixel_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 34), 'ipixel', False)
                    # Processing the call keyword arguments (line 61)
                    kwargs_88 = {}
                    # Getting the type of 'str' (line 61)
                    str_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 30), 'str', False)
                    # Calling str(args, kwargs) (line 61)
                    str_call_result_89 = invoke(stypy.reporting.localization.Localization(__file__, 61, 30), str_86, *[ipixel_87], **kwargs_88)
                    
                    # Getting the type of 'out_line' (line 61)
                    out_line_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'out_line')
                    # Getting the type of 'i' (line 61)
                    i_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 25), 'i')
                    # Storing an element on a container (line 61)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 16), out_line_90, (i_91, str_call_result_89))
                    # SSA join for if statement (line 58)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 56)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 'self' (line 64)
        self_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self')
        # Obtaining the member 'written_lines_count' of a type (line 64)
        written_lines_count_93 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_92, 'written_lines_count')
        int_94 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 36), 'int')
        # Applying the binary operator '+=' (line 64)
        result_iadd_95 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 8), '+=', written_lines_count_93, int_94)
        # Getting the type of 'self' (line 64)
        self_96 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self')
        # Setting the type of the member 'written_lines_count' of a type (line 64)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_96, 'written_lines_count', result_iadd_95)
        
        
        # Getting the type of 'self' (line 65)
        self_97 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 11), 'self')
        # Obtaining the member 'written_lines_count' of a type (line 65)
        written_lines_count_98 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 11), self_97, 'written_lines_count')
        # Getting the type of 'self' (line 65)
        self_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 39), 'self')
        # Obtaining the member 'ny' of a type (line 65)
        ny_100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 39), self_99, 'ny')
        # Applying the binary operator '==' (line 65)
        result_eq_101 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 11), '==', written_lines_count_98, ny_100)
        
        # Testing if the type of an if condition is none (line 65)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 65, 8), result_eq_101):
            pass
        else:
            
            # Testing the type of an if condition (line 65)
            if_condition_102 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 65, 8), result_eq_101)
            # Assigning a type to the variable 'if_condition_102' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'if_condition_102', if_condition_102)
            # SSA begins for if statement (line 65)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to close(...): (line 66)
            # Processing the call keyword arguments (line 66)
            kwargs_106 = {}
            # Getting the type of 'self' (line 66)
            self_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'self', False)
            # Obtaining the member 'out_file' of a type (line 66)
            out_file_104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), self_103, 'out_file')
            # Obtaining the member 'close' of a type (line 66)
            close_105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), out_file_104, 'close')
            # Calling close(args, kwargs) (line 66)
            close_call_result_107 = invoke(stypy.reporting.localization.Localization(__file__, 66, 12), close_105, *[], **kwargs_106)
            
            # SSA join for if statement (line 65)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'saverow(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'saverow' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_108)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'saverow'
        return stypy_return_type_108


# Assigning a type to the variable 'SavePGMlines' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'SavePGMlines', SavePGMlines)

@norecursion
def oliva(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 13), 'int')
    int_110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 13), 'int')
    int_111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 13), 'int')
    float_112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 13), 'float')
    float_113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 13), 'float')
    float_114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 13), 'float')
    float_115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 13), 'float')
    float_116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 13), 'float')
    float_117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 13), 'float')
    float_118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 13), 'float')
    float_119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 13), 'float')
    str_120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 24), 'str', 'oliva.pgm')
    defaults = [int_109, int_110, int_111, float_112, float_113, float_114, float_115, float_116, float_117, float_118, float_119, str_120]
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
    out_file_name_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 26), 'out_file_name', False)
    # Getting the type of 'ny' (line 81)
    ny_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 41), 'ny', False)
    # Processing the call keyword arguments (line 81)
    kwargs_124 = {}
    # Getting the type of 'SavePGMlines' (line 81)
    SavePGMlines_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 13), 'SavePGMlines', False)
    # Calling SavePGMlines(args, kwargs) (line 81)
    SavePGMlines_call_result_125 = invoke(stypy.reporting.localization.Localization(__file__, 81, 13), SavePGMlines_121, *[out_file_name_122, ny_123], **kwargs_124)
    
    # Assigning a type to the variable 'outPGM' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'outPGM', SavePGMlines_call_result_125)
    
    # Assigning a List to a Name (line 83):
    
    # Obtaining an instance of the builtin type 'list' (line 83)
    list_126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 83)
    
    # Assigning a type to the variable 'image_matrix' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'image_matrix', list_126)
    
    # Assigning a Num to a Name (line 84):
    float_127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 8), 'float')
    # Assigning a type to the variable 'c' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'c', float_127)
    
    # Assigning a BinOp to a Name (line 85):
    
    # Obtaining an instance of the builtin type 'list' (line 85)
    list_128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 85)
    # Adding element type (line 85)
    float_129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 8), list_128, float_129)
    
    # Getting the type of 'nx' (line 85)
    nx_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 17), 'nx')
    int_131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 22), 'int')
    # Applying the binary operator '+' (line 85)
    result_add_132 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 17), '+', nx_130, int_131)
    
    # Applying the binary operator '*' (line 85)
    result_mul_133 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 8), '*', list_128, result_add_132)
    
    # Assigning a type to the variable 'a' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'a', result_mul_133)
    
    # Assigning a BinOp to a Name (line 86):
    
    # Obtaining an instance of the builtin type 'list' (line 86)
    list_134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 86)
    # Adding element type (line 86)
    float_135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 8), list_134, float_135)
    
    # Getting the type of 'nx' (line 86)
    nx_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 17), 'nx')
    int_137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 22), 'int')
    # Applying the binary operator '+' (line 86)
    result_add_138 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 17), '+', nx_136, int_137)
    
    # Applying the binary operator '*' (line 86)
    result_mul_139 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 8), '*', list_134, result_add_138)
    
    # Assigning a type to the variable 'b' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'b', result_mul_139)
    
    # Assigning a ListComp to a Name (line 89):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to xrange(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'nx' (line 89)
    nx_150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 55), 'nx', False)
    # Processing the call keyword arguments (line 89)
    kwargs_151 = {}
    # Getting the type of 'xrange' (line 89)
    xrange_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 48), 'xrange', False)
    # Calling xrange(args, kwargs) (line 89)
    xrange_call_result_152 = invoke(stypy.reporting.localization.Localization(__file__, 89, 48), xrange_149, *[nx_150], **kwargs_151)
    
    comprehension_153 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 9), xrange_call_result_152)
    # Assigning a type to the variable 'i' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 9), 'i', comprehension_153)
    # Getting the type of 'ra' (line 89)
    ra_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 9), 'ra')
    float_141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 15), 'float')
    float_142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 22), 'float')
    
    # Call to random(...): (line 89)
    # Processing the call keyword arguments (line 89)
    kwargs_144 = {}
    # Getting the type of 'random' (line 89)
    random_143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 29), 'random', False)
    # Calling random(args, kwargs) (line 89)
    random_call_result_145 = invoke(stypy.reporting.localization.Localization(__file__, 89, 29), random_143, *[], **kwargs_144)
    
    # Applying the binary operator '*' (line 89)
    result_mul_146 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 22), '*', float_142, random_call_result_145)
    
    # Applying the binary operator '+' (line 89)
    result_add_147 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 15), '+', float_141, result_mul_146)
    
    # Applying the binary operator '*' (line 89)
    result_mul_148 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 9), '*', ra_140, result_add_147)
    
    list_154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 9), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 9), list_154, result_mul_148)
    # Assigning a type to the variable 'z' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'z', list_154)
    
    # Assigning a Num to a Name (line 93):
    int_155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 8), 'int')
    # Assigning a type to the variable 'i' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'i', int_155)
    
    
    # Call to xrange(...): (line 94)
    # Processing the call arguments (line 94)
    int_157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 20), 'int')
    # Processing the call keyword arguments (line 94)
    kwargs_158 = {}
    # Getting the type of 'xrange' (line 94)
    xrange_156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 94)
    xrange_call_result_159 = invoke(stypy.reporting.localization.Localization(__file__, 94, 13), xrange_156, *[int_157], **kwargs_158)
    
    # Testing if the for loop is going to be iterated (line 94)
    # Testing the type of a for loop iterable (line 94)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 94, 4), xrange_call_result_159)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 94, 4), xrange_call_result_159):
        # Getting the type of the for loop variable (line 94)
        for_loop_var_160 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 94, 4), xrange_call_result_159)
        # Assigning a type to the variable 'j' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'j', for_loop_var_160)
        # SSA begins for a for statement (line 94)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Num to a Subscript (line 95):
        float_161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 15), 'float')
        # Getting the type of 'a' (line 95)
        a_162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'a')
        # Getting the type of 'i' (line 95)
        i_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 10), 'i')
        # Storing an element on a container (line 95)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 8), a_162, (i_163, float_161))
        
        # Getting the type of 'i' (line 96)
        i_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'i')
        
        # Call to randint(...): (line 96)
        # Processing the call arguments (line 96)
        int_166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 21), 'int')
        int_167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 25), 'int')
        # Processing the call keyword arguments (line 96)
        kwargs_168 = {}
        # Getting the type of 'randint' (line 96)
        randint_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 13), 'randint', False)
        # Calling randint(args, kwargs) (line 96)
        randint_call_result_169 = invoke(stypy.reporting.localization.Localization(__file__, 96, 13), randint_165, *[int_166, int_167], **kwargs_168)
        
        # Applying the binary operator '+=' (line 96)
        result_iadd_170 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 8), '+=', i_164, randint_call_result_169)
        # Assigning a type to the variable 'i' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'i', result_iadd_170)
        
        
        # Getting the type of 'i' (line 97)
        i_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 'i')
        # Getting the type of 'nx' (line 97)
        nx_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 16), 'nx')
        # Applying the binary operator '>=' (line 97)
        result_ge_173 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 11), '>=', i_171, nx_172)
        
        # Testing if the type of an if condition is none (line 97)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 97, 8), result_ge_173):
            pass
        else:
            
            # Testing the type of an if condition (line 97)
            if_condition_174 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 8), result_ge_173)
            # Assigning a type to the variable 'if_condition_174' (line 97)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'if_condition_174', if_condition_174)
            # SSA begins for if statement (line 97)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 97)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a BinOp to a Name (line 102):
    float_175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 10), 'float')
    # Getting the type of 'ra' (line 102)
    ra_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'ra')
    # Applying the binary operator '-' (line 102)
    result_sub_177 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 10), '-', float_175, ra_176)
    
    float_178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 21), 'float')
    # Getting the type of 'da' (line 102)
    da_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 27), 'da')
    # Applying the binary operator '*' (line 102)
    result_mul_180 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 21), '*', float_178, da_179)
    
    # Applying the binary operator '-' (line 102)
    result_sub_181 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 19), '-', result_sub_177, result_mul_180)
    
    # Assigning a type to the variable 'dac' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'dac', result_sub_181)
    
    # Assigning a BinOp to a Name (line 103):
    float_182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 10), 'float')
    # Getting the type of 'rb' (line 103)
    rb_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'rb')
    # Applying the binary operator '-' (line 103)
    result_sub_184 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 10), '-', float_182, rb_183)
    
    float_185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 21), 'float')
    # Getting the type of 'db' (line 103)
    db_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'db')
    # Applying the binary operator '*' (line 103)
    result_mul_187 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 21), '*', float_185, db_186)
    
    # Applying the binary operator '-' (line 103)
    result_sub_188 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 19), '-', result_sub_184, result_mul_187)
    
    # Assigning a type to the variable 'dbc' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'dbc', result_sub_188)
    
    # Assigning a Name to a Name (line 104):
    # Getting the type of 'dbc' (line 104)
    dbc_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 11), 'dbc')
    # Assigning a type to the variable 'dbcc' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'dbcc', dbc_189)
    
    
    # Call to xrange(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'ny' (line 106)
    ny_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 22), 'ny', False)
    # Processing the call keyword arguments (line 106)
    kwargs_192 = {}
    # Getting the type of 'xrange' (line 106)
    xrange_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 15), 'xrange', False)
    # Calling xrange(args, kwargs) (line 106)
    xrange_call_result_193 = invoke(stypy.reporting.localization.Localization(__file__, 106, 15), xrange_190, *[ny_191], **kwargs_192)
    
    # Testing if the for loop is going to be iterated (line 106)
    # Testing the type of a for loop iterable (line 106)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 106, 4), xrange_call_result_193)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 106, 4), xrange_call_result_193):
        # Getting the type of the for loop variable (line 106)
        for_loop_var_194 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 106, 4), xrange_call_result_193)
        # Assigning a type to the variable 'row' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'row', for_loop_var_194)
        # SSA begins for a for statement (line 106)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to xrange(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'kp' (line 108)
        kp_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 28), 'kp', False)
        # Processing the call keyword arguments (line 108)
        kwargs_197 = {}
        # Getting the type of 'xrange' (line 108)
        xrange_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 21), 'xrange', False)
        # Calling xrange(args, kwargs) (line 108)
        xrange_call_result_198 = invoke(stypy.reporting.localization.Localization(__file__, 108, 21), xrange_195, *[kp_196], **kwargs_197)
        
        # Testing if the for loop is going to be iterated (line 108)
        # Testing the type of a for loop iterable (line 108)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 108, 8), xrange_call_result_198)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 108, 8), xrange_call_result_198):
            # Getting the type of the for loop variable (line 108)
            for_loop_var_199 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 108, 8), xrange_call_result_198)
            # Assigning a type to the variable 'niter' (line 108)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'niter', for_loop_var_199)
            # SSA begins for a for statement (line 108)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Name (line 110):
            
            # Obtaining the type of the subscript
            int_200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 19), 'int')
            # Getting the type of 'a' (line 110)
            a_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 17), 'a')
            # Obtaining the member '__getitem__' of a type (line 110)
            getitem___202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 17), a_201, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 110)
            subscript_call_result_203 = invoke(stypy.reporting.localization.Localization(__file__, 110, 17), getitem___202, int_200)
            
            # Assigning a type to the variable 'a1' (line 110)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'a1', subscript_call_result_203)
            
            # Assigning a Subscript to a Name (line 112):
            
            # Obtaining the type of the subscript
            int_204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 19), 'int')
            # Getting the type of 'b' (line 112)
            b_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 17), 'b')
            # Obtaining the member '__getitem__' of a type (line 112)
            getitem___206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 17), b_205, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 112)
            subscript_call_result_207 = invoke(stypy.reporting.localization.Localization(__file__, 112, 17), getitem___206, int_204)
            
            # Assigning a type to the variable 'b1' (line 112)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'b1', subscript_call_result_207)
            
            # Assigning a Subscript to a Subscript (line 113):
            
            # Obtaining the type of the subscript
            # Getting the type of 'nx' (line 113)
            nx_208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 22), 'nx')
            int_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 27), 'int')
            # Applying the binary operator '-' (line 113)
            result_sub_210 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 22), '-', nx_208, int_209)
            
            # Getting the type of 'a' (line 113)
            a_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'a')
            # Obtaining the member '__getitem__' of a type (line 113)
            getitem___212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 20), a_211, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 113)
            subscript_call_result_213 = invoke(stypy.reporting.localization.Localization(__file__, 113, 20), getitem___212, result_sub_210)
            
            # Getting the type of 'a' (line 113)
            a_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'a')
            # Getting the type of 'nx' (line 113)
            nx_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 14), 'nx')
            # Storing an element on a container (line 113)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 12), a_214, (nx_215, subscript_call_result_213))
            
            # Assigning a Subscript to a Subscript (line 114):
            
            # Obtaining the type of the subscript
            # Getting the type of 'nx' (line 114)
            nx_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 22), 'nx')
            int_217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 27), 'int')
            # Applying the binary operator '-' (line 114)
            result_sub_218 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 22), '-', nx_216, int_217)
            
            # Getting the type of 'b' (line 114)
            b_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 20), 'b')
            # Obtaining the member '__getitem__' of a type (line 114)
            getitem___220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 20), b_219, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 114)
            subscript_call_result_221 = invoke(stypy.reporting.localization.Localization(__file__, 114, 20), getitem___220, result_sub_218)
            
            # Getting the type of 'b' (line 114)
            b_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'b')
            # Getting the type of 'nx' (line 114)
            nx_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 14), 'nx')
            # Storing an element on a container (line 114)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 12), b_222, (nx_223, subscript_call_result_221))
            
            # Assigning a Num to a Name (line 115):
            float_224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 18), 'float')
            # Assigning a type to the variable 'bsa' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'bsa', float_224)
            
            
            # Call to xrange(...): (line 118)
            # Processing the call arguments (line 118)
            # Getting the type of 'nx' (line 118)
            nx_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 28), 'nx', False)
            # Processing the call keyword arguments (line 118)
            kwargs_227 = {}
            # Getting the type of 'xrange' (line 118)
            xrange_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 21), 'xrange', False)
            # Calling xrange(args, kwargs) (line 118)
            xrange_call_result_228 = invoke(stypy.reporting.localization.Localization(__file__, 118, 21), xrange_225, *[nx_226], **kwargs_227)
            
            # Testing if the for loop is going to be iterated (line 118)
            # Testing the type of a for loop iterable (line 118)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 118, 12), xrange_call_result_228)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 118, 12), xrange_call_result_228):
                # Getting the type of the for loop variable (line 118)
                for_loop_var_229 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 118, 12), xrange_call_result_228)
                # Assigning a type to the variable 'i' (line 118)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'i', for_loop_var_229)
                # SSA begins for a for statement (line 118)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Subscript to a Name (line 119):
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 119)
                i_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 23), 'i')
                # Getting the type of 'a' (line 119)
                a_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 21), 'a')
                # Obtaining the member '__getitem__' of a type (line 119)
                getitem___232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 21), a_231, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 119)
                subscript_call_result_233 = invoke(stypy.reporting.localization.Localization(__file__, 119, 21), getitem___232, i_230)
                
                # Assigning a type to the variable 'af' (line 119)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'af', subscript_call_result_233)
                
                # Assigning a Subscript to a Name (line 120):
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 120)
                i_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 23), 'i')
                # Getting the type of 'b' (line 120)
                b_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 21), 'b')
                # Obtaining the member '__getitem__' of a type (line 120)
                getitem___236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 21), b_235, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 120)
                subscript_call_result_237 = invoke(stypy.reporting.localization.Localization(__file__, 120, 21), getitem___236, i_234)
                
                # Assigning a type to the variable 'bf' (line 120)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'bf', subscript_call_result_237)
                
                # Assigning a BinOp to a Name (line 121):
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 121)
                i_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'i')
                # Getting the type of 'z' (line 121)
                z_239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 21), 'z')
                # Obtaining the member '__getitem__' of a type (line 121)
                getitem___240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 21), z_239, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 121)
                subscript_call_result_241 = invoke(stypy.reporting.localization.Localization(__file__, 121, 21), getitem___240, i_238)
                
                # Getting the type of 'af' (line 121)
                af_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 28), 'af')
                # Applying the binary operator '*' (line 121)
                result_mul_243 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 21), '*', subscript_call_result_241, af_242)
                
                # Getting the type of 'af' (line 121)
                af_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 33), 'af')
                # Applying the binary operator '*' (line 121)
                result_mul_245 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 31), '*', result_mul_243, af_244)
                
                float_246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 39), 'float')
                # Getting the type of 'sa' (line 121)
                sa_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 45), 'sa')
                # Getting the type of 'af' (line 121)
                af_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 50), 'af')
                # Applying the binary operator '*' (line 121)
                result_mul_249 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 45), '*', sa_247, af_248)
                
                # Getting the type of 'af' (line 121)
                af_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 55), 'af')
                # Applying the binary operator '*' (line 121)
                result_mul_251 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 53), '*', result_mul_249, af_250)
                
                # Applying the binary operator '+' (line 121)
                result_add_252 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 39), '+', float_246, result_mul_251)
                
                # Applying the binary operator 'div' (line 121)
                result_div_253 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 36), 'div', result_mul_245, result_add_252)
                
                # Assigning a type to the variable 'aq' (line 121)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'aq', result_div_253)
                
                # Assigning a BinOp to a Subscript (line 124):
                # Getting the type of 'af' (line 124)
                af_254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 23), 'af')
                # Getting the type of 'dac' (line 124)
                dac_255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 28), 'dac')
                # Applying the binary operator '*' (line 124)
                result_mul_256 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 23), '*', af_254, dac_255)
                
                # Getting the type of 'da' (line 124)
                da_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 34), 'da')
                # Getting the type of 'a1' (line 124)
                a1_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 40), 'a1')
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 124)
                i_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 47), 'i')
                int_260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 51), 'int')
                # Applying the binary operator '+' (line 124)
                result_add_261 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 47), '+', i_259, int_260)
                
                # Getting the type of 'a' (line 124)
                a_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 45), 'a')
                # Obtaining the member '__getitem__' of a type (line 124)
                getitem___263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 45), a_262, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 124)
                subscript_call_result_264 = invoke(stypy.reporting.localization.Localization(__file__, 124, 45), getitem___263, result_add_261)
                
                # Applying the binary operator '+' (line 124)
                result_add_265 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 40), '+', a1_258, subscript_call_result_264)
                
                # Applying the binary operator '*' (line 124)
                result_mul_266 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 34), '*', da_257, result_add_265)
                
                # Applying the binary operator '+' (line 124)
                result_add_267 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 23), '+', result_mul_256, result_mul_266)
                
                # Getting the type of 'aq' (line 124)
                aq_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 57), 'aq')
                # Getting the type of 'sb' (line 124)
                sb_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 63), 'sb')
                # Getting the type of 'bf' (line 124)
                bf_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 68), 'bf')
                # Applying the binary operator '+' (line 124)
                result_add_271 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 63), '+', sb_269, bf_270)
                
                # Applying the binary operator 'div' (line 124)
                result_div_272 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 57), 'div', aq_268, result_add_271)
                
                # Applying the binary operator '+' (line 124)
                result_add_273 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 55), '+', result_add_267, result_div_272)
                
                # Getting the type of 'a' (line 124)
                a_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'a')
                # Getting the type of 'i' (line 124)
                i_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 18), 'i')
                # Storing an element on a container (line 124)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 16), a_274, (i_275, result_add_273))
                
                # Assigning a BinOp to a Subscript (line 126):
                # Getting the type of 'bf' (line 126)
                bf_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 23), 'bf')
                # Getting the type of 'dbcc' (line 126)
                dbcc_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 28), 'dbcc')
                # Applying the binary operator '*' (line 126)
                result_mul_278 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 23), '*', bf_276, dbcc_277)
                
                # Getting the type of 'db' (line 126)
                db_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 35), 'db')
                # Getting the type of 'b1' (line 126)
                b1_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 41), 'b1')
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 126)
                i_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 48), 'i')
                int_282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 52), 'int')
                # Applying the binary operator '+' (line 126)
                result_add_283 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 48), '+', i_281, int_282)
                
                # Getting the type of 'b' (line 126)
                b_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 46), 'b')
                # Obtaining the member '__getitem__' of a type (line 126)
                getitem___285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 46), b_284, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 126)
                subscript_call_result_286 = invoke(stypy.reporting.localization.Localization(__file__, 126, 46), getitem___285, result_add_283)
                
                # Applying the binary operator '+' (line 126)
                result_add_287 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 41), '+', b1_280, subscript_call_result_286)
                
                # Applying the binary operator '*' (line 126)
                result_mul_288 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 35), '*', db_279, result_add_287)
                
                # Applying the binary operator '+' (line 126)
                result_add_289 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 23), '+', result_mul_278, result_mul_288)
                
                # Getting the type of 'aq' (line 126)
                aq_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 58), 'aq')
                # Applying the binary operator '+' (line 126)
                result_add_291 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 56), '+', result_add_289, aq_290)
                
                # Getting the type of 'b' (line 126)
                b_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'b')
                # Getting the type of 'i' (line 126)
                i_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 18), 'i')
                # Storing an element on a container (line 126)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 16), b_292, (i_293, result_add_291))
                
                # Getting the type of 'bsa' (line 127)
                bsa_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'bsa')
                # Getting the type of 'rc' (line 127)
                rc_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'rc')
                # Getting the type of 'af' (line 127)
                af_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 28), 'af')
                # Applying the binary operator '*' (line 127)
                result_mul_297 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 23), '*', rc_295, af_296)
                
                # Applying the binary operator '+=' (line 127)
                result_iadd_298 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 16), '+=', bsa_294, result_mul_297)
                # Assigning a type to the variable 'bsa' (line 127)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'bsa', result_iadd_298)
                
                
                # Assigning a Name to a Name (line 128):
                # Getting the type of 'af' (line 128)
                af_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 21), 'af')
                # Assigning a type to the variable 'a1' (line 128)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'a1', af_299)
                
                # Assigning a Name to a Name (line 129):
                # Getting the type of 'bf' (line 129)
                bf_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 21), 'bf')
                # Assigning a type to the variable 'b1' (line 129)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'b1', bf_300)
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a BinOp to a Name (line 132):
            # Getting the type of 'c' (line 132)
            c_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'c')
            float_302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 21), 'float')
            # Getting the type of 'rc' (line 132)
            rc_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 27), 'rc')
            # Applying the binary operator '-' (line 132)
            result_sub_304 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 21), '-', float_302, rc_303)
            
            # Applying the binary operator '*' (line 132)
            result_mul_305 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 16), '*', c_301, result_sub_304)
            
            # Getting the type of 'bsa' (line 132)
            bsa_306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 33), 'bsa')
            # Getting the type of 'nx' (line 132)
            nx_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 39), 'nx')
            # Applying the binary operator 'div' (line 132)
            result_div_308 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 33), 'div', bsa_306, nx_307)
            
            # Applying the binary operator '+' (line 132)
            result_add_309 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 16), '+', result_mul_305, result_div_308)
            
            # Assigning a type to the variable 'c' (line 132)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'c', result_add_309)
            
            # Assigning a BinOp to a Name (line 133):
            # Getting the type of 'rb' (line 133)
            rb_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 18), 'rb')
            # Getting the type of 'c' (line 133)
            c_311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 23), 'c')
            # Applying the binary operator 'div' (line 133)
            result_div_312 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 18), 'div', rb_310, c_311)
            
            # Assigning a type to the variable 'rbb' (line 133)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'rbb', result_div_312)
            
            # Assigning a BinOp to a Name (line 137):
            float_313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 19), 'float')
            float_314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 25), 'float')
            # Getting the type of 'db' (line 137)
            db_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 31), 'db')
            # Applying the binary operator '*' (line 137)
            result_mul_316 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 25), '*', float_314, db_315)
            
            # Applying the binary operator '-' (line 137)
            result_sub_317 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 19), '-', float_313, result_mul_316)
            
            # Getting the type of 'rbb' (line 137)
            rbb_318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 36), 'rbb')
            # Applying the binary operator '-' (line 137)
            result_sub_319 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 34), '-', result_sub_317, rbb_318)
            
            # Assigning a type to the variable 'dbcc' (line 137)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'dbcc', result_sub_319)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to saverow(...): (line 140)
        # Processing the call arguments (line 140)
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to xrange(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'nx' (line 140)
        nx_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 53), 'nx', False)
        # Processing the call keyword arguments (line 140)
        kwargs_330 = {}
        # Getting the type of 'xrange' (line 140)
        xrange_328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 46), 'xrange', False)
        # Calling xrange(args, kwargs) (line 140)
        xrange_call_result_331 = invoke(stypy.reporting.localization.Localization(__file__, 140, 46), xrange_328, *[nx_329], **kwargs_330)
        
        comprehension_332 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 24), xrange_call_result_331)
        # Assigning a type to the variable 'ix' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 24), 'ix', comprehension_332)
        int_322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 24), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'ix' (line 140)
        ix_323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 32), 'ix', False)
        # Getting the type of 'a' (line 140)
        a_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 30), 'a', False)
        # Obtaining the member '__getitem__' of a type (line 140)
        getitem___325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 30), a_324, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 140)
        subscript_call_result_326 = invoke(stypy.reporting.localization.Localization(__file__, 140, 30), getitem___325, ix_323)
        
        # Applying the binary operator '*' (line 140)
        result_mul_327 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 24), '*', int_322, subscript_call_result_326)
        
        list_333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 24), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 24), list_333, result_mul_327)
        # Processing the call keyword arguments (line 140)
        kwargs_334 = {}
        # Getting the type of 'outPGM' (line 140)
        outPGM_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'outPGM', False)
        # Obtaining the member 'saverow' of a type (line 140)
        saverow_321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), outPGM_320, 'saverow')
        # Calling saverow(args, kwargs) (line 140)
        saverow_call_result_335 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), saverow_321, *[list_333], **kwargs_334)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'oliva(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'oliva' in the type store
    # Getting the type of 'stypy_return_type' (line 69)
    stypy_return_type_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_336)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'oliva'
    return stypy_return_type_336

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
    kwargs_338 = {}
    # Getting the type of 'oliva' (line 144)
    oliva_337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'oliva', False)
    # Calling oliva(args, kwargs) (line 144)
    oliva_call_result_339 = invoke(stypy.reporting.localization.Localization(__file__, 144, 4), oliva_337, *[], **kwargs_338)
    
    # Getting the type of 'True' (line 145)
    True_340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'stypy_return_type', True_340)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 143)
    stypy_return_type_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_341)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_341

# Assigning a type to the variable 'run' (line 143)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 0), 'run', run)

# Call to run(...): (line 148)
# Processing the call keyword arguments (line 148)
kwargs_343 = {}
# Getting the type of 'run' (line 148)
run_342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 0), 'run', False)
# Calling run(args, kwargs) (line 148)
run_call_result_344 = invoke(stypy.reporting.localization.Localization(__file__, 148, 0), run_342, *[], **kwargs_343)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

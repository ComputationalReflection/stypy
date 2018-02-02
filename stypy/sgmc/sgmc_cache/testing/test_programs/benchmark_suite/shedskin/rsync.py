
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: # -*- coding: utf-8 -*-
3: 
4: ### to compile this, copy lib/hashlib.* to the shedskin lib dir, or use shedskin -Llib!
5: 
6: '''
7: This is a pure Python implementation of the [rsync algorithm](TM96).
8: 
9: [TM96] Andrew Tridgell and Paul Mackerras. The rsync algorithm.
10: Technical Report TR-CS-96-05, Canberra 0200 ACT, Australia, 1996.
11: http://samba.anu.edu.au/rsync/.
12: 
13: '''
14: 
15: # taken from: http://code.activestate.com/recipes/577518-rsync-algorithm/
16: 
17: import collections
18: import hashlib
19: import os
20: 
21: 
22: class Element:
23:     def __init__(self, index=-1, data=None):
24:         self.index = index
25:         self.data = data
26: 
27: 
28: def rsyncdelta(datastream, remotesignatures, blocksize):
29:     '''
30:     Generates a binary patch when supplied with the weak and strong
31:     hashes from an unpatched target and a readable stream for the
32:     up-to-date data. The blocksize must be the same as the value
33:     used to generate remotesignatures.
34:     '''
35:     remote_weak, remote_strong = remotesignatures
36: 
37:     match = True
38:     matchblock = -1
39:     delta = []
40: 
41:     while True:
42:         if match and datastream is not None:
43:             # Whenever there is a match or the loop is running for the first
44:             # time, populate the window using weakchecksum instead of rolling
45:             # through every single byte which takes at least twice as long.
46:             window = collections.deque(datastream.read(blocksize))
47:             checksum, a, b = weakchecksum(''.join(window))
48: 
49:         try:
50:             # If there are two identical weak checksums in a file, and the
51:             # matching strong hash does not occur at the first match, it will
52:             # be missed and the data sent over. May fix eventually, but this
53:             # problem arises very rarely.
54:             matchblock = remote_weak.index(checksum, matchblock + 1)
55:             stronghash = hashlib.md5(''.join(window)).hexdigest()
56: 
57:             if remote_strong[matchblock] == stronghash:
58:                 match = True
59:                 delta.append(Element(index=matchblock))
60: 
61:                 if datastream.closed:
62:                     break
63:                 continue
64: 
65:         except ValueError:
66:             # The weakchecksum did not match
67:             match = False
68:             if datastream:
69:                 # Get the next byte and affix to the window
70:                 newchar = datastream.read(1)
71:                 if newchar:
72:                     window.append(newchar)
73:                 else:
74:                     # No more data from the file; the window will slowly shrink.
75:                     # newchar needs to be zero from here on to keep the checksum
76:                     # correct.
77:                     newchar = '\0'
78:                     tailsize = datastream.tell() % blocksize
79:                     datastream = None
80: 
81:             if datastream is None and len(window) <= tailsize:
82:                 # The likelihood that any blocks will match after this is
83:                 # nearly nil so call it quits.
84:                 delta.append(Element(data=list(window)))
85:                 break
86: 
87:             # Yank off the extra byte and calculate the new window checksum
88:             oldchar = window.popleft()
89:             checksum, a, b = rollingchecksum(oldchar, newchar, a, b, blocksize)
90: 
91:             # Add the old byte the file delta. This is data that was not found
92:             # inside of a matching block so it needs to be sent to the target.
93:             if delta:
94:                 delta[-1].data.append(oldchar)
95:             else:
96:                 delta.append(Element(data=[oldchar]))
97: 
98:     return delta
99: 
100: 
101: def blockchecksums(instream, blocksize):
102:     '''
103:     Returns a list of weak and strong hashes for each block of the
104:     defined size for the given data stream.
105:     '''
106:     weakhashes = list()
107:     stronghashes = list()
108:     read = instream.read(blocksize)
109: 
110:     while read:
111:         weakhashes.append(weakchecksum(read)[0])
112:         stronghashes.append(hashlib.md5(read).hexdigest())
113:         read = instream.read(blocksize)
114: 
115:     return weakhashes, stronghashes
116: 
117: 
118: def patchstream(instream, outstream, delta, blocksize):
119:     '''
120:     Patches instream using the supplied delta and write the resultant
121:     data to outstream.
122:     '''
123:     for element in delta:
124:         if element.index != -1:
125:             instream.seek(element.index * blocksize)
126:             data = instream.read(blocksize)
127:         else:
128:             data = ''.join(element.data)
129:         outstream.write(data)
130: 
131: 
132: def rollingchecksum(removed, new, a, b, blocksize):
133:     '''
134:     Generates a new weak checksum when supplied with the internal state
135:     of the checksum calculation for the previous window, the removed
136:     byte, and the added byte.
137:     '''
138:     a -= ord(removed) - ord(new)
139:     b -= ord(removed) * blocksize - a
140:     return (b << 16) | a, a, b
141: 
142: 
143: def weakchecksum(data):
144:     '''
145:     Generates a weak checksum from an iterable set of bytes.
146:     '''
147:     a = b = 0
148:     l = len(data)
149: 
150:     for i in range(l):
151:         n = ord(data[i])
152:         a += n
153:         b += (l - i) * n
154: 
155:     return (b << 16) | a, a, b
156: 
157: 
158: def Relative(path):
159:     return os.path.join(os.path.dirname(__file__), path)
160: 
161: 
162: def run():
163:     blocksize = 4096
164: 
165:     # On the system containing the file that needs to be patched
166:     unpatched = open(Relative("testdata/unpatched.file"), "rb")
167:     hashes = blockchecksums(unpatched, blocksize)
168: 
169:     # On the remote system after having received `hashes`
170:     patchedfile = open(Relative("testdata/patched.file"), "rb")
171:     delta = rsyncdelta(patchedfile, hashes, blocksize)
172: 
173:     # System with the unpatched file after receiving `delta`
174:     unpatched.seek(0)
175:     save_to = open(Relative("testdata/locally-patched.file"), "wb")
176:     patchstream(unpatched, save_to, delta, blocksize)
177:     return True
178: 
179: 
180: run()
181: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, (-1)), 'str', '\nThis is a pure Python implementation of the [rsync algorithm](TM96).\n\n[TM96] Andrew Tridgell and Paul Mackerras. The rsync algorithm.\nTechnical Report TR-CS-96-05, Canberra 0200 ACT, Australia, 1996.\nhttp://samba.anu.edu.au/rsync/.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'import collections' statement (line 17)
import collections

import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'collections', collections, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'import hashlib' statement (line 18)
import hashlib

import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'hashlib', hashlib, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'import os' statement (line 19)
import os

import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'os', os, module_type_store)

# Declaration of the 'Element' class

class Element:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 29), 'int')
        # Getting the type of 'None' (line 23)
        None_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 38), 'None')
        defaults = [int_12, None_13]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 23, 4, False)
        # Assigning a type to the variable 'self' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Element.__init__', ['index', 'data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['index', 'data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 24):
        
        # Assigning a Name to a Attribute (line 24):
        # Getting the type of 'index' (line 24)
        index_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 21), 'index')
        # Getting the type of 'self' (line 24)
        self_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'self')
        # Setting the type of the member 'index' of a type (line 24)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), self_15, 'index', index_14)
        
        # Assigning a Name to a Attribute (line 25):
        
        # Assigning a Name to a Attribute (line 25):
        # Getting the type of 'data' (line 25)
        data_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), 'data')
        # Getting the type of 'self' (line 25)
        self_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self')
        # Setting the type of the member 'data' of a type (line 25)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_17, 'data', data_16)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Element' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'Element', Element)

@norecursion
def rsyncdelta(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'rsyncdelta'
    module_type_store = module_type_store.open_function_context('rsyncdelta', 28, 0, False)
    
    # Passed parameters checking function
    rsyncdelta.stypy_localization = localization
    rsyncdelta.stypy_type_of_self = None
    rsyncdelta.stypy_type_store = module_type_store
    rsyncdelta.stypy_function_name = 'rsyncdelta'
    rsyncdelta.stypy_param_names_list = ['datastream', 'remotesignatures', 'blocksize']
    rsyncdelta.stypy_varargs_param_name = None
    rsyncdelta.stypy_kwargs_param_name = None
    rsyncdelta.stypy_call_defaults = defaults
    rsyncdelta.stypy_call_varargs = varargs
    rsyncdelta.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rsyncdelta', ['datastream', 'remotesignatures', 'blocksize'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rsyncdelta', localization, ['datastream', 'remotesignatures', 'blocksize'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rsyncdelta(...)' code ##################

    str_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, (-1)), 'str', '\n    Generates a binary patch when supplied with the weak and strong\n    hashes from an unpatched target and a readable stream for the\n    up-to-date data. The blocksize must be the same as the value\n    used to generate remotesignatures.\n    ')
    
    # Assigning a Name to a Tuple (line 35):
    
    # Assigning a Subscript to a Name (line 35):
    
    # Obtaining the type of the subscript
    int_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 4), 'int')
    # Getting the type of 'remotesignatures' (line 35)
    remotesignatures_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 33), 'remotesignatures')
    # Obtaining the member '__getitem__' of a type (line 35)
    getitem___21 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 4), remotesignatures_20, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 35)
    subscript_call_result_22 = invoke(stypy.reporting.localization.Localization(__file__, 35, 4), getitem___21, int_19)
    
    # Assigning a type to the variable 'tuple_var_assignment_1' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'tuple_var_assignment_1', subscript_call_result_22)
    
    # Assigning a Subscript to a Name (line 35):
    
    # Obtaining the type of the subscript
    int_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 4), 'int')
    # Getting the type of 'remotesignatures' (line 35)
    remotesignatures_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 33), 'remotesignatures')
    # Obtaining the member '__getitem__' of a type (line 35)
    getitem___25 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 4), remotesignatures_24, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 35)
    subscript_call_result_26 = invoke(stypy.reporting.localization.Localization(__file__, 35, 4), getitem___25, int_23)
    
    # Assigning a type to the variable 'tuple_var_assignment_2' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'tuple_var_assignment_2', subscript_call_result_26)
    
    # Assigning a Name to a Name (line 35):
    # Getting the type of 'tuple_var_assignment_1' (line 35)
    tuple_var_assignment_1_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'tuple_var_assignment_1')
    # Assigning a type to the variable 'remote_weak' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'remote_weak', tuple_var_assignment_1_27)
    
    # Assigning a Name to a Name (line 35):
    # Getting the type of 'tuple_var_assignment_2' (line 35)
    tuple_var_assignment_2_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'tuple_var_assignment_2')
    # Assigning a type to the variable 'remote_strong' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 17), 'remote_strong', tuple_var_assignment_2_28)
    
    # Assigning a Name to a Name (line 37):
    
    # Assigning a Name to a Name (line 37):
    # Getting the type of 'True' (line 37)
    True_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'True')
    # Assigning a type to the variable 'match' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'match', True_29)
    
    # Assigning a Num to a Name (line 38):
    
    # Assigning a Num to a Name (line 38):
    int_30 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 17), 'int')
    # Assigning a type to the variable 'matchblock' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'matchblock', int_30)
    
    # Assigning a List to a Name (line 39):
    
    # Assigning a List to a Name (line 39):
    
    # Obtaining an instance of the builtin type 'list' (line 39)
    list_31 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 39)
    
    # Assigning a type to the variable 'delta' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'delta', list_31)
    
    # Getting the type of 'True' (line 41)
    True_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 10), 'True')
    # Testing if the while is going to be iterated (line 41)
    # Testing the type of an if condition (line 41)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 4), True_32)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 41, 4), True_32):
        # SSA begins for while statement (line 41)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Evaluating a boolean operation
        # Getting the type of 'match' (line 42)
        match_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'match')
        
        # Getting the type of 'datastream' (line 42)
        datastream_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 21), 'datastream')
        # Getting the type of 'None' (line 42)
        None_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 39), 'None')
        # Applying the binary operator 'isnot' (line 42)
        result_is_not_36 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 21), 'isnot', datastream_34, None_35)
        
        # Applying the binary operator 'and' (line 42)
        result_and_keyword_37 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 11), 'and', match_33, result_is_not_36)
        
        # Testing if the type of an if condition is none (line 42)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 42, 8), result_and_keyword_37):
            pass
        else:
            
            # Testing the type of an if condition (line 42)
            if_condition_38 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 8), result_and_keyword_37)
            # Assigning a type to the variable 'if_condition_38' (line 42)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'if_condition_38', if_condition_38)
            # SSA begins for if statement (line 42)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 46):
            
            # Assigning a Call to a Name (line 46):
            
            # Call to deque(...): (line 46)
            # Processing the call arguments (line 46)
            
            # Call to read(...): (line 46)
            # Processing the call arguments (line 46)
            # Getting the type of 'blocksize' (line 46)
            blocksize_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 55), 'blocksize', False)
            # Processing the call keyword arguments (line 46)
            kwargs_44 = {}
            # Getting the type of 'datastream' (line 46)
            datastream_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 39), 'datastream', False)
            # Obtaining the member 'read' of a type (line 46)
            read_42 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 39), datastream_41, 'read')
            # Calling read(args, kwargs) (line 46)
            read_call_result_45 = invoke(stypy.reporting.localization.Localization(__file__, 46, 39), read_42, *[blocksize_43], **kwargs_44)
            
            # Processing the call keyword arguments (line 46)
            kwargs_46 = {}
            # Getting the type of 'collections' (line 46)
            collections_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 21), 'collections', False)
            # Obtaining the member 'deque' of a type (line 46)
            deque_40 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 21), collections_39, 'deque')
            # Calling deque(args, kwargs) (line 46)
            deque_call_result_47 = invoke(stypy.reporting.localization.Localization(__file__, 46, 21), deque_40, *[read_call_result_45], **kwargs_46)
            
            # Assigning a type to the variable 'window' (line 46)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'window', deque_call_result_47)
            
            # Assigning a Call to a Tuple (line 47):
            
            # Assigning a Call to a Name:
            
            # Call to weakchecksum(...): (line 47)
            # Processing the call arguments (line 47)
            
            # Call to join(...): (line 47)
            # Processing the call arguments (line 47)
            # Getting the type of 'window' (line 47)
            window_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 50), 'window', False)
            # Processing the call keyword arguments (line 47)
            kwargs_52 = {}
            str_49 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 42), 'str', '')
            # Obtaining the member 'join' of a type (line 47)
            join_50 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 42), str_49, 'join')
            # Calling join(args, kwargs) (line 47)
            join_call_result_53 = invoke(stypy.reporting.localization.Localization(__file__, 47, 42), join_50, *[window_51], **kwargs_52)
            
            # Processing the call keyword arguments (line 47)
            kwargs_54 = {}
            # Getting the type of 'weakchecksum' (line 47)
            weakchecksum_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 29), 'weakchecksum', False)
            # Calling weakchecksum(args, kwargs) (line 47)
            weakchecksum_call_result_55 = invoke(stypy.reporting.localization.Localization(__file__, 47, 29), weakchecksum_48, *[join_call_result_53], **kwargs_54)
            
            # Assigning a type to the variable 'call_assignment_3' (line 47)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'call_assignment_3', weakchecksum_call_result_55)
            
            # Assigning a Call to a Name (line 47):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_58 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 12), 'int')
            # Processing the call keyword arguments
            kwargs_59 = {}
            # Getting the type of 'call_assignment_3' (line 47)
            call_assignment_3_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'call_assignment_3', False)
            # Obtaining the member '__getitem__' of a type (line 47)
            getitem___57 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), call_assignment_3_56, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_60 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___57, *[int_58], **kwargs_59)
            
            # Assigning a type to the variable 'call_assignment_4' (line 47)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'call_assignment_4', getitem___call_result_60)
            
            # Assigning a Name to a Name (line 47):
            # Getting the type of 'call_assignment_4' (line 47)
            call_assignment_4_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'call_assignment_4')
            # Assigning a type to the variable 'checksum' (line 47)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'checksum', call_assignment_4_61)
            
            # Assigning a Call to a Name (line 47):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_64 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 12), 'int')
            # Processing the call keyword arguments
            kwargs_65 = {}
            # Getting the type of 'call_assignment_3' (line 47)
            call_assignment_3_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'call_assignment_3', False)
            # Obtaining the member '__getitem__' of a type (line 47)
            getitem___63 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), call_assignment_3_62, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_66 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___63, *[int_64], **kwargs_65)
            
            # Assigning a type to the variable 'call_assignment_5' (line 47)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'call_assignment_5', getitem___call_result_66)
            
            # Assigning a Name to a Name (line 47):
            # Getting the type of 'call_assignment_5' (line 47)
            call_assignment_5_67 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'call_assignment_5')
            # Assigning a type to the variable 'a' (line 47)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 22), 'a', call_assignment_5_67)
            
            # Assigning a Call to a Name (line 47):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_70 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 12), 'int')
            # Processing the call keyword arguments
            kwargs_71 = {}
            # Getting the type of 'call_assignment_3' (line 47)
            call_assignment_3_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'call_assignment_3', False)
            # Obtaining the member '__getitem__' of a type (line 47)
            getitem___69 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), call_assignment_3_68, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_72 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___69, *[int_70], **kwargs_71)
            
            # Assigning a type to the variable 'call_assignment_6' (line 47)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'call_assignment_6', getitem___call_result_72)
            
            # Assigning a Name to a Name (line 47):
            # Getting the type of 'call_assignment_6' (line 47)
            call_assignment_6_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'call_assignment_6')
            # Assigning a type to the variable 'b' (line 47)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 25), 'b', call_assignment_6_73)
            # SSA join for if statement (line 42)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # SSA begins for try-except statement (line 49)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 54):
        
        # Assigning a Call to a Name (line 54):
        
        # Call to index(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'checksum' (line 54)
        checksum_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 43), 'checksum', False)
        # Getting the type of 'matchblock' (line 54)
        matchblock_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 53), 'matchblock', False)
        int_78 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 66), 'int')
        # Applying the binary operator '+' (line 54)
        result_add_79 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 53), '+', matchblock_77, int_78)
        
        # Processing the call keyword arguments (line 54)
        kwargs_80 = {}
        # Getting the type of 'remote_weak' (line 54)
        remote_weak_74 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 25), 'remote_weak', False)
        # Obtaining the member 'index' of a type (line 54)
        index_75 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 25), remote_weak_74, 'index')
        # Calling index(args, kwargs) (line 54)
        index_call_result_81 = invoke(stypy.reporting.localization.Localization(__file__, 54, 25), index_75, *[checksum_76, result_add_79], **kwargs_80)
        
        # Assigning a type to the variable 'matchblock' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'matchblock', index_call_result_81)
        
        # Assigning a Call to a Name (line 55):
        
        # Assigning a Call to a Name (line 55):
        
        # Call to hexdigest(...): (line 55)
        # Processing the call keyword arguments (line 55)
        kwargs_92 = {}
        
        # Call to md5(...): (line 55)
        # Processing the call arguments (line 55)
        
        # Call to join(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'window' (line 55)
        window_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 45), 'window', False)
        # Processing the call keyword arguments (line 55)
        kwargs_87 = {}
        str_84 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 37), 'str', '')
        # Obtaining the member 'join' of a type (line 55)
        join_85 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 37), str_84, 'join')
        # Calling join(args, kwargs) (line 55)
        join_call_result_88 = invoke(stypy.reporting.localization.Localization(__file__, 55, 37), join_85, *[window_86], **kwargs_87)
        
        # Processing the call keyword arguments (line 55)
        kwargs_89 = {}
        # Getting the type of 'hashlib' (line 55)
        hashlib_82 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 25), 'hashlib', False)
        # Obtaining the member 'md5' of a type (line 55)
        md5_83 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 25), hashlib_82, 'md5')
        # Calling md5(args, kwargs) (line 55)
        md5_call_result_90 = invoke(stypy.reporting.localization.Localization(__file__, 55, 25), md5_83, *[join_call_result_88], **kwargs_89)
        
        # Obtaining the member 'hexdigest' of a type (line 55)
        hexdigest_91 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 25), md5_call_result_90, 'hexdigest')
        # Calling hexdigest(args, kwargs) (line 55)
        hexdigest_call_result_93 = invoke(stypy.reporting.localization.Localization(__file__, 55, 25), hexdigest_91, *[], **kwargs_92)
        
        # Assigning a type to the variable 'stronghash' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'stronghash', hexdigest_call_result_93)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'matchblock' (line 57)
        matchblock_94 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 29), 'matchblock')
        # Getting the type of 'remote_strong' (line 57)
        remote_strong_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 15), 'remote_strong')
        # Obtaining the member '__getitem__' of a type (line 57)
        getitem___96 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 15), remote_strong_95, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 57)
        subscript_call_result_97 = invoke(stypy.reporting.localization.Localization(__file__, 57, 15), getitem___96, matchblock_94)
        
        # Getting the type of 'stronghash' (line 57)
        stronghash_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 44), 'stronghash')
        # Applying the binary operator '==' (line 57)
        result_eq_99 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 15), '==', subscript_call_result_97, stronghash_98)
        
        # Testing if the type of an if condition is none (line 57)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 57, 12), result_eq_99):
            pass
        else:
            
            # Testing the type of an if condition (line 57)
            if_condition_100 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 12), result_eq_99)
            # Assigning a type to the variable 'if_condition_100' (line 57)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'if_condition_100', if_condition_100)
            # SSA begins for if statement (line 57)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 58):
            
            # Assigning a Name to a Name (line 58):
            # Getting the type of 'True' (line 58)
            True_101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 24), 'True')
            # Assigning a type to the variable 'match' (line 58)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'match', True_101)
            
            # Call to append(...): (line 59)
            # Processing the call arguments (line 59)
            
            # Call to Element(...): (line 59)
            # Processing the call keyword arguments (line 59)
            # Getting the type of 'matchblock' (line 59)
            matchblock_105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 43), 'matchblock', False)
            keyword_106 = matchblock_105
            kwargs_107 = {'index': keyword_106}
            # Getting the type of 'Element' (line 59)
            Element_104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 29), 'Element', False)
            # Calling Element(args, kwargs) (line 59)
            Element_call_result_108 = invoke(stypy.reporting.localization.Localization(__file__, 59, 29), Element_104, *[], **kwargs_107)
            
            # Processing the call keyword arguments (line 59)
            kwargs_109 = {}
            # Getting the type of 'delta' (line 59)
            delta_102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'delta', False)
            # Obtaining the member 'append' of a type (line 59)
            append_103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 16), delta_102, 'append')
            # Calling append(args, kwargs) (line 59)
            append_call_result_110 = invoke(stypy.reporting.localization.Localization(__file__, 59, 16), append_103, *[Element_call_result_108], **kwargs_109)
            
            # Getting the type of 'datastream' (line 61)
            datastream_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 19), 'datastream')
            # Obtaining the member 'closed' of a type (line 61)
            closed_112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 19), datastream_111, 'closed')
            # Testing if the type of an if condition is none (line 61)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 61, 16), closed_112):
                pass
            else:
                
                # Testing the type of an if condition (line 61)
                if_condition_113 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 16), closed_112)
                # Assigning a type to the variable 'if_condition_113' (line 61)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'if_condition_113', if_condition_113)
                # SSA begins for if statement (line 61)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA join for if statement (line 61)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 57)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA branch for the except part of a try statement (line 49)
        # SSA branch for the except 'ValueError' branch of a try statement (line 49)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Name to a Name (line 67):
        
        # Assigning a Name to a Name (line 67):
        # Getting the type of 'False' (line 67)
        False_114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 20), 'False')
        # Assigning a type to the variable 'match' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'match', False_114)
        # Getting the type of 'datastream' (line 68)
        datastream_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 15), 'datastream')
        # Testing if the type of an if condition is none (line 68)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 68, 12), datastream_115):
            pass
        else:
            
            # Testing the type of an if condition (line 68)
            if_condition_116 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 12), datastream_115)
            # Assigning a type to the variable 'if_condition_116' (line 68)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'if_condition_116', if_condition_116)
            # SSA begins for if statement (line 68)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 70):
            
            # Assigning a Call to a Name (line 70):
            
            # Call to read(...): (line 70)
            # Processing the call arguments (line 70)
            int_119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 42), 'int')
            # Processing the call keyword arguments (line 70)
            kwargs_120 = {}
            # Getting the type of 'datastream' (line 70)
            datastream_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 26), 'datastream', False)
            # Obtaining the member 'read' of a type (line 70)
            read_118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 26), datastream_117, 'read')
            # Calling read(args, kwargs) (line 70)
            read_call_result_121 = invoke(stypy.reporting.localization.Localization(__file__, 70, 26), read_118, *[int_119], **kwargs_120)
            
            # Assigning a type to the variable 'newchar' (line 70)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'newchar', read_call_result_121)
            # Getting the type of 'newchar' (line 71)
            newchar_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'newchar')
            # Testing if the type of an if condition is none (line 71)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 71, 16), newchar_122):
                
                # Assigning a Str to a Name (line 77):
                
                # Assigning a Str to a Name (line 77):
                str_129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 30), 'str', '\x00')
                # Assigning a type to the variable 'newchar' (line 77)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 20), 'newchar', str_129)
                
                # Assigning a BinOp to a Name (line 78):
                
                # Assigning a BinOp to a Name (line 78):
                
                # Call to tell(...): (line 78)
                # Processing the call keyword arguments (line 78)
                kwargs_132 = {}
                # Getting the type of 'datastream' (line 78)
                datastream_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 31), 'datastream', False)
                # Obtaining the member 'tell' of a type (line 78)
                tell_131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 31), datastream_130, 'tell')
                # Calling tell(args, kwargs) (line 78)
                tell_call_result_133 = invoke(stypy.reporting.localization.Localization(__file__, 78, 31), tell_131, *[], **kwargs_132)
                
                # Getting the type of 'blocksize' (line 78)
                blocksize_134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 51), 'blocksize')
                # Applying the binary operator '%' (line 78)
                result_mod_135 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 31), '%', tell_call_result_133, blocksize_134)
                
                # Assigning a type to the variable 'tailsize' (line 78)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 20), 'tailsize', result_mod_135)
                
                # Assigning a Name to a Name (line 79):
                
                # Assigning a Name to a Name (line 79):
                # Getting the type of 'None' (line 79)
                None_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 33), 'None')
                # Assigning a type to the variable 'datastream' (line 79)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 20), 'datastream', None_136)
            else:
                
                # Testing the type of an if condition (line 71)
                if_condition_123 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 16), newchar_122)
                # Assigning a type to the variable 'if_condition_123' (line 71)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'if_condition_123', if_condition_123)
                # SSA begins for if statement (line 71)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 72)
                # Processing the call arguments (line 72)
                # Getting the type of 'newchar' (line 72)
                newchar_126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 34), 'newchar', False)
                # Processing the call keyword arguments (line 72)
                kwargs_127 = {}
                # Getting the type of 'window' (line 72)
                window_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 20), 'window', False)
                # Obtaining the member 'append' of a type (line 72)
                append_125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 20), window_124, 'append')
                # Calling append(args, kwargs) (line 72)
                append_call_result_128 = invoke(stypy.reporting.localization.Localization(__file__, 72, 20), append_125, *[newchar_126], **kwargs_127)
                
                # SSA branch for the else part of an if statement (line 71)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Str to a Name (line 77):
                
                # Assigning a Str to a Name (line 77):
                str_129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 30), 'str', '\x00')
                # Assigning a type to the variable 'newchar' (line 77)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 20), 'newchar', str_129)
                
                # Assigning a BinOp to a Name (line 78):
                
                # Assigning a BinOp to a Name (line 78):
                
                # Call to tell(...): (line 78)
                # Processing the call keyword arguments (line 78)
                kwargs_132 = {}
                # Getting the type of 'datastream' (line 78)
                datastream_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 31), 'datastream', False)
                # Obtaining the member 'tell' of a type (line 78)
                tell_131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 31), datastream_130, 'tell')
                # Calling tell(args, kwargs) (line 78)
                tell_call_result_133 = invoke(stypy.reporting.localization.Localization(__file__, 78, 31), tell_131, *[], **kwargs_132)
                
                # Getting the type of 'blocksize' (line 78)
                blocksize_134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 51), 'blocksize')
                # Applying the binary operator '%' (line 78)
                result_mod_135 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 31), '%', tell_call_result_133, blocksize_134)
                
                # Assigning a type to the variable 'tailsize' (line 78)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 20), 'tailsize', result_mod_135)
                
                # Assigning a Name to a Name (line 79):
                
                # Assigning a Name to a Name (line 79):
                # Getting the type of 'None' (line 79)
                None_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 33), 'None')
                # Assigning a type to the variable 'datastream' (line 79)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 20), 'datastream', None_136)
                # SSA join for if statement (line 71)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 68)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Evaluating a boolean operation
        
        # Getting the type of 'datastream' (line 81)
        datastream_137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 15), 'datastream')
        # Getting the type of 'None' (line 81)
        None_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 29), 'None')
        # Applying the binary operator 'is' (line 81)
        result_is__139 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 15), 'is', datastream_137, None_138)
        
        
        
        # Call to len(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'window' (line 81)
        window_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 42), 'window', False)
        # Processing the call keyword arguments (line 81)
        kwargs_142 = {}
        # Getting the type of 'len' (line 81)
        len_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 38), 'len', False)
        # Calling len(args, kwargs) (line 81)
        len_call_result_143 = invoke(stypy.reporting.localization.Localization(__file__, 81, 38), len_140, *[window_141], **kwargs_142)
        
        # Getting the type of 'tailsize' (line 81)
        tailsize_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 53), 'tailsize')
        # Applying the binary operator '<=' (line 81)
        result_le_145 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 38), '<=', len_call_result_143, tailsize_144)
        
        # Applying the binary operator 'and' (line 81)
        result_and_keyword_146 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 15), 'and', result_is__139, result_le_145)
        
        # Testing if the type of an if condition is none (line 81)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 81, 12), result_and_keyword_146):
            pass
        else:
            
            # Testing the type of an if condition (line 81)
            if_condition_147 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 12), result_and_keyword_146)
            # Assigning a type to the variable 'if_condition_147' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'if_condition_147', if_condition_147)
            # SSA begins for if statement (line 81)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 84)
            # Processing the call arguments (line 84)
            
            # Call to Element(...): (line 84)
            # Processing the call keyword arguments (line 84)
            
            # Call to list(...): (line 84)
            # Processing the call arguments (line 84)
            # Getting the type of 'window' (line 84)
            window_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 47), 'window', False)
            # Processing the call keyword arguments (line 84)
            kwargs_153 = {}
            # Getting the type of 'list' (line 84)
            list_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 42), 'list', False)
            # Calling list(args, kwargs) (line 84)
            list_call_result_154 = invoke(stypy.reporting.localization.Localization(__file__, 84, 42), list_151, *[window_152], **kwargs_153)
            
            keyword_155 = list_call_result_154
            kwargs_156 = {'data': keyword_155}
            # Getting the type of 'Element' (line 84)
            Element_150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 29), 'Element', False)
            # Calling Element(args, kwargs) (line 84)
            Element_call_result_157 = invoke(stypy.reporting.localization.Localization(__file__, 84, 29), Element_150, *[], **kwargs_156)
            
            # Processing the call keyword arguments (line 84)
            kwargs_158 = {}
            # Getting the type of 'delta' (line 84)
            delta_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'delta', False)
            # Obtaining the member 'append' of a type (line 84)
            append_149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 16), delta_148, 'append')
            # Calling append(args, kwargs) (line 84)
            append_call_result_159 = invoke(stypy.reporting.localization.Localization(__file__, 84, 16), append_149, *[Element_call_result_157], **kwargs_158)
            
            # SSA join for if statement (line 81)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 88):
        
        # Assigning a Call to a Name (line 88):
        
        # Call to popleft(...): (line 88)
        # Processing the call keyword arguments (line 88)
        kwargs_162 = {}
        # Getting the type of 'window' (line 88)
        window_160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 22), 'window', False)
        # Obtaining the member 'popleft' of a type (line 88)
        popleft_161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 22), window_160, 'popleft')
        # Calling popleft(args, kwargs) (line 88)
        popleft_call_result_163 = invoke(stypy.reporting.localization.Localization(__file__, 88, 22), popleft_161, *[], **kwargs_162)
        
        # Assigning a type to the variable 'oldchar' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'oldchar', popleft_call_result_163)
        
        # Assigning a Call to a Tuple (line 89):
        
        # Assigning a Call to a Name:
        
        # Call to rollingchecksum(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'oldchar' (line 89)
        oldchar_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 45), 'oldchar', False)
        # Getting the type of 'newchar' (line 89)
        newchar_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 54), 'newchar', False)
        # Getting the type of 'a' (line 89)
        a_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 63), 'a', False)
        # Getting the type of 'b' (line 89)
        b_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 66), 'b', False)
        # Getting the type of 'blocksize' (line 89)
        blocksize_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 69), 'blocksize', False)
        # Processing the call keyword arguments (line 89)
        kwargs_170 = {}
        # Getting the type of 'rollingchecksum' (line 89)
        rollingchecksum_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 29), 'rollingchecksum', False)
        # Calling rollingchecksum(args, kwargs) (line 89)
        rollingchecksum_call_result_171 = invoke(stypy.reporting.localization.Localization(__file__, 89, 29), rollingchecksum_164, *[oldchar_165, newchar_166, a_167, b_168, blocksize_169], **kwargs_170)
        
        # Assigning a type to the variable 'call_assignment_7' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'call_assignment_7', rollingchecksum_call_result_171)
        
        # Assigning a Call to a Name (line 89):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 12), 'int')
        # Processing the call keyword arguments
        kwargs_175 = {}
        # Getting the type of 'call_assignment_7' (line 89)
        call_assignment_7_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'call_assignment_7', False)
        # Obtaining the member '__getitem__' of a type (line 89)
        getitem___173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 12), call_assignment_7_172, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_176 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___173, *[int_174], **kwargs_175)
        
        # Assigning a type to the variable 'call_assignment_8' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'call_assignment_8', getitem___call_result_176)
        
        # Assigning a Name to a Name (line 89):
        # Getting the type of 'call_assignment_8' (line 89)
        call_assignment_8_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'call_assignment_8')
        # Assigning a type to the variable 'checksum' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'checksum', call_assignment_8_177)
        
        # Assigning a Call to a Name (line 89):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 12), 'int')
        # Processing the call keyword arguments
        kwargs_181 = {}
        # Getting the type of 'call_assignment_7' (line 89)
        call_assignment_7_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'call_assignment_7', False)
        # Obtaining the member '__getitem__' of a type (line 89)
        getitem___179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 12), call_assignment_7_178, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_182 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___179, *[int_180], **kwargs_181)
        
        # Assigning a type to the variable 'call_assignment_9' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'call_assignment_9', getitem___call_result_182)
        
        # Assigning a Name to a Name (line 89):
        # Getting the type of 'call_assignment_9' (line 89)
        call_assignment_9_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'call_assignment_9')
        # Assigning a type to the variable 'a' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 22), 'a', call_assignment_9_183)
        
        # Assigning a Call to a Name (line 89):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 12), 'int')
        # Processing the call keyword arguments
        kwargs_187 = {}
        # Getting the type of 'call_assignment_7' (line 89)
        call_assignment_7_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'call_assignment_7', False)
        # Obtaining the member '__getitem__' of a type (line 89)
        getitem___185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 12), call_assignment_7_184, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_188 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___185, *[int_186], **kwargs_187)
        
        # Assigning a type to the variable 'call_assignment_10' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'call_assignment_10', getitem___call_result_188)
        
        # Assigning a Name to a Name (line 89):
        # Getting the type of 'call_assignment_10' (line 89)
        call_assignment_10_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'call_assignment_10')
        # Assigning a type to the variable 'b' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 25), 'b', call_assignment_10_189)
        # Getting the type of 'delta' (line 93)
        delta_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 15), 'delta')
        # Testing if the type of an if condition is none (line 93)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 93, 12), delta_190):
            
            # Call to append(...): (line 96)
            # Processing the call arguments (line 96)
            
            # Call to Element(...): (line 96)
            # Processing the call keyword arguments (line 96)
            
            # Obtaining an instance of the builtin type 'list' (line 96)
            list_204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 42), 'list')
            # Adding type elements to the builtin type 'list' instance (line 96)
            # Adding element type (line 96)
            # Getting the type of 'oldchar' (line 96)
            oldchar_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 43), 'oldchar', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 42), list_204, oldchar_205)
            
            keyword_206 = list_204
            kwargs_207 = {'data': keyword_206}
            # Getting the type of 'Element' (line 96)
            Element_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 29), 'Element', False)
            # Calling Element(args, kwargs) (line 96)
            Element_call_result_208 = invoke(stypy.reporting.localization.Localization(__file__, 96, 29), Element_203, *[], **kwargs_207)
            
            # Processing the call keyword arguments (line 96)
            kwargs_209 = {}
            # Getting the type of 'delta' (line 96)
            delta_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'delta', False)
            # Obtaining the member 'append' of a type (line 96)
            append_202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 16), delta_201, 'append')
            # Calling append(args, kwargs) (line 96)
            append_call_result_210 = invoke(stypy.reporting.localization.Localization(__file__, 96, 16), append_202, *[Element_call_result_208], **kwargs_209)
            
        else:
            
            # Testing the type of an if condition (line 93)
            if_condition_191 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 12), delta_190)
            # Assigning a type to the variable 'if_condition_191' (line 93)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'if_condition_191', if_condition_191)
            # SSA begins for if statement (line 93)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 94)
            # Processing the call arguments (line 94)
            # Getting the type of 'oldchar' (line 94)
            oldchar_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 38), 'oldchar', False)
            # Processing the call keyword arguments (line 94)
            kwargs_199 = {}
            
            # Obtaining the type of the subscript
            int_192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 22), 'int')
            # Getting the type of 'delta' (line 94)
            delta_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 16), 'delta', False)
            # Obtaining the member '__getitem__' of a type (line 94)
            getitem___194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 16), delta_193, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 94)
            subscript_call_result_195 = invoke(stypy.reporting.localization.Localization(__file__, 94, 16), getitem___194, int_192)
            
            # Obtaining the member 'data' of a type (line 94)
            data_196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 16), subscript_call_result_195, 'data')
            # Obtaining the member 'append' of a type (line 94)
            append_197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 16), data_196, 'append')
            # Calling append(args, kwargs) (line 94)
            append_call_result_200 = invoke(stypy.reporting.localization.Localization(__file__, 94, 16), append_197, *[oldchar_198], **kwargs_199)
            
            # SSA branch for the else part of an if statement (line 93)
            module_type_store.open_ssa_branch('else')
            
            # Call to append(...): (line 96)
            # Processing the call arguments (line 96)
            
            # Call to Element(...): (line 96)
            # Processing the call keyword arguments (line 96)
            
            # Obtaining an instance of the builtin type 'list' (line 96)
            list_204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 42), 'list')
            # Adding type elements to the builtin type 'list' instance (line 96)
            # Adding element type (line 96)
            # Getting the type of 'oldchar' (line 96)
            oldchar_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 43), 'oldchar', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 42), list_204, oldchar_205)
            
            keyword_206 = list_204
            kwargs_207 = {'data': keyword_206}
            # Getting the type of 'Element' (line 96)
            Element_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 29), 'Element', False)
            # Calling Element(args, kwargs) (line 96)
            Element_call_result_208 = invoke(stypy.reporting.localization.Localization(__file__, 96, 29), Element_203, *[], **kwargs_207)
            
            # Processing the call keyword arguments (line 96)
            kwargs_209 = {}
            # Getting the type of 'delta' (line 96)
            delta_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'delta', False)
            # Obtaining the member 'append' of a type (line 96)
            append_202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 16), delta_201, 'append')
            # Calling append(args, kwargs) (line 96)
            append_call_result_210 = invoke(stypy.reporting.localization.Localization(__file__, 96, 16), append_202, *[Element_call_result_208], **kwargs_209)
            
            # SSA join for if statement (line 93)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for try-except statement (line 49)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 41)
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'delta' (line 98)
    delta_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 11), 'delta')
    # Assigning a type to the variable 'stypy_return_type' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'stypy_return_type', delta_211)
    
    # ################# End of 'rsyncdelta(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rsyncdelta' in the type store
    # Getting the type of 'stypy_return_type' (line 28)
    stypy_return_type_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_212)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rsyncdelta'
    return stypy_return_type_212

# Assigning a type to the variable 'rsyncdelta' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'rsyncdelta', rsyncdelta)

@norecursion
def blockchecksums(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'blockchecksums'
    module_type_store = module_type_store.open_function_context('blockchecksums', 101, 0, False)
    
    # Passed parameters checking function
    blockchecksums.stypy_localization = localization
    blockchecksums.stypy_type_of_self = None
    blockchecksums.stypy_type_store = module_type_store
    blockchecksums.stypy_function_name = 'blockchecksums'
    blockchecksums.stypy_param_names_list = ['instream', 'blocksize']
    blockchecksums.stypy_varargs_param_name = None
    blockchecksums.stypy_kwargs_param_name = None
    blockchecksums.stypy_call_defaults = defaults
    blockchecksums.stypy_call_varargs = varargs
    blockchecksums.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'blockchecksums', ['instream', 'blocksize'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'blockchecksums', localization, ['instream', 'blocksize'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'blockchecksums(...)' code ##################

    str_213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, (-1)), 'str', '\n    Returns a list of weak and strong hashes for each block of the\n    defined size for the given data stream.\n    ')
    
    # Assigning a Call to a Name (line 106):
    
    # Assigning a Call to a Name (line 106):
    
    # Call to list(...): (line 106)
    # Processing the call keyword arguments (line 106)
    kwargs_215 = {}
    # Getting the type of 'list' (line 106)
    list_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 17), 'list', False)
    # Calling list(args, kwargs) (line 106)
    list_call_result_216 = invoke(stypy.reporting.localization.Localization(__file__, 106, 17), list_214, *[], **kwargs_215)
    
    # Assigning a type to the variable 'weakhashes' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'weakhashes', list_call_result_216)
    
    # Assigning a Call to a Name (line 107):
    
    # Assigning a Call to a Name (line 107):
    
    # Call to list(...): (line 107)
    # Processing the call keyword arguments (line 107)
    kwargs_218 = {}
    # Getting the type of 'list' (line 107)
    list_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'list', False)
    # Calling list(args, kwargs) (line 107)
    list_call_result_219 = invoke(stypy.reporting.localization.Localization(__file__, 107, 19), list_217, *[], **kwargs_218)
    
    # Assigning a type to the variable 'stronghashes' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'stronghashes', list_call_result_219)
    
    # Assigning a Call to a Name (line 108):
    
    # Assigning a Call to a Name (line 108):
    
    # Call to read(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'blocksize' (line 108)
    blocksize_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 25), 'blocksize', False)
    # Processing the call keyword arguments (line 108)
    kwargs_223 = {}
    # Getting the type of 'instream' (line 108)
    instream_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 11), 'instream', False)
    # Obtaining the member 'read' of a type (line 108)
    read_221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 11), instream_220, 'read')
    # Calling read(args, kwargs) (line 108)
    read_call_result_224 = invoke(stypy.reporting.localization.Localization(__file__, 108, 11), read_221, *[blocksize_222], **kwargs_223)
    
    # Assigning a type to the variable 'read' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'read', read_call_result_224)
    
    # Getting the type of 'read' (line 110)
    read_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 10), 'read')
    # Testing if the while is going to be iterated (line 110)
    # Testing the type of an if condition (line 110)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 4), read_225)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 110, 4), read_225):
        # SSA begins for while statement (line 110)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Call to append(...): (line 111)
        # Processing the call arguments (line 111)
        
        # Obtaining the type of the subscript
        int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 45), 'int')
        
        # Call to weakchecksum(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'read' (line 111)
        read_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 39), 'read', False)
        # Processing the call keyword arguments (line 111)
        kwargs_231 = {}
        # Getting the type of 'weakchecksum' (line 111)
        weakchecksum_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 26), 'weakchecksum', False)
        # Calling weakchecksum(args, kwargs) (line 111)
        weakchecksum_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 111, 26), weakchecksum_229, *[read_230], **kwargs_231)
        
        # Obtaining the member '__getitem__' of a type (line 111)
        getitem___233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 26), weakchecksum_call_result_232, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 111)
        subscript_call_result_234 = invoke(stypy.reporting.localization.Localization(__file__, 111, 26), getitem___233, int_228)
        
        # Processing the call keyword arguments (line 111)
        kwargs_235 = {}
        # Getting the type of 'weakhashes' (line 111)
        weakhashes_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'weakhashes', False)
        # Obtaining the member 'append' of a type (line 111)
        append_227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), weakhashes_226, 'append')
        # Calling append(args, kwargs) (line 111)
        append_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), append_227, *[subscript_call_result_234], **kwargs_235)
        
        
        # Call to append(...): (line 112)
        # Processing the call arguments (line 112)
        
        # Call to hexdigest(...): (line 112)
        # Processing the call keyword arguments (line 112)
        kwargs_245 = {}
        
        # Call to md5(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'read' (line 112)
        read_241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 40), 'read', False)
        # Processing the call keyword arguments (line 112)
        kwargs_242 = {}
        # Getting the type of 'hashlib' (line 112)
        hashlib_239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 28), 'hashlib', False)
        # Obtaining the member 'md5' of a type (line 112)
        md5_240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 28), hashlib_239, 'md5')
        # Calling md5(args, kwargs) (line 112)
        md5_call_result_243 = invoke(stypy.reporting.localization.Localization(__file__, 112, 28), md5_240, *[read_241], **kwargs_242)
        
        # Obtaining the member 'hexdigest' of a type (line 112)
        hexdigest_244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 28), md5_call_result_243, 'hexdigest')
        # Calling hexdigest(args, kwargs) (line 112)
        hexdigest_call_result_246 = invoke(stypy.reporting.localization.Localization(__file__, 112, 28), hexdigest_244, *[], **kwargs_245)
        
        # Processing the call keyword arguments (line 112)
        kwargs_247 = {}
        # Getting the type of 'stronghashes' (line 112)
        stronghashes_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'stronghashes', False)
        # Obtaining the member 'append' of a type (line 112)
        append_238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), stronghashes_237, 'append')
        # Calling append(args, kwargs) (line 112)
        append_call_result_248 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), append_238, *[hexdigest_call_result_246], **kwargs_247)
        
        
        # Assigning a Call to a Name (line 113):
        
        # Assigning a Call to a Name (line 113):
        
        # Call to read(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'blocksize' (line 113)
        blocksize_251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 29), 'blocksize', False)
        # Processing the call keyword arguments (line 113)
        kwargs_252 = {}
        # Getting the type of 'instream' (line 113)
        instream_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 15), 'instream', False)
        # Obtaining the member 'read' of a type (line 113)
        read_250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 15), instream_249, 'read')
        # Calling read(args, kwargs) (line 113)
        read_call_result_253 = invoke(stypy.reporting.localization.Localization(__file__, 113, 15), read_250, *[blocksize_251], **kwargs_252)
        
        # Assigning a type to the variable 'read' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'read', read_call_result_253)
        # SSA join for while statement (line 110)
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Obtaining an instance of the builtin type 'tuple' (line 115)
    tuple_254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 115)
    # Adding element type (line 115)
    # Getting the type of 'weakhashes' (line 115)
    weakhashes_255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 11), 'weakhashes')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 11), tuple_254, weakhashes_255)
    # Adding element type (line 115)
    # Getting the type of 'stronghashes' (line 115)
    stronghashes_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 23), 'stronghashes')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 11), tuple_254, stronghashes_256)
    
    # Assigning a type to the variable 'stypy_return_type' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type', tuple_254)
    
    # ################# End of 'blockchecksums(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'blockchecksums' in the type store
    # Getting the type of 'stypy_return_type' (line 101)
    stypy_return_type_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_257)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'blockchecksums'
    return stypy_return_type_257

# Assigning a type to the variable 'blockchecksums' (line 101)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), 'blockchecksums', blockchecksums)

@norecursion
def patchstream(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'patchstream'
    module_type_store = module_type_store.open_function_context('patchstream', 118, 0, False)
    
    # Passed parameters checking function
    patchstream.stypy_localization = localization
    patchstream.stypy_type_of_self = None
    patchstream.stypy_type_store = module_type_store
    patchstream.stypy_function_name = 'patchstream'
    patchstream.stypy_param_names_list = ['instream', 'outstream', 'delta', 'blocksize']
    patchstream.stypy_varargs_param_name = None
    patchstream.stypy_kwargs_param_name = None
    patchstream.stypy_call_defaults = defaults
    patchstream.stypy_call_varargs = varargs
    patchstream.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'patchstream', ['instream', 'outstream', 'delta', 'blocksize'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'patchstream', localization, ['instream', 'outstream', 'delta', 'blocksize'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'patchstream(...)' code ##################

    str_258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, (-1)), 'str', '\n    Patches instream using the supplied delta and write the resultant\n    data to outstream.\n    ')
    
    # Getting the type of 'delta' (line 123)
    delta_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 19), 'delta')
    # Testing if the for loop is going to be iterated (line 123)
    # Testing the type of a for loop iterable (line 123)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 123, 4), delta_259)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 123, 4), delta_259):
        # Getting the type of the for loop variable (line 123)
        for_loop_var_260 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 123, 4), delta_259)
        # Assigning a type to the variable 'element' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'element', for_loop_var_260)
        # SSA begins for a for statement (line 123)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'element' (line 124)
        element_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 11), 'element')
        # Obtaining the member 'index' of a type (line 124)
        index_262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 11), element_261, 'index')
        int_263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 28), 'int')
        # Applying the binary operator '!=' (line 124)
        result_ne_264 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 11), '!=', index_262, int_263)
        
        # Testing if the type of an if condition is none (line 124)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 124, 8), result_ne_264):
            
            # Assigning a Call to a Name (line 128):
            
            # Assigning a Call to a Name (line 128):
            
            # Call to join(...): (line 128)
            # Processing the call arguments (line 128)
            # Getting the type of 'element' (line 128)
            element_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 27), 'element', False)
            # Obtaining the member 'data' of a type (line 128)
            data_282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 27), element_281, 'data')
            # Processing the call keyword arguments (line 128)
            kwargs_283 = {}
            str_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 19), 'str', '')
            # Obtaining the member 'join' of a type (line 128)
            join_280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 19), str_279, 'join')
            # Calling join(args, kwargs) (line 128)
            join_call_result_284 = invoke(stypy.reporting.localization.Localization(__file__, 128, 19), join_280, *[data_282], **kwargs_283)
            
            # Assigning a type to the variable 'data' (line 128)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'data', join_call_result_284)
        else:
            
            # Testing the type of an if condition (line 124)
            if_condition_265 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 8), result_ne_264)
            # Assigning a type to the variable 'if_condition_265' (line 124)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'if_condition_265', if_condition_265)
            # SSA begins for if statement (line 124)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to seek(...): (line 125)
            # Processing the call arguments (line 125)
            # Getting the type of 'element' (line 125)
            element_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 26), 'element', False)
            # Obtaining the member 'index' of a type (line 125)
            index_269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 26), element_268, 'index')
            # Getting the type of 'blocksize' (line 125)
            blocksize_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 42), 'blocksize', False)
            # Applying the binary operator '*' (line 125)
            result_mul_271 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 26), '*', index_269, blocksize_270)
            
            # Processing the call keyword arguments (line 125)
            kwargs_272 = {}
            # Getting the type of 'instream' (line 125)
            instream_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'instream', False)
            # Obtaining the member 'seek' of a type (line 125)
            seek_267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 12), instream_266, 'seek')
            # Calling seek(args, kwargs) (line 125)
            seek_call_result_273 = invoke(stypy.reporting.localization.Localization(__file__, 125, 12), seek_267, *[result_mul_271], **kwargs_272)
            
            
            # Assigning a Call to a Name (line 126):
            
            # Assigning a Call to a Name (line 126):
            
            # Call to read(...): (line 126)
            # Processing the call arguments (line 126)
            # Getting the type of 'blocksize' (line 126)
            blocksize_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 33), 'blocksize', False)
            # Processing the call keyword arguments (line 126)
            kwargs_277 = {}
            # Getting the type of 'instream' (line 126)
            instream_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 19), 'instream', False)
            # Obtaining the member 'read' of a type (line 126)
            read_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 19), instream_274, 'read')
            # Calling read(args, kwargs) (line 126)
            read_call_result_278 = invoke(stypy.reporting.localization.Localization(__file__, 126, 19), read_275, *[blocksize_276], **kwargs_277)
            
            # Assigning a type to the variable 'data' (line 126)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'data', read_call_result_278)
            # SSA branch for the else part of an if statement (line 124)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 128):
            
            # Assigning a Call to a Name (line 128):
            
            # Call to join(...): (line 128)
            # Processing the call arguments (line 128)
            # Getting the type of 'element' (line 128)
            element_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 27), 'element', False)
            # Obtaining the member 'data' of a type (line 128)
            data_282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 27), element_281, 'data')
            # Processing the call keyword arguments (line 128)
            kwargs_283 = {}
            str_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 19), 'str', '')
            # Obtaining the member 'join' of a type (line 128)
            join_280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 19), str_279, 'join')
            # Calling join(args, kwargs) (line 128)
            join_call_result_284 = invoke(stypy.reporting.localization.Localization(__file__, 128, 19), join_280, *[data_282], **kwargs_283)
            
            # Assigning a type to the variable 'data' (line 128)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'data', join_call_result_284)
            # SSA join for if statement (line 124)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'data' (line 129)
        data_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 24), 'data', False)
        # Processing the call keyword arguments (line 129)
        kwargs_288 = {}
        # Getting the type of 'outstream' (line 129)
        outstream_285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'outstream', False)
        # Obtaining the member 'write' of a type (line 129)
        write_286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), outstream_285, 'write')
        # Calling write(args, kwargs) (line 129)
        write_call_result_289 = invoke(stypy.reporting.localization.Localization(__file__, 129, 8), write_286, *[data_287], **kwargs_288)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'patchstream(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'patchstream' in the type store
    # Getting the type of 'stypy_return_type' (line 118)
    stypy_return_type_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_290)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'patchstream'
    return stypy_return_type_290

# Assigning a type to the variable 'patchstream' (line 118)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'patchstream', patchstream)

@norecursion
def rollingchecksum(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'rollingchecksum'
    module_type_store = module_type_store.open_function_context('rollingchecksum', 132, 0, False)
    
    # Passed parameters checking function
    rollingchecksum.stypy_localization = localization
    rollingchecksum.stypy_type_of_self = None
    rollingchecksum.stypy_type_store = module_type_store
    rollingchecksum.stypy_function_name = 'rollingchecksum'
    rollingchecksum.stypy_param_names_list = ['removed', 'new', 'a', 'b', 'blocksize']
    rollingchecksum.stypy_varargs_param_name = None
    rollingchecksum.stypy_kwargs_param_name = None
    rollingchecksum.stypy_call_defaults = defaults
    rollingchecksum.stypy_call_varargs = varargs
    rollingchecksum.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rollingchecksum', ['removed', 'new', 'a', 'b', 'blocksize'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rollingchecksum', localization, ['removed', 'new', 'a', 'b', 'blocksize'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rollingchecksum(...)' code ##################

    str_291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, (-1)), 'str', '\n    Generates a new weak checksum when supplied with the internal state\n    of the checksum calculation for the previous window, the removed\n    byte, and the added byte.\n    ')
    
    # Getting the type of 'a' (line 138)
    a_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'a')
    
    # Call to ord(...): (line 138)
    # Processing the call arguments (line 138)
    # Getting the type of 'removed' (line 138)
    removed_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 13), 'removed', False)
    # Processing the call keyword arguments (line 138)
    kwargs_295 = {}
    # Getting the type of 'ord' (line 138)
    ord_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 9), 'ord', False)
    # Calling ord(args, kwargs) (line 138)
    ord_call_result_296 = invoke(stypy.reporting.localization.Localization(__file__, 138, 9), ord_293, *[removed_294], **kwargs_295)
    
    
    # Call to ord(...): (line 138)
    # Processing the call arguments (line 138)
    # Getting the type of 'new' (line 138)
    new_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 28), 'new', False)
    # Processing the call keyword arguments (line 138)
    kwargs_299 = {}
    # Getting the type of 'ord' (line 138)
    ord_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 24), 'ord', False)
    # Calling ord(args, kwargs) (line 138)
    ord_call_result_300 = invoke(stypy.reporting.localization.Localization(__file__, 138, 24), ord_297, *[new_298], **kwargs_299)
    
    # Applying the binary operator '-' (line 138)
    result_sub_301 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 9), '-', ord_call_result_296, ord_call_result_300)
    
    # Applying the binary operator '-=' (line 138)
    result_isub_302 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 4), '-=', a_292, result_sub_301)
    # Assigning a type to the variable 'a' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'a', result_isub_302)
    
    
    # Getting the type of 'b' (line 139)
    b_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'b')
    
    # Call to ord(...): (line 139)
    # Processing the call arguments (line 139)
    # Getting the type of 'removed' (line 139)
    removed_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 13), 'removed', False)
    # Processing the call keyword arguments (line 139)
    kwargs_306 = {}
    # Getting the type of 'ord' (line 139)
    ord_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 9), 'ord', False)
    # Calling ord(args, kwargs) (line 139)
    ord_call_result_307 = invoke(stypy.reporting.localization.Localization(__file__, 139, 9), ord_304, *[removed_305], **kwargs_306)
    
    # Getting the type of 'blocksize' (line 139)
    blocksize_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 24), 'blocksize')
    # Applying the binary operator '*' (line 139)
    result_mul_309 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 9), '*', ord_call_result_307, blocksize_308)
    
    # Getting the type of 'a' (line 139)
    a_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 36), 'a')
    # Applying the binary operator '-' (line 139)
    result_sub_311 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 9), '-', result_mul_309, a_310)
    
    # Applying the binary operator '-=' (line 139)
    result_isub_312 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 4), '-=', b_303, result_sub_311)
    # Assigning a type to the variable 'b' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'b', result_isub_312)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 140)
    tuple_313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 140)
    # Adding element type (line 140)
    # Getting the type of 'b' (line 140)
    b_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'b')
    int_315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 17), 'int')
    # Applying the binary operator '<<' (line 140)
    result_lshift_316 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 12), '<<', b_314, int_315)
    
    # Getting the type of 'a' (line 140)
    a_317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 23), 'a')
    # Applying the binary operator '|' (line 140)
    result_or__318 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 11), '|', result_lshift_316, a_317)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 11), tuple_313, result_or__318)
    # Adding element type (line 140)
    # Getting the type of 'a' (line 140)
    a_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 26), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 11), tuple_313, a_319)
    # Adding element type (line 140)
    # Getting the type of 'b' (line 140)
    b_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 29), 'b')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 11), tuple_313, b_320)
    
    # Assigning a type to the variable 'stypy_return_type' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'stypy_return_type', tuple_313)
    
    # ################# End of 'rollingchecksum(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rollingchecksum' in the type store
    # Getting the type of 'stypy_return_type' (line 132)
    stypy_return_type_321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_321)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rollingchecksum'
    return stypy_return_type_321

# Assigning a type to the variable 'rollingchecksum' (line 132)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 0), 'rollingchecksum', rollingchecksum)

@norecursion
def weakchecksum(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'weakchecksum'
    module_type_store = module_type_store.open_function_context('weakchecksum', 143, 0, False)
    
    # Passed parameters checking function
    weakchecksum.stypy_localization = localization
    weakchecksum.stypy_type_of_self = None
    weakchecksum.stypy_type_store = module_type_store
    weakchecksum.stypy_function_name = 'weakchecksum'
    weakchecksum.stypy_param_names_list = ['data']
    weakchecksum.stypy_varargs_param_name = None
    weakchecksum.stypy_kwargs_param_name = None
    weakchecksum.stypy_call_defaults = defaults
    weakchecksum.stypy_call_varargs = varargs
    weakchecksum.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'weakchecksum', ['data'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'weakchecksum', localization, ['data'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'weakchecksum(...)' code ##################

    str_322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, (-1)), 'str', '\n    Generates a weak checksum from an iterable set of bytes.\n    ')
    
    # Multiple assignment of 2 elements.
    
    # Assigning a Num to a Name (line 147):
    int_323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 12), 'int')
    # Assigning a type to the variable 'b' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'b', int_323)
    
    # Assigning a Name to a Name (line 147):
    # Getting the type of 'b' (line 147)
    b_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'b')
    # Assigning a type to the variable 'a' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'a', b_324)
    
    # Assigning a Call to a Name (line 148):
    
    # Assigning a Call to a Name (line 148):
    
    # Call to len(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'data' (line 148)
    data_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'data', False)
    # Processing the call keyword arguments (line 148)
    kwargs_327 = {}
    # Getting the type of 'len' (line 148)
    len_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'len', False)
    # Calling len(args, kwargs) (line 148)
    len_call_result_328 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), len_325, *[data_326], **kwargs_327)
    
    # Assigning a type to the variable 'l' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'l', len_call_result_328)
    
    
    # Call to range(...): (line 150)
    # Processing the call arguments (line 150)
    # Getting the type of 'l' (line 150)
    l_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 19), 'l', False)
    # Processing the call keyword arguments (line 150)
    kwargs_331 = {}
    # Getting the type of 'range' (line 150)
    range_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 13), 'range', False)
    # Calling range(args, kwargs) (line 150)
    range_call_result_332 = invoke(stypy.reporting.localization.Localization(__file__, 150, 13), range_329, *[l_330], **kwargs_331)
    
    # Testing if the for loop is going to be iterated (line 150)
    # Testing the type of a for loop iterable (line 150)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 150, 4), range_call_result_332)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 150, 4), range_call_result_332):
        # Getting the type of the for loop variable (line 150)
        for_loop_var_333 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 150, 4), range_call_result_332)
        # Assigning a type to the variable 'i' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'i', for_loop_var_333)
        # SSA begins for a for statement (line 150)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 151):
        
        # Assigning a Call to a Name (line 151):
        
        # Call to ord(...): (line 151)
        # Processing the call arguments (line 151)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 151)
        i_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 21), 'i', False)
        # Getting the type of 'data' (line 151)
        data_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 151)
        getitem___337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 16), data_336, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 151)
        subscript_call_result_338 = invoke(stypy.reporting.localization.Localization(__file__, 151, 16), getitem___337, i_335)
        
        # Processing the call keyword arguments (line 151)
        kwargs_339 = {}
        # Getting the type of 'ord' (line 151)
        ord_334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'ord', False)
        # Calling ord(args, kwargs) (line 151)
        ord_call_result_340 = invoke(stypy.reporting.localization.Localization(__file__, 151, 12), ord_334, *[subscript_call_result_338], **kwargs_339)
        
        # Assigning a type to the variable 'n' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'n', ord_call_result_340)
        
        # Getting the type of 'a' (line 152)
        a_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'a')
        # Getting the type of 'n' (line 152)
        n_342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 13), 'n')
        # Applying the binary operator '+=' (line 152)
        result_iadd_343 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 8), '+=', a_341, n_342)
        # Assigning a type to the variable 'a' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'a', result_iadd_343)
        
        
        # Getting the type of 'b' (line 153)
        b_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'b')
        # Getting the type of 'l' (line 153)
        l_345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 14), 'l')
        # Getting the type of 'i' (line 153)
        i_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 18), 'i')
        # Applying the binary operator '-' (line 153)
        result_sub_347 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 14), '-', l_345, i_346)
        
        # Getting the type of 'n' (line 153)
        n_348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 23), 'n')
        # Applying the binary operator '*' (line 153)
        result_mul_349 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 13), '*', result_sub_347, n_348)
        
        # Applying the binary operator '+=' (line 153)
        result_iadd_350 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 8), '+=', b_344, result_mul_349)
        # Assigning a type to the variable 'b' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'b', result_iadd_350)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Obtaining an instance of the builtin type 'tuple' (line 155)
    tuple_351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 155)
    # Adding element type (line 155)
    # Getting the type of 'b' (line 155)
    b_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'b')
    int_353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 17), 'int')
    # Applying the binary operator '<<' (line 155)
    result_lshift_354 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 12), '<<', b_352, int_353)
    
    # Getting the type of 'a' (line 155)
    a_355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 23), 'a')
    # Applying the binary operator '|' (line 155)
    result_or__356 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 11), '|', result_lshift_354, a_355)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 11), tuple_351, result_or__356)
    # Adding element type (line 155)
    # Getting the type of 'a' (line 155)
    a_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 26), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 11), tuple_351, a_357)
    # Adding element type (line 155)
    # Getting the type of 'b' (line 155)
    b_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 29), 'b')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 11), tuple_351, b_358)
    
    # Assigning a type to the variable 'stypy_return_type' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'stypy_return_type', tuple_351)
    
    # ################# End of 'weakchecksum(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'weakchecksum' in the type store
    # Getting the type of 'stypy_return_type' (line 143)
    stypy_return_type_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_359)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'weakchecksum'
    return stypy_return_type_359

# Assigning a type to the variable 'weakchecksum' (line 143)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 0), 'weakchecksum', weakchecksum)

@norecursion
def Relative(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Relative'
    module_type_store = module_type_store.open_function_context('Relative', 158, 0, False)
    
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

    
    # Call to join(...): (line 159)
    # Processing the call arguments (line 159)
    
    # Call to dirname(...): (line 159)
    # Processing the call arguments (line 159)
    # Getting the type of '__file__' (line 159)
    file___366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 40), '__file__', False)
    # Processing the call keyword arguments (line 159)
    kwargs_367 = {}
    # Getting the type of 'os' (line 159)
    os_363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 159)
    path_364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 24), os_363, 'path')
    # Obtaining the member 'dirname' of a type (line 159)
    dirname_365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 24), path_364, 'dirname')
    # Calling dirname(args, kwargs) (line 159)
    dirname_call_result_368 = invoke(stypy.reporting.localization.Localization(__file__, 159, 24), dirname_365, *[file___366], **kwargs_367)
    
    # Getting the type of 'path' (line 159)
    path_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 51), 'path', False)
    # Processing the call keyword arguments (line 159)
    kwargs_370 = {}
    # Getting the type of 'os' (line 159)
    os_360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 159)
    path_361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 11), os_360, 'path')
    # Obtaining the member 'join' of a type (line 159)
    join_362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 11), path_361, 'join')
    # Calling join(args, kwargs) (line 159)
    join_call_result_371 = invoke(stypy.reporting.localization.Localization(__file__, 159, 11), join_362, *[dirname_call_result_368, path_369], **kwargs_370)
    
    # Assigning a type to the variable 'stypy_return_type' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'stypy_return_type', join_call_result_371)
    
    # ################# End of 'Relative(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Relative' in the type store
    # Getting the type of 'stypy_return_type' (line 158)
    stypy_return_type_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_372)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Relative'
    return stypy_return_type_372

# Assigning a type to the variable 'Relative' (line 158)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), 'Relative', Relative)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 162, 0, False)
    
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

    
    # Assigning a Num to a Name (line 163):
    
    # Assigning a Num to a Name (line 163):
    int_373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 16), 'int')
    # Assigning a type to the variable 'blocksize' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'blocksize', int_373)
    
    # Assigning a Call to a Name (line 166):
    
    # Assigning a Call to a Name (line 166):
    
    # Call to open(...): (line 166)
    # Processing the call arguments (line 166)
    
    # Call to Relative(...): (line 166)
    # Processing the call arguments (line 166)
    str_376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 30), 'str', 'testdata/unpatched.file')
    # Processing the call keyword arguments (line 166)
    kwargs_377 = {}
    # Getting the type of 'Relative' (line 166)
    Relative_375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 21), 'Relative', False)
    # Calling Relative(args, kwargs) (line 166)
    Relative_call_result_378 = invoke(stypy.reporting.localization.Localization(__file__, 166, 21), Relative_375, *[str_376], **kwargs_377)
    
    str_379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 58), 'str', 'rb')
    # Processing the call keyword arguments (line 166)
    kwargs_380 = {}
    # Getting the type of 'open' (line 166)
    open_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'open', False)
    # Calling open(args, kwargs) (line 166)
    open_call_result_381 = invoke(stypy.reporting.localization.Localization(__file__, 166, 16), open_374, *[Relative_call_result_378, str_379], **kwargs_380)
    
    # Assigning a type to the variable 'unpatched' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'unpatched', open_call_result_381)
    
    # Assigning a Call to a Name (line 167):
    
    # Assigning a Call to a Name (line 167):
    
    # Call to blockchecksums(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'unpatched' (line 167)
    unpatched_383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 28), 'unpatched', False)
    # Getting the type of 'blocksize' (line 167)
    blocksize_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 39), 'blocksize', False)
    # Processing the call keyword arguments (line 167)
    kwargs_385 = {}
    # Getting the type of 'blockchecksums' (line 167)
    blockchecksums_382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 13), 'blockchecksums', False)
    # Calling blockchecksums(args, kwargs) (line 167)
    blockchecksums_call_result_386 = invoke(stypy.reporting.localization.Localization(__file__, 167, 13), blockchecksums_382, *[unpatched_383, blocksize_384], **kwargs_385)
    
    # Assigning a type to the variable 'hashes' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'hashes', blockchecksums_call_result_386)
    
    # Assigning a Call to a Name (line 170):
    
    # Assigning a Call to a Name (line 170):
    
    # Call to open(...): (line 170)
    # Processing the call arguments (line 170)
    
    # Call to Relative(...): (line 170)
    # Processing the call arguments (line 170)
    str_389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 32), 'str', 'testdata/patched.file')
    # Processing the call keyword arguments (line 170)
    kwargs_390 = {}
    # Getting the type of 'Relative' (line 170)
    Relative_388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 23), 'Relative', False)
    # Calling Relative(args, kwargs) (line 170)
    Relative_call_result_391 = invoke(stypy.reporting.localization.Localization(__file__, 170, 23), Relative_388, *[str_389], **kwargs_390)
    
    str_392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 58), 'str', 'rb')
    # Processing the call keyword arguments (line 170)
    kwargs_393 = {}
    # Getting the type of 'open' (line 170)
    open_387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 18), 'open', False)
    # Calling open(args, kwargs) (line 170)
    open_call_result_394 = invoke(stypy.reporting.localization.Localization(__file__, 170, 18), open_387, *[Relative_call_result_391, str_392], **kwargs_393)
    
    # Assigning a type to the variable 'patchedfile' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'patchedfile', open_call_result_394)
    
    # Assigning a Call to a Name (line 171):
    
    # Assigning a Call to a Name (line 171):
    
    # Call to rsyncdelta(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'patchedfile' (line 171)
    patchedfile_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 23), 'patchedfile', False)
    # Getting the type of 'hashes' (line 171)
    hashes_397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 36), 'hashes', False)
    # Getting the type of 'blocksize' (line 171)
    blocksize_398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 44), 'blocksize', False)
    # Processing the call keyword arguments (line 171)
    kwargs_399 = {}
    # Getting the type of 'rsyncdelta' (line 171)
    rsyncdelta_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'rsyncdelta', False)
    # Calling rsyncdelta(args, kwargs) (line 171)
    rsyncdelta_call_result_400 = invoke(stypy.reporting.localization.Localization(__file__, 171, 12), rsyncdelta_395, *[patchedfile_396, hashes_397, blocksize_398], **kwargs_399)
    
    # Assigning a type to the variable 'delta' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'delta', rsyncdelta_call_result_400)
    
    # Call to seek(...): (line 174)
    # Processing the call arguments (line 174)
    int_403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 19), 'int')
    # Processing the call keyword arguments (line 174)
    kwargs_404 = {}
    # Getting the type of 'unpatched' (line 174)
    unpatched_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'unpatched', False)
    # Obtaining the member 'seek' of a type (line 174)
    seek_402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 4), unpatched_401, 'seek')
    # Calling seek(args, kwargs) (line 174)
    seek_call_result_405 = invoke(stypy.reporting.localization.Localization(__file__, 174, 4), seek_402, *[int_403], **kwargs_404)
    
    
    # Assigning a Call to a Name (line 175):
    
    # Assigning a Call to a Name (line 175):
    
    # Call to open(...): (line 175)
    # Processing the call arguments (line 175)
    
    # Call to Relative(...): (line 175)
    # Processing the call arguments (line 175)
    str_408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 28), 'str', 'testdata/locally-patched.file')
    # Processing the call keyword arguments (line 175)
    kwargs_409 = {}
    # Getting the type of 'Relative' (line 175)
    Relative_407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 19), 'Relative', False)
    # Calling Relative(args, kwargs) (line 175)
    Relative_call_result_410 = invoke(stypy.reporting.localization.Localization(__file__, 175, 19), Relative_407, *[str_408], **kwargs_409)
    
    str_411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 62), 'str', 'wb')
    # Processing the call keyword arguments (line 175)
    kwargs_412 = {}
    # Getting the type of 'open' (line 175)
    open_406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 14), 'open', False)
    # Calling open(args, kwargs) (line 175)
    open_call_result_413 = invoke(stypy.reporting.localization.Localization(__file__, 175, 14), open_406, *[Relative_call_result_410, str_411], **kwargs_412)
    
    # Assigning a type to the variable 'save_to' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'save_to', open_call_result_413)
    
    # Call to patchstream(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of 'unpatched' (line 176)
    unpatched_415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 16), 'unpatched', False)
    # Getting the type of 'save_to' (line 176)
    save_to_416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 27), 'save_to', False)
    # Getting the type of 'delta' (line 176)
    delta_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 36), 'delta', False)
    # Getting the type of 'blocksize' (line 176)
    blocksize_418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 43), 'blocksize', False)
    # Processing the call keyword arguments (line 176)
    kwargs_419 = {}
    # Getting the type of 'patchstream' (line 176)
    patchstream_414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'patchstream', False)
    # Calling patchstream(args, kwargs) (line 176)
    patchstream_call_result_420 = invoke(stypy.reporting.localization.Localization(__file__, 176, 4), patchstream_414, *[unpatched_415, save_to_416, delta_417, blocksize_418], **kwargs_419)
    
    # Getting the type of 'True' (line 177)
    True_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'stypy_return_type', True_421)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 162)
    stypy_return_type_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_422)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_422

# Assigning a type to the variable 'run' (line 162)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 0), 'run', run)

# Call to run(...): (line 180)
# Processing the call keyword arguments (line 180)
kwargs_424 = {}
# Getting the type of 'run' (line 180)
run_423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 0), 'run', False)
# Calling run(args, kwargs) (line 180)
run_call_result_425 = invoke(stypy.reporting.localization.Localization(__file__, 180, 0), run_423, *[], **kwargs_424)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

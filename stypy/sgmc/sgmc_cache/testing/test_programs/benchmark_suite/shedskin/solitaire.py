
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/bin/env python
2: 
3: '''
4: Python implementation of Bruce Schneier's Solitaire Encryption
5: Algorithm (http://www.counterpane.com/solitaire.html).
6: 
7: John Dell'Aquila <jbd@alum.mit.edu>
8: '''
9: 
10: import string, sys
11: 
12: 
13: def toNumber(c):
14:     '''
15:     Convert letter to number: Aa->1, Bb->2, ..., Zz->26.
16:     Non-letters are treated as X's.
17:     '''
18:     if c in string.letters:
19:         return ord(string.upper(c)) - 64
20:     return 24  # 'X'
21: 
22: 
23: def toChar(n):
24:     '''
25:     Convert number to letter: 1->A,  2->B, ..., 26->Z,
26:     27->A, 28->B, ... ad infitum
27:     '''
28:     return chr((n - 1) % 26 + 65)
29: 
30: 
31: class Solitaire:
32:     ''' Solitaire Encryption Algorithm
33:     http://www.counterpane.com/solitaire.html
34:     '''
35: 
36:     def _setKey(self, passphrase):
37:         '''
38:         Order deck according to passphrase.
39:         '''
40:         self.deck = range(1, 55)
41:         # card numbering:
42:         #  1, 2,...,13 are A,2,...,K of clubs
43:         # 14,15,...,26 are A,2,...,K of diamonds
44:         # 27,28,...,39 are A,2,...,K of hearts
45:         # 40,41,...,52 are A,2,...,K of spades
46:         # 53 & 54 are the A & B jokers
47:         for c in passphrase:
48:             self._round()
49:             self._countCut(toNumber(c))
50: 
51:     def _down1(self, card):
52:         '''
53:         Move designated card down 1 position, treating
54:         deck as circular.
55:         '''
56:         d = self.deck
57:         n = d.index(card)
58:         if n < 53:  # not last card - swap with successor
59:             d[n], d[n + 1] = d[n + 1], d[n]
60:         else:  # last card - move below first card
61:             d[1:] = d[-1:] + d[1:-1]
62: 
63:     def _tripleCut(self):
64:         '''
65:         Swap cards above first joker with cards below
66:         second joker.
67:         '''
68:         d = self.deck
69:         a, b = d.index(53), d.index(54)
70:         if a > b:
71:             a, b = b, a
72:         d[:] = d[b + 1:] + d[a:b + 1] + d[:a]
73: 
74:     def _countCut(self, n):
75:         '''
76:         Cut after the n-th card, leaving the bottom
77:         card in place.
78:         '''
79:         d = self.deck
80:         n = min(n, 53)  # either joker is 53
81:         d[:-1] = d[n:-1] + d[:n]
82: 
83:     def _round(self):
84:         '''
85:         Perform one round of keystream generation.
86:         '''
87:         self._down1(53)  # move A joker down 1
88:         self._down1(54)  # move B joker down 2
89:         self._down1(54)
90:         self._tripleCut()
91:         self._countCut(self.deck[-1])
92: 
93:     def _output(self):
94:         '''
95:         Return next output card.
96:         '''
97:         d = self.deck
98:         while 1:
99:             self._round()
100:             topCard = min(d[0], 53)  # either joker is 53
101:             if d[topCard] < 53:  # don't return a joker
102:                 return d[topCard]
103: 
104:     def encrypt(self, txt, key):
105:         '''
106:         Return 'txt' encrypted using 'key'.
107:         '''
108:         self._setKey(key)
109:         # pad with X's to multiple of 5
110:         txt = txt + 'X' * ((5 - len(txt)) % 5)
111:         cipher = [None] * len(txt)
112:         for n in range(len(txt)):
113:             cipher[n] = toChar(toNumber(txt[n]) + self._output())
114:         # add spaces to make 5 letter blocks
115:         for n in range(len(cipher) - 5, 4, -5):
116:             cipher[n:n] = [' ']
117:         return string.join(cipher, '')
118: 
119:     def decrypt(self, cipher, key):
120:         '''
121:         Return 'cipher' decrypted using 'key'.
122:         '''
123:         self._setKey(key)
124:         # remove white space between code blocks
125:         cipher = string.join(string.split(cipher), '')
126:         txt = [None] * len(cipher)
127:         for n in range(len(cipher)):
128:             txt[n] = toChar(toNumber(cipher[n]) - self._output())
129:         return string.join(txt, '')
130: 
131: 
132: testCases = (  # test vectors from Schneier paper
133:     ('AAAAAAAAAAAAAAA', '', 'EXKYI ZSGEH UNTIQ'),
134:     ('AAAAAAAAAAAAAAA', 'f', 'XYIUQ BMHKK JBEGY'),
135:     ('AAAAAAAAAAAAAAA', 'fo', 'TUJYM BERLG XNDIW'),
136:     ('AAAAAAAAAAAAAAA', 'foo', 'ITHZU JIWGR FARMW'),
137:     ('AAAAAAAAAAAAAAA', 'a', 'XODAL GSCUL IQNSC'),
138:     ('AAAAAAAAAAAAAAA', 'aa', 'OHGWM XXCAI MCIQP'),
139:     ('AAAAAAAAAAAAAAA', 'aaa', 'DCSQY HBQZN GDRUT'),
140:     ('AAAAAAAAAAAAAAA', 'b', 'XQEEM OITLZ VDSQS'),
141:     ('AAAAAAAAAAAAAAA', 'bc', 'QNGRK QIHCL GWSCE'),
142:     ('AAAAAAAAAAAAAAA', 'bcd', 'FMUBY BMAXH NQXCJ'),
143:     ('AAAAAAAAAAAAAAAAAAAAAAAAA', 'cryptonomicon',
144:      'SUGSR SXSWQ RMXOH IPBFP XARYQ'),
145:     ('SOLITAIRE', 'cryptonomicon', 'KIRAK SFJAN')
146: )
147: 
148: 
149: def usage():
150:     print '''Usage:
151:     sol.py {-e | -d} _key_ < _file_
152:     sol.py -test
153:     
154:     N.B. WinNT requires "python sol.py ..."
155:     for input redirection to work (NT bug).
156:     '''
157:     sys.exit(2)
158: 
159: 
160: def main():
161:     ##    args = sys.argv
162:     ##    if len(args) < 2:
163:     ##        usage()
164:     ##    elif args[1] == '-test':
165:     s = Solitaire()
166:     for txt, key, cipher in testCases:
167:         coded = s.encrypt(txt, key)
168:         assert cipher == coded
169:         decoded = s.decrypt(coded, key)
170:         assert decoded[:len(txt)] == string.upper(txt)
171: 
172: 
173: ##        print 'All tests passed.'
174: ##    elif len(args) < 3:
175: ##        usage()
176: ##    elif args[1] == '-e':
177: ##        print Solitaire().encrypt(sys.stdin.read(), args[2])
178: ##    elif args[1] == '-d':
179: ##        print Solitaire().decrypt(sys.stdin.read(), args[2])
180: ##    else:
181: ##        usage()
182: 
183: def run():
184:     for i in range(100):
185:         main()
186:     return True
187: 
188: 
189: run()
190: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, (-1)), 'str', "\nPython implementation of Bruce Schneier's Solitaire Encryption\nAlgorithm (http://www.counterpane.com/solitaire.html).\n\nJohn Dell'Aquila <jbd@alum.mit.edu>\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# Multiple import statement. import string (1/2) (line 10)
import string

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'string', string, module_type_store)
# Multiple import statement. import sys (2/2) (line 10)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'sys', sys, module_type_store)


@norecursion
def toNumber(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'toNumber'
    module_type_store = module_type_store.open_function_context('toNumber', 13, 0, False)
    
    # Passed parameters checking function
    toNumber.stypy_localization = localization
    toNumber.stypy_type_of_self = None
    toNumber.stypy_type_store = module_type_store
    toNumber.stypy_function_name = 'toNumber'
    toNumber.stypy_param_names_list = ['c']
    toNumber.stypy_varargs_param_name = None
    toNumber.stypy_kwargs_param_name = None
    toNumber.stypy_call_defaults = defaults
    toNumber.stypy_call_varargs = varargs
    toNumber.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'toNumber', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'toNumber', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'toNumber(...)' code ##################

    str_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', "\n    Convert letter to number: Aa->1, Bb->2, ..., Zz->26.\n    Non-letters are treated as X's.\n    ")
    
    # Getting the type of 'c' (line 18)
    c_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 7), 'c')
    # Getting the type of 'string' (line 18)
    string_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 12), 'string')
    # Obtaining the member 'letters' of a type (line 18)
    letters_11 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 12), string_10, 'letters')
    # Applying the binary operator 'in' (line 18)
    result_contains_12 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 7), 'in', c_9, letters_11)
    
    # Testing if the type of an if condition is none (line 18)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 18, 4), result_contains_12):
        pass
    else:
        
        # Testing the type of an if condition (line 18)
        if_condition_13 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 18, 4), result_contains_12)
        # Assigning a type to the variable 'if_condition_13' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'if_condition_13', if_condition_13)
        # SSA begins for if statement (line 18)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ord(...): (line 19)
        # Processing the call arguments (line 19)
        
        # Call to upper(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'c' (line 19)
        c_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 32), 'c', False)
        # Processing the call keyword arguments (line 19)
        kwargs_18 = {}
        # Getting the type of 'string' (line 19)
        string_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 19), 'string', False)
        # Obtaining the member 'upper' of a type (line 19)
        upper_16 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 19), string_15, 'upper')
        # Calling upper(args, kwargs) (line 19)
        upper_call_result_19 = invoke(stypy.reporting.localization.Localization(__file__, 19, 19), upper_16, *[c_17], **kwargs_18)
        
        # Processing the call keyword arguments (line 19)
        kwargs_20 = {}
        # Getting the type of 'ord' (line 19)
        ord_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), 'ord', False)
        # Calling ord(args, kwargs) (line 19)
        ord_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 19, 15), ord_14, *[upper_call_result_19], **kwargs_20)
        
        int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 38), 'int')
        # Applying the binary operator '-' (line 19)
        result_sub_23 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 15), '-', ord_call_result_21, int_22)
        
        # Assigning a type to the variable 'stypy_return_type' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'stypy_return_type', result_sub_23)
        # SSA join for if statement (line 18)
        module_type_store = module_type_store.join_ssa_context()
        

    int_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 11), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type', int_24)
    
    # ################# End of 'toNumber(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'toNumber' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'toNumber'
    return stypy_return_type_25

# Assigning a type to the variable 'toNumber' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'toNumber', toNumber)

@norecursion
def toChar(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'toChar'
    module_type_store = module_type_store.open_function_context('toChar', 23, 0, False)
    
    # Passed parameters checking function
    toChar.stypy_localization = localization
    toChar.stypy_type_of_self = None
    toChar.stypy_type_store = module_type_store
    toChar.stypy_function_name = 'toChar'
    toChar.stypy_param_names_list = ['n']
    toChar.stypy_varargs_param_name = None
    toChar.stypy_kwargs_param_name = None
    toChar.stypy_call_defaults = defaults
    toChar.stypy_call_varargs = varargs
    toChar.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'toChar', ['n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'toChar', localization, ['n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'toChar(...)' code ##################

    str_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, (-1)), 'str', '\n    Convert number to letter: 1->A,  2->B, ..., 26->Z,\n    27->A, 28->B, ... ad infitum\n    ')
    
    # Call to chr(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'n' (line 28)
    n_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), 'n', False)
    int_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 20), 'int')
    # Applying the binary operator '-' (line 28)
    result_sub_30 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 16), '-', n_28, int_29)
    
    int_31 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 25), 'int')
    # Applying the binary operator '%' (line 28)
    result_mod_32 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 15), '%', result_sub_30, int_31)
    
    int_33 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 30), 'int')
    # Applying the binary operator '+' (line 28)
    result_add_34 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 15), '+', result_mod_32, int_33)
    
    # Processing the call keyword arguments (line 28)
    kwargs_35 = {}
    # Getting the type of 'chr' (line 28)
    chr_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 11), 'chr', False)
    # Calling chr(args, kwargs) (line 28)
    chr_call_result_36 = invoke(stypy.reporting.localization.Localization(__file__, 28, 11), chr_27, *[result_add_34], **kwargs_35)
    
    # Assigning a type to the variable 'stypy_return_type' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type', chr_call_result_36)
    
    # ################# End of 'toChar(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'toChar' in the type store
    # Getting the type of 'stypy_return_type' (line 23)
    stypy_return_type_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_37)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'toChar'
    return stypy_return_type_37

# Assigning a type to the variable 'toChar' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'toChar', toChar)
# Declaration of the 'Solitaire' class

class Solitaire:
    str_38 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, (-1)), 'str', ' Solitaire Encryption Algorithm\n    http://www.counterpane.com/solitaire.html\n    ')

    @norecursion
    def _setKey(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_setKey'
        module_type_store = module_type_store.open_function_context('_setKey', 36, 4, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Solitaire._setKey.__dict__.__setitem__('stypy_localization', localization)
        Solitaire._setKey.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Solitaire._setKey.__dict__.__setitem__('stypy_type_store', module_type_store)
        Solitaire._setKey.__dict__.__setitem__('stypy_function_name', 'Solitaire._setKey')
        Solitaire._setKey.__dict__.__setitem__('stypy_param_names_list', ['passphrase'])
        Solitaire._setKey.__dict__.__setitem__('stypy_varargs_param_name', None)
        Solitaire._setKey.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Solitaire._setKey.__dict__.__setitem__('stypy_call_defaults', defaults)
        Solitaire._setKey.__dict__.__setitem__('stypy_call_varargs', varargs)
        Solitaire._setKey.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Solitaire._setKey.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Solitaire._setKey', ['passphrase'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_setKey', localization, ['passphrase'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_setKey(...)' code ##################

        str_39 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, (-1)), 'str', '\n        Order deck according to passphrase.\n        ')
        
        # Assigning a Call to a Attribute (line 40):
        
        # Assigning a Call to a Attribute (line 40):
        
        # Call to range(...): (line 40)
        # Processing the call arguments (line 40)
        int_41 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 26), 'int')
        int_42 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 29), 'int')
        # Processing the call keyword arguments (line 40)
        kwargs_43 = {}
        # Getting the type of 'range' (line 40)
        range_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 20), 'range', False)
        # Calling range(args, kwargs) (line 40)
        range_call_result_44 = invoke(stypy.reporting.localization.Localization(__file__, 40, 20), range_40, *[int_41, int_42], **kwargs_43)
        
        # Getting the type of 'self' (line 40)
        self_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'self')
        # Setting the type of the member 'deck' of a type (line 40)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), self_45, 'deck', range_call_result_44)
        
        # Getting the type of 'passphrase' (line 47)
        passphrase_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 17), 'passphrase')
        # Assigning a type to the variable 'passphrase_46' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'passphrase_46', passphrase_46)
        # Testing if the for loop is going to be iterated (line 47)
        # Testing the type of a for loop iterable (line 47)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 47, 8), passphrase_46)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 47, 8), passphrase_46):
            # Getting the type of the for loop variable (line 47)
            for_loop_var_47 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 47, 8), passphrase_46)
            # Assigning a type to the variable 'c' (line 47)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'c', for_loop_var_47)
            # SSA begins for a for statement (line 47)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to _round(...): (line 48)
            # Processing the call keyword arguments (line 48)
            kwargs_50 = {}
            # Getting the type of 'self' (line 48)
            self_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'self', False)
            # Obtaining the member '_round' of a type (line 48)
            _round_49 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 12), self_48, '_round')
            # Calling _round(args, kwargs) (line 48)
            _round_call_result_51 = invoke(stypy.reporting.localization.Localization(__file__, 48, 12), _round_49, *[], **kwargs_50)
            
            
            # Call to _countCut(...): (line 49)
            # Processing the call arguments (line 49)
            
            # Call to toNumber(...): (line 49)
            # Processing the call arguments (line 49)
            # Getting the type of 'c' (line 49)
            c_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 36), 'c', False)
            # Processing the call keyword arguments (line 49)
            kwargs_56 = {}
            # Getting the type of 'toNumber' (line 49)
            toNumber_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 27), 'toNumber', False)
            # Calling toNumber(args, kwargs) (line 49)
            toNumber_call_result_57 = invoke(stypy.reporting.localization.Localization(__file__, 49, 27), toNumber_54, *[c_55], **kwargs_56)
            
            # Processing the call keyword arguments (line 49)
            kwargs_58 = {}
            # Getting the type of 'self' (line 49)
            self_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'self', False)
            # Obtaining the member '_countCut' of a type (line 49)
            _countCut_53 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), self_52, '_countCut')
            # Calling _countCut(args, kwargs) (line 49)
            _countCut_call_result_59 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), _countCut_53, *[toNumber_call_result_57], **kwargs_58)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of '_setKey(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_setKey' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_setKey'
        return stypy_return_type_60


    @norecursion
    def _down1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_down1'
        module_type_store = module_type_store.open_function_context('_down1', 51, 4, False)
        # Assigning a type to the variable 'self' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Solitaire._down1.__dict__.__setitem__('stypy_localization', localization)
        Solitaire._down1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Solitaire._down1.__dict__.__setitem__('stypy_type_store', module_type_store)
        Solitaire._down1.__dict__.__setitem__('stypy_function_name', 'Solitaire._down1')
        Solitaire._down1.__dict__.__setitem__('stypy_param_names_list', ['card'])
        Solitaire._down1.__dict__.__setitem__('stypy_varargs_param_name', None)
        Solitaire._down1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Solitaire._down1.__dict__.__setitem__('stypy_call_defaults', defaults)
        Solitaire._down1.__dict__.__setitem__('stypy_call_varargs', varargs)
        Solitaire._down1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Solitaire._down1.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Solitaire._down1', ['card'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_down1', localization, ['card'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_down1(...)' code ##################

        str_61 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, (-1)), 'str', '\n        Move designated card down 1 position, treating\n        deck as circular.\n        ')
        
        # Assigning a Attribute to a Name (line 56):
        
        # Assigning a Attribute to a Name (line 56):
        # Getting the type of 'self' (line 56)
        self_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'self')
        # Obtaining the member 'deck' of a type (line 56)
        deck_63 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 12), self_62, 'deck')
        # Assigning a type to the variable 'd' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'd', deck_63)
        
        # Assigning a Call to a Name (line 57):
        
        # Assigning a Call to a Name (line 57):
        
        # Call to index(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'card' (line 57)
        card_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 20), 'card', False)
        # Processing the call keyword arguments (line 57)
        kwargs_67 = {}
        # Getting the type of 'd' (line 57)
        d_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'd', False)
        # Obtaining the member 'index' of a type (line 57)
        index_65 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), d_64, 'index')
        # Calling index(args, kwargs) (line 57)
        index_call_result_68 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), index_65, *[card_66], **kwargs_67)
        
        # Assigning a type to the variable 'n' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'n', index_call_result_68)
        
        # Getting the type of 'n' (line 58)
        n_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 11), 'n')
        int_70 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 15), 'int')
        # Applying the binary operator '<' (line 58)
        result_lt_71 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 11), '<', n_69, int_70)
        
        # Testing if the type of an if condition is none (line 58)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 58, 8), result_lt_71):
            
            # Assigning a BinOp to a Subscript (line 61):
            
            # Assigning a BinOp to a Subscript (line 61):
            
            # Obtaining the type of the subscript
            int_91 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 22), 'int')
            slice_92 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 61, 20), int_91, None, None)
            # Getting the type of 'd' (line 61)
            d_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 20), 'd')
            # Obtaining the member '__getitem__' of a type (line 61)
            getitem___94 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 20), d_93, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 61)
            subscript_call_result_95 = invoke(stypy.reporting.localization.Localization(__file__, 61, 20), getitem___94, slice_92)
            
            
            # Obtaining the type of the subscript
            int_96 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 31), 'int')
            int_97 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 33), 'int')
            slice_98 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 61, 29), int_96, int_97, None)
            # Getting the type of 'd' (line 61)
            d_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 29), 'd')
            # Obtaining the member '__getitem__' of a type (line 61)
            getitem___100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 29), d_99, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 61)
            subscript_call_result_101 = invoke(stypy.reporting.localization.Localization(__file__, 61, 29), getitem___100, slice_98)
            
            # Applying the binary operator '+' (line 61)
            result_add_102 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 20), '+', subscript_call_result_95, subscript_call_result_101)
            
            # Getting the type of 'd' (line 61)
            d_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'd')
            int_104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 14), 'int')
            slice_105 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 61, 12), int_104, None, None)
            # Storing an element on a container (line 61)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 12), d_103, (slice_105, result_add_102))
        else:
            
            # Testing the type of an if condition (line 58)
            if_condition_72 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 8), result_lt_71)
            # Assigning a type to the variable 'if_condition_72' (line 58)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'if_condition_72', if_condition_72)
            # SSA begins for if statement (line 58)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Tuple to a Tuple (line 59):
            
            # Assigning a Subscript to a Name (line 59):
            
            # Obtaining the type of the subscript
            # Getting the type of 'n' (line 59)
            n_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 31), 'n')
            int_74 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 35), 'int')
            # Applying the binary operator '+' (line 59)
            result_add_75 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 31), '+', n_73, int_74)
            
            # Getting the type of 'd' (line 59)
            d_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 29), 'd')
            # Obtaining the member '__getitem__' of a type (line 59)
            getitem___77 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 29), d_76, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 59)
            subscript_call_result_78 = invoke(stypy.reporting.localization.Localization(__file__, 59, 29), getitem___77, result_add_75)
            
            # Assigning a type to the variable 'tuple_assignment_1' (line 59)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'tuple_assignment_1', subscript_call_result_78)
            
            # Assigning a Subscript to a Name (line 59):
            
            # Obtaining the type of the subscript
            # Getting the type of 'n' (line 59)
            n_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 41), 'n')
            # Getting the type of 'd' (line 59)
            d_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 39), 'd')
            # Obtaining the member '__getitem__' of a type (line 59)
            getitem___81 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 39), d_80, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 59)
            subscript_call_result_82 = invoke(stypy.reporting.localization.Localization(__file__, 59, 39), getitem___81, n_79)
            
            # Assigning a type to the variable 'tuple_assignment_2' (line 59)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'tuple_assignment_2', subscript_call_result_82)
            
            # Assigning a Name to a Subscript (line 59):
            # Getting the type of 'tuple_assignment_1' (line 59)
            tuple_assignment_1_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'tuple_assignment_1')
            # Getting the type of 'd' (line 59)
            d_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'd')
            # Getting the type of 'n' (line 59)
            n_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 14), 'n')
            # Storing an element on a container (line 59)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 12), d_84, (n_85, tuple_assignment_1_83))
            
            # Assigning a Name to a Subscript (line 59):
            # Getting the type of 'tuple_assignment_2' (line 59)
            tuple_assignment_2_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'tuple_assignment_2')
            # Getting the type of 'd' (line 59)
            d_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 18), 'd')
            # Getting the type of 'n' (line 59)
            n_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 'n')
            int_89 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 24), 'int')
            # Applying the binary operator '+' (line 59)
            result_add_90 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 20), '+', n_88, int_89)
            
            # Storing an element on a container (line 59)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 18), d_87, (result_add_90, tuple_assignment_2_86))
            # SSA branch for the else part of an if statement (line 58)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Subscript (line 61):
            
            # Assigning a BinOp to a Subscript (line 61):
            
            # Obtaining the type of the subscript
            int_91 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 22), 'int')
            slice_92 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 61, 20), int_91, None, None)
            # Getting the type of 'd' (line 61)
            d_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 20), 'd')
            # Obtaining the member '__getitem__' of a type (line 61)
            getitem___94 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 20), d_93, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 61)
            subscript_call_result_95 = invoke(stypy.reporting.localization.Localization(__file__, 61, 20), getitem___94, slice_92)
            
            
            # Obtaining the type of the subscript
            int_96 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 31), 'int')
            int_97 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 33), 'int')
            slice_98 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 61, 29), int_96, int_97, None)
            # Getting the type of 'd' (line 61)
            d_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 29), 'd')
            # Obtaining the member '__getitem__' of a type (line 61)
            getitem___100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 29), d_99, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 61)
            subscript_call_result_101 = invoke(stypy.reporting.localization.Localization(__file__, 61, 29), getitem___100, slice_98)
            
            # Applying the binary operator '+' (line 61)
            result_add_102 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 20), '+', subscript_call_result_95, subscript_call_result_101)
            
            # Getting the type of 'd' (line 61)
            d_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'd')
            int_104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 14), 'int')
            slice_105 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 61, 12), int_104, None, None)
            # Storing an element on a container (line 61)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 12), d_103, (slice_105, result_add_102))
            # SSA join for if statement (line 58)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of '_down1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_down1' in the type store
        # Getting the type of 'stypy_return_type' (line 51)
        stypy_return_type_106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_106)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_down1'
        return stypy_return_type_106


    @norecursion
    def _tripleCut(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_tripleCut'
        module_type_store = module_type_store.open_function_context('_tripleCut', 63, 4, False)
        # Assigning a type to the variable 'self' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Solitaire._tripleCut.__dict__.__setitem__('stypy_localization', localization)
        Solitaire._tripleCut.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Solitaire._tripleCut.__dict__.__setitem__('stypy_type_store', module_type_store)
        Solitaire._tripleCut.__dict__.__setitem__('stypy_function_name', 'Solitaire._tripleCut')
        Solitaire._tripleCut.__dict__.__setitem__('stypy_param_names_list', [])
        Solitaire._tripleCut.__dict__.__setitem__('stypy_varargs_param_name', None)
        Solitaire._tripleCut.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Solitaire._tripleCut.__dict__.__setitem__('stypy_call_defaults', defaults)
        Solitaire._tripleCut.__dict__.__setitem__('stypy_call_varargs', varargs)
        Solitaire._tripleCut.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Solitaire._tripleCut.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Solitaire._tripleCut', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_tripleCut', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_tripleCut(...)' code ##################

        str_107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, (-1)), 'str', '\n        Swap cards above first joker with cards below\n        second joker.\n        ')
        
        # Assigning a Attribute to a Name (line 68):
        
        # Assigning a Attribute to a Name (line 68):
        # Getting the type of 'self' (line 68)
        self_108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'self')
        # Obtaining the member 'deck' of a type (line 68)
        deck_109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), self_108, 'deck')
        # Assigning a type to the variable 'd' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'd', deck_109)
        
        # Assigning a Tuple to a Tuple (line 69):
        
        # Assigning a Call to a Name (line 69):
        
        # Call to index(...): (line 69)
        # Processing the call arguments (line 69)
        int_112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 23), 'int')
        # Processing the call keyword arguments (line 69)
        kwargs_113 = {}
        # Getting the type of 'd' (line 69)
        d_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 15), 'd', False)
        # Obtaining the member 'index' of a type (line 69)
        index_111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 15), d_110, 'index')
        # Calling index(args, kwargs) (line 69)
        index_call_result_114 = invoke(stypy.reporting.localization.Localization(__file__, 69, 15), index_111, *[int_112], **kwargs_113)
        
        # Assigning a type to the variable 'tuple_assignment_3' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'tuple_assignment_3', index_call_result_114)
        
        # Assigning a Call to a Name (line 69):
        
        # Call to index(...): (line 69)
        # Processing the call arguments (line 69)
        int_117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 36), 'int')
        # Processing the call keyword arguments (line 69)
        kwargs_118 = {}
        # Getting the type of 'd' (line 69)
        d_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 28), 'd', False)
        # Obtaining the member 'index' of a type (line 69)
        index_116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 28), d_115, 'index')
        # Calling index(args, kwargs) (line 69)
        index_call_result_119 = invoke(stypy.reporting.localization.Localization(__file__, 69, 28), index_116, *[int_117], **kwargs_118)
        
        # Assigning a type to the variable 'tuple_assignment_4' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'tuple_assignment_4', index_call_result_119)
        
        # Assigning a Name to a Name (line 69):
        # Getting the type of 'tuple_assignment_3' (line 69)
        tuple_assignment_3_120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'tuple_assignment_3')
        # Assigning a type to the variable 'a' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'a', tuple_assignment_3_120)
        
        # Assigning a Name to a Name (line 69):
        # Getting the type of 'tuple_assignment_4' (line 69)
        tuple_assignment_4_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'tuple_assignment_4')
        # Assigning a type to the variable 'b' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 11), 'b', tuple_assignment_4_121)
        
        # Getting the type of 'a' (line 70)
        a_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 11), 'a')
        # Getting the type of 'b' (line 70)
        b_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 15), 'b')
        # Applying the binary operator '>' (line 70)
        result_gt_124 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 11), '>', a_122, b_123)
        
        # Testing if the type of an if condition is none (line 70)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 70, 8), result_gt_124):
            pass
        else:
            
            # Testing the type of an if condition (line 70)
            if_condition_125 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 8), result_gt_124)
            # Assigning a type to the variable 'if_condition_125' (line 70)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'if_condition_125', if_condition_125)
            # SSA begins for if statement (line 70)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Tuple to a Tuple (line 71):
            
            # Assigning a Name to a Name (line 71):
            # Getting the type of 'b' (line 71)
            b_126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'b')
            # Assigning a type to the variable 'tuple_assignment_5' (line 71)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'tuple_assignment_5', b_126)
            
            # Assigning a Name to a Name (line 71):
            # Getting the type of 'a' (line 71)
            a_127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 22), 'a')
            # Assigning a type to the variable 'tuple_assignment_6' (line 71)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'tuple_assignment_6', a_127)
            
            # Assigning a Name to a Name (line 71):
            # Getting the type of 'tuple_assignment_5' (line 71)
            tuple_assignment_5_128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'tuple_assignment_5')
            # Assigning a type to the variable 'a' (line 71)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'a', tuple_assignment_5_128)
            
            # Assigning a Name to a Name (line 71):
            # Getting the type of 'tuple_assignment_6' (line 71)
            tuple_assignment_6_129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'tuple_assignment_6')
            # Assigning a type to the variable 'b' (line 71)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 15), 'b', tuple_assignment_6_129)
            # SSA join for if statement (line 70)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Subscript (line 72):
        
        # Assigning a BinOp to a Subscript (line 72):
        
        # Obtaining the type of the subscript
        # Getting the type of 'b' (line 72)
        b_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 17), 'b')
        int_131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 21), 'int')
        # Applying the binary operator '+' (line 72)
        result_add_132 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 17), '+', b_130, int_131)
        
        slice_133 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 72, 15), result_add_132, None, None)
        # Getting the type of 'd' (line 72)
        d_134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 'd')
        # Obtaining the member '__getitem__' of a type (line 72)
        getitem___135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 15), d_134, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 72)
        subscript_call_result_136 = invoke(stypy.reporting.localization.Localization(__file__, 72, 15), getitem___135, slice_133)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'a' (line 72)
        a_137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 29), 'a')
        # Getting the type of 'b' (line 72)
        b_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 31), 'b')
        int_139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 35), 'int')
        # Applying the binary operator '+' (line 72)
        result_add_140 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 31), '+', b_138, int_139)
        
        slice_141 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 72, 27), a_137, result_add_140, None)
        # Getting the type of 'd' (line 72)
        d_142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 27), 'd')
        # Obtaining the member '__getitem__' of a type (line 72)
        getitem___143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 27), d_142, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 72)
        subscript_call_result_144 = invoke(stypy.reporting.localization.Localization(__file__, 72, 27), getitem___143, slice_141)
        
        # Applying the binary operator '+' (line 72)
        result_add_145 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 15), '+', subscript_call_result_136, subscript_call_result_144)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'a' (line 72)
        a_146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 43), 'a')
        slice_147 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 72, 40), None, a_146, None)
        # Getting the type of 'd' (line 72)
        d_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 40), 'd')
        # Obtaining the member '__getitem__' of a type (line 72)
        getitem___149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 40), d_148, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 72)
        subscript_call_result_150 = invoke(stypy.reporting.localization.Localization(__file__, 72, 40), getitem___149, slice_147)
        
        # Applying the binary operator '+' (line 72)
        result_add_151 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 38), '+', result_add_145, subscript_call_result_150)
        
        # Getting the type of 'd' (line 72)
        d_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'd')
        slice_153 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 72, 8), None, None, None)
        # Storing an element on a container (line 72)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 8), d_152, (slice_153, result_add_151))
        
        # ################# End of '_tripleCut(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_tripleCut' in the type store
        # Getting the type of 'stypy_return_type' (line 63)
        stypy_return_type_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_154)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_tripleCut'
        return stypy_return_type_154


    @norecursion
    def _countCut(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_countCut'
        module_type_store = module_type_store.open_function_context('_countCut', 74, 4, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Solitaire._countCut.__dict__.__setitem__('stypy_localization', localization)
        Solitaire._countCut.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Solitaire._countCut.__dict__.__setitem__('stypy_type_store', module_type_store)
        Solitaire._countCut.__dict__.__setitem__('stypy_function_name', 'Solitaire._countCut')
        Solitaire._countCut.__dict__.__setitem__('stypy_param_names_list', ['n'])
        Solitaire._countCut.__dict__.__setitem__('stypy_varargs_param_name', None)
        Solitaire._countCut.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Solitaire._countCut.__dict__.__setitem__('stypy_call_defaults', defaults)
        Solitaire._countCut.__dict__.__setitem__('stypy_call_varargs', varargs)
        Solitaire._countCut.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Solitaire._countCut.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Solitaire._countCut', ['n'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_countCut', localization, ['n'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_countCut(...)' code ##################

        str_155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, (-1)), 'str', '\n        Cut after the n-th card, leaving the bottom\n        card in place.\n        ')
        
        # Assigning a Attribute to a Name (line 79):
        
        # Assigning a Attribute to a Name (line 79):
        # Getting the type of 'self' (line 79)
        self_156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'self')
        # Obtaining the member 'deck' of a type (line 79)
        deck_157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), self_156, 'deck')
        # Assigning a type to the variable 'd' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'd', deck_157)
        
        # Assigning a Call to a Name (line 80):
        
        # Assigning a Call to a Name (line 80):
        
        # Call to min(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'n' (line 80)
        n_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'n', False)
        int_160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 19), 'int')
        # Processing the call keyword arguments (line 80)
        kwargs_161 = {}
        # Getting the type of 'min' (line 80)
        min_158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'min', False)
        # Calling min(args, kwargs) (line 80)
        min_call_result_162 = invoke(stypy.reporting.localization.Localization(__file__, 80, 12), min_158, *[n_159, int_160], **kwargs_161)
        
        # Assigning a type to the variable 'n' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'n', min_call_result_162)
        
        # Assigning a BinOp to a Subscript (line 81):
        
        # Assigning a BinOp to a Subscript (line 81):
        
        # Obtaining the type of the subscript
        # Getting the type of 'n' (line 81)
        n_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 19), 'n')
        int_164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 21), 'int')
        slice_165 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 81, 17), n_163, int_164, None)
        # Getting the type of 'd' (line 81)
        d_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 17), 'd')
        # Obtaining the member '__getitem__' of a type (line 81)
        getitem___167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 17), d_166, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 81)
        subscript_call_result_168 = invoke(stypy.reporting.localization.Localization(__file__, 81, 17), getitem___167, slice_165)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'n' (line 81)
        n_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 30), 'n')
        slice_170 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 81, 27), None, n_169, None)
        # Getting the type of 'd' (line 81)
        d_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 27), 'd')
        # Obtaining the member '__getitem__' of a type (line 81)
        getitem___172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 27), d_171, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 81)
        subscript_call_result_173 = invoke(stypy.reporting.localization.Localization(__file__, 81, 27), getitem___172, slice_170)
        
        # Applying the binary operator '+' (line 81)
        result_add_174 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 17), '+', subscript_call_result_168, subscript_call_result_173)
        
        # Getting the type of 'd' (line 81)
        d_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'd')
        int_176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 11), 'int')
        slice_177 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 81, 8), None, int_176, None)
        # Storing an element on a container (line 81)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 8), d_175, (slice_177, result_add_174))
        
        # ################# End of '_countCut(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_countCut' in the type store
        # Getting the type of 'stypy_return_type' (line 74)
        stypy_return_type_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_178)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_countCut'
        return stypy_return_type_178


    @norecursion
    def _round(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_round'
        module_type_store = module_type_store.open_function_context('_round', 83, 4, False)
        # Assigning a type to the variable 'self' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Solitaire._round.__dict__.__setitem__('stypy_localization', localization)
        Solitaire._round.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Solitaire._round.__dict__.__setitem__('stypy_type_store', module_type_store)
        Solitaire._round.__dict__.__setitem__('stypy_function_name', 'Solitaire._round')
        Solitaire._round.__dict__.__setitem__('stypy_param_names_list', [])
        Solitaire._round.__dict__.__setitem__('stypy_varargs_param_name', None)
        Solitaire._round.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Solitaire._round.__dict__.__setitem__('stypy_call_defaults', defaults)
        Solitaire._round.__dict__.__setitem__('stypy_call_varargs', varargs)
        Solitaire._round.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Solitaire._round.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Solitaire._round', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_round', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_round(...)' code ##################

        str_179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, (-1)), 'str', '\n        Perform one round of keystream generation.\n        ')
        
        # Call to _down1(...): (line 87)
        # Processing the call arguments (line 87)
        int_182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 20), 'int')
        # Processing the call keyword arguments (line 87)
        kwargs_183 = {}
        # Getting the type of 'self' (line 87)
        self_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'self', False)
        # Obtaining the member '_down1' of a type (line 87)
        _down1_181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), self_180, '_down1')
        # Calling _down1(args, kwargs) (line 87)
        _down1_call_result_184 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), _down1_181, *[int_182], **kwargs_183)
        
        
        # Call to _down1(...): (line 88)
        # Processing the call arguments (line 88)
        int_187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 20), 'int')
        # Processing the call keyword arguments (line 88)
        kwargs_188 = {}
        # Getting the type of 'self' (line 88)
        self_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'self', False)
        # Obtaining the member '_down1' of a type (line 88)
        _down1_186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), self_185, '_down1')
        # Calling _down1(args, kwargs) (line 88)
        _down1_call_result_189 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), _down1_186, *[int_187], **kwargs_188)
        
        
        # Call to _down1(...): (line 89)
        # Processing the call arguments (line 89)
        int_192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 20), 'int')
        # Processing the call keyword arguments (line 89)
        kwargs_193 = {}
        # Getting the type of 'self' (line 89)
        self_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'self', False)
        # Obtaining the member '_down1' of a type (line 89)
        _down1_191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), self_190, '_down1')
        # Calling _down1(args, kwargs) (line 89)
        _down1_call_result_194 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), _down1_191, *[int_192], **kwargs_193)
        
        
        # Call to _tripleCut(...): (line 90)
        # Processing the call keyword arguments (line 90)
        kwargs_197 = {}
        # Getting the type of 'self' (line 90)
        self_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'self', False)
        # Obtaining the member '_tripleCut' of a type (line 90)
        _tripleCut_196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), self_195, '_tripleCut')
        # Calling _tripleCut(args, kwargs) (line 90)
        _tripleCut_call_result_198 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), _tripleCut_196, *[], **kwargs_197)
        
        
        # Call to _countCut(...): (line 91)
        # Processing the call arguments (line 91)
        
        # Obtaining the type of the subscript
        int_201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 33), 'int')
        # Getting the type of 'self' (line 91)
        self_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 23), 'self', False)
        # Obtaining the member 'deck' of a type (line 91)
        deck_203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 23), self_202, 'deck')
        # Obtaining the member '__getitem__' of a type (line 91)
        getitem___204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 23), deck_203, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 91)
        subscript_call_result_205 = invoke(stypy.reporting.localization.Localization(__file__, 91, 23), getitem___204, int_201)
        
        # Processing the call keyword arguments (line 91)
        kwargs_206 = {}
        # Getting the type of 'self' (line 91)
        self_199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'self', False)
        # Obtaining the member '_countCut' of a type (line 91)
        _countCut_200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), self_199, '_countCut')
        # Calling _countCut(args, kwargs) (line 91)
        _countCut_call_result_207 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), _countCut_200, *[subscript_call_result_205], **kwargs_206)
        
        
        # ################# End of '_round(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_round' in the type store
        # Getting the type of 'stypy_return_type' (line 83)
        stypy_return_type_208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_round'
        return stypy_return_type_208


    @norecursion
    def _output(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_output'
        module_type_store = module_type_store.open_function_context('_output', 93, 4, False)
        # Assigning a type to the variable 'self' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Solitaire._output.__dict__.__setitem__('stypy_localization', localization)
        Solitaire._output.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Solitaire._output.__dict__.__setitem__('stypy_type_store', module_type_store)
        Solitaire._output.__dict__.__setitem__('stypy_function_name', 'Solitaire._output')
        Solitaire._output.__dict__.__setitem__('stypy_param_names_list', [])
        Solitaire._output.__dict__.__setitem__('stypy_varargs_param_name', None)
        Solitaire._output.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Solitaire._output.__dict__.__setitem__('stypy_call_defaults', defaults)
        Solitaire._output.__dict__.__setitem__('stypy_call_varargs', varargs)
        Solitaire._output.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Solitaire._output.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Solitaire._output', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_output', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_output(...)' code ##################

        str_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, (-1)), 'str', '\n        Return next output card.\n        ')
        
        # Assigning a Attribute to a Name (line 97):
        
        # Assigning a Attribute to a Name (line 97):
        # Getting the type of 'self' (line 97)
        self_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'self')
        # Obtaining the member 'deck' of a type (line 97)
        deck_211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 12), self_210, 'deck')
        # Assigning a type to the variable 'd' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'd', deck_211)
        
        int_212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 14), 'int')
        # Assigning a type to the variable 'int_212' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'int_212', int_212)
        # Testing if the while is going to be iterated (line 98)
        # Testing the type of an if condition (line 98)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 8), int_212)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 98, 8), int_212):
            # SSA begins for while statement (line 98)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Call to _round(...): (line 99)
            # Processing the call keyword arguments (line 99)
            kwargs_215 = {}
            # Getting the type of 'self' (line 99)
            self_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'self', False)
            # Obtaining the member '_round' of a type (line 99)
            _round_214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 12), self_213, '_round')
            # Calling _round(args, kwargs) (line 99)
            _round_call_result_216 = invoke(stypy.reporting.localization.Localization(__file__, 99, 12), _round_214, *[], **kwargs_215)
            
            
            # Assigning a Call to a Name (line 100):
            
            # Assigning a Call to a Name (line 100):
            
            # Call to min(...): (line 100)
            # Processing the call arguments (line 100)
            
            # Obtaining the type of the subscript
            int_218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 28), 'int')
            # Getting the type of 'd' (line 100)
            d_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 26), 'd', False)
            # Obtaining the member '__getitem__' of a type (line 100)
            getitem___220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 26), d_219, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 100)
            subscript_call_result_221 = invoke(stypy.reporting.localization.Localization(__file__, 100, 26), getitem___220, int_218)
            
            int_222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 32), 'int')
            # Processing the call keyword arguments (line 100)
            kwargs_223 = {}
            # Getting the type of 'min' (line 100)
            min_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 22), 'min', False)
            # Calling min(args, kwargs) (line 100)
            min_call_result_224 = invoke(stypy.reporting.localization.Localization(__file__, 100, 22), min_217, *[subscript_call_result_221, int_222], **kwargs_223)
            
            # Assigning a type to the variable 'topCard' (line 100)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'topCard', min_call_result_224)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'topCard' (line 101)
            topCard_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 17), 'topCard')
            # Getting the type of 'd' (line 101)
            d_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'd')
            # Obtaining the member '__getitem__' of a type (line 101)
            getitem___227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 15), d_226, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 101)
            subscript_call_result_228 = invoke(stypy.reporting.localization.Localization(__file__, 101, 15), getitem___227, topCard_225)
            
            int_229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 28), 'int')
            # Applying the binary operator '<' (line 101)
            result_lt_230 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 15), '<', subscript_call_result_228, int_229)
            
            # Testing if the type of an if condition is none (line 101)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 101, 12), result_lt_230):
                pass
            else:
                
                # Testing the type of an if condition (line 101)
                if_condition_231 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 12), result_lt_230)
                # Assigning a type to the variable 'if_condition_231' (line 101)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'if_condition_231', if_condition_231)
                # SSA begins for if statement (line 101)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Obtaining the type of the subscript
                # Getting the type of 'topCard' (line 102)
                topCard_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 25), 'topCard')
                # Getting the type of 'd' (line 102)
                d_233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 23), 'd')
                # Obtaining the member '__getitem__' of a type (line 102)
                getitem___234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 23), d_233, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 102)
                subscript_call_result_235 = invoke(stypy.reporting.localization.Localization(__file__, 102, 23), getitem___234, topCard_232)
                
                # Assigning a type to the variable 'stypy_return_type' (line 102)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'stypy_return_type', subscript_call_result_235)
                # SSA join for if statement (line 101)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for while statement (line 98)
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of '_output(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_output' in the type store
        # Getting the type of 'stypy_return_type' (line 93)
        stypy_return_type_236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_236)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_output'
        return stypy_return_type_236


    @norecursion
    def encrypt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'encrypt'
        module_type_store = module_type_store.open_function_context('encrypt', 104, 4, False)
        # Assigning a type to the variable 'self' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Solitaire.encrypt.__dict__.__setitem__('stypy_localization', localization)
        Solitaire.encrypt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Solitaire.encrypt.__dict__.__setitem__('stypy_type_store', module_type_store)
        Solitaire.encrypt.__dict__.__setitem__('stypy_function_name', 'Solitaire.encrypt')
        Solitaire.encrypt.__dict__.__setitem__('stypy_param_names_list', ['txt', 'key'])
        Solitaire.encrypt.__dict__.__setitem__('stypy_varargs_param_name', None)
        Solitaire.encrypt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Solitaire.encrypt.__dict__.__setitem__('stypy_call_defaults', defaults)
        Solitaire.encrypt.__dict__.__setitem__('stypy_call_varargs', varargs)
        Solitaire.encrypt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Solitaire.encrypt.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Solitaire.encrypt', ['txt', 'key'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'encrypt', localization, ['txt', 'key'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'encrypt(...)' code ##################

        str_237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, (-1)), 'str', "\n        Return 'txt' encrypted using 'key'.\n        ")
        
        # Call to _setKey(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'key' (line 108)
        key_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 21), 'key', False)
        # Processing the call keyword arguments (line 108)
        kwargs_241 = {}
        # Getting the type of 'self' (line 108)
        self_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'self', False)
        # Obtaining the member '_setKey' of a type (line 108)
        _setKey_239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), self_238, '_setKey')
        # Calling _setKey(args, kwargs) (line 108)
        _setKey_call_result_242 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), _setKey_239, *[key_240], **kwargs_241)
        
        
        # Assigning a BinOp to a Name (line 110):
        
        # Assigning a BinOp to a Name (line 110):
        # Getting the type of 'txt' (line 110)
        txt_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 14), 'txt')
        str_244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 20), 'str', 'X')
        int_245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 28), 'int')
        
        # Call to len(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'txt' (line 110)
        txt_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 36), 'txt', False)
        # Processing the call keyword arguments (line 110)
        kwargs_248 = {}
        # Getting the type of 'len' (line 110)
        len_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 32), 'len', False)
        # Calling len(args, kwargs) (line 110)
        len_call_result_249 = invoke(stypy.reporting.localization.Localization(__file__, 110, 32), len_246, *[txt_247], **kwargs_248)
        
        # Applying the binary operator '-' (line 110)
        result_sub_250 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 28), '-', int_245, len_call_result_249)
        
        int_251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 44), 'int')
        # Applying the binary operator '%' (line 110)
        result_mod_252 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 27), '%', result_sub_250, int_251)
        
        # Applying the binary operator '*' (line 110)
        result_mul_253 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 20), '*', str_244, result_mod_252)
        
        # Applying the binary operator '+' (line 110)
        result_add_254 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 14), '+', txt_243, result_mul_253)
        
        # Assigning a type to the variable 'txt' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'txt', result_add_254)
        
        # Assigning a BinOp to a Name (line 111):
        
        # Assigning a BinOp to a Name (line 111):
        
        # Obtaining an instance of the builtin type 'list' (line 111)
        list_255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 111)
        # Adding element type (line 111)
        # Getting the type of 'None' (line 111)
        None_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 18), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 17), list_255, None_256)
        
        
        # Call to len(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'txt' (line 111)
        txt_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 30), 'txt', False)
        # Processing the call keyword arguments (line 111)
        kwargs_259 = {}
        # Getting the type of 'len' (line 111)
        len_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 26), 'len', False)
        # Calling len(args, kwargs) (line 111)
        len_call_result_260 = invoke(stypy.reporting.localization.Localization(__file__, 111, 26), len_257, *[txt_258], **kwargs_259)
        
        # Applying the binary operator '*' (line 111)
        result_mul_261 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 17), '*', list_255, len_call_result_260)
        
        # Assigning a type to the variable 'cipher' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'cipher', result_mul_261)
        
        
        # Call to range(...): (line 112)
        # Processing the call arguments (line 112)
        
        # Call to len(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'txt' (line 112)
        txt_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 27), 'txt', False)
        # Processing the call keyword arguments (line 112)
        kwargs_265 = {}
        # Getting the type of 'len' (line 112)
        len_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 23), 'len', False)
        # Calling len(args, kwargs) (line 112)
        len_call_result_266 = invoke(stypy.reporting.localization.Localization(__file__, 112, 23), len_263, *[txt_264], **kwargs_265)
        
        # Processing the call keyword arguments (line 112)
        kwargs_267 = {}
        # Getting the type of 'range' (line 112)
        range_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 17), 'range', False)
        # Calling range(args, kwargs) (line 112)
        range_call_result_268 = invoke(stypy.reporting.localization.Localization(__file__, 112, 17), range_262, *[len_call_result_266], **kwargs_267)
        
        # Assigning a type to the variable 'range_call_result_268' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'range_call_result_268', range_call_result_268)
        # Testing if the for loop is going to be iterated (line 112)
        # Testing the type of a for loop iterable (line 112)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 112, 8), range_call_result_268)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 112, 8), range_call_result_268):
            # Getting the type of the for loop variable (line 112)
            for_loop_var_269 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 112, 8), range_call_result_268)
            # Assigning a type to the variable 'n' (line 112)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'n', for_loop_var_269)
            # SSA begins for a for statement (line 112)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Subscript (line 113):
            
            # Assigning a Call to a Subscript (line 113):
            
            # Call to toChar(...): (line 113)
            # Processing the call arguments (line 113)
            
            # Call to toNumber(...): (line 113)
            # Processing the call arguments (line 113)
            
            # Obtaining the type of the subscript
            # Getting the type of 'n' (line 113)
            n_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 44), 'n', False)
            # Getting the type of 'txt' (line 113)
            txt_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 40), 'txt', False)
            # Obtaining the member '__getitem__' of a type (line 113)
            getitem___274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 40), txt_273, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 113)
            subscript_call_result_275 = invoke(stypy.reporting.localization.Localization(__file__, 113, 40), getitem___274, n_272)
            
            # Processing the call keyword arguments (line 113)
            kwargs_276 = {}
            # Getting the type of 'toNumber' (line 113)
            toNumber_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 31), 'toNumber', False)
            # Calling toNumber(args, kwargs) (line 113)
            toNumber_call_result_277 = invoke(stypy.reporting.localization.Localization(__file__, 113, 31), toNumber_271, *[subscript_call_result_275], **kwargs_276)
            
            
            # Call to _output(...): (line 113)
            # Processing the call keyword arguments (line 113)
            kwargs_280 = {}
            # Getting the type of 'self' (line 113)
            self_278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 50), 'self', False)
            # Obtaining the member '_output' of a type (line 113)
            _output_279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 50), self_278, '_output')
            # Calling _output(args, kwargs) (line 113)
            _output_call_result_281 = invoke(stypy.reporting.localization.Localization(__file__, 113, 50), _output_279, *[], **kwargs_280)
            
            # Applying the binary operator '+' (line 113)
            result_add_282 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 31), '+', toNumber_call_result_277, _output_call_result_281)
            
            # Processing the call keyword arguments (line 113)
            kwargs_283 = {}
            # Getting the type of 'toChar' (line 113)
            toChar_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 24), 'toChar', False)
            # Calling toChar(args, kwargs) (line 113)
            toChar_call_result_284 = invoke(stypy.reporting.localization.Localization(__file__, 113, 24), toChar_270, *[result_add_282], **kwargs_283)
            
            # Getting the type of 'cipher' (line 113)
            cipher_285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'cipher')
            # Getting the type of 'n' (line 113)
            n_286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'n')
            # Storing an element on a container (line 113)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 12), cipher_285, (n_286, toChar_call_result_284))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to range(...): (line 115)
        # Processing the call arguments (line 115)
        
        # Call to len(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'cipher' (line 115)
        cipher_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 27), 'cipher', False)
        # Processing the call keyword arguments (line 115)
        kwargs_290 = {}
        # Getting the type of 'len' (line 115)
        len_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 23), 'len', False)
        # Calling len(args, kwargs) (line 115)
        len_call_result_291 = invoke(stypy.reporting.localization.Localization(__file__, 115, 23), len_288, *[cipher_289], **kwargs_290)
        
        int_292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 37), 'int')
        # Applying the binary operator '-' (line 115)
        result_sub_293 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 23), '-', len_call_result_291, int_292)
        
        int_294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 40), 'int')
        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 43), 'int')
        # Processing the call keyword arguments (line 115)
        kwargs_296 = {}
        # Getting the type of 'range' (line 115)
        range_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 17), 'range', False)
        # Calling range(args, kwargs) (line 115)
        range_call_result_297 = invoke(stypy.reporting.localization.Localization(__file__, 115, 17), range_287, *[result_sub_293, int_294, int_295], **kwargs_296)
        
        # Assigning a type to the variable 'range_call_result_297' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'range_call_result_297', range_call_result_297)
        # Testing if the for loop is going to be iterated (line 115)
        # Testing the type of a for loop iterable (line 115)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 115, 8), range_call_result_297)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 115, 8), range_call_result_297):
            # Getting the type of the for loop variable (line 115)
            for_loop_var_298 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 115, 8), range_call_result_297)
            # Assigning a type to the variable 'n' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'n', for_loop_var_298)
            # SSA begins for a for statement (line 115)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a List to a Subscript (line 116):
            
            # Assigning a List to a Subscript (line 116):
            
            # Obtaining an instance of the builtin type 'list' (line 116)
            list_299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 26), 'list')
            # Adding type elements to the builtin type 'list' instance (line 116)
            # Adding element type (line 116)
            str_300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 27), 'str', ' ')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 26), list_299, str_300)
            
            # Getting the type of 'cipher' (line 116)
            cipher_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'cipher')
            # Getting the type of 'n' (line 116)
            n_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 19), 'n')
            # Getting the type of 'n' (line 116)
            n_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 21), 'n')
            slice_304 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 116, 12), n_302, n_303, None)
            # Storing an element on a container (line 116)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 12), cipher_301, (slice_304, list_299))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to join(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'cipher' (line 117)
        cipher_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 27), 'cipher', False)
        str_308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 35), 'str', '')
        # Processing the call keyword arguments (line 117)
        kwargs_309 = {}
        # Getting the type of 'string' (line 117)
        string_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'string', False)
        # Obtaining the member 'join' of a type (line 117)
        join_306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 15), string_305, 'join')
        # Calling join(args, kwargs) (line 117)
        join_call_result_310 = invoke(stypy.reporting.localization.Localization(__file__, 117, 15), join_306, *[cipher_307, str_308], **kwargs_309)
        
        # Assigning a type to the variable 'stypy_return_type' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'stypy_return_type', join_call_result_310)
        
        # ################# End of 'encrypt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'encrypt' in the type store
        # Getting the type of 'stypy_return_type' (line 104)
        stypy_return_type_311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_311)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'encrypt'
        return stypy_return_type_311


    @norecursion
    def decrypt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'decrypt'
        module_type_store = module_type_store.open_function_context('decrypt', 119, 4, False)
        # Assigning a type to the variable 'self' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Solitaire.decrypt.__dict__.__setitem__('stypy_localization', localization)
        Solitaire.decrypt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Solitaire.decrypt.__dict__.__setitem__('stypy_type_store', module_type_store)
        Solitaire.decrypt.__dict__.__setitem__('stypy_function_name', 'Solitaire.decrypt')
        Solitaire.decrypt.__dict__.__setitem__('stypy_param_names_list', ['cipher', 'key'])
        Solitaire.decrypt.__dict__.__setitem__('stypy_varargs_param_name', None)
        Solitaire.decrypt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Solitaire.decrypt.__dict__.__setitem__('stypy_call_defaults', defaults)
        Solitaire.decrypt.__dict__.__setitem__('stypy_call_varargs', varargs)
        Solitaire.decrypt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Solitaire.decrypt.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Solitaire.decrypt', ['cipher', 'key'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'decrypt', localization, ['cipher', 'key'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'decrypt(...)' code ##################

        str_312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, (-1)), 'str', "\n        Return 'cipher' decrypted using 'key'.\n        ")
        
        # Call to _setKey(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'key' (line 123)
        key_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 21), 'key', False)
        # Processing the call keyword arguments (line 123)
        kwargs_316 = {}
        # Getting the type of 'self' (line 123)
        self_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'self', False)
        # Obtaining the member '_setKey' of a type (line 123)
        _setKey_314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), self_313, '_setKey')
        # Calling _setKey(args, kwargs) (line 123)
        _setKey_call_result_317 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), _setKey_314, *[key_315], **kwargs_316)
        
        
        # Assigning a Call to a Name (line 125):
        
        # Assigning a Call to a Name (line 125):
        
        # Call to join(...): (line 125)
        # Processing the call arguments (line 125)
        
        # Call to split(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'cipher' (line 125)
        cipher_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 42), 'cipher', False)
        # Processing the call keyword arguments (line 125)
        kwargs_323 = {}
        # Getting the type of 'string' (line 125)
        string_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 29), 'string', False)
        # Obtaining the member 'split' of a type (line 125)
        split_321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 29), string_320, 'split')
        # Calling split(args, kwargs) (line 125)
        split_call_result_324 = invoke(stypy.reporting.localization.Localization(__file__, 125, 29), split_321, *[cipher_322], **kwargs_323)
        
        str_325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 51), 'str', '')
        # Processing the call keyword arguments (line 125)
        kwargs_326 = {}
        # Getting the type of 'string' (line 125)
        string_318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 17), 'string', False)
        # Obtaining the member 'join' of a type (line 125)
        join_319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 17), string_318, 'join')
        # Calling join(args, kwargs) (line 125)
        join_call_result_327 = invoke(stypy.reporting.localization.Localization(__file__, 125, 17), join_319, *[split_call_result_324, str_325], **kwargs_326)
        
        # Assigning a type to the variable 'cipher' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'cipher', join_call_result_327)
        
        # Assigning a BinOp to a Name (line 126):
        
        # Assigning a BinOp to a Name (line 126):
        
        # Obtaining an instance of the builtin type 'list' (line 126)
        list_328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 126)
        # Adding element type (line 126)
        # Getting the type of 'None' (line 126)
        None_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 15), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 14), list_328, None_329)
        
        
        # Call to len(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'cipher' (line 126)
        cipher_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 27), 'cipher', False)
        # Processing the call keyword arguments (line 126)
        kwargs_332 = {}
        # Getting the type of 'len' (line 126)
        len_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 23), 'len', False)
        # Calling len(args, kwargs) (line 126)
        len_call_result_333 = invoke(stypy.reporting.localization.Localization(__file__, 126, 23), len_330, *[cipher_331], **kwargs_332)
        
        # Applying the binary operator '*' (line 126)
        result_mul_334 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 14), '*', list_328, len_call_result_333)
        
        # Assigning a type to the variable 'txt' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'txt', result_mul_334)
        
        
        # Call to range(...): (line 127)
        # Processing the call arguments (line 127)
        
        # Call to len(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'cipher' (line 127)
        cipher_337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 27), 'cipher', False)
        # Processing the call keyword arguments (line 127)
        kwargs_338 = {}
        # Getting the type of 'len' (line 127)
        len_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'len', False)
        # Calling len(args, kwargs) (line 127)
        len_call_result_339 = invoke(stypy.reporting.localization.Localization(__file__, 127, 23), len_336, *[cipher_337], **kwargs_338)
        
        # Processing the call keyword arguments (line 127)
        kwargs_340 = {}
        # Getting the type of 'range' (line 127)
        range_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 17), 'range', False)
        # Calling range(args, kwargs) (line 127)
        range_call_result_341 = invoke(stypy.reporting.localization.Localization(__file__, 127, 17), range_335, *[len_call_result_339], **kwargs_340)
        
        # Assigning a type to the variable 'range_call_result_341' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'range_call_result_341', range_call_result_341)
        # Testing if the for loop is going to be iterated (line 127)
        # Testing the type of a for loop iterable (line 127)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 127, 8), range_call_result_341)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 127, 8), range_call_result_341):
            # Getting the type of the for loop variable (line 127)
            for_loop_var_342 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 127, 8), range_call_result_341)
            # Assigning a type to the variable 'n' (line 127)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'n', for_loop_var_342)
            # SSA begins for a for statement (line 127)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Subscript (line 128):
            
            # Assigning a Call to a Subscript (line 128):
            
            # Call to toChar(...): (line 128)
            # Processing the call arguments (line 128)
            
            # Call to toNumber(...): (line 128)
            # Processing the call arguments (line 128)
            
            # Obtaining the type of the subscript
            # Getting the type of 'n' (line 128)
            n_345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 44), 'n', False)
            # Getting the type of 'cipher' (line 128)
            cipher_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 37), 'cipher', False)
            # Obtaining the member '__getitem__' of a type (line 128)
            getitem___347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 37), cipher_346, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 128)
            subscript_call_result_348 = invoke(stypy.reporting.localization.Localization(__file__, 128, 37), getitem___347, n_345)
            
            # Processing the call keyword arguments (line 128)
            kwargs_349 = {}
            # Getting the type of 'toNumber' (line 128)
            toNumber_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 28), 'toNumber', False)
            # Calling toNumber(args, kwargs) (line 128)
            toNumber_call_result_350 = invoke(stypy.reporting.localization.Localization(__file__, 128, 28), toNumber_344, *[subscript_call_result_348], **kwargs_349)
            
            
            # Call to _output(...): (line 128)
            # Processing the call keyword arguments (line 128)
            kwargs_353 = {}
            # Getting the type of 'self' (line 128)
            self_351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 50), 'self', False)
            # Obtaining the member '_output' of a type (line 128)
            _output_352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 50), self_351, '_output')
            # Calling _output(args, kwargs) (line 128)
            _output_call_result_354 = invoke(stypy.reporting.localization.Localization(__file__, 128, 50), _output_352, *[], **kwargs_353)
            
            # Applying the binary operator '-' (line 128)
            result_sub_355 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 28), '-', toNumber_call_result_350, _output_call_result_354)
            
            # Processing the call keyword arguments (line 128)
            kwargs_356 = {}
            # Getting the type of 'toChar' (line 128)
            toChar_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 21), 'toChar', False)
            # Calling toChar(args, kwargs) (line 128)
            toChar_call_result_357 = invoke(stypy.reporting.localization.Localization(__file__, 128, 21), toChar_343, *[result_sub_355], **kwargs_356)
            
            # Getting the type of 'txt' (line 128)
            txt_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'txt')
            # Getting the type of 'n' (line 128)
            n_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'n')
            # Storing an element on a container (line 128)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 12), txt_358, (n_359, toChar_call_result_357))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to join(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'txt' (line 129)
        txt_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 27), 'txt', False)
        str_363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 32), 'str', '')
        # Processing the call keyword arguments (line 129)
        kwargs_364 = {}
        # Getting the type of 'string' (line 129)
        string_360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 15), 'string', False)
        # Obtaining the member 'join' of a type (line 129)
        join_361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 15), string_360, 'join')
        # Calling join(args, kwargs) (line 129)
        join_call_result_365 = invoke(stypy.reporting.localization.Localization(__file__, 129, 15), join_361, *[txt_362, str_363], **kwargs_364)
        
        # Assigning a type to the variable 'stypy_return_type' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'stypy_return_type', join_call_result_365)
        
        # ################# End of 'decrypt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'decrypt' in the type store
        # Getting the type of 'stypy_return_type' (line 119)
        stypy_return_type_366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_366)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'decrypt'
        return stypy_return_type_366


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 31, 0, False)
        # Assigning a type to the variable 'self' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Solitaire.__init__', [], None, None, defaults, varargs, kwargs)

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

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Solitaire' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'Solitaire', Solitaire)

# Assigning a Tuple to a Name (line 132):

# Assigning a Tuple to a Name (line 132):

# Obtaining an instance of the builtin type 'tuple' (line 133)
tuple_367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 4), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 133)
# Adding element type (line 133)

# Obtaining an instance of the builtin type 'tuple' (line 133)
tuple_368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 133)
# Adding element type (line 133)
str_369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 5), 'str', 'AAAAAAAAAAAAAAA')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 5), tuple_368, str_369)
# Adding element type (line 133)
str_370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 24), 'str', '')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 5), tuple_368, str_370)
# Adding element type (line 133)
str_371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 28), 'str', 'EXKYI ZSGEH UNTIQ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 5), tuple_368, str_371)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 4), tuple_367, tuple_368)
# Adding element type (line 133)

# Obtaining an instance of the builtin type 'tuple' (line 134)
tuple_372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 134)
# Adding element type (line 134)
str_373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 5), 'str', 'AAAAAAAAAAAAAAA')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 5), tuple_372, str_373)
# Adding element type (line 134)
str_374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 24), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 5), tuple_372, str_374)
# Adding element type (line 134)
str_375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 29), 'str', 'XYIUQ BMHKK JBEGY')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 5), tuple_372, str_375)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 4), tuple_367, tuple_372)
# Adding element type (line 133)

# Obtaining an instance of the builtin type 'tuple' (line 135)
tuple_376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 135)
# Adding element type (line 135)
str_377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 5), 'str', 'AAAAAAAAAAAAAAA')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 5), tuple_376, str_377)
# Adding element type (line 135)
str_378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 24), 'str', 'fo')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 5), tuple_376, str_378)
# Adding element type (line 135)
str_379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 30), 'str', 'TUJYM BERLG XNDIW')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 5), tuple_376, str_379)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 4), tuple_367, tuple_376)
# Adding element type (line 133)

# Obtaining an instance of the builtin type 'tuple' (line 136)
tuple_380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 136)
# Adding element type (line 136)
str_381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 5), 'str', 'AAAAAAAAAAAAAAA')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 5), tuple_380, str_381)
# Adding element type (line 136)
str_382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 24), 'str', 'foo')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 5), tuple_380, str_382)
# Adding element type (line 136)
str_383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 31), 'str', 'ITHZU JIWGR FARMW')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 5), tuple_380, str_383)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 4), tuple_367, tuple_380)
# Adding element type (line 133)

# Obtaining an instance of the builtin type 'tuple' (line 137)
tuple_384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 137)
# Adding element type (line 137)
str_385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 5), 'str', 'AAAAAAAAAAAAAAA')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 5), tuple_384, str_385)
# Adding element type (line 137)
str_386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 24), 'str', 'a')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 5), tuple_384, str_386)
# Adding element type (line 137)
str_387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 29), 'str', 'XODAL GSCUL IQNSC')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 5), tuple_384, str_387)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 4), tuple_367, tuple_384)
# Adding element type (line 133)

# Obtaining an instance of the builtin type 'tuple' (line 138)
tuple_388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 138)
# Adding element type (line 138)
str_389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 5), 'str', 'AAAAAAAAAAAAAAA')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 5), tuple_388, str_389)
# Adding element type (line 138)
str_390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 24), 'str', 'aa')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 5), tuple_388, str_390)
# Adding element type (line 138)
str_391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 30), 'str', 'OHGWM XXCAI MCIQP')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 5), tuple_388, str_391)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 4), tuple_367, tuple_388)
# Adding element type (line 133)

# Obtaining an instance of the builtin type 'tuple' (line 139)
tuple_392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 139)
# Adding element type (line 139)
str_393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 5), 'str', 'AAAAAAAAAAAAAAA')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 5), tuple_392, str_393)
# Adding element type (line 139)
str_394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 24), 'str', 'aaa')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 5), tuple_392, str_394)
# Adding element type (line 139)
str_395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 31), 'str', 'DCSQY HBQZN GDRUT')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 5), tuple_392, str_395)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 4), tuple_367, tuple_392)
# Adding element type (line 133)

# Obtaining an instance of the builtin type 'tuple' (line 140)
tuple_396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 140)
# Adding element type (line 140)
str_397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 5), 'str', 'AAAAAAAAAAAAAAA')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 5), tuple_396, str_397)
# Adding element type (line 140)
str_398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 24), 'str', 'b')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 5), tuple_396, str_398)
# Adding element type (line 140)
str_399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 29), 'str', 'XQEEM OITLZ VDSQS')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 5), tuple_396, str_399)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 4), tuple_367, tuple_396)
# Adding element type (line 133)

# Obtaining an instance of the builtin type 'tuple' (line 141)
tuple_400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 141)
# Adding element type (line 141)
str_401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 5), 'str', 'AAAAAAAAAAAAAAA')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 5), tuple_400, str_401)
# Adding element type (line 141)
str_402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 24), 'str', 'bc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 5), tuple_400, str_402)
# Adding element type (line 141)
str_403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 30), 'str', 'QNGRK QIHCL GWSCE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 5), tuple_400, str_403)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 4), tuple_367, tuple_400)
# Adding element type (line 133)

# Obtaining an instance of the builtin type 'tuple' (line 142)
tuple_404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 142)
# Adding element type (line 142)
str_405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 5), 'str', 'AAAAAAAAAAAAAAA')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 5), tuple_404, str_405)
# Adding element type (line 142)
str_406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 24), 'str', 'bcd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 5), tuple_404, str_406)
# Adding element type (line 142)
str_407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 31), 'str', 'FMUBY BMAXH NQXCJ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 5), tuple_404, str_407)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 4), tuple_367, tuple_404)
# Adding element type (line 133)

# Obtaining an instance of the builtin type 'tuple' (line 143)
tuple_408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 143)
# Adding element type (line 143)
str_409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 5), 'str', 'AAAAAAAAAAAAAAAAAAAAAAAAA')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 5), tuple_408, str_409)
# Adding element type (line 143)
str_410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 34), 'str', 'cryptonomicon')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 5), tuple_408, str_410)
# Adding element type (line 143)
str_411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 5), 'str', 'SUGSR SXSWQ RMXOH IPBFP XARYQ')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 5), tuple_408, str_411)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 4), tuple_367, tuple_408)
# Adding element type (line 133)

# Obtaining an instance of the builtin type 'tuple' (line 145)
tuple_412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 145)
# Adding element type (line 145)
str_413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 5), 'str', 'SOLITAIRE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 5), tuple_412, str_413)
# Adding element type (line 145)
str_414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 18), 'str', 'cryptonomicon')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 5), tuple_412, str_414)
# Adding element type (line 145)
str_415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 35), 'str', 'KIRAK SFJAN')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 5), tuple_412, str_415)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 4), tuple_367, tuple_412)

# Assigning a type to the variable 'testCases' (line 132)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 0), 'testCases', tuple_367)

@norecursion
def usage(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'usage'
    module_type_store = module_type_store.open_function_context('usage', 149, 0, False)
    
    # Passed parameters checking function
    usage.stypy_localization = localization
    usage.stypy_type_of_self = None
    usage.stypy_type_store = module_type_store
    usage.stypy_function_name = 'usage'
    usage.stypy_param_names_list = []
    usage.stypy_varargs_param_name = None
    usage.stypy_kwargs_param_name = None
    usage.stypy_call_defaults = defaults
    usage.stypy_call_varargs = varargs
    usage.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'usage', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'usage', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'usage(...)' code ##################

    str_416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, (-1)), 'str', 'Usage:\n    sol.py {-e | -d} _key_ < _file_\n    sol.py -test\n    \n    N.B. WinNT requires "python sol.py ..."\n    for input redirection to work (NT bug).\n    ')
    
    # Call to exit(...): (line 157)
    # Processing the call arguments (line 157)
    int_419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 13), 'int')
    # Processing the call keyword arguments (line 157)
    kwargs_420 = {}
    # Getting the type of 'sys' (line 157)
    sys_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'sys', False)
    # Obtaining the member 'exit' of a type (line 157)
    exit_418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 4), sys_417, 'exit')
    # Calling exit(args, kwargs) (line 157)
    exit_call_result_421 = invoke(stypy.reporting.localization.Localization(__file__, 157, 4), exit_418, *[int_419], **kwargs_420)
    
    
    # ################# End of 'usage(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'usage' in the type store
    # Getting the type of 'stypy_return_type' (line 149)
    stypy_return_type_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_422)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'usage'
    return stypy_return_type_422

# Assigning a type to the variable 'usage' (line 149)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 0), 'usage', usage)

@norecursion
def main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'main'
    module_type_store = module_type_store.open_function_context('main', 160, 0, False)
    
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

    
    # Assigning a Call to a Name (line 165):
    
    # Assigning a Call to a Name (line 165):
    
    # Call to Solitaire(...): (line 165)
    # Processing the call keyword arguments (line 165)
    kwargs_424 = {}
    # Getting the type of 'Solitaire' (line 165)
    Solitaire_423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'Solitaire', False)
    # Calling Solitaire(args, kwargs) (line 165)
    Solitaire_call_result_425 = invoke(stypy.reporting.localization.Localization(__file__, 165, 8), Solitaire_423, *[], **kwargs_424)
    
    # Assigning a type to the variable 's' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 's', Solitaire_call_result_425)
    
    # Getting the type of 'testCases' (line 166)
    testCases_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 28), 'testCases')
    # Assigning a type to the variable 'testCases_426' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'testCases_426', testCases_426)
    # Testing if the for loop is going to be iterated (line 166)
    # Testing the type of a for loop iterable (line 166)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 166, 4), testCases_426)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 166, 4), testCases_426):
        # Getting the type of the for loop variable (line 166)
        for_loop_var_427 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 166, 4), testCases_426)
        # Assigning a type to the variable 'txt' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'txt', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 4), for_loop_var_427))
        # Assigning a type to the variable 'key' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 4), for_loop_var_427))
        # Assigning a type to the variable 'cipher' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'cipher', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 4), for_loop_var_427))
        # SSA begins for a for statement (line 166)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 167):
        
        # Assigning a Call to a Name (line 167):
        
        # Call to encrypt(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'txt' (line 167)
        txt_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 26), 'txt', False)
        # Getting the type of 'key' (line 167)
        key_431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 31), 'key', False)
        # Processing the call keyword arguments (line 167)
        kwargs_432 = {}
        # Getting the type of 's' (line 167)
        s_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 's', False)
        # Obtaining the member 'encrypt' of a type (line 167)
        encrypt_429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 16), s_428, 'encrypt')
        # Calling encrypt(args, kwargs) (line 167)
        encrypt_call_result_433 = invoke(stypy.reporting.localization.Localization(__file__, 167, 16), encrypt_429, *[txt_430, key_431], **kwargs_432)
        
        # Assigning a type to the variable 'coded' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'coded', encrypt_call_result_433)
        # Evaluating assert statement condition
        
        # Getting the type of 'cipher' (line 168)
        cipher_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 15), 'cipher')
        # Getting the type of 'coded' (line 168)
        coded_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 25), 'coded')
        # Applying the binary operator '==' (line 168)
        result_eq_436 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 15), '==', cipher_434, coded_435)
        
        assert_437 = result_eq_436
        # Assigning a type to the variable 'assert_437' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'assert_437', result_eq_436)
        
        # Assigning a Call to a Name (line 169):
        
        # Assigning a Call to a Name (line 169):
        
        # Call to decrypt(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'coded' (line 169)
        coded_440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 28), 'coded', False)
        # Getting the type of 'key' (line 169)
        key_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 35), 'key', False)
        # Processing the call keyword arguments (line 169)
        kwargs_442 = {}
        # Getting the type of 's' (line 169)
        s_438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 18), 's', False)
        # Obtaining the member 'decrypt' of a type (line 169)
        decrypt_439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 18), s_438, 'decrypt')
        # Calling decrypt(args, kwargs) (line 169)
        decrypt_call_result_443 = invoke(stypy.reporting.localization.Localization(__file__, 169, 18), decrypt_439, *[coded_440, key_441], **kwargs_442)
        
        # Assigning a type to the variable 'decoded' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'decoded', decrypt_call_result_443)
        # Evaluating assert statement condition
        
        
        # Obtaining the type of the subscript
        
        # Call to len(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'txt' (line 170)
        txt_445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 28), 'txt', False)
        # Processing the call keyword arguments (line 170)
        kwargs_446 = {}
        # Getting the type of 'len' (line 170)
        len_444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 24), 'len', False)
        # Calling len(args, kwargs) (line 170)
        len_call_result_447 = invoke(stypy.reporting.localization.Localization(__file__, 170, 24), len_444, *[txt_445], **kwargs_446)
        
        slice_448 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 170, 15), None, len_call_result_447, None)
        # Getting the type of 'decoded' (line 170)
        decoded_449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 15), 'decoded')
        # Obtaining the member '__getitem__' of a type (line 170)
        getitem___450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 15), decoded_449, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 170)
        subscript_call_result_451 = invoke(stypy.reporting.localization.Localization(__file__, 170, 15), getitem___450, slice_448)
        
        
        # Call to upper(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'txt' (line 170)
        txt_454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 50), 'txt', False)
        # Processing the call keyword arguments (line 170)
        kwargs_455 = {}
        # Getting the type of 'string' (line 170)
        string_452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 37), 'string', False)
        # Obtaining the member 'upper' of a type (line 170)
        upper_453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 37), string_452, 'upper')
        # Calling upper(args, kwargs) (line 170)
        upper_call_result_456 = invoke(stypy.reporting.localization.Localization(__file__, 170, 37), upper_453, *[txt_454], **kwargs_455)
        
        # Applying the binary operator '==' (line 170)
        result_eq_457 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 15), '==', subscript_call_result_451, upper_call_result_456)
        
        assert_458 = result_eq_457
        # Assigning a type to the variable 'assert_458' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'assert_458', result_eq_457)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 160)
    stypy_return_type_459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_459)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_459

# Assigning a type to the variable 'main' (line 160)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 0), 'main', main)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 183, 0, False)
    
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

    
    
    # Call to range(...): (line 184)
    # Processing the call arguments (line 184)
    int_461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 19), 'int')
    # Processing the call keyword arguments (line 184)
    kwargs_462 = {}
    # Getting the type of 'range' (line 184)
    range_460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 13), 'range', False)
    # Calling range(args, kwargs) (line 184)
    range_call_result_463 = invoke(stypy.reporting.localization.Localization(__file__, 184, 13), range_460, *[int_461], **kwargs_462)
    
    # Assigning a type to the variable 'range_call_result_463' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'range_call_result_463', range_call_result_463)
    # Testing if the for loop is going to be iterated (line 184)
    # Testing the type of a for loop iterable (line 184)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 184, 4), range_call_result_463)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 184, 4), range_call_result_463):
        # Getting the type of the for loop variable (line 184)
        for_loop_var_464 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 184, 4), range_call_result_463)
        # Assigning a type to the variable 'i' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'i', for_loop_var_464)
        # SSA begins for a for statement (line 184)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to main(...): (line 185)
        # Processing the call keyword arguments (line 185)
        kwargs_466 = {}
        # Getting the type of 'main' (line 185)
        main_465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'main', False)
        # Calling main(args, kwargs) (line 185)
        main_call_result_467 = invoke(stypy.reporting.localization.Localization(__file__, 185, 8), main_465, *[], **kwargs_466)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'True' (line 186)
    True_468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'stypy_return_type', True_468)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 183)
    stypy_return_type_469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_469)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_469

# Assigning a type to the variable 'run' (line 183)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), 'run', run)

# Call to run(...): (line 189)
# Processing the call keyword arguments (line 189)
kwargs_471 = {}
# Getting the type of 'run' (line 189)
run_470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 0), 'run', False)
# Calling run(args, kwargs) (line 189)
run_call_result_472 = invoke(stypy.reporting.localization.Localization(__file__, 189, 0), run_470, *[], **kwargs_471)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

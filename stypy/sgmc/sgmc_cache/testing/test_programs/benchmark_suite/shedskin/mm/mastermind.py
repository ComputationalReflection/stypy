
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import game
2: import code
3: import peg
4: import colour
5: import re
6: import sys
7: 
8: ''' copyright Sean McCarthy, license GPL v2 or later '''
9: 
10: class Mastermind:
11: 
12:     '''
13:         The game of Mastermind
14: 
15:         This class provides a function "play" for invoking a new game.
16: 
17:         The objective of the game is to guess a code composed of 4 coloured
18:         pegs. The code can be composed of any combination of the six colours
19:         (red, green, purple, yellow, white, black), and can include duplicates.
20: 
21:         For each guess a result code may be returned composed of black and/or
22:         white pegs. A black peg indicates a peg of the right colour and in the
23:         right position. A white peg indicates a peg of the right colour but in
24:         the wrong position. The arrangement of the pegs does not correspond to
25:         the pegs in the guess- black pegs will always be shown first, followed
26:         but white pegs.
27: 
28:         The game ends with either a correct guess or after running out of 
29:         guesses.
30:     '''
31: 
32:     def play(self,guesses=8):
33:         self.greeting()
34:         gm = game.Game(guesses)
35:         while not gm.isOver():
36: ##            print "Guess: ",gm.getTries()+1,"/",gm.getMaxTries()
37:             gm.makeGuess(self.__readGuess())
38: ##            print "--------Board--------"
39:             self.display(gm)
40: ##            print "---------------------"
41: 	
42:         if gm.isWon():
43: ##            print "Congratulations!"
44:             pass
45:         else:
46: ##            print "Secret Code: ",
47:             self.displaySecret(gm)
48: 
49:     def greeting(self):
50:         pass
51: ##        print ""
52: ##        print "---------------------"
53: ##        print "Welcome to Mastermind"
54: ##        print "---------------------"
55: ##        print ""
56: ##        print "Each guess should be 4 colours from any of:"
57: ##        print "red yellow green purple black white"
58: ##        print ""
59: 
60:     def display(self,game):
61:         for r in game.getBoard().getRows():
62:             for p in r.getGuess().getPegs():
63:                 pass#print str(colour.getColourName(p.getColour())).rjust(6),
64: ##            print " | Result:\t",
65:             for p in r.getResult().getPegs():
66:                 pass#print str(colour.getColourName(p.getColour())).rjust(6),
67: ##            print ""
68: 
69:     def displaySecret(self,game):
70:         for p in game.getSecretCode().getPegs():
71:             pass#print str(colour.getColourName(p.getColour())).rjust(6),
72: 
73:     fakeInput = ["r y p g", "r y p g", "r y p g", "r y p g", "r y p g", "r y p g", "r y p g", "r y p g", "r y p g", "r y p g", "r y p g", "r y p g"]
74:     
75:     def __readGuess(self):
76:         guessPegs = []
77: ##        print "Type four (space seperated) colours from:"
78: ##        print "[r]ed [y]ellow [g]reen [p]urple [b]lack [w]hite"
79: 
80:         inputOk = False
81:         fakeInputIndex = 0
82:         while not inputOk:
83:             #re.split("\\s", raw_input("> "), 4)
84:             colours = re.split("\\s", self.fakeInput[fakeInputIndex], 4)
85:             fakeInputIndex+=1
86:             for c in colours:
87:                 peg = self.__parseColour(c)
88:                 if peg is not None:
89:                     guessPegs.append(peg)
90:             inputOk = (len(guessPegs) == 4)
91:             if not inputOk:
92: ##                print "Invalid input, use colours as stated"
93:                 guessPegs = []
94:         return code.Code(guessPegs)
95: 
96:     def __parseColour(self,s):
97: ##        print s
98:         if (re.search("^r",s) is not None):
99:             return peg.Peg(colour.Colours.red)
100:         elif (re.search("^p",s) is not None):
101:             return peg.Peg(colour.Colours.purple)
102:         elif (re.search("^g",s) is not None):
103:             return peg.Peg(colour.Colours.green)
104:         elif (re.search("^y",s) is not None):
105:             return peg.Peg(colour.Colours.yellow)
106:         elif (re.search("^w",s) is not None):
107:             return peg.Peg(colour.Colours.white)
108:         elif (re.search("^b",s) is not None):
109:             return peg.Peg(colour.Colours.black)
110:         else:
111:             return None
112: 
113: '''
114:     Instantiate mastermind and invoke play method to play game
115: 
116: '''
117: 
118: def main():
119:     m = Mastermind()
120:     guesses = 8
121:     if len(sys.argv) > 1 and re.match("\d", sys.argv[1]) is not None:
122:         guesses = int(sys.argv[1])
123:     m.play(guesses)
124: 
125: ##if __name__ == "__main__":
126: ##    main()
127: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import game' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/mm/')
import_430 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'game')

if (type(import_430) is not StypyTypeError):

    if (import_430 != 'pyd_module'):
        __import__(import_430)
        sys_modules_431 = sys.modules[import_430]
        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'game', sys_modules_431.module_type_store, module_type_store)
    else:
        import game

        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'game', game, module_type_store)

else:
    # Assigning a type to the variable 'game' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'game', import_430)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/mm/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import code' statement (line 2)
import code

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'code', code, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import peg' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/mm/')
import_432 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'peg')

if (type(import_432) is not StypyTypeError):

    if (import_432 != 'pyd_module'):
        __import__(import_432)
        sys_modules_433 = sys.modules[import_432]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'peg', sys_modules_433.module_type_store, module_type_store)
    else:
        import peg

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'peg', peg, module_type_store)

else:
    # Assigning a type to the variable 'peg' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'peg', import_432)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/mm/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import colour' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/mm/')
import_434 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'colour')

if (type(import_434) is not StypyTypeError):

    if (import_434 != 'pyd_module'):
        __import__(import_434)
        sys_modules_435 = sys.modules[import_434]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'colour', sys_modules_435.module_type_store, module_type_store)
    else:
        import colour

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'colour', colour, module_type_store)

else:
    # Assigning a type to the variable 'colour' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'colour', import_434)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/mm/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import re' statement (line 5)
import re

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import sys' statement (line 6)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'sys', sys, module_type_store)

str_436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 0), 'str', ' copyright Sean McCarthy, license GPL v2 or later ')
# Declaration of the 'Mastermind' class

class Mastermind:
    str_437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, (-1)), 'str', '\n        The game of Mastermind\n\n        This class provides a function "play" for invoking a new game.\n\n        The objective of the game is to guess a code composed of 4 coloured\n        pegs. The code can be composed of any combination of the six colours\n        (red, green, purple, yellow, white, black), and can include duplicates.\n\n        For each guess a result code may be returned composed of black and/or\n        white pegs. A black peg indicates a peg of the right colour and in the\n        right position. A white peg indicates a peg of the right colour but in\n        the wrong position. The arrangement of the pegs does not correspond to\n        the pegs in the guess- black pegs will always be shown first, followed\n        but white pegs.\n\n        The game ends with either a correct guess or after running out of \n        guesses.\n    ')

    @norecursion
    def play(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 26), 'int')
        defaults = [int_438]
        # Create a new context for function 'play'
        module_type_store = module_type_store.open_function_context('play', 32, 4, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Mastermind.play.__dict__.__setitem__('stypy_localization', localization)
        Mastermind.play.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Mastermind.play.__dict__.__setitem__('stypy_type_store', module_type_store)
        Mastermind.play.__dict__.__setitem__('stypy_function_name', 'Mastermind.play')
        Mastermind.play.__dict__.__setitem__('stypy_param_names_list', ['guesses'])
        Mastermind.play.__dict__.__setitem__('stypy_varargs_param_name', None)
        Mastermind.play.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Mastermind.play.__dict__.__setitem__('stypy_call_defaults', defaults)
        Mastermind.play.__dict__.__setitem__('stypy_call_varargs', varargs)
        Mastermind.play.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Mastermind.play.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Mastermind.play', ['guesses'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'play', localization, ['guesses'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'play(...)' code ##################

        
        # Call to greeting(...): (line 33)
        # Processing the call keyword arguments (line 33)
        kwargs_441 = {}
        # Getting the type of 'self' (line 33)
        self_439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self', False)
        # Obtaining the member 'greeting' of a type (line 33)
        greeting_440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_439, 'greeting')
        # Calling greeting(args, kwargs) (line 33)
        greeting_call_result_442 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), greeting_440, *[], **kwargs_441)
        
        
        # Assigning a Call to a Name (line 34):
        
        # Call to Game(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'guesses' (line 34)
        guesses_445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 23), 'guesses', False)
        # Processing the call keyword arguments (line 34)
        kwargs_446 = {}
        # Getting the type of 'game' (line 34)
        game_443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 13), 'game', False)
        # Obtaining the member 'Game' of a type (line 34)
        Game_444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 13), game_443, 'Game')
        # Calling Game(args, kwargs) (line 34)
        Game_call_result_447 = invoke(stypy.reporting.localization.Localization(__file__, 34, 13), Game_444, *[guesses_445], **kwargs_446)
        
        # Assigning a type to the variable 'gm' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'gm', Game_call_result_447)
        
        
        
        # Call to isOver(...): (line 35)
        # Processing the call keyword arguments (line 35)
        kwargs_450 = {}
        # Getting the type of 'gm' (line 35)
        gm_448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 18), 'gm', False)
        # Obtaining the member 'isOver' of a type (line 35)
        isOver_449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 18), gm_448, 'isOver')
        # Calling isOver(args, kwargs) (line 35)
        isOver_call_result_451 = invoke(stypy.reporting.localization.Localization(__file__, 35, 18), isOver_449, *[], **kwargs_450)
        
        # Applying the 'not' unary operator (line 35)
        result_not__452 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 14), 'not', isOver_call_result_451)
        
        # Assigning a type to the variable 'result_not__452' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'result_not__452', result_not__452)
        # Testing if the while is going to be iterated (line 35)
        # Testing the type of an if condition (line 35)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 8), result_not__452)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 35, 8), result_not__452):
            # SSA begins for while statement (line 35)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Call to makeGuess(...): (line 37)
            # Processing the call arguments (line 37)
            
            # Call to __readGuess(...): (line 37)
            # Processing the call keyword arguments (line 37)
            kwargs_457 = {}
            # Getting the type of 'self' (line 37)
            self_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 25), 'self', False)
            # Obtaining the member '__readGuess' of a type (line 37)
            readGuess_456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 25), self_455, '__readGuess')
            # Calling __readGuess(args, kwargs) (line 37)
            readGuess_call_result_458 = invoke(stypy.reporting.localization.Localization(__file__, 37, 25), readGuess_456, *[], **kwargs_457)
            
            # Processing the call keyword arguments (line 37)
            kwargs_459 = {}
            # Getting the type of 'gm' (line 37)
            gm_453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'gm', False)
            # Obtaining the member 'makeGuess' of a type (line 37)
            makeGuess_454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 12), gm_453, 'makeGuess')
            # Calling makeGuess(args, kwargs) (line 37)
            makeGuess_call_result_460 = invoke(stypy.reporting.localization.Localization(__file__, 37, 12), makeGuess_454, *[readGuess_call_result_458], **kwargs_459)
            
            
            # Call to display(...): (line 39)
            # Processing the call arguments (line 39)
            # Getting the type of 'gm' (line 39)
            gm_463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 25), 'gm', False)
            # Processing the call keyword arguments (line 39)
            kwargs_464 = {}
            # Getting the type of 'self' (line 39)
            self_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'self', False)
            # Obtaining the member 'display' of a type (line 39)
            display_462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 12), self_461, 'display')
            # Calling display(args, kwargs) (line 39)
            display_call_result_465 = invoke(stypy.reporting.localization.Localization(__file__, 39, 12), display_462, *[gm_463], **kwargs_464)
            
            # SSA join for while statement (line 35)
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to isWon(...): (line 42)
        # Processing the call keyword arguments (line 42)
        kwargs_468 = {}
        # Getting the type of 'gm' (line 42)
        gm_466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'gm', False)
        # Obtaining the member 'isWon' of a type (line 42)
        isWon_467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 11), gm_466, 'isWon')
        # Calling isWon(args, kwargs) (line 42)
        isWon_call_result_469 = invoke(stypy.reporting.localization.Localization(__file__, 42, 11), isWon_467, *[], **kwargs_468)
        
        # Testing if the type of an if condition is none (line 42)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 42, 8), isWon_call_result_469):
            
            # Call to displaySecret(...): (line 47)
            # Processing the call arguments (line 47)
            # Getting the type of 'gm' (line 47)
            gm_473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 31), 'gm', False)
            # Processing the call keyword arguments (line 47)
            kwargs_474 = {}
            # Getting the type of 'self' (line 47)
            self_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'self', False)
            # Obtaining the member 'displaySecret' of a type (line 47)
            displaySecret_472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), self_471, 'displaySecret')
            # Calling displaySecret(args, kwargs) (line 47)
            displaySecret_call_result_475 = invoke(stypy.reporting.localization.Localization(__file__, 47, 12), displaySecret_472, *[gm_473], **kwargs_474)
            
        else:
            
            # Testing the type of an if condition (line 42)
            if_condition_470 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 8), isWon_call_result_469)
            # Assigning a type to the variable 'if_condition_470' (line 42)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'if_condition_470', if_condition_470)
            # SSA begins for if statement (line 42)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            pass
            # SSA branch for the else part of an if statement (line 42)
            module_type_store.open_ssa_branch('else')
            
            # Call to displaySecret(...): (line 47)
            # Processing the call arguments (line 47)
            # Getting the type of 'gm' (line 47)
            gm_473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 31), 'gm', False)
            # Processing the call keyword arguments (line 47)
            kwargs_474 = {}
            # Getting the type of 'self' (line 47)
            self_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'self', False)
            # Obtaining the member 'displaySecret' of a type (line 47)
            displaySecret_472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), self_471, 'displaySecret')
            # Calling displaySecret(args, kwargs) (line 47)
            displaySecret_call_result_475 = invoke(stypy.reporting.localization.Localization(__file__, 47, 12), displaySecret_472, *[gm_473], **kwargs_474)
            
            # SSA join for if statement (line 42)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'play(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'play' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_476)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'play'
        return stypy_return_type_476


    @norecursion
    def greeting(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'greeting'
        module_type_store = module_type_store.open_function_context('greeting', 49, 4, False)
        # Assigning a type to the variable 'self' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Mastermind.greeting.__dict__.__setitem__('stypy_localization', localization)
        Mastermind.greeting.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Mastermind.greeting.__dict__.__setitem__('stypy_type_store', module_type_store)
        Mastermind.greeting.__dict__.__setitem__('stypy_function_name', 'Mastermind.greeting')
        Mastermind.greeting.__dict__.__setitem__('stypy_param_names_list', [])
        Mastermind.greeting.__dict__.__setitem__('stypy_varargs_param_name', None)
        Mastermind.greeting.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Mastermind.greeting.__dict__.__setitem__('stypy_call_defaults', defaults)
        Mastermind.greeting.__dict__.__setitem__('stypy_call_varargs', varargs)
        Mastermind.greeting.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Mastermind.greeting.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Mastermind.greeting', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'greeting', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'greeting(...)' code ##################

        pass
        
        # ################# End of 'greeting(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'greeting' in the type store
        # Getting the type of 'stypy_return_type' (line 49)
        stypy_return_type_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_477)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'greeting'
        return stypy_return_type_477


    @norecursion
    def display(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'display'
        module_type_store = module_type_store.open_function_context('display', 60, 4, False)
        # Assigning a type to the variable 'self' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Mastermind.display.__dict__.__setitem__('stypy_localization', localization)
        Mastermind.display.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Mastermind.display.__dict__.__setitem__('stypy_type_store', module_type_store)
        Mastermind.display.__dict__.__setitem__('stypy_function_name', 'Mastermind.display')
        Mastermind.display.__dict__.__setitem__('stypy_param_names_list', ['game'])
        Mastermind.display.__dict__.__setitem__('stypy_varargs_param_name', None)
        Mastermind.display.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Mastermind.display.__dict__.__setitem__('stypy_call_defaults', defaults)
        Mastermind.display.__dict__.__setitem__('stypy_call_varargs', varargs)
        Mastermind.display.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Mastermind.display.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Mastermind.display', ['game'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'display', localization, ['game'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'display(...)' code ##################

        
        
        # Call to getRows(...): (line 61)
        # Processing the call keyword arguments (line 61)
        kwargs_483 = {}
        
        # Call to getBoard(...): (line 61)
        # Processing the call keyword arguments (line 61)
        kwargs_480 = {}
        # Getting the type of 'game' (line 61)
        game_478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 17), 'game', False)
        # Obtaining the member 'getBoard' of a type (line 61)
        getBoard_479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 17), game_478, 'getBoard')
        # Calling getBoard(args, kwargs) (line 61)
        getBoard_call_result_481 = invoke(stypy.reporting.localization.Localization(__file__, 61, 17), getBoard_479, *[], **kwargs_480)
        
        # Obtaining the member 'getRows' of a type (line 61)
        getRows_482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 17), getBoard_call_result_481, 'getRows')
        # Calling getRows(args, kwargs) (line 61)
        getRows_call_result_484 = invoke(stypy.reporting.localization.Localization(__file__, 61, 17), getRows_482, *[], **kwargs_483)
        
        # Assigning a type to the variable 'getRows_call_result_484' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'getRows_call_result_484', getRows_call_result_484)
        # Testing if the for loop is going to be iterated (line 61)
        # Testing the type of a for loop iterable (line 61)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 61, 8), getRows_call_result_484)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 61, 8), getRows_call_result_484):
            # Getting the type of the for loop variable (line 61)
            for_loop_var_485 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 61, 8), getRows_call_result_484)
            # Assigning a type to the variable 'r' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'r', for_loop_var_485)
            # SSA begins for a for statement (line 61)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to getPegs(...): (line 62)
            # Processing the call keyword arguments (line 62)
            kwargs_491 = {}
            
            # Call to getGuess(...): (line 62)
            # Processing the call keyword arguments (line 62)
            kwargs_488 = {}
            # Getting the type of 'r' (line 62)
            r_486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 21), 'r', False)
            # Obtaining the member 'getGuess' of a type (line 62)
            getGuess_487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 21), r_486, 'getGuess')
            # Calling getGuess(args, kwargs) (line 62)
            getGuess_call_result_489 = invoke(stypy.reporting.localization.Localization(__file__, 62, 21), getGuess_487, *[], **kwargs_488)
            
            # Obtaining the member 'getPegs' of a type (line 62)
            getPegs_490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 21), getGuess_call_result_489, 'getPegs')
            # Calling getPegs(args, kwargs) (line 62)
            getPegs_call_result_492 = invoke(stypy.reporting.localization.Localization(__file__, 62, 21), getPegs_490, *[], **kwargs_491)
            
            # Assigning a type to the variable 'getPegs_call_result_492' (line 62)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'getPegs_call_result_492', getPegs_call_result_492)
            # Testing if the for loop is going to be iterated (line 62)
            # Testing the type of a for loop iterable (line 62)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 62, 12), getPegs_call_result_492)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 62, 12), getPegs_call_result_492):
                # Getting the type of the for loop variable (line 62)
                for_loop_var_493 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 62, 12), getPegs_call_result_492)
                # Assigning a type to the variable 'p' (line 62)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'p', for_loop_var_493)
                # SSA begins for a for statement (line 62)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                pass
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            
            # Call to getPegs(...): (line 65)
            # Processing the call keyword arguments (line 65)
            kwargs_499 = {}
            
            # Call to getResult(...): (line 65)
            # Processing the call keyword arguments (line 65)
            kwargs_496 = {}
            # Getting the type of 'r' (line 65)
            r_494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 21), 'r', False)
            # Obtaining the member 'getResult' of a type (line 65)
            getResult_495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 21), r_494, 'getResult')
            # Calling getResult(args, kwargs) (line 65)
            getResult_call_result_497 = invoke(stypy.reporting.localization.Localization(__file__, 65, 21), getResult_495, *[], **kwargs_496)
            
            # Obtaining the member 'getPegs' of a type (line 65)
            getPegs_498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 21), getResult_call_result_497, 'getPegs')
            # Calling getPegs(args, kwargs) (line 65)
            getPegs_call_result_500 = invoke(stypy.reporting.localization.Localization(__file__, 65, 21), getPegs_498, *[], **kwargs_499)
            
            # Assigning a type to the variable 'getPegs_call_result_500' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'getPegs_call_result_500', getPegs_call_result_500)
            # Testing if the for loop is going to be iterated (line 65)
            # Testing the type of a for loop iterable (line 65)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 65, 12), getPegs_call_result_500)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 65, 12), getPegs_call_result_500):
                # Getting the type of the for loop variable (line 65)
                for_loop_var_501 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 65, 12), getPegs_call_result_500)
                # Assigning a type to the variable 'p' (line 65)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'p', for_loop_var_501)
                # SSA begins for a for statement (line 65)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                pass
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'display(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'display' in the type store
        # Getting the type of 'stypy_return_type' (line 60)
        stypy_return_type_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_502)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'display'
        return stypy_return_type_502


    @norecursion
    def displaySecret(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'displaySecret'
        module_type_store = module_type_store.open_function_context('displaySecret', 69, 4, False)
        # Assigning a type to the variable 'self' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Mastermind.displaySecret.__dict__.__setitem__('stypy_localization', localization)
        Mastermind.displaySecret.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Mastermind.displaySecret.__dict__.__setitem__('stypy_type_store', module_type_store)
        Mastermind.displaySecret.__dict__.__setitem__('stypy_function_name', 'Mastermind.displaySecret')
        Mastermind.displaySecret.__dict__.__setitem__('stypy_param_names_list', ['game'])
        Mastermind.displaySecret.__dict__.__setitem__('stypy_varargs_param_name', None)
        Mastermind.displaySecret.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Mastermind.displaySecret.__dict__.__setitem__('stypy_call_defaults', defaults)
        Mastermind.displaySecret.__dict__.__setitem__('stypy_call_varargs', varargs)
        Mastermind.displaySecret.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Mastermind.displaySecret.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Mastermind.displaySecret', ['game'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'displaySecret', localization, ['game'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'displaySecret(...)' code ##################

        
        
        # Call to getPegs(...): (line 70)
        # Processing the call keyword arguments (line 70)
        kwargs_508 = {}
        
        # Call to getSecretCode(...): (line 70)
        # Processing the call keyword arguments (line 70)
        kwargs_505 = {}
        # Getting the type of 'game' (line 70)
        game_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 17), 'game', False)
        # Obtaining the member 'getSecretCode' of a type (line 70)
        getSecretCode_504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 17), game_503, 'getSecretCode')
        # Calling getSecretCode(args, kwargs) (line 70)
        getSecretCode_call_result_506 = invoke(stypy.reporting.localization.Localization(__file__, 70, 17), getSecretCode_504, *[], **kwargs_505)
        
        # Obtaining the member 'getPegs' of a type (line 70)
        getPegs_507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 17), getSecretCode_call_result_506, 'getPegs')
        # Calling getPegs(args, kwargs) (line 70)
        getPegs_call_result_509 = invoke(stypy.reporting.localization.Localization(__file__, 70, 17), getPegs_507, *[], **kwargs_508)
        
        # Assigning a type to the variable 'getPegs_call_result_509' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'getPegs_call_result_509', getPegs_call_result_509)
        # Testing if the for loop is going to be iterated (line 70)
        # Testing the type of a for loop iterable (line 70)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 70, 8), getPegs_call_result_509)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 70, 8), getPegs_call_result_509):
            # Getting the type of the for loop variable (line 70)
            for_loop_var_510 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 70, 8), getPegs_call_result_509)
            # Assigning a type to the variable 'p' (line 70)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'p', for_loop_var_510)
            # SSA begins for a for statement (line 70)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            pass
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'displaySecret(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'displaySecret' in the type store
        # Getting the type of 'stypy_return_type' (line 69)
        stypy_return_type_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_511)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'displaySecret'
        return stypy_return_type_511


    @norecursion
    def __readGuess(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__readGuess'
        module_type_store = module_type_store.open_function_context('__readGuess', 75, 4, False)
        # Assigning a type to the variable 'self' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Mastermind.__readGuess.__dict__.__setitem__('stypy_localization', localization)
        Mastermind.__readGuess.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Mastermind.__readGuess.__dict__.__setitem__('stypy_type_store', module_type_store)
        Mastermind.__readGuess.__dict__.__setitem__('stypy_function_name', 'Mastermind.__readGuess')
        Mastermind.__readGuess.__dict__.__setitem__('stypy_param_names_list', [])
        Mastermind.__readGuess.__dict__.__setitem__('stypy_varargs_param_name', None)
        Mastermind.__readGuess.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Mastermind.__readGuess.__dict__.__setitem__('stypy_call_defaults', defaults)
        Mastermind.__readGuess.__dict__.__setitem__('stypy_call_varargs', varargs)
        Mastermind.__readGuess.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Mastermind.__readGuess.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Mastermind.__readGuess', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__readGuess', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__readGuess(...)' code ##################

        
        # Assigning a List to a Name (line 76):
        
        # Obtaining an instance of the builtin type 'list' (line 76)
        list_512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 76)
        
        # Assigning a type to the variable 'guessPegs' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'guessPegs', list_512)
        
        # Assigning a Name to a Name (line 80):
        # Getting the type of 'False' (line 80)
        False_513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 18), 'False')
        # Assigning a type to the variable 'inputOk' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'inputOk', False_513)
        
        # Assigning a Num to a Name (line 81):
        int_514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 25), 'int')
        # Assigning a type to the variable 'fakeInputIndex' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'fakeInputIndex', int_514)
        
        
        # Getting the type of 'inputOk' (line 82)
        inputOk_515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 18), 'inputOk')
        # Applying the 'not' unary operator (line 82)
        result_not__516 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 14), 'not', inputOk_515)
        
        # Assigning a type to the variable 'result_not__516' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'result_not__516', result_not__516)
        # Testing if the while is going to be iterated (line 82)
        # Testing the type of an if condition (line 82)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 8), result_not__516)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 82, 8), result_not__516):
            # SSA begins for while statement (line 82)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a Call to a Name (line 84):
            
            # Call to split(...): (line 84)
            # Processing the call arguments (line 84)
            str_519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 31), 'str', '\\s')
            
            # Obtaining the type of the subscript
            # Getting the type of 'fakeInputIndex' (line 84)
            fakeInputIndex_520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 53), 'fakeInputIndex', False)
            # Getting the type of 'self' (line 84)
            self_521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 38), 'self', False)
            # Obtaining the member 'fakeInput' of a type (line 84)
            fakeInput_522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 38), self_521, 'fakeInput')
            # Obtaining the member '__getitem__' of a type (line 84)
            getitem___523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 38), fakeInput_522, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 84)
            subscript_call_result_524 = invoke(stypy.reporting.localization.Localization(__file__, 84, 38), getitem___523, fakeInputIndex_520)
            
            int_525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 70), 'int')
            # Processing the call keyword arguments (line 84)
            kwargs_526 = {}
            # Getting the type of 're' (line 84)
            re_517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 22), 're', False)
            # Obtaining the member 'split' of a type (line 84)
            split_518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 22), re_517, 'split')
            # Calling split(args, kwargs) (line 84)
            split_call_result_527 = invoke(stypy.reporting.localization.Localization(__file__, 84, 22), split_518, *[str_519, subscript_call_result_524, int_525], **kwargs_526)
            
            # Assigning a type to the variable 'colours' (line 84)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'colours', split_call_result_527)
            
            # Getting the type of 'fakeInputIndex' (line 85)
            fakeInputIndex_528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'fakeInputIndex')
            int_529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 28), 'int')
            # Applying the binary operator '+=' (line 85)
            result_iadd_530 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 12), '+=', fakeInputIndex_528, int_529)
            # Assigning a type to the variable 'fakeInputIndex' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'fakeInputIndex', result_iadd_530)
            
            
            # Getting the type of 'colours' (line 86)
            colours_531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 21), 'colours')
            # Assigning a type to the variable 'colours_531' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'colours_531', colours_531)
            # Testing if the for loop is going to be iterated (line 86)
            # Testing the type of a for loop iterable (line 86)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 86, 12), colours_531)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 86, 12), colours_531):
                # Getting the type of the for loop variable (line 86)
                for_loop_var_532 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 86, 12), colours_531)
                # Assigning a type to the variable 'c' (line 86)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'c', for_loop_var_532)
                # SSA begins for a for statement (line 86)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Call to a Name (line 87):
                
                # Call to __parseColour(...): (line 87)
                # Processing the call arguments (line 87)
                # Getting the type of 'c' (line 87)
                c_535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 41), 'c', False)
                # Processing the call keyword arguments (line 87)
                kwargs_536 = {}
                # Getting the type of 'self' (line 87)
                self_533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 22), 'self', False)
                # Obtaining the member '__parseColour' of a type (line 87)
                parseColour_534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 22), self_533, '__parseColour')
                # Calling __parseColour(args, kwargs) (line 87)
                parseColour_call_result_537 = invoke(stypy.reporting.localization.Localization(__file__, 87, 22), parseColour_534, *[c_535], **kwargs_536)
                
                # Assigning a type to the variable 'peg' (line 87)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'peg', parseColour_call_result_537)
                
                # Type idiom detected: calculating its left and rigth part (line 88)
                # Getting the type of 'peg' (line 88)
                peg_538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'peg')
                # Getting the type of 'None' (line 88)
                None_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 30), 'None')
                
                (may_be_540, more_types_in_union_541) = may_not_be_none(peg_538, None_539)

                if may_be_540:

                    if more_types_in_union_541:
                        # Runtime conditional SSA (line 88)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    
                    # Call to append(...): (line 89)
                    # Processing the call arguments (line 89)
                    # Getting the type of 'peg' (line 89)
                    peg_544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 37), 'peg', False)
                    # Processing the call keyword arguments (line 89)
                    kwargs_545 = {}
                    # Getting the type of 'guessPegs' (line 89)
                    guessPegs_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 20), 'guessPegs', False)
                    # Obtaining the member 'append' of a type (line 89)
                    append_543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 20), guessPegs_542, 'append')
                    # Calling append(args, kwargs) (line 89)
                    append_call_result_546 = invoke(stypy.reporting.localization.Localization(__file__, 89, 20), append_543, *[peg_544], **kwargs_545)
                    

                    if more_types_in_union_541:
                        # SSA join for if statement (line 88)
                        module_type_store = module_type_store.join_ssa_context()


                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a Compare to a Name (line 90):
            
            
            # Call to len(...): (line 90)
            # Processing the call arguments (line 90)
            # Getting the type of 'guessPegs' (line 90)
            guessPegs_548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 27), 'guessPegs', False)
            # Processing the call keyword arguments (line 90)
            kwargs_549 = {}
            # Getting the type of 'len' (line 90)
            len_547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 23), 'len', False)
            # Calling len(args, kwargs) (line 90)
            len_call_result_550 = invoke(stypy.reporting.localization.Localization(__file__, 90, 23), len_547, *[guessPegs_548], **kwargs_549)
            
            int_551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 41), 'int')
            # Applying the binary operator '==' (line 90)
            result_eq_552 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 23), '==', len_call_result_550, int_551)
            
            # Assigning a type to the variable 'inputOk' (line 90)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'inputOk', result_eq_552)
            
            # Getting the type of 'inputOk' (line 91)
            inputOk_553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 19), 'inputOk')
            # Applying the 'not' unary operator (line 91)
            result_not__554 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 15), 'not', inputOk_553)
            
            # Testing if the type of an if condition is none (line 91)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 91, 12), result_not__554):
                pass
            else:
                
                # Testing the type of an if condition (line 91)
                if_condition_555 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 12), result_not__554)
                # Assigning a type to the variable 'if_condition_555' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'if_condition_555', if_condition_555)
                # SSA begins for if statement (line 91)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a List to a Name (line 93):
                
                # Obtaining an instance of the builtin type 'list' (line 93)
                list_556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 28), 'list')
                # Adding type elements to the builtin type 'list' instance (line 93)
                
                # Assigning a type to the variable 'guessPegs' (line 93)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'guessPegs', list_556)
                # SSA join for if statement (line 91)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for while statement (line 82)
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to Code(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'guessPegs' (line 94)
        guessPegs_559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 25), 'guessPegs', False)
        # Processing the call keyword arguments (line 94)
        kwargs_560 = {}
        # Getting the type of 'code' (line 94)
        code_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 15), 'code', False)
        # Obtaining the member 'Code' of a type (line 94)
        Code_558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 15), code_557, 'Code')
        # Calling Code(args, kwargs) (line 94)
        Code_call_result_561 = invoke(stypy.reporting.localization.Localization(__file__, 94, 15), Code_558, *[guessPegs_559], **kwargs_560)
        
        # Assigning a type to the variable 'stypy_return_type' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'stypy_return_type', Code_call_result_561)
        
        # ################# End of '__readGuess(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__readGuess' in the type store
        # Getting the type of 'stypy_return_type' (line 75)
        stypy_return_type_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_562)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__readGuess'
        return stypy_return_type_562


    @norecursion
    def __parseColour(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__parseColour'
        module_type_store = module_type_store.open_function_context('__parseColour', 96, 4, False)
        # Assigning a type to the variable 'self' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Mastermind.__parseColour.__dict__.__setitem__('stypy_localization', localization)
        Mastermind.__parseColour.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Mastermind.__parseColour.__dict__.__setitem__('stypy_type_store', module_type_store)
        Mastermind.__parseColour.__dict__.__setitem__('stypy_function_name', 'Mastermind.__parseColour')
        Mastermind.__parseColour.__dict__.__setitem__('stypy_param_names_list', ['s'])
        Mastermind.__parseColour.__dict__.__setitem__('stypy_varargs_param_name', None)
        Mastermind.__parseColour.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Mastermind.__parseColour.__dict__.__setitem__('stypy_call_defaults', defaults)
        Mastermind.__parseColour.__dict__.__setitem__('stypy_call_varargs', varargs)
        Mastermind.__parseColour.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Mastermind.__parseColour.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Mastermind.__parseColour', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__parseColour', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__parseColour(...)' code ##################

        
        
        # Call to search(...): (line 98)
        # Processing the call arguments (line 98)
        str_565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 22), 'str', '^r')
        # Getting the type of 's' (line 98)
        s_566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 27), 's', False)
        # Processing the call keyword arguments (line 98)
        kwargs_567 = {}
        # Getting the type of 're' (line 98)
        re_563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 're', False)
        # Obtaining the member 'search' of a type (line 98)
        search_564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 12), re_563, 'search')
        # Calling search(args, kwargs) (line 98)
        search_call_result_568 = invoke(stypy.reporting.localization.Localization(__file__, 98, 12), search_564, *[str_565, s_566], **kwargs_567)
        
        # Getting the type of 'None' (line 98)
        None_569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 37), 'None')
        # Applying the binary operator 'isnot' (line 98)
        result_is_not_570 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 12), 'isnot', search_call_result_568, None_569)
        
        # Testing if the type of an if condition is none (line 98)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 98, 8), result_is_not_570):
            
            
            # Call to search(...): (line 100)
            # Processing the call arguments (line 100)
            str_581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 24), 'str', '^p')
            # Getting the type of 's' (line 100)
            s_582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 29), 's', False)
            # Processing the call keyword arguments (line 100)
            kwargs_583 = {}
            # Getting the type of 're' (line 100)
            re_579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 14), 're', False)
            # Obtaining the member 'search' of a type (line 100)
            search_580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 14), re_579, 'search')
            # Calling search(args, kwargs) (line 100)
            search_call_result_584 = invoke(stypy.reporting.localization.Localization(__file__, 100, 14), search_580, *[str_581, s_582], **kwargs_583)
            
            # Getting the type of 'None' (line 100)
            None_585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 39), 'None')
            # Applying the binary operator 'isnot' (line 100)
            result_is_not_586 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 14), 'isnot', search_call_result_584, None_585)
            
            # Testing if the type of an if condition is none (line 100)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 100, 13), result_is_not_586):
                
                
                # Call to search(...): (line 102)
                # Processing the call arguments (line 102)
                str_597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 24), 'str', '^g')
                # Getting the type of 's' (line 102)
                s_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 29), 's', False)
                # Processing the call keyword arguments (line 102)
                kwargs_599 = {}
                # Getting the type of 're' (line 102)
                re_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 14), 're', False)
                # Obtaining the member 'search' of a type (line 102)
                search_596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 14), re_595, 'search')
                # Calling search(args, kwargs) (line 102)
                search_call_result_600 = invoke(stypy.reporting.localization.Localization(__file__, 102, 14), search_596, *[str_597, s_598], **kwargs_599)
                
                # Getting the type of 'None' (line 102)
                None_601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 39), 'None')
                # Applying the binary operator 'isnot' (line 102)
                result_is_not_602 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 14), 'isnot', search_call_result_600, None_601)
                
                # Testing if the type of an if condition is none (line 102)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 102, 13), result_is_not_602):
                    
                    
                    # Call to search(...): (line 104)
                    # Processing the call arguments (line 104)
                    str_613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 24), 'str', '^y')
                    # Getting the type of 's' (line 104)
                    s_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 29), 's', False)
                    # Processing the call keyword arguments (line 104)
                    kwargs_615 = {}
                    # Getting the type of 're' (line 104)
                    re_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 14), 're', False)
                    # Obtaining the member 'search' of a type (line 104)
                    search_612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 14), re_611, 'search')
                    # Calling search(args, kwargs) (line 104)
                    search_call_result_616 = invoke(stypy.reporting.localization.Localization(__file__, 104, 14), search_612, *[str_613, s_614], **kwargs_615)
                    
                    # Getting the type of 'None' (line 104)
                    None_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 39), 'None')
                    # Applying the binary operator 'isnot' (line 104)
                    result_is_not_618 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 14), 'isnot', search_call_result_616, None_617)
                    
                    # Testing if the type of an if condition is none (line 104)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 104, 13), result_is_not_618):
                        
                        
                        # Call to search(...): (line 106)
                        # Processing the call arguments (line 106)
                        str_629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 24), 'str', '^w')
                        # Getting the type of 's' (line 106)
                        s_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 's', False)
                        # Processing the call keyword arguments (line 106)
                        kwargs_631 = {}
                        # Getting the type of 're' (line 106)
                        re_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 14), 're', False)
                        # Obtaining the member 'search' of a type (line 106)
                        search_628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 14), re_627, 'search')
                        # Calling search(args, kwargs) (line 106)
                        search_call_result_632 = invoke(stypy.reporting.localization.Localization(__file__, 106, 14), search_628, *[str_629, s_630], **kwargs_631)
                        
                        # Getting the type of 'None' (line 106)
                        None_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 39), 'None')
                        # Applying the binary operator 'isnot' (line 106)
                        result_is_not_634 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 14), 'isnot', search_call_result_632, None_633)
                        
                        # Testing if the type of an if condition is none (line 106)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634):
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 106)
                            if_condition_635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634)
                            # Assigning a type to the variable 'if_condition_635' (line 106)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 13), 'if_condition_635', if_condition_635)
                            # SSA begins for if statement (line 106)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to Peg(...): (line 107)
                            # Processing the call arguments (line 107)
                            # Getting the type of 'colour' (line 107)
                            colour_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 27), 'colour', False)
                            # Obtaining the member 'Colours' of a type (line 107)
                            Colours_639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), colour_638, 'Colours')
                            # Obtaining the member 'white' of a type (line 107)
                            white_640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), Colours_639, 'white')
                            # Processing the call keyword arguments (line 107)
                            kwargs_641 = {}
                            # Getting the type of 'peg' (line 107)
                            peg_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'peg', False)
                            # Obtaining the member 'Peg' of a type (line 107)
                            Peg_637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 19), peg_636, 'Peg')
                            # Calling Peg(args, kwargs) (line 107)
                            Peg_call_result_642 = invoke(stypy.reporting.localization.Localization(__file__, 107, 19), Peg_637, *[white_640], **kwargs_641)
                            
                            # Assigning a type to the variable 'stypy_return_type' (line 107)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'stypy_return_type', Peg_call_result_642)
                            # SSA branch for the else part of an if statement (line 106)
                            module_type_store.open_ssa_branch('else')
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 106)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 104)
                        if_condition_619 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 13), result_is_not_618)
                        # Assigning a type to the variable 'if_condition_619' (line 104)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 13), 'if_condition_619', if_condition_619)
                        # SSA begins for if statement (line 104)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to Peg(...): (line 105)
                        # Processing the call arguments (line 105)
                        # Getting the type of 'colour' (line 105)
                        colour_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 27), 'colour', False)
                        # Obtaining the member 'Colours' of a type (line 105)
                        Colours_623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 27), colour_622, 'Colours')
                        # Obtaining the member 'yellow' of a type (line 105)
                        yellow_624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 27), Colours_623, 'yellow')
                        # Processing the call keyword arguments (line 105)
                        kwargs_625 = {}
                        # Getting the type of 'peg' (line 105)
                        peg_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 19), 'peg', False)
                        # Obtaining the member 'Peg' of a type (line 105)
                        Peg_621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 19), peg_620, 'Peg')
                        # Calling Peg(args, kwargs) (line 105)
                        Peg_call_result_626 = invoke(stypy.reporting.localization.Localization(__file__, 105, 19), Peg_621, *[yellow_624], **kwargs_625)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 105)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'stypy_return_type', Peg_call_result_626)
                        # SSA branch for the else part of an if statement (line 104)
                        module_type_store.open_ssa_branch('else')
                        
                        
                        # Call to search(...): (line 106)
                        # Processing the call arguments (line 106)
                        str_629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 24), 'str', '^w')
                        # Getting the type of 's' (line 106)
                        s_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 's', False)
                        # Processing the call keyword arguments (line 106)
                        kwargs_631 = {}
                        # Getting the type of 're' (line 106)
                        re_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 14), 're', False)
                        # Obtaining the member 'search' of a type (line 106)
                        search_628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 14), re_627, 'search')
                        # Calling search(args, kwargs) (line 106)
                        search_call_result_632 = invoke(stypy.reporting.localization.Localization(__file__, 106, 14), search_628, *[str_629, s_630], **kwargs_631)
                        
                        # Getting the type of 'None' (line 106)
                        None_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 39), 'None')
                        # Applying the binary operator 'isnot' (line 106)
                        result_is_not_634 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 14), 'isnot', search_call_result_632, None_633)
                        
                        # Testing if the type of an if condition is none (line 106)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634):
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 106)
                            if_condition_635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634)
                            # Assigning a type to the variable 'if_condition_635' (line 106)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 13), 'if_condition_635', if_condition_635)
                            # SSA begins for if statement (line 106)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to Peg(...): (line 107)
                            # Processing the call arguments (line 107)
                            # Getting the type of 'colour' (line 107)
                            colour_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 27), 'colour', False)
                            # Obtaining the member 'Colours' of a type (line 107)
                            Colours_639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), colour_638, 'Colours')
                            # Obtaining the member 'white' of a type (line 107)
                            white_640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), Colours_639, 'white')
                            # Processing the call keyword arguments (line 107)
                            kwargs_641 = {}
                            # Getting the type of 'peg' (line 107)
                            peg_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'peg', False)
                            # Obtaining the member 'Peg' of a type (line 107)
                            Peg_637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 19), peg_636, 'Peg')
                            # Calling Peg(args, kwargs) (line 107)
                            Peg_call_result_642 = invoke(stypy.reporting.localization.Localization(__file__, 107, 19), Peg_637, *[white_640], **kwargs_641)
                            
                            # Assigning a type to the variable 'stypy_return_type' (line 107)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'stypy_return_type', Peg_call_result_642)
                            # SSA branch for the else part of an if statement (line 106)
                            module_type_store.open_ssa_branch('else')
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 106)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 104)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 102)
                    if_condition_603 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 13), result_is_not_602)
                    # Assigning a type to the variable 'if_condition_603' (line 102)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 13), 'if_condition_603', if_condition_603)
                    # SSA begins for if statement (line 102)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to Peg(...): (line 103)
                    # Processing the call arguments (line 103)
                    # Getting the type of 'colour' (line 103)
                    colour_606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'colour', False)
                    # Obtaining the member 'Colours' of a type (line 103)
                    Colours_607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 27), colour_606, 'Colours')
                    # Obtaining the member 'green' of a type (line 103)
                    green_608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 27), Colours_607, 'green')
                    # Processing the call keyword arguments (line 103)
                    kwargs_609 = {}
                    # Getting the type of 'peg' (line 103)
                    peg_604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 19), 'peg', False)
                    # Obtaining the member 'Peg' of a type (line 103)
                    Peg_605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 19), peg_604, 'Peg')
                    # Calling Peg(args, kwargs) (line 103)
                    Peg_call_result_610 = invoke(stypy.reporting.localization.Localization(__file__, 103, 19), Peg_605, *[green_608], **kwargs_609)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 103)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'stypy_return_type', Peg_call_result_610)
                    # SSA branch for the else part of an if statement (line 102)
                    module_type_store.open_ssa_branch('else')
                    
                    
                    # Call to search(...): (line 104)
                    # Processing the call arguments (line 104)
                    str_613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 24), 'str', '^y')
                    # Getting the type of 's' (line 104)
                    s_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 29), 's', False)
                    # Processing the call keyword arguments (line 104)
                    kwargs_615 = {}
                    # Getting the type of 're' (line 104)
                    re_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 14), 're', False)
                    # Obtaining the member 'search' of a type (line 104)
                    search_612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 14), re_611, 'search')
                    # Calling search(args, kwargs) (line 104)
                    search_call_result_616 = invoke(stypy.reporting.localization.Localization(__file__, 104, 14), search_612, *[str_613, s_614], **kwargs_615)
                    
                    # Getting the type of 'None' (line 104)
                    None_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 39), 'None')
                    # Applying the binary operator 'isnot' (line 104)
                    result_is_not_618 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 14), 'isnot', search_call_result_616, None_617)
                    
                    # Testing if the type of an if condition is none (line 104)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 104, 13), result_is_not_618):
                        
                        
                        # Call to search(...): (line 106)
                        # Processing the call arguments (line 106)
                        str_629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 24), 'str', '^w')
                        # Getting the type of 's' (line 106)
                        s_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 's', False)
                        # Processing the call keyword arguments (line 106)
                        kwargs_631 = {}
                        # Getting the type of 're' (line 106)
                        re_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 14), 're', False)
                        # Obtaining the member 'search' of a type (line 106)
                        search_628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 14), re_627, 'search')
                        # Calling search(args, kwargs) (line 106)
                        search_call_result_632 = invoke(stypy.reporting.localization.Localization(__file__, 106, 14), search_628, *[str_629, s_630], **kwargs_631)
                        
                        # Getting the type of 'None' (line 106)
                        None_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 39), 'None')
                        # Applying the binary operator 'isnot' (line 106)
                        result_is_not_634 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 14), 'isnot', search_call_result_632, None_633)
                        
                        # Testing if the type of an if condition is none (line 106)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634):
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 106)
                            if_condition_635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634)
                            # Assigning a type to the variable 'if_condition_635' (line 106)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 13), 'if_condition_635', if_condition_635)
                            # SSA begins for if statement (line 106)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to Peg(...): (line 107)
                            # Processing the call arguments (line 107)
                            # Getting the type of 'colour' (line 107)
                            colour_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 27), 'colour', False)
                            # Obtaining the member 'Colours' of a type (line 107)
                            Colours_639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), colour_638, 'Colours')
                            # Obtaining the member 'white' of a type (line 107)
                            white_640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), Colours_639, 'white')
                            # Processing the call keyword arguments (line 107)
                            kwargs_641 = {}
                            # Getting the type of 'peg' (line 107)
                            peg_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'peg', False)
                            # Obtaining the member 'Peg' of a type (line 107)
                            Peg_637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 19), peg_636, 'Peg')
                            # Calling Peg(args, kwargs) (line 107)
                            Peg_call_result_642 = invoke(stypy.reporting.localization.Localization(__file__, 107, 19), Peg_637, *[white_640], **kwargs_641)
                            
                            # Assigning a type to the variable 'stypy_return_type' (line 107)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'stypy_return_type', Peg_call_result_642)
                            # SSA branch for the else part of an if statement (line 106)
                            module_type_store.open_ssa_branch('else')
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 106)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 104)
                        if_condition_619 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 13), result_is_not_618)
                        # Assigning a type to the variable 'if_condition_619' (line 104)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 13), 'if_condition_619', if_condition_619)
                        # SSA begins for if statement (line 104)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to Peg(...): (line 105)
                        # Processing the call arguments (line 105)
                        # Getting the type of 'colour' (line 105)
                        colour_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 27), 'colour', False)
                        # Obtaining the member 'Colours' of a type (line 105)
                        Colours_623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 27), colour_622, 'Colours')
                        # Obtaining the member 'yellow' of a type (line 105)
                        yellow_624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 27), Colours_623, 'yellow')
                        # Processing the call keyword arguments (line 105)
                        kwargs_625 = {}
                        # Getting the type of 'peg' (line 105)
                        peg_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 19), 'peg', False)
                        # Obtaining the member 'Peg' of a type (line 105)
                        Peg_621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 19), peg_620, 'Peg')
                        # Calling Peg(args, kwargs) (line 105)
                        Peg_call_result_626 = invoke(stypy.reporting.localization.Localization(__file__, 105, 19), Peg_621, *[yellow_624], **kwargs_625)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 105)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'stypy_return_type', Peg_call_result_626)
                        # SSA branch for the else part of an if statement (line 104)
                        module_type_store.open_ssa_branch('else')
                        
                        
                        # Call to search(...): (line 106)
                        # Processing the call arguments (line 106)
                        str_629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 24), 'str', '^w')
                        # Getting the type of 's' (line 106)
                        s_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 's', False)
                        # Processing the call keyword arguments (line 106)
                        kwargs_631 = {}
                        # Getting the type of 're' (line 106)
                        re_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 14), 're', False)
                        # Obtaining the member 'search' of a type (line 106)
                        search_628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 14), re_627, 'search')
                        # Calling search(args, kwargs) (line 106)
                        search_call_result_632 = invoke(stypy.reporting.localization.Localization(__file__, 106, 14), search_628, *[str_629, s_630], **kwargs_631)
                        
                        # Getting the type of 'None' (line 106)
                        None_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 39), 'None')
                        # Applying the binary operator 'isnot' (line 106)
                        result_is_not_634 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 14), 'isnot', search_call_result_632, None_633)
                        
                        # Testing if the type of an if condition is none (line 106)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634):
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 106)
                            if_condition_635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634)
                            # Assigning a type to the variable 'if_condition_635' (line 106)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 13), 'if_condition_635', if_condition_635)
                            # SSA begins for if statement (line 106)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to Peg(...): (line 107)
                            # Processing the call arguments (line 107)
                            # Getting the type of 'colour' (line 107)
                            colour_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 27), 'colour', False)
                            # Obtaining the member 'Colours' of a type (line 107)
                            Colours_639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), colour_638, 'Colours')
                            # Obtaining the member 'white' of a type (line 107)
                            white_640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), Colours_639, 'white')
                            # Processing the call keyword arguments (line 107)
                            kwargs_641 = {}
                            # Getting the type of 'peg' (line 107)
                            peg_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'peg', False)
                            # Obtaining the member 'Peg' of a type (line 107)
                            Peg_637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 19), peg_636, 'Peg')
                            # Calling Peg(args, kwargs) (line 107)
                            Peg_call_result_642 = invoke(stypy.reporting.localization.Localization(__file__, 107, 19), Peg_637, *[white_640], **kwargs_641)
                            
                            # Assigning a type to the variable 'stypy_return_type' (line 107)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'stypy_return_type', Peg_call_result_642)
                            # SSA branch for the else part of an if statement (line 106)
                            module_type_store.open_ssa_branch('else')
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 106)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 104)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 102)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 100)
                if_condition_587 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 13), result_is_not_586)
                # Assigning a type to the variable 'if_condition_587' (line 100)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 13), 'if_condition_587', if_condition_587)
                # SSA begins for if statement (line 100)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to Peg(...): (line 101)
                # Processing the call arguments (line 101)
                # Getting the type of 'colour' (line 101)
                colour_590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 27), 'colour', False)
                # Obtaining the member 'Colours' of a type (line 101)
                Colours_591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 27), colour_590, 'Colours')
                # Obtaining the member 'purple' of a type (line 101)
                purple_592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 27), Colours_591, 'purple')
                # Processing the call keyword arguments (line 101)
                kwargs_593 = {}
                # Getting the type of 'peg' (line 101)
                peg_588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 19), 'peg', False)
                # Obtaining the member 'Peg' of a type (line 101)
                Peg_589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 19), peg_588, 'Peg')
                # Calling Peg(args, kwargs) (line 101)
                Peg_call_result_594 = invoke(stypy.reporting.localization.Localization(__file__, 101, 19), Peg_589, *[purple_592], **kwargs_593)
                
                # Assigning a type to the variable 'stypy_return_type' (line 101)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'stypy_return_type', Peg_call_result_594)
                # SSA branch for the else part of an if statement (line 100)
                module_type_store.open_ssa_branch('else')
                
                
                # Call to search(...): (line 102)
                # Processing the call arguments (line 102)
                str_597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 24), 'str', '^g')
                # Getting the type of 's' (line 102)
                s_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 29), 's', False)
                # Processing the call keyword arguments (line 102)
                kwargs_599 = {}
                # Getting the type of 're' (line 102)
                re_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 14), 're', False)
                # Obtaining the member 'search' of a type (line 102)
                search_596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 14), re_595, 'search')
                # Calling search(args, kwargs) (line 102)
                search_call_result_600 = invoke(stypy.reporting.localization.Localization(__file__, 102, 14), search_596, *[str_597, s_598], **kwargs_599)
                
                # Getting the type of 'None' (line 102)
                None_601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 39), 'None')
                # Applying the binary operator 'isnot' (line 102)
                result_is_not_602 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 14), 'isnot', search_call_result_600, None_601)
                
                # Testing if the type of an if condition is none (line 102)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 102, 13), result_is_not_602):
                    
                    
                    # Call to search(...): (line 104)
                    # Processing the call arguments (line 104)
                    str_613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 24), 'str', '^y')
                    # Getting the type of 's' (line 104)
                    s_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 29), 's', False)
                    # Processing the call keyword arguments (line 104)
                    kwargs_615 = {}
                    # Getting the type of 're' (line 104)
                    re_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 14), 're', False)
                    # Obtaining the member 'search' of a type (line 104)
                    search_612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 14), re_611, 'search')
                    # Calling search(args, kwargs) (line 104)
                    search_call_result_616 = invoke(stypy.reporting.localization.Localization(__file__, 104, 14), search_612, *[str_613, s_614], **kwargs_615)
                    
                    # Getting the type of 'None' (line 104)
                    None_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 39), 'None')
                    # Applying the binary operator 'isnot' (line 104)
                    result_is_not_618 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 14), 'isnot', search_call_result_616, None_617)
                    
                    # Testing if the type of an if condition is none (line 104)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 104, 13), result_is_not_618):
                        
                        
                        # Call to search(...): (line 106)
                        # Processing the call arguments (line 106)
                        str_629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 24), 'str', '^w')
                        # Getting the type of 's' (line 106)
                        s_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 's', False)
                        # Processing the call keyword arguments (line 106)
                        kwargs_631 = {}
                        # Getting the type of 're' (line 106)
                        re_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 14), 're', False)
                        # Obtaining the member 'search' of a type (line 106)
                        search_628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 14), re_627, 'search')
                        # Calling search(args, kwargs) (line 106)
                        search_call_result_632 = invoke(stypy.reporting.localization.Localization(__file__, 106, 14), search_628, *[str_629, s_630], **kwargs_631)
                        
                        # Getting the type of 'None' (line 106)
                        None_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 39), 'None')
                        # Applying the binary operator 'isnot' (line 106)
                        result_is_not_634 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 14), 'isnot', search_call_result_632, None_633)
                        
                        # Testing if the type of an if condition is none (line 106)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634):
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 106)
                            if_condition_635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634)
                            # Assigning a type to the variable 'if_condition_635' (line 106)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 13), 'if_condition_635', if_condition_635)
                            # SSA begins for if statement (line 106)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to Peg(...): (line 107)
                            # Processing the call arguments (line 107)
                            # Getting the type of 'colour' (line 107)
                            colour_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 27), 'colour', False)
                            # Obtaining the member 'Colours' of a type (line 107)
                            Colours_639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), colour_638, 'Colours')
                            # Obtaining the member 'white' of a type (line 107)
                            white_640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), Colours_639, 'white')
                            # Processing the call keyword arguments (line 107)
                            kwargs_641 = {}
                            # Getting the type of 'peg' (line 107)
                            peg_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'peg', False)
                            # Obtaining the member 'Peg' of a type (line 107)
                            Peg_637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 19), peg_636, 'Peg')
                            # Calling Peg(args, kwargs) (line 107)
                            Peg_call_result_642 = invoke(stypy.reporting.localization.Localization(__file__, 107, 19), Peg_637, *[white_640], **kwargs_641)
                            
                            # Assigning a type to the variable 'stypy_return_type' (line 107)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'stypy_return_type', Peg_call_result_642)
                            # SSA branch for the else part of an if statement (line 106)
                            module_type_store.open_ssa_branch('else')
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 106)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 104)
                        if_condition_619 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 13), result_is_not_618)
                        # Assigning a type to the variable 'if_condition_619' (line 104)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 13), 'if_condition_619', if_condition_619)
                        # SSA begins for if statement (line 104)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to Peg(...): (line 105)
                        # Processing the call arguments (line 105)
                        # Getting the type of 'colour' (line 105)
                        colour_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 27), 'colour', False)
                        # Obtaining the member 'Colours' of a type (line 105)
                        Colours_623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 27), colour_622, 'Colours')
                        # Obtaining the member 'yellow' of a type (line 105)
                        yellow_624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 27), Colours_623, 'yellow')
                        # Processing the call keyword arguments (line 105)
                        kwargs_625 = {}
                        # Getting the type of 'peg' (line 105)
                        peg_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 19), 'peg', False)
                        # Obtaining the member 'Peg' of a type (line 105)
                        Peg_621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 19), peg_620, 'Peg')
                        # Calling Peg(args, kwargs) (line 105)
                        Peg_call_result_626 = invoke(stypy.reporting.localization.Localization(__file__, 105, 19), Peg_621, *[yellow_624], **kwargs_625)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 105)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'stypy_return_type', Peg_call_result_626)
                        # SSA branch for the else part of an if statement (line 104)
                        module_type_store.open_ssa_branch('else')
                        
                        
                        # Call to search(...): (line 106)
                        # Processing the call arguments (line 106)
                        str_629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 24), 'str', '^w')
                        # Getting the type of 's' (line 106)
                        s_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 's', False)
                        # Processing the call keyword arguments (line 106)
                        kwargs_631 = {}
                        # Getting the type of 're' (line 106)
                        re_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 14), 're', False)
                        # Obtaining the member 'search' of a type (line 106)
                        search_628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 14), re_627, 'search')
                        # Calling search(args, kwargs) (line 106)
                        search_call_result_632 = invoke(stypy.reporting.localization.Localization(__file__, 106, 14), search_628, *[str_629, s_630], **kwargs_631)
                        
                        # Getting the type of 'None' (line 106)
                        None_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 39), 'None')
                        # Applying the binary operator 'isnot' (line 106)
                        result_is_not_634 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 14), 'isnot', search_call_result_632, None_633)
                        
                        # Testing if the type of an if condition is none (line 106)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634):
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 106)
                            if_condition_635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634)
                            # Assigning a type to the variable 'if_condition_635' (line 106)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 13), 'if_condition_635', if_condition_635)
                            # SSA begins for if statement (line 106)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to Peg(...): (line 107)
                            # Processing the call arguments (line 107)
                            # Getting the type of 'colour' (line 107)
                            colour_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 27), 'colour', False)
                            # Obtaining the member 'Colours' of a type (line 107)
                            Colours_639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), colour_638, 'Colours')
                            # Obtaining the member 'white' of a type (line 107)
                            white_640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), Colours_639, 'white')
                            # Processing the call keyword arguments (line 107)
                            kwargs_641 = {}
                            # Getting the type of 'peg' (line 107)
                            peg_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'peg', False)
                            # Obtaining the member 'Peg' of a type (line 107)
                            Peg_637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 19), peg_636, 'Peg')
                            # Calling Peg(args, kwargs) (line 107)
                            Peg_call_result_642 = invoke(stypy.reporting.localization.Localization(__file__, 107, 19), Peg_637, *[white_640], **kwargs_641)
                            
                            # Assigning a type to the variable 'stypy_return_type' (line 107)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'stypy_return_type', Peg_call_result_642)
                            # SSA branch for the else part of an if statement (line 106)
                            module_type_store.open_ssa_branch('else')
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 106)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 104)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 102)
                    if_condition_603 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 13), result_is_not_602)
                    # Assigning a type to the variable 'if_condition_603' (line 102)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 13), 'if_condition_603', if_condition_603)
                    # SSA begins for if statement (line 102)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to Peg(...): (line 103)
                    # Processing the call arguments (line 103)
                    # Getting the type of 'colour' (line 103)
                    colour_606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'colour', False)
                    # Obtaining the member 'Colours' of a type (line 103)
                    Colours_607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 27), colour_606, 'Colours')
                    # Obtaining the member 'green' of a type (line 103)
                    green_608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 27), Colours_607, 'green')
                    # Processing the call keyword arguments (line 103)
                    kwargs_609 = {}
                    # Getting the type of 'peg' (line 103)
                    peg_604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 19), 'peg', False)
                    # Obtaining the member 'Peg' of a type (line 103)
                    Peg_605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 19), peg_604, 'Peg')
                    # Calling Peg(args, kwargs) (line 103)
                    Peg_call_result_610 = invoke(stypy.reporting.localization.Localization(__file__, 103, 19), Peg_605, *[green_608], **kwargs_609)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 103)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'stypy_return_type', Peg_call_result_610)
                    # SSA branch for the else part of an if statement (line 102)
                    module_type_store.open_ssa_branch('else')
                    
                    
                    # Call to search(...): (line 104)
                    # Processing the call arguments (line 104)
                    str_613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 24), 'str', '^y')
                    # Getting the type of 's' (line 104)
                    s_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 29), 's', False)
                    # Processing the call keyword arguments (line 104)
                    kwargs_615 = {}
                    # Getting the type of 're' (line 104)
                    re_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 14), 're', False)
                    # Obtaining the member 'search' of a type (line 104)
                    search_612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 14), re_611, 'search')
                    # Calling search(args, kwargs) (line 104)
                    search_call_result_616 = invoke(stypy.reporting.localization.Localization(__file__, 104, 14), search_612, *[str_613, s_614], **kwargs_615)
                    
                    # Getting the type of 'None' (line 104)
                    None_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 39), 'None')
                    # Applying the binary operator 'isnot' (line 104)
                    result_is_not_618 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 14), 'isnot', search_call_result_616, None_617)
                    
                    # Testing if the type of an if condition is none (line 104)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 104, 13), result_is_not_618):
                        
                        
                        # Call to search(...): (line 106)
                        # Processing the call arguments (line 106)
                        str_629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 24), 'str', '^w')
                        # Getting the type of 's' (line 106)
                        s_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 's', False)
                        # Processing the call keyword arguments (line 106)
                        kwargs_631 = {}
                        # Getting the type of 're' (line 106)
                        re_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 14), 're', False)
                        # Obtaining the member 'search' of a type (line 106)
                        search_628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 14), re_627, 'search')
                        # Calling search(args, kwargs) (line 106)
                        search_call_result_632 = invoke(stypy.reporting.localization.Localization(__file__, 106, 14), search_628, *[str_629, s_630], **kwargs_631)
                        
                        # Getting the type of 'None' (line 106)
                        None_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 39), 'None')
                        # Applying the binary operator 'isnot' (line 106)
                        result_is_not_634 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 14), 'isnot', search_call_result_632, None_633)
                        
                        # Testing if the type of an if condition is none (line 106)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634):
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 106)
                            if_condition_635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634)
                            # Assigning a type to the variable 'if_condition_635' (line 106)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 13), 'if_condition_635', if_condition_635)
                            # SSA begins for if statement (line 106)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to Peg(...): (line 107)
                            # Processing the call arguments (line 107)
                            # Getting the type of 'colour' (line 107)
                            colour_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 27), 'colour', False)
                            # Obtaining the member 'Colours' of a type (line 107)
                            Colours_639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), colour_638, 'Colours')
                            # Obtaining the member 'white' of a type (line 107)
                            white_640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), Colours_639, 'white')
                            # Processing the call keyword arguments (line 107)
                            kwargs_641 = {}
                            # Getting the type of 'peg' (line 107)
                            peg_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'peg', False)
                            # Obtaining the member 'Peg' of a type (line 107)
                            Peg_637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 19), peg_636, 'Peg')
                            # Calling Peg(args, kwargs) (line 107)
                            Peg_call_result_642 = invoke(stypy.reporting.localization.Localization(__file__, 107, 19), Peg_637, *[white_640], **kwargs_641)
                            
                            # Assigning a type to the variable 'stypy_return_type' (line 107)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'stypy_return_type', Peg_call_result_642)
                            # SSA branch for the else part of an if statement (line 106)
                            module_type_store.open_ssa_branch('else')
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 106)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 104)
                        if_condition_619 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 13), result_is_not_618)
                        # Assigning a type to the variable 'if_condition_619' (line 104)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 13), 'if_condition_619', if_condition_619)
                        # SSA begins for if statement (line 104)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to Peg(...): (line 105)
                        # Processing the call arguments (line 105)
                        # Getting the type of 'colour' (line 105)
                        colour_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 27), 'colour', False)
                        # Obtaining the member 'Colours' of a type (line 105)
                        Colours_623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 27), colour_622, 'Colours')
                        # Obtaining the member 'yellow' of a type (line 105)
                        yellow_624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 27), Colours_623, 'yellow')
                        # Processing the call keyword arguments (line 105)
                        kwargs_625 = {}
                        # Getting the type of 'peg' (line 105)
                        peg_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 19), 'peg', False)
                        # Obtaining the member 'Peg' of a type (line 105)
                        Peg_621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 19), peg_620, 'Peg')
                        # Calling Peg(args, kwargs) (line 105)
                        Peg_call_result_626 = invoke(stypy.reporting.localization.Localization(__file__, 105, 19), Peg_621, *[yellow_624], **kwargs_625)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 105)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'stypy_return_type', Peg_call_result_626)
                        # SSA branch for the else part of an if statement (line 104)
                        module_type_store.open_ssa_branch('else')
                        
                        
                        # Call to search(...): (line 106)
                        # Processing the call arguments (line 106)
                        str_629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 24), 'str', '^w')
                        # Getting the type of 's' (line 106)
                        s_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 's', False)
                        # Processing the call keyword arguments (line 106)
                        kwargs_631 = {}
                        # Getting the type of 're' (line 106)
                        re_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 14), 're', False)
                        # Obtaining the member 'search' of a type (line 106)
                        search_628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 14), re_627, 'search')
                        # Calling search(args, kwargs) (line 106)
                        search_call_result_632 = invoke(stypy.reporting.localization.Localization(__file__, 106, 14), search_628, *[str_629, s_630], **kwargs_631)
                        
                        # Getting the type of 'None' (line 106)
                        None_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 39), 'None')
                        # Applying the binary operator 'isnot' (line 106)
                        result_is_not_634 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 14), 'isnot', search_call_result_632, None_633)
                        
                        # Testing if the type of an if condition is none (line 106)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634):
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 106)
                            if_condition_635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634)
                            # Assigning a type to the variable 'if_condition_635' (line 106)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 13), 'if_condition_635', if_condition_635)
                            # SSA begins for if statement (line 106)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to Peg(...): (line 107)
                            # Processing the call arguments (line 107)
                            # Getting the type of 'colour' (line 107)
                            colour_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 27), 'colour', False)
                            # Obtaining the member 'Colours' of a type (line 107)
                            Colours_639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), colour_638, 'Colours')
                            # Obtaining the member 'white' of a type (line 107)
                            white_640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), Colours_639, 'white')
                            # Processing the call keyword arguments (line 107)
                            kwargs_641 = {}
                            # Getting the type of 'peg' (line 107)
                            peg_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'peg', False)
                            # Obtaining the member 'Peg' of a type (line 107)
                            Peg_637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 19), peg_636, 'Peg')
                            # Calling Peg(args, kwargs) (line 107)
                            Peg_call_result_642 = invoke(stypy.reporting.localization.Localization(__file__, 107, 19), Peg_637, *[white_640], **kwargs_641)
                            
                            # Assigning a type to the variable 'stypy_return_type' (line 107)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'stypy_return_type', Peg_call_result_642)
                            # SSA branch for the else part of an if statement (line 106)
                            module_type_store.open_ssa_branch('else')
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 106)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 104)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 102)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 100)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 98)
            if_condition_571 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 8), result_is_not_570)
            # Assigning a type to the variable 'if_condition_571' (line 98)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'if_condition_571', if_condition_571)
            # SSA begins for if statement (line 98)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to Peg(...): (line 99)
            # Processing the call arguments (line 99)
            # Getting the type of 'colour' (line 99)
            colour_574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 27), 'colour', False)
            # Obtaining the member 'Colours' of a type (line 99)
            Colours_575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 27), colour_574, 'Colours')
            # Obtaining the member 'red' of a type (line 99)
            red_576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 27), Colours_575, 'red')
            # Processing the call keyword arguments (line 99)
            kwargs_577 = {}
            # Getting the type of 'peg' (line 99)
            peg_572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 19), 'peg', False)
            # Obtaining the member 'Peg' of a type (line 99)
            Peg_573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 19), peg_572, 'Peg')
            # Calling Peg(args, kwargs) (line 99)
            Peg_call_result_578 = invoke(stypy.reporting.localization.Localization(__file__, 99, 19), Peg_573, *[red_576], **kwargs_577)
            
            # Assigning a type to the variable 'stypy_return_type' (line 99)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'stypy_return_type', Peg_call_result_578)
            # SSA branch for the else part of an if statement (line 98)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to search(...): (line 100)
            # Processing the call arguments (line 100)
            str_581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 24), 'str', '^p')
            # Getting the type of 's' (line 100)
            s_582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 29), 's', False)
            # Processing the call keyword arguments (line 100)
            kwargs_583 = {}
            # Getting the type of 're' (line 100)
            re_579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 14), 're', False)
            # Obtaining the member 'search' of a type (line 100)
            search_580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 14), re_579, 'search')
            # Calling search(args, kwargs) (line 100)
            search_call_result_584 = invoke(stypy.reporting.localization.Localization(__file__, 100, 14), search_580, *[str_581, s_582], **kwargs_583)
            
            # Getting the type of 'None' (line 100)
            None_585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 39), 'None')
            # Applying the binary operator 'isnot' (line 100)
            result_is_not_586 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 14), 'isnot', search_call_result_584, None_585)
            
            # Testing if the type of an if condition is none (line 100)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 100, 13), result_is_not_586):
                
                
                # Call to search(...): (line 102)
                # Processing the call arguments (line 102)
                str_597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 24), 'str', '^g')
                # Getting the type of 's' (line 102)
                s_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 29), 's', False)
                # Processing the call keyword arguments (line 102)
                kwargs_599 = {}
                # Getting the type of 're' (line 102)
                re_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 14), 're', False)
                # Obtaining the member 'search' of a type (line 102)
                search_596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 14), re_595, 'search')
                # Calling search(args, kwargs) (line 102)
                search_call_result_600 = invoke(stypy.reporting.localization.Localization(__file__, 102, 14), search_596, *[str_597, s_598], **kwargs_599)
                
                # Getting the type of 'None' (line 102)
                None_601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 39), 'None')
                # Applying the binary operator 'isnot' (line 102)
                result_is_not_602 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 14), 'isnot', search_call_result_600, None_601)
                
                # Testing if the type of an if condition is none (line 102)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 102, 13), result_is_not_602):
                    
                    
                    # Call to search(...): (line 104)
                    # Processing the call arguments (line 104)
                    str_613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 24), 'str', '^y')
                    # Getting the type of 's' (line 104)
                    s_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 29), 's', False)
                    # Processing the call keyword arguments (line 104)
                    kwargs_615 = {}
                    # Getting the type of 're' (line 104)
                    re_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 14), 're', False)
                    # Obtaining the member 'search' of a type (line 104)
                    search_612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 14), re_611, 'search')
                    # Calling search(args, kwargs) (line 104)
                    search_call_result_616 = invoke(stypy.reporting.localization.Localization(__file__, 104, 14), search_612, *[str_613, s_614], **kwargs_615)
                    
                    # Getting the type of 'None' (line 104)
                    None_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 39), 'None')
                    # Applying the binary operator 'isnot' (line 104)
                    result_is_not_618 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 14), 'isnot', search_call_result_616, None_617)
                    
                    # Testing if the type of an if condition is none (line 104)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 104, 13), result_is_not_618):
                        
                        
                        # Call to search(...): (line 106)
                        # Processing the call arguments (line 106)
                        str_629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 24), 'str', '^w')
                        # Getting the type of 's' (line 106)
                        s_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 's', False)
                        # Processing the call keyword arguments (line 106)
                        kwargs_631 = {}
                        # Getting the type of 're' (line 106)
                        re_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 14), 're', False)
                        # Obtaining the member 'search' of a type (line 106)
                        search_628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 14), re_627, 'search')
                        # Calling search(args, kwargs) (line 106)
                        search_call_result_632 = invoke(stypy.reporting.localization.Localization(__file__, 106, 14), search_628, *[str_629, s_630], **kwargs_631)
                        
                        # Getting the type of 'None' (line 106)
                        None_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 39), 'None')
                        # Applying the binary operator 'isnot' (line 106)
                        result_is_not_634 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 14), 'isnot', search_call_result_632, None_633)
                        
                        # Testing if the type of an if condition is none (line 106)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634):
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 106)
                            if_condition_635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634)
                            # Assigning a type to the variable 'if_condition_635' (line 106)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 13), 'if_condition_635', if_condition_635)
                            # SSA begins for if statement (line 106)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to Peg(...): (line 107)
                            # Processing the call arguments (line 107)
                            # Getting the type of 'colour' (line 107)
                            colour_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 27), 'colour', False)
                            # Obtaining the member 'Colours' of a type (line 107)
                            Colours_639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), colour_638, 'Colours')
                            # Obtaining the member 'white' of a type (line 107)
                            white_640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), Colours_639, 'white')
                            # Processing the call keyword arguments (line 107)
                            kwargs_641 = {}
                            # Getting the type of 'peg' (line 107)
                            peg_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'peg', False)
                            # Obtaining the member 'Peg' of a type (line 107)
                            Peg_637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 19), peg_636, 'Peg')
                            # Calling Peg(args, kwargs) (line 107)
                            Peg_call_result_642 = invoke(stypy.reporting.localization.Localization(__file__, 107, 19), Peg_637, *[white_640], **kwargs_641)
                            
                            # Assigning a type to the variable 'stypy_return_type' (line 107)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'stypy_return_type', Peg_call_result_642)
                            # SSA branch for the else part of an if statement (line 106)
                            module_type_store.open_ssa_branch('else')
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 106)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 104)
                        if_condition_619 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 13), result_is_not_618)
                        # Assigning a type to the variable 'if_condition_619' (line 104)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 13), 'if_condition_619', if_condition_619)
                        # SSA begins for if statement (line 104)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to Peg(...): (line 105)
                        # Processing the call arguments (line 105)
                        # Getting the type of 'colour' (line 105)
                        colour_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 27), 'colour', False)
                        # Obtaining the member 'Colours' of a type (line 105)
                        Colours_623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 27), colour_622, 'Colours')
                        # Obtaining the member 'yellow' of a type (line 105)
                        yellow_624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 27), Colours_623, 'yellow')
                        # Processing the call keyword arguments (line 105)
                        kwargs_625 = {}
                        # Getting the type of 'peg' (line 105)
                        peg_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 19), 'peg', False)
                        # Obtaining the member 'Peg' of a type (line 105)
                        Peg_621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 19), peg_620, 'Peg')
                        # Calling Peg(args, kwargs) (line 105)
                        Peg_call_result_626 = invoke(stypy.reporting.localization.Localization(__file__, 105, 19), Peg_621, *[yellow_624], **kwargs_625)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 105)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'stypy_return_type', Peg_call_result_626)
                        # SSA branch for the else part of an if statement (line 104)
                        module_type_store.open_ssa_branch('else')
                        
                        
                        # Call to search(...): (line 106)
                        # Processing the call arguments (line 106)
                        str_629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 24), 'str', '^w')
                        # Getting the type of 's' (line 106)
                        s_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 's', False)
                        # Processing the call keyword arguments (line 106)
                        kwargs_631 = {}
                        # Getting the type of 're' (line 106)
                        re_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 14), 're', False)
                        # Obtaining the member 'search' of a type (line 106)
                        search_628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 14), re_627, 'search')
                        # Calling search(args, kwargs) (line 106)
                        search_call_result_632 = invoke(stypy.reporting.localization.Localization(__file__, 106, 14), search_628, *[str_629, s_630], **kwargs_631)
                        
                        # Getting the type of 'None' (line 106)
                        None_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 39), 'None')
                        # Applying the binary operator 'isnot' (line 106)
                        result_is_not_634 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 14), 'isnot', search_call_result_632, None_633)
                        
                        # Testing if the type of an if condition is none (line 106)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634):
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 106)
                            if_condition_635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634)
                            # Assigning a type to the variable 'if_condition_635' (line 106)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 13), 'if_condition_635', if_condition_635)
                            # SSA begins for if statement (line 106)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to Peg(...): (line 107)
                            # Processing the call arguments (line 107)
                            # Getting the type of 'colour' (line 107)
                            colour_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 27), 'colour', False)
                            # Obtaining the member 'Colours' of a type (line 107)
                            Colours_639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), colour_638, 'Colours')
                            # Obtaining the member 'white' of a type (line 107)
                            white_640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), Colours_639, 'white')
                            # Processing the call keyword arguments (line 107)
                            kwargs_641 = {}
                            # Getting the type of 'peg' (line 107)
                            peg_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'peg', False)
                            # Obtaining the member 'Peg' of a type (line 107)
                            Peg_637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 19), peg_636, 'Peg')
                            # Calling Peg(args, kwargs) (line 107)
                            Peg_call_result_642 = invoke(stypy.reporting.localization.Localization(__file__, 107, 19), Peg_637, *[white_640], **kwargs_641)
                            
                            # Assigning a type to the variable 'stypy_return_type' (line 107)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'stypy_return_type', Peg_call_result_642)
                            # SSA branch for the else part of an if statement (line 106)
                            module_type_store.open_ssa_branch('else')
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 106)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 104)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 102)
                    if_condition_603 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 13), result_is_not_602)
                    # Assigning a type to the variable 'if_condition_603' (line 102)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 13), 'if_condition_603', if_condition_603)
                    # SSA begins for if statement (line 102)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to Peg(...): (line 103)
                    # Processing the call arguments (line 103)
                    # Getting the type of 'colour' (line 103)
                    colour_606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'colour', False)
                    # Obtaining the member 'Colours' of a type (line 103)
                    Colours_607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 27), colour_606, 'Colours')
                    # Obtaining the member 'green' of a type (line 103)
                    green_608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 27), Colours_607, 'green')
                    # Processing the call keyword arguments (line 103)
                    kwargs_609 = {}
                    # Getting the type of 'peg' (line 103)
                    peg_604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 19), 'peg', False)
                    # Obtaining the member 'Peg' of a type (line 103)
                    Peg_605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 19), peg_604, 'Peg')
                    # Calling Peg(args, kwargs) (line 103)
                    Peg_call_result_610 = invoke(stypy.reporting.localization.Localization(__file__, 103, 19), Peg_605, *[green_608], **kwargs_609)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 103)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'stypy_return_type', Peg_call_result_610)
                    # SSA branch for the else part of an if statement (line 102)
                    module_type_store.open_ssa_branch('else')
                    
                    
                    # Call to search(...): (line 104)
                    # Processing the call arguments (line 104)
                    str_613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 24), 'str', '^y')
                    # Getting the type of 's' (line 104)
                    s_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 29), 's', False)
                    # Processing the call keyword arguments (line 104)
                    kwargs_615 = {}
                    # Getting the type of 're' (line 104)
                    re_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 14), 're', False)
                    # Obtaining the member 'search' of a type (line 104)
                    search_612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 14), re_611, 'search')
                    # Calling search(args, kwargs) (line 104)
                    search_call_result_616 = invoke(stypy.reporting.localization.Localization(__file__, 104, 14), search_612, *[str_613, s_614], **kwargs_615)
                    
                    # Getting the type of 'None' (line 104)
                    None_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 39), 'None')
                    # Applying the binary operator 'isnot' (line 104)
                    result_is_not_618 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 14), 'isnot', search_call_result_616, None_617)
                    
                    # Testing if the type of an if condition is none (line 104)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 104, 13), result_is_not_618):
                        
                        
                        # Call to search(...): (line 106)
                        # Processing the call arguments (line 106)
                        str_629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 24), 'str', '^w')
                        # Getting the type of 's' (line 106)
                        s_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 's', False)
                        # Processing the call keyword arguments (line 106)
                        kwargs_631 = {}
                        # Getting the type of 're' (line 106)
                        re_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 14), 're', False)
                        # Obtaining the member 'search' of a type (line 106)
                        search_628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 14), re_627, 'search')
                        # Calling search(args, kwargs) (line 106)
                        search_call_result_632 = invoke(stypy.reporting.localization.Localization(__file__, 106, 14), search_628, *[str_629, s_630], **kwargs_631)
                        
                        # Getting the type of 'None' (line 106)
                        None_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 39), 'None')
                        # Applying the binary operator 'isnot' (line 106)
                        result_is_not_634 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 14), 'isnot', search_call_result_632, None_633)
                        
                        # Testing if the type of an if condition is none (line 106)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634):
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 106)
                            if_condition_635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634)
                            # Assigning a type to the variable 'if_condition_635' (line 106)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 13), 'if_condition_635', if_condition_635)
                            # SSA begins for if statement (line 106)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to Peg(...): (line 107)
                            # Processing the call arguments (line 107)
                            # Getting the type of 'colour' (line 107)
                            colour_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 27), 'colour', False)
                            # Obtaining the member 'Colours' of a type (line 107)
                            Colours_639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), colour_638, 'Colours')
                            # Obtaining the member 'white' of a type (line 107)
                            white_640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), Colours_639, 'white')
                            # Processing the call keyword arguments (line 107)
                            kwargs_641 = {}
                            # Getting the type of 'peg' (line 107)
                            peg_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'peg', False)
                            # Obtaining the member 'Peg' of a type (line 107)
                            Peg_637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 19), peg_636, 'Peg')
                            # Calling Peg(args, kwargs) (line 107)
                            Peg_call_result_642 = invoke(stypy.reporting.localization.Localization(__file__, 107, 19), Peg_637, *[white_640], **kwargs_641)
                            
                            # Assigning a type to the variable 'stypy_return_type' (line 107)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'stypy_return_type', Peg_call_result_642)
                            # SSA branch for the else part of an if statement (line 106)
                            module_type_store.open_ssa_branch('else')
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 106)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 104)
                        if_condition_619 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 13), result_is_not_618)
                        # Assigning a type to the variable 'if_condition_619' (line 104)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 13), 'if_condition_619', if_condition_619)
                        # SSA begins for if statement (line 104)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to Peg(...): (line 105)
                        # Processing the call arguments (line 105)
                        # Getting the type of 'colour' (line 105)
                        colour_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 27), 'colour', False)
                        # Obtaining the member 'Colours' of a type (line 105)
                        Colours_623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 27), colour_622, 'Colours')
                        # Obtaining the member 'yellow' of a type (line 105)
                        yellow_624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 27), Colours_623, 'yellow')
                        # Processing the call keyword arguments (line 105)
                        kwargs_625 = {}
                        # Getting the type of 'peg' (line 105)
                        peg_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 19), 'peg', False)
                        # Obtaining the member 'Peg' of a type (line 105)
                        Peg_621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 19), peg_620, 'Peg')
                        # Calling Peg(args, kwargs) (line 105)
                        Peg_call_result_626 = invoke(stypy.reporting.localization.Localization(__file__, 105, 19), Peg_621, *[yellow_624], **kwargs_625)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 105)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'stypy_return_type', Peg_call_result_626)
                        # SSA branch for the else part of an if statement (line 104)
                        module_type_store.open_ssa_branch('else')
                        
                        
                        # Call to search(...): (line 106)
                        # Processing the call arguments (line 106)
                        str_629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 24), 'str', '^w')
                        # Getting the type of 's' (line 106)
                        s_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 's', False)
                        # Processing the call keyword arguments (line 106)
                        kwargs_631 = {}
                        # Getting the type of 're' (line 106)
                        re_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 14), 're', False)
                        # Obtaining the member 'search' of a type (line 106)
                        search_628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 14), re_627, 'search')
                        # Calling search(args, kwargs) (line 106)
                        search_call_result_632 = invoke(stypy.reporting.localization.Localization(__file__, 106, 14), search_628, *[str_629, s_630], **kwargs_631)
                        
                        # Getting the type of 'None' (line 106)
                        None_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 39), 'None')
                        # Applying the binary operator 'isnot' (line 106)
                        result_is_not_634 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 14), 'isnot', search_call_result_632, None_633)
                        
                        # Testing if the type of an if condition is none (line 106)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634):
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 106)
                            if_condition_635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634)
                            # Assigning a type to the variable 'if_condition_635' (line 106)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 13), 'if_condition_635', if_condition_635)
                            # SSA begins for if statement (line 106)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to Peg(...): (line 107)
                            # Processing the call arguments (line 107)
                            # Getting the type of 'colour' (line 107)
                            colour_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 27), 'colour', False)
                            # Obtaining the member 'Colours' of a type (line 107)
                            Colours_639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), colour_638, 'Colours')
                            # Obtaining the member 'white' of a type (line 107)
                            white_640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), Colours_639, 'white')
                            # Processing the call keyword arguments (line 107)
                            kwargs_641 = {}
                            # Getting the type of 'peg' (line 107)
                            peg_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'peg', False)
                            # Obtaining the member 'Peg' of a type (line 107)
                            Peg_637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 19), peg_636, 'Peg')
                            # Calling Peg(args, kwargs) (line 107)
                            Peg_call_result_642 = invoke(stypy.reporting.localization.Localization(__file__, 107, 19), Peg_637, *[white_640], **kwargs_641)
                            
                            # Assigning a type to the variable 'stypy_return_type' (line 107)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'stypy_return_type', Peg_call_result_642)
                            # SSA branch for the else part of an if statement (line 106)
                            module_type_store.open_ssa_branch('else')
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 106)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 104)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 102)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 100)
                if_condition_587 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 13), result_is_not_586)
                # Assigning a type to the variable 'if_condition_587' (line 100)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 13), 'if_condition_587', if_condition_587)
                # SSA begins for if statement (line 100)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to Peg(...): (line 101)
                # Processing the call arguments (line 101)
                # Getting the type of 'colour' (line 101)
                colour_590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 27), 'colour', False)
                # Obtaining the member 'Colours' of a type (line 101)
                Colours_591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 27), colour_590, 'Colours')
                # Obtaining the member 'purple' of a type (line 101)
                purple_592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 27), Colours_591, 'purple')
                # Processing the call keyword arguments (line 101)
                kwargs_593 = {}
                # Getting the type of 'peg' (line 101)
                peg_588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 19), 'peg', False)
                # Obtaining the member 'Peg' of a type (line 101)
                Peg_589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 19), peg_588, 'Peg')
                # Calling Peg(args, kwargs) (line 101)
                Peg_call_result_594 = invoke(stypy.reporting.localization.Localization(__file__, 101, 19), Peg_589, *[purple_592], **kwargs_593)
                
                # Assigning a type to the variable 'stypy_return_type' (line 101)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'stypy_return_type', Peg_call_result_594)
                # SSA branch for the else part of an if statement (line 100)
                module_type_store.open_ssa_branch('else')
                
                
                # Call to search(...): (line 102)
                # Processing the call arguments (line 102)
                str_597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 24), 'str', '^g')
                # Getting the type of 's' (line 102)
                s_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 29), 's', False)
                # Processing the call keyword arguments (line 102)
                kwargs_599 = {}
                # Getting the type of 're' (line 102)
                re_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 14), 're', False)
                # Obtaining the member 'search' of a type (line 102)
                search_596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 14), re_595, 'search')
                # Calling search(args, kwargs) (line 102)
                search_call_result_600 = invoke(stypy.reporting.localization.Localization(__file__, 102, 14), search_596, *[str_597, s_598], **kwargs_599)
                
                # Getting the type of 'None' (line 102)
                None_601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 39), 'None')
                # Applying the binary operator 'isnot' (line 102)
                result_is_not_602 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 14), 'isnot', search_call_result_600, None_601)
                
                # Testing if the type of an if condition is none (line 102)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 102, 13), result_is_not_602):
                    
                    
                    # Call to search(...): (line 104)
                    # Processing the call arguments (line 104)
                    str_613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 24), 'str', '^y')
                    # Getting the type of 's' (line 104)
                    s_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 29), 's', False)
                    # Processing the call keyword arguments (line 104)
                    kwargs_615 = {}
                    # Getting the type of 're' (line 104)
                    re_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 14), 're', False)
                    # Obtaining the member 'search' of a type (line 104)
                    search_612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 14), re_611, 'search')
                    # Calling search(args, kwargs) (line 104)
                    search_call_result_616 = invoke(stypy.reporting.localization.Localization(__file__, 104, 14), search_612, *[str_613, s_614], **kwargs_615)
                    
                    # Getting the type of 'None' (line 104)
                    None_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 39), 'None')
                    # Applying the binary operator 'isnot' (line 104)
                    result_is_not_618 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 14), 'isnot', search_call_result_616, None_617)
                    
                    # Testing if the type of an if condition is none (line 104)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 104, 13), result_is_not_618):
                        
                        
                        # Call to search(...): (line 106)
                        # Processing the call arguments (line 106)
                        str_629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 24), 'str', '^w')
                        # Getting the type of 's' (line 106)
                        s_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 's', False)
                        # Processing the call keyword arguments (line 106)
                        kwargs_631 = {}
                        # Getting the type of 're' (line 106)
                        re_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 14), 're', False)
                        # Obtaining the member 'search' of a type (line 106)
                        search_628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 14), re_627, 'search')
                        # Calling search(args, kwargs) (line 106)
                        search_call_result_632 = invoke(stypy.reporting.localization.Localization(__file__, 106, 14), search_628, *[str_629, s_630], **kwargs_631)
                        
                        # Getting the type of 'None' (line 106)
                        None_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 39), 'None')
                        # Applying the binary operator 'isnot' (line 106)
                        result_is_not_634 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 14), 'isnot', search_call_result_632, None_633)
                        
                        # Testing if the type of an if condition is none (line 106)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634):
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 106)
                            if_condition_635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634)
                            # Assigning a type to the variable 'if_condition_635' (line 106)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 13), 'if_condition_635', if_condition_635)
                            # SSA begins for if statement (line 106)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to Peg(...): (line 107)
                            # Processing the call arguments (line 107)
                            # Getting the type of 'colour' (line 107)
                            colour_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 27), 'colour', False)
                            # Obtaining the member 'Colours' of a type (line 107)
                            Colours_639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), colour_638, 'Colours')
                            # Obtaining the member 'white' of a type (line 107)
                            white_640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), Colours_639, 'white')
                            # Processing the call keyword arguments (line 107)
                            kwargs_641 = {}
                            # Getting the type of 'peg' (line 107)
                            peg_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'peg', False)
                            # Obtaining the member 'Peg' of a type (line 107)
                            Peg_637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 19), peg_636, 'Peg')
                            # Calling Peg(args, kwargs) (line 107)
                            Peg_call_result_642 = invoke(stypy.reporting.localization.Localization(__file__, 107, 19), Peg_637, *[white_640], **kwargs_641)
                            
                            # Assigning a type to the variable 'stypy_return_type' (line 107)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'stypy_return_type', Peg_call_result_642)
                            # SSA branch for the else part of an if statement (line 106)
                            module_type_store.open_ssa_branch('else')
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 106)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 104)
                        if_condition_619 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 13), result_is_not_618)
                        # Assigning a type to the variable 'if_condition_619' (line 104)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 13), 'if_condition_619', if_condition_619)
                        # SSA begins for if statement (line 104)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to Peg(...): (line 105)
                        # Processing the call arguments (line 105)
                        # Getting the type of 'colour' (line 105)
                        colour_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 27), 'colour', False)
                        # Obtaining the member 'Colours' of a type (line 105)
                        Colours_623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 27), colour_622, 'Colours')
                        # Obtaining the member 'yellow' of a type (line 105)
                        yellow_624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 27), Colours_623, 'yellow')
                        # Processing the call keyword arguments (line 105)
                        kwargs_625 = {}
                        # Getting the type of 'peg' (line 105)
                        peg_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 19), 'peg', False)
                        # Obtaining the member 'Peg' of a type (line 105)
                        Peg_621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 19), peg_620, 'Peg')
                        # Calling Peg(args, kwargs) (line 105)
                        Peg_call_result_626 = invoke(stypy.reporting.localization.Localization(__file__, 105, 19), Peg_621, *[yellow_624], **kwargs_625)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 105)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'stypy_return_type', Peg_call_result_626)
                        # SSA branch for the else part of an if statement (line 104)
                        module_type_store.open_ssa_branch('else')
                        
                        
                        # Call to search(...): (line 106)
                        # Processing the call arguments (line 106)
                        str_629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 24), 'str', '^w')
                        # Getting the type of 's' (line 106)
                        s_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 's', False)
                        # Processing the call keyword arguments (line 106)
                        kwargs_631 = {}
                        # Getting the type of 're' (line 106)
                        re_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 14), 're', False)
                        # Obtaining the member 'search' of a type (line 106)
                        search_628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 14), re_627, 'search')
                        # Calling search(args, kwargs) (line 106)
                        search_call_result_632 = invoke(stypy.reporting.localization.Localization(__file__, 106, 14), search_628, *[str_629, s_630], **kwargs_631)
                        
                        # Getting the type of 'None' (line 106)
                        None_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 39), 'None')
                        # Applying the binary operator 'isnot' (line 106)
                        result_is_not_634 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 14), 'isnot', search_call_result_632, None_633)
                        
                        # Testing if the type of an if condition is none (line 106)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634):
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 106)
                            if_condition_635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634)
                            # Assigning a type to the variable 'if_condition_635' (line 106)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 13), 'if_condition_635', if_condition_635)
                            # SSA begins for if statement (line 106)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to Peg(...): (line 107)
                            # Processing the call arguments (line 107)
                            # Getting the type of 'colour' (line 107)
                            colour_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 27), 'colour', False)
                            # Obtaining the member 'Colours' of a type (line 107)
                            Colours_639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), colour_638, 'Colours')
                            # Obtaining the member 'white' of a type (line 107)
                            white_640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), Colours_639, 'white')
                            # Processing the call keyword arguments (line 107)
                            kwargs_641 = {}
                            # Getting the type of 'peg' (line 107)
                            peg_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'peg', False)
                            # Obtaining the member 'Peg' of a type (line 107)
                            Peg_637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 19), peg_636, 'Peg')
                            # Calling Peg(args, kwargs) (line 107)
                            Peg_call_result_642 = invoke(stypy.reporting.localization.Localization(__file__, 107, 19), Peg_637, *[white_640], **kwargs_641)
                            
                            # Assigning a type to the variable 'stypy_return_type' (line 107)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'stypy_return_type', Peg_call_result_642)
                            # SSA branch for the else part of an if statement (line 106)
                            module_type_store.open_ssa_branch('else')
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 106)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 104)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 102)
                    if_condition_603 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 13), result_is_not_602)
                    # Assigning a type to the variable 'if_condition_603' (line 102)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 13), 'if_condition_603', if_condition_603)
                    # SSA begins for if statement (line 102)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to Peg(...): (line 103)
                    # Processing the call arguments (line 103)
                    # Getting the type of 'colour' (line 103)
                    colour_606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'colour', False)
                    # Obtaining the member 'Colours' of a type (line 103)
                    Colours_607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 27), colour_606, 'Colours')
                    # Obtaining the member 'green' of a type (line 103)
                    green_608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 27), Colours_607, 'green')
                    # Processing the call keyword arguments (line 103)
                    kwargs_609 = {}
                    # Getting the type of 'peg' (line 103)
                    peg_604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 19), 'peg', False)
                    # Obtaining the member 'Peg' of a type (line 103)
                    Peg_605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 19), peg_604, 'Peg')
                    # Calling Peg(args, kwargs) (line 103)
                    Peg_call_result_610 = invoke(stypy.reporting.localization.Localization(__file__, 103, 19), Peg_605, *[green_608], **kwargs_609)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 103)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'stypy_return_type', Peg_call_result_610)
                    # SSA branch for the else part of an if statement (line 102)
                    module_type_store.open_ssa_branch('else')
                    
                    
                    # Call to search(...): (line 104)
                    # Processing the call arguments (line 104)
                    str_613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 24), 'str', '^y')
                    # Getting the type of 's' (line 104)
                    s_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 29), 's', False)
                    # Processing the call keyword arguments (line 104)
                    kwargs_615 = {}
                    # Getting the type of 're' (line 104)
                    re_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 14), 're', False)
                    # Obtaining the member 'search' of a type (line 104)
                    search_612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 14), re_611, 'search')
                    # Calling search(args, kwargs) (line 104)
                    search_call_result_616 = invoke(stypy.reporting.localization.Localization(__file__, 104, 14), search_612, *[str_613, s_614], **kwargs_615)
                    
                    # Getting the type of 'None' (line 104)
                    None_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 39), 'None')
                    # Applying the binary operator 'isnot' (line 104)
                    result_is_not_618 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 14), 'isnot', search_call_result_616, None_617)
                    
                    # Testing if the type of an if condition is none (line 104)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 104, 13), result_is_not_618):
                        
                        
                        # Call to search(...): (line 106)
                        # Processing the call arguments (line 106)
                        str_629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 24), 'str', '^w')
                        # Getting the type of 's' (line 106)
                        s_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 's', False)
                        # Processing the call keyword arguments (line 106)
                        kwargs_631 = {}
                        # Getting the type of 're' (line 106)
                        re_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 14), 're', False)
                        # Obtaining the member 'search' of a type (line 106)
                        search_628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 14), re_627, 'search')
                        # Calling search(args, kwargs) (line 106)
                        search_call_result_632 = invoke(stypy.reporting.localization.Localization(__file__, 106, 14), search_628, *[str_629, s_630], **kwargs_631)
                        
                        # Getting the type of 'None' (line 106)
                        None_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 39), 'None')
                        # Applying the binary operator 'isnot' (line 106)
                        result_is_not_634 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 14), 'isnot', search_call_result_632, None_633)
                        
                        # Testing if the type of an if condition is none (line 106)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634):
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 106)
                            if_condition_635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634)
                            # Assigning a type to the variable 'if_condition_635' (line 106)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 13), 'if_condition_635', if_condition_635)
                            # SSA begins for if statement (line 106)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to Peg(...): (line 107)
                            # Processing the call arguments (line 107)
                            # Getting the type of 'colour' (line 107)
                            colour_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 27), 'colour', False)
                            # Obtaining the member 'Colours' of a type (line 107)
                            Colours_639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), colour_638, 'Colours')
                            # Obtaining the member 'white' of a type (line 107)
                            white_640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), Colours_639, 'white')
                            # Processing the call keyword arguments (line 107)
                            kwargs_641 = {}
                            # Getting the type of 'peg' (line 107)
                            peg_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'peg', False)
                            # Obtaining the member 'Peg' of a type (line 107)
                            Peg_637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 19), peg_636, 'Peg')
                            # Calling Peg(args, kwargs) (line 107)
                            Peg_call_result_642 = invoke(stypy.reporting.localization.Localization(__file__, 107, 19), Peg_637, *[white_640], **kwargs_641)
                            
                            # Assigning a type to the variable 'stypy_return_type' (line 107)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'stypy_return_type', Peg_call_result_642)
                            # SSA branch for the else part of an if statement (line 106)
                            module_type_store.open_ssa_branch('else')
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 106)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 104)
                        if_condition_619 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 13), result_is_not_618)
                        # Assigning a type to the variable 'if_condition_619' (line 104)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 13), 'if_condition_619', if_condition_619)
                        # SSA begins for if statement (line 104)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to Peg(...): (line 105)
                        # Processing the call arguments (line 105)
                        # Getting the type of 'colour' (line 105)
                        colour_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 27), 'colour', False)
                        # Obtaining the member 'Colours' of a type (line 105)
                        Colours_623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 27), colour_622, 'Colours')
                        # Obtaining the member 'yellow' of a type (line 105)
                        yellow_624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 27), Colours_623, 'yellow')
                        # Processing the call keyword arguments (line 105)
                        kwargs_625 = {}
                        # Getting the type of 'peg' (line 105)
                        peg_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 19), 'peg', False)
                        # Obtaining the member 'Peg' of a type (line 105)
                        Peg_621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 19), peg_620, 'Peg')
                        # Calling Peg(args, kwargs) (line 105)
                        Peg_call_result_626 = invoke(stypy.reporting.localization.Localization(__file__, 105, 19), Peg_621, *[yellow_624], **kwargs_625)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 105)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'stypy_return_type', Peg_call_result_626)
                        # SSA branch for the else part of an if statement (line 104)
                        module_type_store.open_ssa_branch('else')
                        
                        
                        # Call to search(...): (line 106)
                        # Processing the call arguments (line 106)
                        str_629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 24), 'str', '^w')
                        # Getting the type of 's' (line 106)
                        s_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 's', False)
                        # Processing the call keyword arguments (line 106)
                        kwargs_631 = {}
                        # Getting the type of 're' (line 106)
                        re_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 14), 're', False)
                        # Obtaining the member 'search' of a type (line 106)
                        search_628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 14), re_627, 'search')
                        # Calling search(args, kwargs) (line 106)
                        search_call_result_632 = invoke(stypy.reporting.localization.Localization(__file__, 106, 14), search_628, *[str_629, s_630], **kwargs_631)
                        
                        # Getting the type of 'None' (line 106)
                        None_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 39), 'None')
                        # Applying the binary operator 'isnot' (line 106)
                        result_is_not_634 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 14), 'isnot', search_call_result_632, None_633)
                        
                        # Testing if the type of an if condition is none (line 106)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634):
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                        else:
                            
                            # Testing the type of an if condition (line 106)
                            if_condition_635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 13), result_is_not_634)
                            # Assigning a type to the variable 'if_condition_635' (line 106)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 13), 'if_condition_635', if_condition_635)
                            # SSA begins for if statement (line 106)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Call to Peg(...): (line 107)
                            # Processing the call arguments (line 107)
                            # Getting the type of 'colour' (line 107)
                            colour_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 27), 'colour', False)
                            # Obtaining the member 'Colours' of a type (line 107)
                            Colours_639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), colour_638, 'Colours')
                            # Obtaining the member 'white' of a type (line 107)
                            white_640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 27), Colours_639, 'white')
                            # Processing the call keyword arguments (line 107)
                            kwargs_641 = {}
                            # Getting the type of 'peg' (line 107)
                            peg_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'peg', False)
                            # Obtaining the member 'Peg' of a type (line 107)
                            Peg_637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 19), peg_636, 'Peg')
                            # Calling Peg(args, kwargs) (line 107)
                            Peg_call_result_642 = invoke(stypy.reporting.localization.Localization(__file__, 107, 19), Peg_637, *[white_640], **kwargs_641)
                            
                            # Assigning a type to the variable 'stypy_return_type' (line 107)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'stypy_return_type', Peg_call_result_642)
                            # SSA branch for the else part of an if statement (line 106)
                            module_type_store.open_ssa_branch('else')
                            
                            
                            # Call to search(...): (line 108)
                            # Processing the call arguments (line 108)
                            str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'str', '^b')
                            # Getting the type of 's' (line 108)
                            s_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 's', False)
                            # Processing the call keyword arguments (line 108)
                            kwargs_647 = {}
                            # Getting the type of 're' (line 108)
                            re_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 're', False)
                            # Obtaining the member 'search' of a type (line 108)
                            search_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), re_643, 'search')
                            # Calling search(args, kwargs) (line 108)
                            search_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), search_644, *[str_645, s_646], **kwargs_647)
                            
                            # Getting the type of 'None' (line 108)
                            None_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'None')
                            # Applying the binary operator 'isnot' (line 108)
                            result_is_not_650 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), 'isnot', search_call_result_648, None_649)
                            
                            # Testing if the type of an if condition is none (line 108)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650):
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                            else:
                                
                                # Testing the type of an if condition (line 108)
                                if_condition_651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_is_not_650)
                                # Assigning a type to the variable 'if_condition_651' (line 108)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_651', if_condition_651)
                                # SSA begins for if statement (line 108)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Call to Peg(...): (line 109)
                                # Processing the call arguments (line 109)
                                # Getting the type of 'colour' (line 109)
                                colour_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'colour', False)
                                # Obtaining the member 'Colours' of a type (line 109)
                                Colours_655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), colour_654, 'Colours')
                                # Obtaining the member 'black' of a type (line 109)
                                black_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), Colours_655, 'black')
                                # Processing the call keyword arguments (line 109)
                                kwargs_657 = {}
                                # Getting the type of 'peg' (line 109)
                                peg_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'peg', False)
                                # Obtaining the member 'Peg' of a type (line 109)
                                Peg_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), peg_652, 'Peg')
                                # Calling Peg(args, kwargs) (line 109)
                                Peg_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), Peg_653, *[black_656], **kwargs_657)
                                
                                # Assigning a type to the variable 'stypy_return_type' (line 109)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'stypy_return_type', Peg_call_result_658)
                                # SSA branch for the else part of an if statement (line 108)
                                module_type_store.open_ssa_branch('else')
                                # Getting the type of 'None' (line 111)
                                None_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None')
                                # Assigning a type to the variable 'stypy_return_type' (line 111)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', None_659)
                                # SSA join for if statement (line 108)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for if statement (line 106)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 104)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 102)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 100)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 98)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of '__parseColour(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__parseColour' in the type store
        # Getting the type of 'stypy_return_type' (line 96)
        stypy_return_type_660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_660)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__parseColour'
        return stypy_return_type_660


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 10, 0, False)
        # Assigning a type to the variable 'self' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Mastermind.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Mastermind' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'Mastermind', Mastermind)

# Assigning a List to a Name (line 73):

# Obtaining an instance of the builtin type 'list' (line 73)
list_661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 73)
# Adding element type (line 73)
str_662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 17), 'str', 'r y p g')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 16), list_661, str_662)
# Adding element type (line 73)
str_663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 28), 'str', 'r y p g')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 16), list_661, str_663)
# Adding element type (line 73)
str_664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 39), 'str', 'r y p g')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 16), list_661, str_664)
# Adding element type (line 73)
str_665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 50), 'str', 'r y p g')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 16), list_661, str_665)
# Adding element type (line 73)
str_666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 61), 'str', 'r y p g')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 16), list_661, str_666)
# Adding element type (line 73)
str_667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 72), 'str', 'r y p g')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 16), list_661, str_667)
# Adding element type (line 73)
str_668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 83), 'str', 'r y p g')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 16), list_661, str_668)
# Adding element type (line 73)
str_669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 94), 'str', 'r y p g')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 16), list_661, str_669)
# Adding element type (line 73)
str_670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 105), 'str', 'r y p g')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 16), list_661, str_670)
# Adding element type (line 73)
str_671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 116), 'str', 'r y p g')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 16), list_661, str_671)
# Adding element type (line 73)
str_672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 127), 'str', 'r y p g')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 16), list_661, str_672)
# Adding element type (line 73)
str_673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 138), 'str', 'r y p g')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 16), list_661, str_673)

# Getting the type of 'Mastermind'
Mastermind_674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Mastermind')
# Setting the type of the member 'fakeInput' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Mastermind_674, 'fakeInput', list_661)
str_675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, (-1)), 'str', '\n    Instantiate mastermind and invoke play method to play game\n\n')

@norecursion
def main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'main'
    module_type_store = module_type_store.open_function_context('main', 118, 0, False)
    
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

    
    # Assigning a Call to a Name (line 119):
    
    # Call to Mastermind(...): (line 119)
    # Processing the call keyword arguments (line 119)
    kwargs_677 = {}
    # Getting the type of 'Mastermind' (line 119)
    Mastermind_676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'Mastermind', False)
    # Calling Mastermind(args, kwargs) (line 119)
    Mastermind_call_result_678 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), Mastermind_676, *[], **kwargs_677)
    
    # Assigning a type to the variable 'm' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'm', Mastermind_call_result_678)
    
    # Assigning a Num to a Name (line 120):
    int_679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 14), 'int')
    # Assigning a type to the variable 'guesses' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'guesses', int_679)
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'sys' (line 121)
    sys_681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 11), 'sys', False)
    # Obtaining the member 'argv' of a type (line 121)
    argv_682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 11), sys_681, 'argv')
    # Processing the call keyword arguments (line 121)
    kwargs_683 = {}
    # Getting the type of 'len' (line 121)
    len_680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 7), 'len', False)
    # Calling len(args, kwargs) (line 121)
    len_call_result_684 = invoke(stypy.reporting.localization.Localization(__file__, 121, 7), len_680, *[argv_682], **kwargs_683)
    
    int_685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 23), 'int')
    # Applying the binary operator '>' (line 121)
    result_gt_686 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 7), '>', len_call_result_684, int_685)
    
    
    
    # Call to match(...): (line 121)
    # Processing the call arguments (line 121)
    str_689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 38), 'str', '\\d')
    
    # Obtaining the type of the subscript
    int_690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 53), 'int')
    # Getting the type of 'sys' (line 121)
    sys_691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 44), 'sys', False)
    # Obtaining the member 'argv' of a type (line 121)
    argv_692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 44), sys_691, 'argv')
    # Obtaining the member '__getitem__' of a type (line 121)
    getitem___693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 44), argv_692, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 121)
    subscript_call_result_694 = invoke(stypy.reporting.localization.Localization(__file__, 121, 44), getitem___693, int_690)
    
    # Processing the call keyword arguments (line 121)
    kwargs_695 = {}
    # Getting the type of 're' (line 121)
    re_687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 29), 're', False)
    # Obtaining the member 'match' of a type (line 121)
    match_688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 29), re_687, 'match')
    # Calling match(args, kwargs) (line 121)
    match_call_result_696 = invoke(stypy.reporting.localization.Localization(__file__, 121, 29), match_688, *[str_689, subscript_call_result_694], **kwargs_695)
    
    # Getting the type of 'None' (line 121)
    None_697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 64), 'None')
    # Applying the binary operator 'isnot' (line 121)
    result_is_not_698 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 29), 'isnot', match_call_result_696, None_697)
    
    # Applying the binary operator 'and' (line 121)
    result_and_keyword_699 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 7), 'and', result_gt_686, result_is_not_698)
    
    # Testing if the type of an if condition is none (line 121)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 121, 4), result_and_keyword_699):
        pass
    else:
        
        # Testing the type of an if condition (line 121)
        if_condition_700 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 4), result_and_keyword_699)
        # Assigning a type to the variable 'if_condition_700' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'if_condition_700', if_condition_700)
        # SSA begins for if statement (line 121)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 122):
        
        # Call to int(...): (line 122)
        # Processing the call arguments (line 122)
        
        # Obtaining the type of the subscript
        int_702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 31), 'int')
        # Getting the type of 'sys' (line 122)
        sys_703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 22), 'sys', False)
        # Obtaining the member 'argv' of a type (line 122)
        argv_704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 22), sys_703, 'argv')
        # Obtaining the member '__getitem__' of a type (line 122)
        getitem___705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 22), argv_704, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 122)
        subscript_call_result_706 = invoke(stypy.reporting.localization.Localization(__file__, 122, 22), getitem___705, int_702)
        
        # Processing the call keyword arguments (line 122)
        kwargs_707 = {}
        # Getting the type of 'int' (line 122)
        int_701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 18), 'int', False)
        # Calling int(args, kwargs) (line 122)
        int_call_result_708 = invoke(stypy.reporting.localization.Localization(__file__, 122, 18), int_701, *[subscript_call_result_706], **kwargs_707)
        
        # Assigning a type to the variable 'guesses' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'guesses', int_call_result_708)
        # SSA join for if statement (line 121)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to play(...): (line 123)
    # Processing the call arguments (line 123)
    # Getting the type of 'guesses' (line 123)
    guesses_711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 11), 'guesses', False)
    # Processing the call keyword arguments (line 123)
    kwargs_712 = {}
    # Getting the type of 'm' (line 123)
    m_709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'm', False)
    # Obtaining the member 'play' of a type (line 123)
    play_710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 4), m_709, 'play')
    # Calling play(args, kwargs) (line 123)
    play_call_result_713 = invoke(stypy.reporting.localization.Localization(__file__, 123, 4), play_710, *[guesses_711], **kwargs_712)
    
    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 118)
    stypy_return_type_714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_714)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_714

# Assigning a type to the variable 'main' (line 118)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'main', main)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import random
2: import peg
3: import colour
4: 
5: ''' copyright Sean McCarthy, license GPL v2 or later '''
6: 
7: class Code:
8:     '''Class representing a collection of pegs'''
9: 
10:     #__defaultCodeSize = 4
11:     #__pegList = []
12: 
13:     def __init__(self, __pegList=None):
14:     #    self.__pegList = __pegList
15:         self.__defaultCodeSize = 4
16:         self.__pegList = __pegList
17: 
18:     def setPegs(self, __pegList):
19:         self.__pegList = __pegList
20: 
21:     def setRandomCode(self, codeSize=-1):
22:         if codeSize == -1:
23:             codeSize = self.__defaultCodeSize
24:         random.seed()
25:         self.__pegList = []
26:         for i in range(codeSize):
27:             #Avoid to guess the fake code (change lower limit from 0 to 1)
28:             x = peg.Peg(random.randint(1,colour.Colours.numberOfColours-1))
29:             self.__pegList.append(x)
30: 
31:     def getPegs(self):
32:         return self.__pegList
33: 
34:     def equals(self,code):
35:         c1 = code.getPegs();
36:         for i in range(4):
37:             if (not c1[i].equals(self.__pegList[i])):
38:                 return False
39:         return True
40: 
41:     def compare(self,code):
42:         resultPegs = []
43:         secretUsed = [] 
44:         guessUsed = []
45:         count = 0
46:         codeLength = len(self.__pegList)
47:         for i in range(codeLength):
48:             secretUsed.append(False)
49:             guessUsed.append(False)
50: 
51:         '''
52:            Black pegs first: correct colour in correct position
53: 
54:         '''
55:         for i in range(codeLength):
56:             if (self.__pegList[i].equals(code.getPegs()[i])):
57:                 secretUsed[i] = True
58:                 guessUsed[i] = True
59:                 resultPegs.append(peg.Peg(colour.Colours.black))
60:                 count += 1
61: 
62:         '''
63:            White pegs: trickier
64: 
65:            White pegs are for pegs of the correct colour, but in the wrong
66:            place. Each peg should only be evaluated once, hence the "used"
67:            lists.
68: 
69:            Condition below is a shortcut- if there were 3 blacks pegs
70:            then the remaining peg can't be a correct colour (think about it)
71: 
72:         '''
73:         if (count < codeLength-1):
74:             for i in range(codeLength):
75:                 if (guessUsed[i]):
76:                     continue
77:                 for j in range(codeLength):
78:                     if (i != j and not secretUsed[j] \
79:                     and not guessUsed[i] \
80:                     and self.__pegList[j].equals(code.getPegs()[i])):
81:                         resultPegs.append(peg.Peg(colour.Colours.white))
82:                         secretUsed[j] = True
83:                         guessUsed[i] = True
84: 
85:         return Code(resultPegs)
86: 
87: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import random' statement (line 1)
import random

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'random', random, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import peg' statement (line 2)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/mm/')
import_44 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'peg')

if (type(import_44) is not StypyTypeError):

    if (import_44 != 'pyd_module'):
        __import__(import_44)
        sys_modules_45 = sys.modules[import_44]
        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'peg', sys_modules_45.module_type_store, module_type_store)
    else:
        import peg

        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'peg', peg, module_type_store)

else:
    # Assigning a type to the variable 'peg' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'peg', import_44)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/mm/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import colour' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/mm/')
import_46 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'colour')

if (type(import_46) is not StypyTypeError):

    if (import_46 != 'pyd_module'):
        __import__(import_46)
        sys_modules_47 = sys.modules[import_46]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'colour', sys_modules_47.module_type_store, module_type_store)
    else:
        import colour

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'colour', colour, module_type_store)

else:
    # Assigning a type to the variable 'colour' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'colour', import_46)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/mm/')

str_48 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 0), 'str', ' copyright Sean McCarthy, license GPL v2 or later ')
# Declaration of the 'Code' class

class Code:
    str_49 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 4), 'str', 'Class representing a collection of pegs')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 13)
        None_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 33), 'None')
        defaults = [None_50]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 13, 4, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Code.__init__', ['__pegList'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['__pegList'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Num to a Attribute (line 15):
        int_51 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 33), 'int')
        # Getting the type of 'self' (line 15)
        self_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'self')
        # Setting the type of the member '__defaultCodeSize' of a type (line 15)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), self_52, '__defaultCodeSize', int_51)
        
        # Assigning a Name to a Attribute (line 16):
        # Getting the type of '__pegList' (line 16)
        pegList_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 25), '__pegList')
        # Getting the type of 'self' (line 16)
        self_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'self')
        # Setting the type of the member '__pegList' of a type (line 16)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), self_54, '__pegList', pegList_53)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def setPegs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setPegs'
        module_type_store = module_type_store.open_function_context('setPegs', 18, 4, False)
        # Assigning a type to the variable 'self' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Code.setPegs.__dict__.__setitem__('stypy_localization', localization)
        Code.setPegs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Code.setPegs.__dict__.__setitem__('stypy_type_store', module_type_store)
        Code.setPegs.__dict__.__setitem__('stypy_function_name', 'Code.setPegs')
        Code.setPegs.__dict__.__setitem__('stypy_param_names_list', ['__pegList'])
        Code.setPegs.__dict__.__setitem__('stypy_varargs_param_name', None)
        Code.setPegs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Code.setPegs.__dict__.__setitem__('stypy_call_defaults', defaults)
        Code.setPegs.__dict__.__setitem__('stypy_call_varargs', varargs)
        Code.setPegs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Code.setPegs.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Code.setPegs', ['__pegList'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setPegs', localization, ['__pegList'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setPegs(...)' code ##################

        
        # Assigning a Name to a Attribute (line 19):
        # Getting the type of '__pegList' (line 19)
        pegList_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 25), '__pegList')
        # Getting the type of 'self' (line 19)
        self_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'self')
        # Setting the type of the member '__pegList' of a type (line 19)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 8), self_56, '__pegList', pegList_55)
        
        # ################# End of 'setPegs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setPegs' in the type store
        # Getting the type of 'stypy_return_type' (line 18)
        stypy_return_type_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_57)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setPegs'
        return stypy_return_type_57


    @norecursion
    def setRandomCode(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_58 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 37), 'int')
        defaults = [int_58]
        # Create a new context for function 'setRandomCode'
        module_type_store = module_type_store.open_function_context('setRandomCode', 21, 4, False)
        # Assigning a type to the variable 'self' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Code.setRandomCode.__dict__.__setitem__('stypy_localization', localization)
        Code.setRandomCode.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Code.setRandomCode.__dict__.__setitem__('stypy_type_store', module_type_store)
        Code.setRandomCode.__dict__.__setitem__('stypy_function_name', 'Code.setRandomCode')
        Code.setRandomCode.__dict__.__setitem__('stypy_param_names_list', ['codeSize'])
        Code.setRandomCode.__dict__.__setitem__('stypy_varargs_param_name', None)
        Code.setRandomCode.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Code.setRandomCode.__dict__.__setitem__('stypy_call_defaults', defaults)
        Code.setRandomCode.__dict__.__setitem__('stypy_call_varargs', varargs)
        Code.setRandomCode.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Code.setRandomCode.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Code.setRandomCode', ['codeSize'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setRandomCode', localization, ['codeSize'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setRandomCode(...)' code ##################

        
        # Getting the type of 'codeSize' (line 22)
        codeSize_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), 'codeSize')
        int_60 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 23), 'int')
        # Applying the binary operator '==' (line 22)
        result_eq_61 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 11), '==', codeSize_59, int_60)
        
        # Testing if the type of an if condition is none (line 22)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 22, 8), result_eq_61):
            pass
        else:
            
            # Testing the type of an if condition (line 22)
            if_condition_62 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 22, 8), result_eq_61)
            # Assigning a type to the variable 'if_condition_62' (line 22)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'if_condition_62', if_condition_62)
            # SSA begins for if statement (line 22)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 23):
            # Getting the type of 'self' (line 23)
            self_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 23), 'self')
            # Obtaining the member '__defaultCodeSize' of a type (line 23)
            defaultCodeSize_64 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 23), self_63, '__defaultCodeSize')
            # Assigning a type to the variable 'codeSize' (line 23)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'codeSize', defaultCodeSize_64)
            # SSA join for if statement (line 22)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to seed(...): (line 24)
        # Processing the call keyword arguments (line 24)
        kwargs_67 = {}
        # Getting the type of 'random' (line 24)
        random_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'random', False)
        # Obtaining the member 'seed' of a type (line 24)
        seed_66 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), random_65, 'seed')
        # Calling seed(args, kwargs) (line 24)
        seed_call_result_68 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), seed_66, *[], **kwargs_67)
        
        
        # Assigning a List to a Attribute (line 25):
        
        # Obtaining an instance of the builtin type 'list' (line 25)
        list_69 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 25)
        
        # Getting the type of 'self' (line 25)
        self_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self')
        # Setting the type of the member '__pegList' of a type (line 25)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_70, '__pegList', list_69)
        
        
        # Call to range(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'codeSize' (line 26)
        codeSize_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 23), 'codeSize', False)
        # Processing the call keyword arguments (line 26)
        kwargs_73 = {}
        # Getting the type of 'range' (line 26)
        range_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 17), 'range', False)
        # Calling range(args, kwargs) (line 26)
        range_call_result_74 = invoke(stypy.reporting.localization.Localization(__file__, 26, 17), range_71, *[codeSize_72], **kwargs_73)
        
        # Assigning a type to the variable 'range_call_result_74' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'range_call_result_74', range_call_result_74)
        # Testing if the for loop is going to be iterated (line 26)
        # Testing the type of a for loop iterable (line 26)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 26, 8), range_call_result_74)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 26, 8), range_call_result_74):
            # Getting the type of the for loop variable (line 26)
            for_loop_var_75 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 26, 8), range_call_result_74)
            # Assigning a type to the variable 'i' (line 26)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'i', for_loop_var_75)
            # SSA begins for a for statement (line 26)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 28):
            
            # Call to Peg(...): (line 28)
            # Processing the call arguments (line 28)
            
            # Call to randint(...): (line 28)
            # Processing the call arguments (line 28)
            int_80 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 39), 'int')
            # Getting the type of 'colour' (line 28)
            colour_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 41), 'colour', False)
            # Obtaining the member 'Colours' of a type (line 28)
            Colours_82 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 41), colour_81, 'Colours')
            # Obtaining the member 'numberOfColours' of a type (line 28)
            numberOfColours_83 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 41), Colours_82, 'numberOfColours')
            int_84 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 72), 'int')
            # Applying the binary operator '-' (line 28)
            result_sub_85 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 41), '-', numberOfColours_83, int_84)
            
            # Processing the call keyword arguments (line 28)
            kwargs_86 = {}
            # Getting the type of 'random' (line 28)
            random_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 24), 'random', False)
            # Obtaining the member 'randint' of a type (line 28)
            randint_79 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 24), random_78, 'randint')
            # Calling randint(args, kwargs) (line 28)
            randint_call_result_87 = invoke(stypy.reporting.localization.Localization(__file__, 28, 24), randint_79, *[int_80, result_sub_85], **kwargs_86)
            
            # Processing the call keyword arguments (line 28)
            kwargs_88 = {}
            # Getting the type of 'peg' (line 28)
            peg_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), 'peg', False)
            # Obtaining the member 'Peg' of a type (line 28)
            Peg_77 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 16), peg_76, 'Peg')
            # Calling Peg(args, kwargs) (line 28)
            Peg_call_result_89 = invoke(stypy.reporting.localization.Localization(__file__, 28, 16), Peg_77, *[randint_call_result_87], **kwargs_88)
            
            # Assigning a type to the variable 'x' (line 28)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'x', Peg_call_result_89)
            
            # Call to append(...): (line 29)
            # Processing the call arguments (line 29)
            # Getting the type of 'x' (line 29)
            x_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 34), 'x', False)
            # Processing the call keyword arguments (line 29)
            kwargs_94 = {}
            # Getting the type of 'self' (line 29)
            self_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'self', False)
            # Obtaining the member '__pegList' of a type (line 29)
            pegList_91 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 12), self_90, '__pegList')
            # Obtaining the member 'append' of a type (line 29)
            append_92 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 12), pegList_91, 'append')
            # Calling append(args, kwargs) (line 29)
            append_call_result_95 = invoke(stypy.reporting.localization.Localization(__file__, 29, 12), append_92, *[x_93], **kwargs_94)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'setRandomCode(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setRandomCode' in the type store
        # Getting the type of 'stypy_return_type' (line 21)
        stypy_return_type_96 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_96)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setRandomCode'
        return stypy_return_type_96


    @norecursion
    def getPegs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getPegs'
        module_type_store = module_type_store.open_function_context('getPegs', 31, 4, False)
        # Assigning a type to the variable 'self' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Code.getPegs.__dict__.__setitem__('stypy_localization', localization)
        Code.getPegs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Code.getPegs.__dict__.__setitem__('stypy_type_store', module_type_store)
        Code.getPegs.__dict__.__setitem__('stypy_function_name', 'Code.getPegs')
        Code.getPegs.__dict__.__setitem__('stypy_param_names_list', [])
        Code.getPegs.__dict__.__setitem__('stypy_varargs_param_name', None)
        Code.getPegs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Code.getPegs.__dict__.__setitem__('stypy_call_defaults', defaults)
        Code.getPegs.__dict__.__setitem__('stypy_call_varargs', varargs)
        Code.getPegs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Code.getPegs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Code.getPegs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getPegs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getPegs(...)' code ##################

        # Getting the type of 'self' (line 32)
        self_97 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'self')
        # Obtaining the member '__pegList' of a type (line 32)
        pegList_98 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 15), self_97, '__pegList')
        # Assigning a type to the variable 'stypy_return_type' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'stypy_return_type', pegList_98)
        
        # ################# End of 'getPegs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getPegs' in the type store
        # Getting the type of 'stypy_return_type' (line 31)
        stypy_return_type_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_99)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getPegs'
        return stypy_return_type_99


    @norecursion
    def equals(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'equals'
        module_type_store = module_type_store.open_function_context('equals', 34, 4, False)
        # Assigning a type to the variable 'self' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Code.equals.__dict__.__setitem__('stypy_localization', localization)
        Code.equals.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Code.equals.__dict__.__setitem__('stypy_type_store', module_type_store)
        Code.equals.__dict__.__setitem__('stypy_function_name', 'Code.equals')
        Code.equals.__dict__.__setitem__('stypy_param_names_list', ['code'])
        Code.equals.__dict__.__setitem__('stypy_varargs_param_name', None)
        Code.equals.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Code.equals.__dict__.__setitem__('stypy_call_defaults', defaults)
        Code.equals.__dict__.__setitem__('stypy_call_varargs', varargs)
        Code.equals.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Code.equals.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Code.equals', ['code'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'equals', localization, ['code'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'equals(...)' code ##################

        
        # Assigning a Call to a Name (line 35):
        
        # Call to getPegs(...): (line 35)
        # Processing the call keyword arguments (line 35)
        kwargs_102 = {}
        # Getting the type of 'code' (line 35)
        code_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 13), 'code', False)
        # Obtaining the member 'getPegs' of a type (line 35)
        getPegs_101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 13), code_100, 'getPegs')
        # Calling getPegs(args, kwargs) (line 35)
        getPegs_call_result_103 = invoke(stypy.reporting.localization.Localization(__file__, 35, 13), getPegs_101, *[], **kwargs_102)
        
        # Assigning a type to the variable 'c1' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'c1', getPegs_call_result_103)
        
        
        # Call to range(...): (line 36)
        # Processing the call arguments (line 36)
        int_105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 23), 'int')
        # Processing the call keyword arguments (line 36)
        kwargs_106 = {}
        # Getting the type of 'range' (line 36)
        range_104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 17), 'range', False)
        # Calling range(args, kwargs) (line 36)
        range_call_result_107 = invoke(stypy.reporting.localization.Localization(__file__, 36, 17), range_104, *[int_105], **kwargs_106)
        
        # Assigning a type to the variable 'range_call_result_107' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'range_call_result_107', range_call_result_107)
        # Testing if the for loop is going to be iterated (line 36)
        # Testing the type of a for loop iterable (line 36)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 36, 8), range_call_result_107)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 36, 8), range_call_result_107):
            # Getting the type of the for loop variable (line 36)
            for_loop_var_108 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 36, 8), range_call_result_107)
            # Assigning a type to the variable 'i' (line 36)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'i', for_loop_var_108)
            # SSA begins for a for statement (line 36)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to equals(...): (line 37)
            # Processing the call arguments (line 37)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 37)
            i_114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 48), 'i', False)
            # Getting the type of 'self' (line 37)
            self_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 33), 'self', False)
            # Obtaining the member '__pegList' of a type (line 37)
            pegList_116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 33), self_115, '__pegList')
            # Obtaining the member '__getitem__' of a type (line 37)
            getitem___117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 33), pegList_116, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 37)
            subscript_call_result_118 = invoke(stypy.reporting.localization.Localization(__file__, 37, 33), getitem___117, i_114)
            
            # Processing the call keyword arguments (line 37)
            kwargs_119 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 37)
            i_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 23), 'i', False)
            # Getting the type of 'c1' (line 37)
            c1_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 20), 'c1', False)
            # Obtaining the member '__getitem__' of a type (line 37)
            getitem___111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 20), c1_110, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 37)
            subscript_call_result_112 = invoke(stypy.reporting.localization.Localization(__file__, 37, 20), getitem___111, i_109)
            
            # Obtaining the member 'equals' of a type (line 37)
            equals_113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 20), subscript_call_result_112, 'equals')
            # Calling equals(args, kwargs) (line 37)
            equals_call_result_120 = invoke(stypy.reporting.localization.Localization(__file__, 37, 20), equals_113, *[subscript_call_result_118], **kwargs_119)
            
            # Applying the 'not' unary operator (line 37)
            result_not__121 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 16), 'not', equals_call_result_120)
            
            # Testing if the type of an if condition is none (line 37)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 37, 12), result_not__121):
                pass
            else:
                
                # Testing the type of an if condition (line 37)
                if_condition_122 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 37, 12), result_not__121)
                # Assigning a type to the variable 'if_condition_122' (line 37)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'if_condition_122', if_condition_122)
                # SSA begins for if statement (line 37)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'False' (line 38)
                False_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 23), 'False')
                # Assigning a type to the variable 'stypy_return_type' (line 38)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'stypy_return_type', False_123)
                # SSA join for if statement (line 37)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'True' (line 39)
        True_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'stypy_return_type', True_124)
        
        # ################# End of 'equals(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'equals' in the type store
        # Getting the type of 'stypy_return_type' (line 34)
        stypy_return_type_125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_125)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'equals'
        return stypy_return_type_125


    @norecursion
    def compare(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'compare'
        module_type_store = module_type_store.open_function_context('compare', 41, 4, False)
        # Assigning a type to the variable 'self' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Code.compare.__dict__.__setitem__('stypy_localization', localization)
        Code.compare.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Code.compare.__dict__.__setitem__('stypy_type_store', module_type_store)
        Code.compare.__dict__.__setitem__('stypy_function_name', 'Code.compare')
        Code.compare.__dict__.__setitem__('stypy_param_names_list', ['code'])
        Code.compare.__dict__.__setitem__('stypy_varargs_param_name', None)
        Code.compare.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Code.compare.__dict__.__setitem__('stypy_call_defaults', defaults)
        Code.compare.__dict__.__setitem__('stypy_call_varargs', varargs)
        Code.compare.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Code.compare.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Code.compare', ['code'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'compare', localization, ['code'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'compare(...)' code ##################

        
        # Assigning a List to a Name (line 42):
        
        # Obtaining an instance of the builtin type 'list' (line 42)
        list_126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 42)
        
        # Assigning a type to the variable 'resultPegs' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'resultPegs', list_126)
        
        # Assigning a List to a Name (line 43):
        
        # Obtaining an instance of the builtin type 'list' (line 43)
        list_127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 43)
        
        # Assigning a type to the variable 'secretUsed' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'secretUsed', list_127)
        
        # Assigning a List to a Name (line 44):
        
        # Obtaining an instance of the builtin type 'list' (line 44)
        list_128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 44)
        
        # Assigning a type to the variable 'guessUsed' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'guessUsed', list_128)
        
        # Assigning a Num to a Name (line 45):
        int_129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 16), 'int')
        # Assigning a type to the variable 'count' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'count', int_129)
        
        # Assigning a Call to a Name (line 46):
        
        # Call to len(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'self' (line 46)
        self_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 25), 'self', False)
        # Obtaining the member '__pegList' of a type (line 46)
        pegList_132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 25), self_131, '__pegList')
        # Processing the call keyword arguments (line 46)
        kwargs_133 = {}
        # Getting the type of 'len' (line 46)
        len_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 21), 'len', False)
        # Calling len(args, kwargs) (line 46)
        len_call_result_134 = invoke(stypy.reporting.localization.Localization(__file__, 46, 21), len_130, *[pegList_132], **kwargs_133)
        
        # Assigning a type to the variable 'codeLength' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'codeLength', len_call_result_134)
        
        
        # Call to range(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'codeLength' (line 47)
        codeLength_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 23), 'codeLength', False)
        # Processing the call keyword arguments (line 47)
        kwargs_137 = {}
        # Getting the type of 'range' (line 47)
        range_135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 17), 'range', False)
        # Calling range(args, kwargs) (line 47)
        range_call_result_138 = invoke(stypy.reporting.localization.Localization(__file__, 47, 17), range_135, *[codeLength_136], **kwargs_137)
        
        # Assigning a type to the variable 'range_call_result_138' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'range_call_result_138', range_call_result_138)
        # Testing if the for loop is going to be iterated (line 47)
        # Testing the type of a for loop iterable (line 47)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 47, 8), range_call_result_138)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 47, 8), range_call_result_138):
            # Getting the type of the for loop variable (line 47)
            for_loop_var_139 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 47, 8), range_call_result_138)
            # Assigning a type to the variable 'i' (line 47)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'i', for_loop_var_139)
            # SSA begins for a for statement (line 47)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to append(...): (line 48)
            # Processing the call arguments (line 48)
            # Getting the type of 'False' (line 48)
            False_142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 30), 'False', False)
            # Processing the call keyword arguments (line 48)
            kwargs_143 = {}
            # Getting the type of 'secretUsed' (line 48)
            secretUsed_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'secretUsed', False)
            # Obtaining the member 'append' of a type (line 48)
            append_141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 12), secretUsed_140, 'append')
            # Calling append(args, kwargs) (line 48)
            append_call_result_144 = invoke(stypy.reporting.localization.Localization(__file__, 48, 12), append_141, *[False_142], **kwargs_143)
            
            
            # Call to append(...): (line 49)
            # Processing the call arguments (line 49)
            # Getting the type of 'False' (line 49)
            False_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 29), 'False', False)
            # Processing the call keyword arguments (line 49)
            kwargs_148 = {}
            # Getting the type of 'guessUsed' (line 49)
            guessUsed_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'guessUsed', False)
            # Obtaining the member 'append' of a type (line 49)
            append_146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), guessUsed_145, 'append')
            # Calling append(args, kwargs) (line 49)
            append_call_result_149 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), append_146, *[False_147], **kwargs_148)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        str_150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, (-1)), 'str', '\n           Black pegs first: correct colour in correct position\n\n        ')
        
        
        # Call to range(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'codeLength' (line 55)
        codeLength_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 23), 'codeLength', False)
        # Processing the call keyword arguments (line 55)
        kwargs_153 = {}
        # Getting the type of 'range' (line 55)
        range_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 17), 'range', False)
        # Calling range(args, kwargs) (line 55)
        range_call_result_154 = invoke(stypy.reporting.localization.Localization(__file__, 55, 17), range_151, *[codeLength_152], **kwargs_153)
        
        # Assigning a type to the variable 'range_call_result_154' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'range_call_result_154', range_call_result_154)
        # Testing if the for loop is going to be iterated (line 55)
        # Testing the type of a for loop iterable (line 55)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 55, 8), range_call_result_154)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 55, 8), range_call_result_154):
            # Getting the type of the for loop variable (line 55)
            for_loop_var_155 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 55, 8), range_call_result_154)
            # Assigning a type to the variable 'i' (line 55)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'i', for_loop_var_155)
            # SSA begins for a for statement (line 55)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to equals(...): (line 56)
            # Processing the call arguments (line 56)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 56)
            i_162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 56), 'i', False)
            
            # Call to getPegs(...): (line 56)
            # Processing the call keyword arguments (line 56)
            kwargs_165 = {}
            # Getting the type of 'code' (line 56)
            code_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 41), 'code', False)
            # Obtaining the member 'getPegs' of a type (line 56)
            getPegs_164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 41), code_163, 'getPegs')
            # Calling getPegs(args, kwargs) (line 56)
            getPegs_call_result_166 = invoke(stypy.reporting.localization.Localization(__file__, 56, 41), getPegs_164, *[], **kwargs_165)
            
            # Obtaining the member '__getitem__' of a type (line 56)
            getitem___167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 41), getPegs_call_result_166, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 56)
            subscript_call_result_168 = invoke(stypy.reporting.localization.Localization(__file__, 56, 41), getitem___167, i_162)
            
            # Processing the call keyword arguments (line 56)
            kwargs_169 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 56)
            i_156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 31), 'i', False)
            # Getting the type of 'self' (line 56)
            self_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 16), 'self', False)
            # Obtaining the member '__pegList' of a type (line 56)
            pegList_158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 16), self_157, '__pegList')
            # Obtaining the member '__getitem__' of a type (line 56)
            getitem___159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 16), pegList_158, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 56)
            subscript_call_result_160 = invoke(stypy.reporting.localization.Localization(__file__, 56, 16), getitem___159, i_156)
            
            # Obtaining the member 'equals' of a type (line 56)
            equals_161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 16), subscript_call_result_160, 'equals')
            # Calling equals(args, kwargs) (line 56)
            equals_call_result_170 = invoke(stypy.reporting.localization.Localization(__file__, 56, 16), equals_161, *[subscript_call_result_168], **kwargs_169)
            
            # Testing if the type of an if condition is none (line 56)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 56, 12), equals_call_result_170):
                pass
            else:
                
                # Testing the type of an if condition (line 56)
                if_condition_171 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 56, 12), equals_call_result_170)
                # Assigning a type to the variable 'if_condition_171' (line 56)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'if_condition_171', if_condition_171)
                # SSA begins for if statement (line 56)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Subscript (line 57):
                # Getting the type of 'True' (line 57)
                True_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 32), 'True')
                # Getting the type of 'secretUsed' (line 57)
                secretUsed_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'secretUsed')
                # Getting the type of 'i' (line 57)
                i_174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 27), 'i')
                # Storing an element on a container (line 57)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 16), secretUsed_173, (i_174, True_172))
                
                # Assigning a Name to a Subscript (line 58):
                # Getting the type of 'True' (line 58)
                True_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 31), 'True')
                # Getting the type of 'guessUsed' (line 58)
                guessUsed_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'guessUsed')
                # Getting the type of 'i' (line 58)
                i_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 26), 'i')
                # Storing an element on a container (line 58)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 16), guessUsed_176, (i_177, True_175))
                
                # Call to append(...): (line 59)
                # Processing the call arguments (line 59)
                
                # Call to Peg(...): (line 59)
                # Processing the call arguments (line 59)
                # Getting the type of 'colour' (line 59)
                colour_182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 42), 'colour', False)
                # Obtaining the member 'Colours' of a type (line 59)
                Colours_183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 42), colour_182, 'Colours')
                # Obtaining the member 'black' of a type (line 59)
                black_184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 42), Colours_183, 'black')
                # Processing the call keyword arguments (line 59)
                kwargs_185 = {}
                # Getting the type of 'peg' (line 59)
                peg_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 34), 'peg', False)
                # Obtaining the member 'Peg' of a type (line 59)
                Peg_181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 34), peg_180, 'Peg')
                # Calling Peg(args, kwargs) (line 59)
                Peg_call_result_186 = invoke(stypy.reporting.localization.Localization(__file__, 59, 34), Peg_181, *[black_184], **kwargs_185)
                
                # Processing the call keyword arguments (line 59)
                kwargs_187 = {}
                # Getting the type of 'resultPegs' (line 59)
                resultPegs_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'resultPegs', False)
                # Obtaining the member 'append' of a type (line 59)
                append_179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 16), resultPegs_178, 'append')
                # Calling append(args, kwargs) (line 59)
                append_call_result_188 = invoke(stypy.reporting.localization.Localization(__file__, 59, 16), append_179, *[Peg_call_result_186], **kwargs_187)
                
                
                # Getting the type of 'count' (line 60)
                count_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'count')
                int_190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 25), 'int')
                # Applying the binary operator '+=' (line 60)
                result_iadd_191 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 16), '+=', count_189, int_190)
                # Assigning a type to the variable 'count' (line 60)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'count', result_iadd_191)
                
                # SSA join for if statement (line 56)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        str_192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, (-1)), 'str', '\n           White pegs: trickier\n\n           White pegs are for pegs of the correct colour, but in the wrong\n           place. Each peg should only be evaluated once, hence the "used"\n           lists.\n\n           Condition below is a shortcut- if there were 3 blacks pegs\n           then the remaining peg can\'t be a correct colour (think about it)\n\n        ')
        
        # Getting the type of 'count' (line 73)
        count_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'count')
        # Getting the type of 'codeLength' (line 73)
        codeLength_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 20), 'codeLength')
        int_195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 31), 'int')
        # Applying the binary operator '-' (line 73)
        result_sub_196 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 20), '-', codeLength_194, int_195)
        
        # Applying the binary operator '<' (line 73)
        result_lt_197 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 12), '<', count_193, result_sub_196)
        
        # Testing if the type of an if condition is none (line 73)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 73, 8), result_lt_197):
            pass
        else:
            
            # Testing the type of an if condition (line 73)
            if_condition_198 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 8), result_lt_197)
            # Assigning a type to the variable 'if_condition_198' (line 73)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'if_condition_198', if_condition_198)
            # SSA begins for if statement (line 73)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to range(...): (line 74)
            # Processing the call arguments (line 74)
            # Getting the type of 'codeLength' (line 74)
            codeLength_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 27), 'codeLength', False)
            # Processing the call keyword arguments (line 74)
            kwargs_201 = {}
            # Getting the type of 'range' (line 74)
            range_199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 21), 'range', False)
            # Calling range(args, kwargs) (line 74)
            range_call_result_202 = invoke(stypy.reporting.localization.Localization(__file__, 74, 21), range_199, *[codeLength_200], **kwargs_201)
            
            # Assigning a type to the variable 'range_call_result_202' (line 74)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'range_call_result_202', range_call_result_202)
            # Testing if the for loop is going to be iterated (line 74)
            # Testing the type of a for loop iterable (line 74)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 74, 12), range_call_result_202)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 74, 12), range_call_result_202):
                # Getting the type of the for loop variable (line 74)
                for_loop_var_203 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 74, 12), range_call_result_202)
                # Assigning a type to the variable 'i' (line 74)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'i', for_loop_var_203)
                # SSA begins for a for statement (line 74)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 75)
                i_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 30), 'i')
                # Getting the type of 'guessUsed' (line 75)
                guessUsed_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 20), 'guessUsed')
                # Obtaining the member '__getitem__' of a type (line 75)
                getitem___206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 20), guessUsed_205, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                subscript_call_result_207 = invoke(stypy.reporting.localization.Localization(__file__, 75, 20), getitem___206, i_204)
                
                # Testing if the type of an if condition is none (line 75)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 16), subscript_call_result_207):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 75)
                    if_condition_208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 16), subscript_call_result_207)
                    # Assigning a type to the variable 'if_condition_208' (line 75)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'if_condition_208', if_condition_208)
                    # SSA begins for if statement (line 75)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # SSA join for if statement (line 75)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                
                # Call to range(...): (line 77)
                # Processing the call arguments (line 77)
                # Getting the type of 'codeLength' (line 77)
                codeLength_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 31), 'codeLength', False)
                # Processing the call keyword arguments (line 77)
                kwargs_211 = {}
                # Getting the type of 'range' (line 77)
                range_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 25), 'range', False)
                # Calling range(args, kwargs) (line 77)
                range_call_result_212 = invoke(stypy.reporting.localization.Localization(__file__, 77, 25), range_209, *[codeLength_210], **kwargs_211)
                
                # Assigning a type to the variable 'range_call_result_212' (line 77)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 16), 'range_call_result_212', range_call_result_212)
                # Testing if the for loop is going to be iterated (line 77)
                # Testing the type of a for loop iterable (line 77)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 77, 16), range_call_result_212)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 77, 16), range_call_result_212):
                    # Getting the type of the for loop variable (line 77)
                    for_loop_var_213 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 77, 16), range_call_result_212)
                    # Assigning a type to the variable 'j' (line 77)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 16), 'j', for_loop_var_213)
                    # SSA begins for a for statement (line 77)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Evaluating a boolean operation
                    
                    # Getting the type of 'i' (line 78)
                    i_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 24), 'i')
                    # Getting the type of 'j' (line 78)
                    j_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 29), 'j')
                    # Applying the binary operator '!=' (line 78)
                    result_ne_216 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 24), '!=', i_214, j_215)
                    
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'j' (line 78)
                    j_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 50), 'j')
                    # Getting the type of 'secretUsed' (line 78)
                    secretUsed_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 39), 'secretUsed')
                    # Obtaining the member '__getitem__' of a type (line 78)
                    getitem___219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 39), secretUsed_218, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                    subscript_call_result_220 = invoke(stypy.reporting.localization.Localization(__file__, 78, 39), getitem___219, j_217)
                    
                    # Applying the 'not' unary operator (line 78)
                    result_not__221 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 35), 'not', subscript_call_result_220)
                    
                    # Applying the binary operator 'and' (line 78)
                    result_and_keyword_222 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 24), 'and', result_ne_216, result_not__221)
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 79)
                    i_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 38), 'i')
                    # Getting the type of 'guessUsed' (line 79)
                    guessUsed_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 28), 'guessUsed')
                    # Obtaining the member '__getitem__' of a type (line 79)
                    getitem___225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 28), guessUsed_224, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 79)
                    subscript_call_result_226 = invoke(stypy.reporting.localization.Localization(__file__, 79, 28), getitem___225, i_223)
                    
                    # Applying the 'not' unary operator (line 79)
                    result_not__227 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 24), 'not', subscript_call_result_226)
                    
                    # Applying the binary operator 'and' (line 78)
                    result_and_keyword_228 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 24), 'and', result_and_keyword_222, result_not__227)
                    
                    # Call to equals(...): (line 80)
                    # Processing the call arguments (line 80)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 80)
                    i_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 64), 'i', False)
                    
                    # Call to getPegs(...): (line 80)
                    # Processing the call keyword arguments (line 80)
                    kwargs_238 = {}
                    # Getting the type of 'code' (line 80)
                    code_236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 49), 'code', False)
                    # Obtaining the member 'getPegs' of a type (line 80)
                    getPegs_237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 49), code_236, 'getPegs')
                    # Calling getPegs(args, kwargs) (line 80)
                    getPegs_call_result_239 = invoke(stypy.reporting.localization.Localization(__file__, 80, 49), getPegs_237, *[], **kwargs_238)
                    
                    # Obtaining the member '__getitem__' of a type (line 80)
                    getitem___240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 49), getPegs_call_result_239, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 80)
                    subscript_call_result_241 = invoke(stypy.reporting.localization.Localization(__file__, 80, 49), getitem___240, i_235)
                    
                    # Processing the call keyword arguments (line 80)
                    kwargs_242 = {}
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'j' (line 80)
                    j_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 39), 'j', False)
                    # Getting the type of 'self' (line 80)
                    self_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 24), 'self', False)
                    # Obtaining the member '__pegList' of a type (line 80)
                    pegList_231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 24), self_230, '__pegList')
                    # Obtaining the member '__getitem__' of a type (line 80)
                    getitem___232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 24), pegList_231, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 80)
                    subscript_call_result_233 = invoke(stypy.reporting.localization.Localization(__file__, 80, 24), getitem___232, j_229)
                    
                    # Obtaining the member 'equals' of a type (line 80)
                    equals_234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 24), subscript_call_result_233, 'equals')
                    # Calling equals(args, kwargs) (line 80)
                    equals_call_result_243 = invoke(stypy.reporting.localization.Localization(__file__, 80, 24), equals_234, *[subscript_call_result_241], **kwargs_242)
                    
                    # Applying the binary operator 'and' (line 78)
                    result_and_keyword_244 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 24), 'and', result_and_keyword_228, equals_call_result_243)
                    
                    # Testing if the type of an if condition is none (line 78)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 78, 20), result_and_keyword_244):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 78)
                        if_condition_245 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 20), result_and_keyword_244)
                        # Assigning a type to the variable 'if_condition_245' (line 78)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 20), 'if_condition_245', if_condition_245)
                        # SSA begins for if statement (line 78)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to append(...): (line 81)
                        # Processing the call arguments (line 81)
                        
                        # Call to Peg(...): (line 81)
                        # Processing the call arguments (line 81)
                        # Getting the type of 'colour' (line 81)
                        colour_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 50), 'colour', False)
                        # Obtaining the member 'Colours' of a type (line 81)
                        Colours_251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 50), colour_250, 'Colours')
                        # Obtaining the member 'white' of a type (line 81)
                        white_252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 50), Colours_251, 'white')
                        # Processing the call keyword arguments (line 81)
                        kwargs_253 = {}
                        # Getting the type of 'peg' (line 81)
                        peg_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 42), 'peg', False)
                        # Obtaining the member 'Peg' of a type (line 81)
                        Peg_249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 42), peg_248, 'Peg')
                        # Calling Peg(args, kwargs) (line 81)
                        Peg_call_result_254 = invoke(stypy.reporting.localization.Localization(__file__, 81, 42), Peg_249, *[white_252], **kwargs_253)
                        
                        # Processing the call keyword arguments (line 81)
                        kwargs_255 = {}
                        # Getting the type of 'resultPegs' (line 81)
                        resultPegs_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 24), 'resultPegs', False)
                        # Obtaining the member 'append' of a type (line 81)
                        append_247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 24), resultPegs_246, 'append')
                        # Calling append(args, kwargs) (line 81)
                        append_call_result_256 = invoke(stypy.reporting.localization.Localization(__file__, 81, 24), append_247, *[Peg_call_result_254], **kwargs_255)
                        
                        
                        # Assigning a Name to a Subscript (line 82):
                        # Getting the type of 'True' (line 82)
                        True_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 40), 'True')
                        # Getting the type of 'secretUsed' (line 82)
                        secretUsed_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 24), 'secretUsed')
                        # Getting the type of 'j' (line 82)
                        j_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 35), 'j')
                        # Storing an element on a container (line 82)
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 24), secretUsed_258, (j_259, True_257))
                        
                        # Assigning a Name to a Subscript (line 83):
                        # Getting the type of 'True' (line 83)
                        True_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 39), 'True')
                        # Getting the type of 'guessUsed' (line 83)
                        guessUsed_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 24), 'guessUsed')
                        # Getting the type of 'i' (line 83)
                        i_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 34), 'i')
                        # Storing an element on a container (line 83)
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 24), guessUsed_261, (i_262, True_260))
                        # SSA join for if statement (line 78)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 73)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to Code(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'resultPegs' (line 85)
        resultPegs_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 20), 'resultPegs', False)
        # Processing the call keyword arguments (line 85)
        kwargs_265 = {}
        # Getting the type of 'Code' (line 85)
        Code_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 15), 'Code', False)
        # Calling Code(args, kwargs) (line 85)
        Code_call_result_266 = invoke(stypy.reporting.localization.Localization(__file__, 85, 15), Code_263, *[resultPegs_264], **kwargs_265)
        
        # Assigning a type to the variable 'stypy_return_type' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'stypy_return_type', Code_call_result_266)
        
        # ################# End of 'compare(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'compare' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_267)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'compare'
        return stypy_return_type_267


# Assigning a type to the variable 'Code' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'Code', Code)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

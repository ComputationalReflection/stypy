
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import board
2: import row
3: import code
4: 
5: ''' copyright Sean McCarthy, license GPL v2 or later '''
6: 
7: class Game:
8:     '''Class Game, provides functions for playing'''
9: 
10:     def __init__(self,maxguesses=16):
11:         secret = code.Code()
12:         secret.setRandomCode()
13:         self.__secretCode = secret
14:         self.__board = board.Board()
15:         self.__maxguesses = maxguesses
16:         self.__tries = 0
17: 
18:     def getBoard(self):
19:         return self.__board
20: 
21:     def getSecretCode(self):
22:         return self.__secretCode
23: 
24:     def makeGuess(self,guessCode):
25:         self.__tries += 1
26:         self.__board.addRow(row.Row(guessCode, self.getResult(guessCode)))
27: 
28:     def getResult(self,guessCode):
29:         return self.__secretCode.compare(guessCode)
30: 
31:     def lastGuess(self):
32:         return self.__board.getRow(self.__tries-1).getGuess()
33: 
34:     def isOver(self):
35:         if self.__tries > 0:
36:             return self.__tries >= self.__maxguesses \
37:             or self.lastGuess().equals(self.__secretCode)
38:         return False
39: 
40:     def isWon(self):
41:         return self.lastGuess().equals(self.getSecretCode())
42: 
43:     def getTries(self):
44:         return self.__tries
45: 
46:     def getMaxTries(self):
47:         return self.__maxguesses
48: 
49: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import board' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/mm/')
import_317 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'board')

if (type(import_317) is not StypyTypeError):

    if (import_317 != 'pyd_module'):
        __import__(import_317)
        sys_modules_318 = sys.modules[import_317]
        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'board', sys_modules_318.module_type_store, module_type_store)
    else:
        import board

        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'board', board, module_type_store)

else:
    # Assigning a type to the variable 'board' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'board', import_317)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/mm/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import row' statement (line 2)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/mm/')
import_319 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'row')

if (type(import_319) is not StypyTypeError):

    if (import_319 != 'pyd_module'):
        __import__(import_319)
        sys_modules_320 = sys.modules[import_319]
        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'row', sys_modules_320.module_type_store, module_type_store)
    else:
        import row

        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'row', row, module_type_store)

else:
    # Assigning a type to the variable 'row' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'row', import_319)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/mm/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import code' statement (line 3)
import code

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'code', code, module_type_store)

str_321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 0), 'str', ' copyright Sean McCarthy, license GPL v2 or later ')
# Declaration of the 'Game' class

class Game:
    str_322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 4), 'str', 'Class Game, provides functions for playing')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 33), 'int')
        defaults = [int_323]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 10, 4, False)
        # Assigning a type to the variable 'self' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Game.__init__', ['maxguesses'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['maxguesses'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Name (line 11):
        
        # Call to Code(...): (line 11)
        # Processing the call keyword arguments (line 11)
        kwargs_326 = {}
        # Getting the type of 'code' (line 11)
        code_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 17), 'code', False)
        # Obtaining the member 'Code' of a type (line 11)
        Code_325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 17), code_324, 'Code')
        # Calling Code(args, kwargs) (line 11)
        Code_call_result_327 = invoke(stypy.reporting.localization.Localization(__file__, 11, 17), Code_325, *[], **kwargs_326)
        
        # Assigning a type to the variable 'secret' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'secret', Code_call_result_327)
        
        # Call to setRandomCode(...): (line 12)
        # Processing the call keyword arguments (line 12)
        kwargs_330 = {}
        # Getting the type of 'secret' (line 12)
        secret_328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'secret', False)
        # Obtaining the member 'setRandomCode' of a type (line 12)
        setRandomCode_329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 8), secret_328, 'setRandomCode')
        # Calling setRandomCode(args, kwargs) (line 12)
        setRandomCode_call_result_331 = invoke(stypy.reporting.localization.Localization(__file__, 12, 8), setRandomCode_329, *[], **kwargs_330)
        
        
        # Assigning a Name to a Attribute (line 13):
        # Getting the type of 'secret' (line 13)
        secret_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 28), 'secret')
        # Getting the type of 'self' (line 13)
        self_333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'self')
        # Setting the type of the member '__secretCode' of a type (line 13)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 8), self_333, '__secretCode', secret_332)
        
        # Assigning a Call to a Attribute (line 14):
        
        # Call to Board(...): (line 14)
        # Processing the call keyword arguments (line 14)
        kwargs_336 = {}
        # Getting the type of 'board' (line 14)
        board_334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 23), 'board', False)
        # Obtaining the member 'Board' of a type (line 14)
        Board_335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 23), board_334, 'Board')
        # Calling Board(args, kwargs) (line 14)
        Board_call_result_337 = invoke(stypy.reporting.localization.Localization(__file__, 14, 23), Board_335, *[], **kwargs_336)
        
        # Getting the type of 'self' (line 14)
        self_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'self')
        # Setting the type of the member '__board' of a type (line 14)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), self_338, '__board', Board_call_result_337)
        
        # Assigning a Name to a Attribute (line 15):
        # Getting the type of 'maxguesses' (line 15)
        maxguesses_339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 28), 'maxguesses')
        # Getting the type of 'self' (line 15)
        self_340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'self')
        # Setting the type of the member '__maxguesses' of a type (line 15)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), self_340, '__maxguesses', maxguesses_339)
        
        # Assigning a Num to a Attribute (line 16):
        int_341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 23), 'int')
        # Getting the type of 'self' (line 16)
        self_342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'self')
        # Setting the type of the member '__tries' of a type (line 16)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), self_342, '__tries', int_341)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def getBoard(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getBoard'
        module_type_store = module_type_store.open_function_context('getBoard', 18, 4, False)
        # Assigning a type to the variable 'self' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Game.getBoard.__dict__.__setitem__('stypy_localization', localization)
        Game.getBoard.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Game.getBoard.__dict__.__setitem__('stypy_type_store', module_type_store)
        Game.getBoard.__dict__.__setitem__('stypy_function_name', 'Game.getBoard')
        Game.getBoard.__dict__.__setitem__('stypy_param_names_list', [])
        Game.getBoard.__dict__.__setitem__('stypy_varargs_param_name', None)
        Game.getBoard.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Game.getBoard.__dict__.__setitem__('stypy_call_defaults', defaults)
        Game.getBoard.__dict__.__setitem__('stypy_call_varargs', varargs)
        Game.getBoard.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Game.getBoard.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Game.getBoard', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getBoard', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getBoard(...)' code ##################

        # Getting the type of 'self' (line 19)
        self_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), 'self')
        # Obtaining the member '__board' of a type (line 19)
        board_344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 15), self_343, '__board')
        # Assigning a type to the variable 'stypy_return_type' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'stypy_return_type', board_344)
        
        # ################# End of 'getBoard(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getBoard' in the type store
        # Getting the type of 'stypy_return_type' (line 18)
        stypy_return_type_345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_345)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getBoard'
        return stypy_return_type_345


    @norecursion
    def getSecretCode(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getSecretCode'
        module_type_store = module_type_store.open_function_context('getSecretCode', 21, 4, False)
        # Assigning a type to the variable 'self' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Game.getSecretCode.__dict__.__setitem__('stypy_localization', localization)
        Game.getSecretCode.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Game.getSecretCode.__dict__.__setitem__('stypy_type_store', module_type_store)
        Game.getSecretCode.__dict__.__setitem__('stypy_function_name', 'Game.getSecretCode')
        Game.getSecretCode.__dict__.__setitem__('stypy_param_names_list', [])
        Game.getSecretCode.__dict__.__setitem__('stypy_varargs_param_name', None)
        Game.getSecretCode.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Game.getSecretCode.__dict__.__setitem__('stypy_call_defaults', defaults)
        Game.getSecretCode.__dict__.__setitem__('stypy_call_varargs', varargs)
        Game.getSecretCode.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Game.getSecretCode.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Game.getSecretCode', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getSecretCode', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getSecretCode(...)' code ##################

        # Getting the type of 'self' (line 22)
        self_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 15), 'self')
        # Obtaining the member '__secretCode' of a type (line 22)
        secretCode_347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 15), self_346, '__secretCode')
        # Assigning a type to the variable 'stypy_return_type' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'stypy_return_type', secretCode_347)
        
        # ################# End of 'getSecretCode(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getSecretCode' in the type store
        # Getting the type of 'stypy_return_type' (line 21)
        stypy_return_type_348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_348)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getSecretCode'
        return stypy_return_type_348


    @norecursion
    def makeGuess(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'makeGuess'
        module_type_store = module_type_store.open_function_context('makeGuess', 24, 4, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Game.makeGuess.__dict__.__setitem__('stypy_localization', localization)
        Game.makeGuess.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Game.makeGuess.__dict__.__setitem__('stypy_type_store', module_type_store)
        Game.makeGuess.__dict__.__setitem__('stypy_function_name', 'Game.makeGuess')
        Game.makeGuess.__dict__.__setitem__('stypy_param_names_list', ['guessCode'])
        Game.makeGuess.__dict__.__setitem__('stypy_varargs_param_name', None)
        Game.makeGuess.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Game.makeGuess.__dict__.__setitem__('stypy_call_defaults', defaults)
        Game.makeGuess.__dict__.__setitem__('stypy_call_varargs', varargs)
        Game.makeGuess.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Game.makeGuess.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Game.makeGuess', ['guessCode'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'makeGuess', localization, ['guessCode'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'makeGuess(...)' code ##################

        
        # Getting the type of 'self' (line 25)
        self_349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self')
        # Obtaining the member '__tries' of a type (line 25)
        tries_350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_349, '__tries')
        int_351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 24), 'int')
        # Applying the binary operator '+=' (line 25)
        result_iadd_352 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 8), '+=', tries_350, int_351)
        # Getting the type of 'self' (line 25)
        self_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self')
        # Setting the type of the member '__tries' of a type (line 25)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_353, '__tries', result_iadd_352)
        
        
        # Call to addRow(...): (line 26)
        # Processing the call arguments (line 26)
        
        # Call to Row(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'guessCode' (line 26)
        guessCode_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 36), 'guessCode', False)
        
        # Call to getResult(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'guessCode' (line 26)
        guessCode_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 62), 'guessCode', False)
        # Processing the call keyword arguments (line 26)
        kwargs_363 = {}
        # Getting the type of 'self' (line 26)
        self_360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 47), 'self', False)
        # Obtaining the member 'getResult' of a type (line 26)
        getResult_361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 47), self_360, 'getResult')
        # Calling getResult(args, kwargs) (line 26)
        getResult_call_result_364 = invoke(stypy.reporting.localization.Localization(__file__, 26, 47), getResult_361, *[guessCode_362], **kwargs_363)
        
        # Processing the call keyword arguments (line 26)
        kwargs_365 = {}
        # Getting the type of 'row' (line 26)
        row_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 28), 'row', False)
        # Obtaining the member 'Row' of a type (line 26)
        Row_358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 28), row_357, 'Row')
        # Calling Row(args, kwargs) (line 26)
        Row_call_result_366 = invoke(stypy.reporting.localization.Localization(__file__, 26, 28), Row_358, *[guessCode_359, getResult_call_result_364], **kwargs_365)
        
        # Processing the call keyword arguments (line 26)
        kwargs_367 = {}
        # Getting the type of 'self' (line 26)
        self_354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self', False)
        # Obtaining the member '__board' of a type (line 26)
        board_355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), self_354, '__board')
        # Obtaining the member 'addRow' of a type (line 26)
        addRow_356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), board_355, 'addRow')
        # Calling addRow(args, kwargs) (line 26)
        addRow_call_result_368 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), addRow_356, *[Row_call_result_366], **kwargs_367)
        
        
        # ################# End of 'makeGuess(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'makeGuess' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_369)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'makeGuess'
        return stypy_return_type_369


    @norecursion
    def getResult(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getResult'
        module_type_store = module_type_store.open_function_context('getResult', 28, 4, False)
        # Assigning a type to the variable 'self' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Game.getResult.__dict__.__setitem__('stypy_localization', localization)
        Game.getResult.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Game.getResult.__dict__.__setitem__('stypy_type_store', module_type_store)
        Game.getResult.__dict__.__setitem__('stypy_function_name', 'Game.getResult')
        Game.getResult.__dict__.__setitem__('stypy_param_names_list', ['guessCode'])
        Game.getResult.__dict__.__setitem__('stypy_varargs_param_name', None)
        Game.getResult.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Game.getResult.__dict__.__setitem__('stypy_call_defaults', defaults)
        Game.getResult.__dict__.__setitem__('stypy_call_varargs', varargs)
        Game.getResult.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Game.getResult.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Game.getResult', ['guessCode'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getResult', localization, ['guessCode'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getResult(...)' code ##################

        
        # Call to compare(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'guessCode' (line 29)
        guessCode_373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 41), 'guessCode', False)
        # Processing the call keyword arguments (line 29)
        kwargs_374 = {}
        # Getting the type of 'self' (line 29)
        self_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'self', False)
        # Obtaining the member '__secretCode' of a type (line 29)
        secretCode_371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 15), self_370, '__secretCode')
        # Obtaining the member 'compare' of a type (line 29)
        compare_372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 15), secretCode_371, 'compare')
        # Calling compare(args, kwargs) (line 29)
        compare_call_result_375 = invoke(stypy.reporting.localization.Localization(__file__, 29, 15), compare_372, *[guessCode_373], **kwargs_374)
        
        # Assigning a type to the variable 'stypy_return_type' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'stypy_return_type', compare_call_result_375)
        
        # ################# End of 'getResult(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getResult' in the type store
        # Getting the type of 'stypy_return_type' (line 28)
        stypy_return_type_376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_376)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getResult'
        return stypy_return_type_376


    @norecursion
    def lastGuess(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'lastGuess'
        module_type_store = module_type_store.open_function_context('lastGuess', 31, 4, False)
        # Assigning a type to the variable 'self' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Game.lastGuess.__dict__.__setitem__('stypy_localization', localization)
        Game.lastGuess.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Game.lastGuess.__dict__.__setitem__('stypy_type_store', module_type_store)
        Game.lastGuess.__dict__.__setitem__('stypy_function_name', 'Game.lastGuess')
        Game.lastGuess.__dict__.__setitem__('stypy_param_names_list', [])
        Game.lastGuess.__dict__.__setitem__('stypy_varargs_param_name', None)
        Game.lastGuess.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Game.lastGuess.__dict__.__setitem__('stypy_call_defaults', defaults)
        Game.lastGuess.__dict__.__setitem__('stypy_call_varargs', varargs)
        Game.lastGuess.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Game.lastGuess.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Game.lastGuess', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'lastGuess', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'lastGuess(...)' code ##################

        
        # Call to getGuess(...): (line 32)
        # Processing the call keyword arguments (line 32)
        kwargs_387 = {}
        
        # Call to getRow(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'self' (line 32)
        self_380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 35), 'self', False)
        # Obtaining the member '__tries' of a type (line 32)
        tries_381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 35), self_380, '__tries')
        int_382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 48), 'int')
        # Applying the binary operator '-' (line 32)
        result_sub_383 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 35), '-', tries_381, int_382)
        
        # Processing the call keyword arguments (line 32)
        kwargs_384 = {}
        # Getting the type of 'self' (line 32)
        self_377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'self', False)
        # Obtaining the member '__board' of a type (line 32)
        board_378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 15), self_377, '__board')
        # Obtaining the member 'getRow' of a type (line 32)
        getRow_379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 15), board_378, 'getRow')
        # Calling getRow(args, kwargs) (line 32)
        getRow_call_result_385 = invoke(stypy.reporting.localization.Localization(__file__, 32, 15), getRow_379, *[result_sub_383], **kwargs_384)
        
        # Obtaining the member 'getGuess' of a type (line 32)
        getGuess_386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 15), getRow_call_result_385, 'getGuess')
        # Calling getGuess(args, kwargs) (line 32)
        getGuess_call_result_388 = invoke(stypy.reporting.localization.Localization(__file__, 32, 15), getGuess_386, *[], **kwargs_387)
        
        # Assigning a type to the variable 'stypy_return_type' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'stypy_return_type', getGuess_call_result_388)
        
        # ################# End of 'lastGuess(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'lastGuess' in the type store
        # Getting the type of 'stypy_return_type' (line 31)
        stypy_return_type_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_389)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'lastGuess'
        return stypy_return_type_389


    @norecursion
    def isOver(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'isOver'
        module_type_store = module_type_store.open_function_context('isOver', 34, 4, False)
        # Assigning a type to the variable 'self' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Game.isOver.__dict__.__setitem__('stypy_localization', localization)
        Game.isOver.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Game.isOver.__dict__.__setitem__('stypy_type_store', module_type_store)
        Game.isOver.__dict__.__setitem__('stypy_function_name', 'Game.isOver')
        Game.isOver.__dict__.__setitem__('stypy_param_names_list', [])
        Game.isOver.__dict__.__setitem__('stypy_varargs_param_name', None)
        Game.isOver.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Game.isOver.__dict__.__setitem__('stypy_call_defaults', defaults)
        Game.isOver.__dict__.__setitem__('stypy_call_varargs', varargs)
        Game.isOver.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Game.isOver.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Game.isOver', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'isOver', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'isOver(...)' code ##################

        
        # Getting the type of 'self' (line 35)
        self_390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 11), 'self')
        # Obtaining the member '__tries' of a type (line 35)
        tries_391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 11), self_390, '__tries')
        int_392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 26), 'int')
        # Applying the binary operator '>' (line 35)
        result_gt_393 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 11), '>', tries_391, int_392)
        
        # Testing if the type of an if condition is none (line 35)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 35, 8), result_gt_393):
            pass
        else:
            
            # Testing the type of an if condition (line 35)
            if_condition_394 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 8), result_gt_393)
            # Assigning a type to the variable 'if_condition_394' (line 35)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'if_condition_394', if_condition_394)
            # SSA begins for if statement (line 35)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Evaluating a boolean operation
            
            # Getting the type of 'self' (line 36)
            self_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 19), 'self')
            # Obtaining the member '__tries' of a type (line 36)
            tries_396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 19), self_395, '__tries')
            # Getting the type of 'self' (line 36)
            self_397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 35), 'self')
            # Obtaining the member '__maxguesses' of a type (line 36)
            maxguesses_398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 35), self_397, '__maxguesses')
            # Applying the binary operator '>=' (line 36)
            result_ge_399 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 19), '>=', tries_396, maxguesses_398)
            
            
            # Call to equals(...): (line 37)
            # Processing the call arguments (line 37)
            # Getting the type of 'self' (line 37)
            self_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 39), 'self', False)
            # Obtaining the member '__secretCode' of a type (line 37)
            secretCode_406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 39), self_405, '__secretCode')
            # Processing the call keyword arguments (line 37)
            kwargs_407 = {}
            
            # Call to lastGuess(...): (line 37)
            # Processing the call keyword arguments (line 37)
            kwargs_402 = {}
            # Getting the type of 'self' (line 37)
            self_400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 15), 'self', False)
            # Obtaining the member 'lastGuess' of a type (line 37)
            lastGuess_401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 15), self_400, 'lastGuess')
            # Calling lastGuess(args, kwargs) (line 37)
            lastGuess_call_result_403 = invoke(stypy.reporting.localization.Localization(__file__, 37, 15), lastGuess_401, *[], **kwargs_402)
            
            # Obtaining the member 'equals' of a type (line 37)
            equals_404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 15), lastGuess_call_result_403, 'equals')
            # Calling equals(args, kwargs) (line 37)
            equals_call_result_408 = invoke(stypy.reporting.localization.Localization(__file__, 37, 15), equals_404, *[secretCode_406], **kwargs_407)
            
            # Applying the binary operator 'or' (line 36)
            result_or_keyword_409 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 19), 'or', result_ge_399, equals_call_result_408)
            
            # Assigning a type to the variable 'stypy_return_type' (line 36)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'stypy_return_type', result_or_keyword_409)
            # SSA join for if statement (line 35)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'False' (line 38)
        False_410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'stypy_return_type', False_410)
        
        # ################# End of 'isOver(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'isOver' in the type store
        # Getting the type of 'stypy_return_type' (line 34)
        stypy_return_type_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_411)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'isOver'
        return stypy_return_type_411


    @norecursion
    def isWon(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'isWon'
        module_type_store = module_type_store.open_function_context('isWon', 40, 4, False)
        # Assigning a type to the variable 'self' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Game.isWon.__dict__.__setitem__('stypy_localization', localization)
        Game.isWon.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Game.isWon.__dict__.__setitem__('stypy_type_store', module_type_store)
        Game.isWon.__dict__.__setitem__('stypy_function_name', 'Game.isWon')
        Game.isWon.__dict__.__setitem__('stypy_param_names_list', [])
        Game.isWon.__dict__.__setitem__('stypy_varargs_param_name', None)
        Game.isWon.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Game.isWon.__dict__.__setitem__('stypy_call_defaults', defaults)
        Game.isWon.__dict__.__setitem__('stypy_call_varargs', varargs)
        Game.isWon.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Game.isWon.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Game.isWon', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'isWon', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'isWon(...)' code ##################

        
        # Call to equals(...): (line 41)
        # Processing the call arguments (line 41)
        
        # Call to getSecretCode(...): (line 41)
        # Processing the call keyword arguments (line 41)
        kwargs_419 = {}
        # Getting the type of 'self' (line 41)
        self_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 39), 'self', False)
        # Obtaining the member 'getSecretCode' of a type (line 41)
        getSecretCode_418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 39), self_417, 'getSecretCode')
        # Calling getSecretCode(args, kwargs) (line 41)
        getSecretCode_call_result_420 = invoke(stypy.reporting.localization.Localization(__file__, 41, 39), getSecretCode_418, *[], **kwargs_419)
        
        # Processing the call keyword arguments (line 41)
        kwargs_421 = {}
        
        # Call to lastGuess(...): (line 41)
        # Processing the call keyword arguments (line 41)
        kwargs_414 = {}
        # Getting the type of 'self' (line 41)
        self_412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 15), 'self', False)
        # Obtaining the member 'lastGuess' of a type (line 41)
        lastGuess_413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 15), self_412, 'lastGuess')
        # Calling lastGuess(args, kwargs) (line 41)
        lastGuess_call_result_415 = invoke(stypy.reporting.localization.Localization(__file__, 41, 15), lastGuess_413, *[], **kwargs_414)
        
        # Obtaining the member 'equals' of a type (line 41)
        equals_416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 15), lastGuess_call_result_415, 'equals')
        # Calling equals(args, kwargs) (line 41)
        equals_call_result_422 = invoke(stypy.reporting.localization.Localization(__file__, 41, 15), equals_416, *[getSecretCode_call_result_420], **kwargs_421)
        
        # Assigning a type to the variable 'stypy_return_type' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'stypy_return_type', equals_call_result_422)
        
        # ################# End of 'isWon(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'isWon' in the type store
        # Getting the type of 'stypy_return_type' (line 40)
        stypy_return_type_423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_423)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'isWon'
        return stypy_return_type_423


    @norecursion
    def getTries(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getTries'
        module_type_store = module_type_store.open_function_context('getTries', 43, 4, False)
        # Assigning a type to the variable 'self' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Game.getTries.__dict__.__setitem__('stypy_localization', localization)
        Game.getTries.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Game.getTries.__dict__.__setitem__('stypy_type_store', module_type_store)
        Game.getTries.__dict__.__setitem__('stypy_function_name', 'Game.getTries')
        Game.getTries.__dict__.__setitem__('stypy_param_names_list', [])
        Game.getTries.__dict__.__setitem__('stypy_varargs_param_name', None)
        Game.getTries.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Game.getTries.__dict__.__setitem__('stypy_call_defaults', defaults)
        Game.getTries.__dict__.__setitem__('stypy_call_varargs', varargs)
        Game.getTries.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Game.getTries.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Game.getTries', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getTries', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getTries(...)' code ##################

        # Getting the type of 'self' (line 44)
        self_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'self')
        # Obtaining the member '__tries' of a type (line 44)
        tries_425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 15), self_424, '__tries')
        # Assigning a type to the variable 'stypy_return_type' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'stypy_return_type', tries_425)
        
        # ################# End of 'getTries(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getTries' in the type store
        # Getting the type of 'stypy_return_type' (line 43)
        stypy_return_type_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_426)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getTries'
        return stypy_return_type_426


    @norecursion
    def getMaxTries(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getMaxTries'
        module_type_store = module_type_store.open_function_context('getMaxTries', 46, 4, False)
        # Assigning a type to the variable 'self' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Game.getMaxTries.__dict__.__setitem__('stypy_localization', localization)
        Game.getMaxTries.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Game.getMaxTries.__dict__.__setitem__('stypy_type_store', module_type_store)
        Game.getMaxTries.__dict__.__setitem__('stypy_function_name', 'Game.getMaxTries')
        Game.getMaxTries.__dict__.__setitem__('stypy_param_names_list', [])
        Game.getMaxTries.__dict__.__setitem__('stypy_varargs_param_name', None)
        Game.getMaxTries.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Game.getMaxTries.__dict__.__setitem__('stypy_call_defaults', defaults)
        Game.getMaxTries.__dict__.__setitem__('stypy_call_varargs', varargs)
        Game.getMaxTries.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Game.getMaxTries.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Game.getMaxTries', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getMaxTries', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getMaxTries(...)' code ##################

        # Getting the type of 'self' (line 47)
        self_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 15), 'self')
        # Obtaining the member '__maxguesses' of a type (line 47)
        maxguesses_428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 15), self_427, '__maxguesses')
        # Assigning a type to the variable 'stypy_return_type' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'stypy_return_type', maxguesses_428)
        
        # ################# End of 'getMaxTries(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getMaxTries' in the type store
        # Getting the type of 'stypy_return_type' (line 46)
        stypy_return_type_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_429)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getMaxTries'
        return stypy_return_type_429


# Assigning a type to the variable 'Game' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'Game', Game)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

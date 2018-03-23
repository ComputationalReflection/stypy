
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' copyright Sean McCarthy, license GPL v2 or later '''
2: 
3: def getColourName(i):
4:     if i == 0:
5:         return "red"
6:     elif i == 1:
7:         return "green"
8:     elif i == 2:
9:         return "purple"
10:     elif i == 3:
11:         return "yellow"
12:     elif i == 4:
13:         return "white"
14:     elif i == 5:
15:         return "black"
16: 
17: class Colours:
18:     numberOfColours = 6
19:     red = 0
20:     green = 1
21:     purple = 2
22:     yellow = 3
23:     white = 4
24:     black = 5 
25: 
26:     def getNumberOfColours(self):
27:         return self.numberOfColours
28: 
29: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', ' copyright Sean McCarthy, license GPL v2 or later ')

@norecursion
def getColourName(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'getColourName'
    module_type_store = module_type_store.open_function_context('getColourName', 3, 0, False)
    
    # Passed parameters checking function
    getColourName.stypy_localization = localization
    getColourName.stypy_type_of_self = None
    getColourName.stypy_type_store = module_type_store
    getColourName.stypy_function_name = 'getColourName'
    getColourName.stypy_param_names_list = ['i']
    getColourName.stypy_varargs_param_name = None
    getColourName.stypy_kwargs_param_name = None
    getColourName.stypy_call_defaults = defaults
    getColourName.stypy_call_varargs = varargs
    getColourName.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getColourName', ['i'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getColourName', localization, ['i'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getColourName(...)' code ##################

    
    # Getting the type of 'i' (line 4)
    i_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 7), 'i')
    int_270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 12), 'int')
    # Applying the binary operator '==' (line 4)
    result_eq_271 = python_operator(stypy.reporting.localization.Localization(__file__, 4, 7), '==', i_269, int_270)
    
    # Testing if the type of an if condition is none (line 4)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 4, 4), result_eq_271):
        
        # Getting the type of 'i' (line 6)
        i_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 9), 'i')
        int_275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 14), 'int')
        # Applying the binary operator '==' (line 6)
        result_eq_276 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 9), '==', i_274, int_275)
        
        # Testing if the type of an if condition is none (line 6)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 6, 9), result_eq_276):
            
            # Getting the type of 'i' (line 8)
            i_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 9), 'i')
            int_280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 14), 'int')
            # Applying the binary operator '==' (line 8)
            result_eq_281 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 9), '==', i_279, int_280)
            
            # Testing if the type of an if condition is none (line 8)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 8, 9), result_eq_281):
                
                # Getting the type of 'i' (line 10)
                i_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'i')
                int_285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 14), 'int')
                # Applying the binary operator '==' (line 10)
                result_eq_286 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 9), '==', i_284, int_285)
                
                # Testing if the type of an if condition is none (line 10)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 10, 9), result_eq_286):
                    
                    # Getting the type of 'i' (line 12)
                    i_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'i')
                    int_290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'int')
                    # Applying the binary operator '==' (line 12)
                    result_eq_291 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 9), '==', i_289, int_290)
                    
                    # Testing if the type of an if condition is none (line 12)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291):
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 12)
                        if_condition_292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291)
                        # Assigning a type to the variable 'if_condition_292' (line 12)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'if_condition_292', if_condition_292)
                        # SSA begins for if statement (line 12)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        str_293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'str', 'white')
                        # Assigning a type to the variable 'stypy_return_type' (line 13)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type', str_293)
                        # SSA branch for the else part of an if statement (line 12)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 12)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 10)
                    if_condition_287 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 10, 9), result_eq_286)
                    # Assigning a type to the variable 'if_condition_287' (line 10)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'if_condition_287', if_condition_287)
                    # SSA begins for if statement (line 10)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    str_288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 15), 'str', 'yellow')
                    # Assigning a type to the variable 'stypy_return_type' (line 11)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'stypy_return_type', str_288)
                    # SSA branch for the else part of an if statement (line 10)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'i' (line 12)
                    i_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'i')
                    int_290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'int')
                    # Applying the binary operator '==' (line 12)
                    result_eq_291 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 9), '==', i_289, int_290)
                    
                    # Testing if the type of an if condition is none (line 12)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291):
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 12)
                        if_condition_292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291)
                        # Assigning a type to the variable 'if_condition_292' (line 12)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'if_condition_292', if_condition_292)
                        # SSA begins for if statement (line 12)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        str_293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'str', 'white')
                        # Assigning a type to the variable 'stypy_return_type' (line 13)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type', str_293)
                        # SSA branch for the else part of an if statement (line 12)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 12)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 10)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 8)
                if_condition_282 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 8, 9), result_eq_281)
                # Assigning a type to the variable 'if_condition_282' (line 8)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 9), 'if_condition_282', if_condition_282)
                # SSA begins for if statement (line 8)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                str_283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 15), 'str', 'purple')
                # Assigning a type to the variable 'stypy_return_type' (line 9)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'stypy_return_type', str_283)
                # SSA branch for the else part of an if statement (line 8)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'i' (line 10)
                i_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'i')
                int_285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 14), 'int')
                # Applying the binary operator '==' (line 10)
                result_eq_286 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 9), '==', i_284, int_285)
                
                # Testing if the type of an if condition is none (line 10)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 10, 9), result_eq_286):
                    
                    # Getting the type of 'i' (line 12)
                    i_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'i')
                    int_290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'int')
                    # Applying the binary operator '==' (line 12)
                    result_eq_291 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 9), '==', i_289, int_290)
                    
                    # Testing if the type of an if condition is none (line 12)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291):
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 12)
                        if_condition_292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291)
                        # Assigning a type to the variable 'if_condition_292' (line 12)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'if_condition_292', if_condition_292)
                        # SSA begins for if statement (line 12)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        str_293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'str', 'white')
                        # Assigning a type to the variable 'stypy_return_type' (line 13)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type', str_293)
                        # SSA branch for the else part of an if statement (line 12)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 12)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 10)
                    if_condition_287 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 10, 9), result_eq_286)
                    # Assigning a type to the variable 'if_condition_287' (line 10)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'if_condition_287', if_condition_287)
                    # SSA begins for if statement (line 10)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    str_288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 15), 'str', 'yellow')
                    # Assigning a type to the variable 'stypy_return_type' (line 11)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'stypy_return_type', str_288)
                    # SSA branch for the else part of an if statement (line 10)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'i' (line 12)
                    i_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'i')
                    int_290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'int')
                    # Applying the binary operator '==' (line 12)
                    result_eq_291 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 9), '==', i_289, int_290)
                    
                    # Testing if the type of an if condition is none (line 12)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291):
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 12)
                        if_condition_292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291)
                        # Assigning a type to the variable 'if_condition_292' (line 12)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'if_condition_292', if_condition_292)
                        # SSA begins for if statement (line 12)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        str_293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'str', 'white')
                        # Assigning a type to the variable 'stypy_return_type' (line 13)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type', str_293)
                        # SSA branch for the else part of an if statement (line 12)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 12)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 10)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 8)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 6)
            if_condition_277 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 6, 9), result_eq_276)
            # Assigning a type to the variable 'if_condition_277' (line 6)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 9), 'if_condition_277', if_condition_277)
            # SSA begins for if statement (line 6)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'str', 'green')
            # Assigning a type to the variable 'stypy_return_type' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'stypy_return_type', str_278)
            # SSA branch for the else part of an if statement (line 6)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'i' (line 8)
            i_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 9), 'i')
            int_280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 14), 'int')
            # Applying the binary operator '==' (line 8)
            result_eq_281 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 9), '==', i_279, int_280)
            
            # Testing if the type of an if condition is none (line 8)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 8, 9), result_eq_281):
                
                # Getting the type of 'i' (line 10)
                i_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'i')
                int_285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 14), 'int')
                # Applying the binary operator '==' (line 10)
                result_eq_286 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 9), '==', i_284, int_285)
                
                # Testing if the type of an if condition is none (line 10)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 10, 9), result_eq_286):
                    
                    # Getting the type of 'i' (line 12)
                    i_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'i')
                    int_290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'int')
                    # Applying the binary operator '==' (line 12)
                    result_eq_291 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 9), '==', i_289, int_290)
                    
                    # Testing if the type of an if condition is none (line 12)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291):
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 12)
                        if_condition_292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291)
                        # Assigning a type to the variable 'if_condition_292' (line 12)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'if_condition_292', if_condition_292)
                        # SSA begins for if statement (line 12)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        str_293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'str', 'white')
                        # Assigning a type to the variable 'stypy_return_type' (line 13)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type', str_293)
                        # SSA branch for the else part of an if statement (line 12)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 12)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 10)
                    if_condition_287 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 10, 9), result_eq_286)
                    # Assigning a type to the variable 'if_condition_287' (line 10)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'if_condition_287', if_condition_287)
                    # SSA begins for if statement (line 10)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    str_288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 15), 'str', 'yellow')
                    # Assigning a type to the variable 'stypy_return_type' (line 11)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'stypy_return_type', str_288)
                    # SSA branch for the else part of an if statement (line 10)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'i' (line 12)
                    i_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'i')
                    int_290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'int')
                    # Applying the binary operator '==' (line 12)
                    result_eq_291 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 9), '==', i_289, int_290)
                    
                    # Testing if the type of an if condition is none (line 12)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291):
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 12)
                        if_condition_292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291)
                        # Assigning a type to the variable 'if_condition_292' (line 12)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'if_condition_292', if_condition_292)
                        # SSA begins for if statement (line 12)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        str_293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'str', 'white')
                        # Assigning a type to the variable 'stypy_return_type' (line 13)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type', str_293)
                        # SSA branch for the else part of an if statement (line 12)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 12)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 10)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 8)
                if_condition_282 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 8, 9), result_eq_281)
                # Assigning a type to the variable 'if_condition_282' (line 8)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 9), 'if_condition_282', if_condition_282)
                # SSA begins for if statement (line 8)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                str_283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 15), 'str', 'purple')
                # Assigning a type to the variable 'stypy_return_type' (line 9)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'stypy_return_type', str_283)
                # SSA branch for the else part of an if statement (line 8)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'i' (line 10)
                i_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'i')
                int_285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 14), 'int')
                # Applying the binary operator '==' (line 10)
                result_eq_286 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 9), '==', i_284, int_285)
                
                # Testing if the type of an if condition is none (line 10)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 10, 9), result_eq_286):
                    
                    # Getting the type of 'i' (line 12)
                    i_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'i')
                    int_290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'int')
                    # Applying the binary operator '==' (line 12)
                    result_eq_291 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 9), '==', i_289, int_290)
                    
                    # Testing if the type of an if condition is none (line 12)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291):
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 12)
                        if_condition_292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291)
                        # Assigning a type to the variable 'if_condition_292' (line 12)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'if_condition_292', if_condition_292)
                        # SSA begins for if statement (line 12)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        str_293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'str', 'white')
                        # Assigning a type to the variable 'stypy_return_type' (line 13)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type', str_293)
                        # SSA branch for the else part of an if statement (line 12)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 12)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 10)
                    if_condition_287 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 10, 9), result_eq_286)
                    # Assigning a type to the variable 'if_condition_287' (line 10)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'if_condition_287', if_condition_287)
                    # SSA begins for if statement (line 10)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    str_288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 15), 'str', 'yellow')
                    # Assigning a type to the variable 'stypy_return_type' (line 11)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'stypy_return_type', str_288)
                    # SSA branch for the else part of an if statement (line 10)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'i' (line 12)
                    i_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'i')
                    int_290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'int')
                    # Applying the binary operator '==' (line 12)
                    result_eq_291 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 9), '==', i_289, int_290)
                    
                    # Testing if the type of an if condition is none (line 12)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291):
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 12)
                        if_condition_292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291)
                        # Assigning a type to the variable 'if_condition_292' (line 12)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'if_condition_292', if_condition_292)
                        # SSA begins for if statement (line 12)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        str_293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'str', 'white')
                        # Assigning a type to the variable 'stypy_return_type' (line 13)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type', str_293)
                        # SSA branch for the else part of an if statement (line 12)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 12)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 10)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 8)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 6)
            module_type_store = module_type_store.join_ssa_context()
            

    else:
        
        # Testing the type of an if condition (line 4)
        if_condition_272 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 4, 4), result_eq_271)
        # Assigning a type to the variable 'if_condition_272' (line 4)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'if_condition_272', if_condition_272)
        # SSA begins for if statement (line 4)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'str', 'red')
        # Assigning a type to the variable 'stypy_return_type' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 8), 'stypy_return_type', str_273)
        # SSA branch for the else part of an if statement (line 4)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'i' (line 6)
        i_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 9), 'i')
        int_275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 14), 'int')
        # Applying the binary operator '==' (line 6)
        result_eq_276 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 9), '==', i_274, int_275)
        
        # Testing if the type of an if condition is none (line 6)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 6, 9), result_eq_276):
            
            # Getting the type of 'i' (line 8)
            i_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 9), 'i')
            int_280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 14), 'int')
            # Applying the binary operator '==' (line 8)
            result_eq_281 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 9), '==', i_279, int_280)
            
            # Testing if the type of an if condition is none (line 8)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 8, 9), result_eq_281):
                
                # Getting the type of 'i' (line 10)
                i_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'i')
                int_285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 14), 'int')
                # Applying the binary operator '==' (line 10)
                result_eq_286 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 9), '==', i_284, int_285)
                
                # Testing if the type of an if condition is none (line 10)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 10, 9), result_eq_286):
                    
                    # Getting the type of 'i' (line 12)
                    i_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'i')
                    int_290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'int')
                    # Applying the binary operator '==' (line 12)
                    result_eq_291 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 9), '==', i_289, int_290)
                    
                    # Testing if the type of an if condition is none (line 12)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291):
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 12)
                        if_condition_292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291)
                        # Assigning a type to the variable 'if_condition_292' (line 12)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'if_condition_292', if_condition_292)
                        # SSA begins for if statement (line 12)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        str_293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'str', 'white')
                        # Assigning a type to the variable 'stypy_return_type' (line 13)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type', str_293)
                        # SSA branch for the else part of an if statement (line 12)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 12)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 10)
                    if_condition_287 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 10, 9), result_eq_286)
                    # Assigning a type to the variable 'if_condition_287' (line 10)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'if_condition_287', if_condition_287)
                    # SSA begins for if statement (line 10)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    str_288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 15), 'str', 'yellow')
                    # Assigning a type to the variable 'stypy_return_type' (line 11)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'stypy_return_type', str_288)
                    # SSA branch for the else part of an if statement (line 10)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'i' (line 12)
                    i_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'i')
                    int_290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'int')
                    # Applying the binary operator '==' (line 12)
                    result_eq_291 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 9), '==', i_289, int_290)
                    
                    # Testing if the type of an if condition is none (line 12)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291):
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 12)
                        if_condition_292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291)
                        # Assigning a type to the variable 'if_condition_292' (line 12)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'if_condition_292', if_condition_292)
                        # SSA begins for if statement (line 12)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        str_293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'str', 'white')
                        # Assigning a type to the variable 'stypy_return_type' (line 13)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type', str_293)
                        # SSA branch for the else part of an if statement (line 12)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 12)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 10)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 8)
                if_condition_282 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 8, 9), result_eq_281)
                # Assigning a type to the variable 'if_condition_282' (line 8)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 9), 'if_condition_282', if_condition_282)
                # SSA begins for if statement (line 8)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                str_283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 15), 'str', 'purple')
                # Assigning a type to the variable 'stypy_return_type' (line 9)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'stypy_return_type', str_283)
                # SSA branch for the else part of an if statement (line 8)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'i' (line 10)
                i_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'i')
                int_285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 14), 'int')
                # Applying the binary operator '==' (line 10)
                result_eq_286 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 9), '==', i_284, int_285)
                
                # Testing if the type of an if condition is none (line 10)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 10, 9), result_eq_286):
                    
                    # Getting the type of 'i' (line 12)
                    i_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'i')
                    int_290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'int')
                    # Applying the binary operator '==' (line 12)
                    result_eq_291 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 9), '==', i_289, int_290)
                    
                    # Testing if the type of an if condition is none (line 12)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291):
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 12)
                        if_condition_292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291)
                        # Assigning a type to the variable 'if_condition_292' (line 12)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'if_condition_292', if_condition_292)
                        # SSA begins for if statement (line 12)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        str_293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'str', 'white')
                        # Assigning a type to the variable 'stypy_return_type' (line 13)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type', str_293)
                        # SSA branch for the else part of an if statement (line 12)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 12)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 10)
                    if_condition_287 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 10, 9), result_eq_286)
                    # Assigning a type to the variable 'if_condition_287' (line 10)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'if_condition_287', if_condition_287)
                    # SSA begins for if statement (line 10)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    str_288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 15), 'str', 'yellow')
                    # Assigning a type to the variable 'stypy_return_type' (line 11)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'stypy_return_type', str_288)
                    # SSA branch for the else part of an if statement (line 10)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'i' (line 12)
                    i_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'i')
                    int_290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'int')
                    # Applying the binary operator '==' (line 12)
                    result_eq_291 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 9), '==', i_289, int_290)
                    
                    # Testing if the type of an if condition is none (line 12)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291):
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 12)
                        if_condition_292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291)
                        # Assigning a type to the variable 'if_condition_292' (line 12)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'if_condition_292', if_condition_292)
                        # SSA begins for if statement (line 12)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        str_293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'str', 'white')
                        # Assigning a type to the variable 'stypy_return_type' (line 13)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type', str_293)
                        # SSA branch for the else part of an if statement (line 12)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 12)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 10)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 8)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 6)
            if_condition_277 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 6, 9), result_eq_276)
            # Assigning a type to the variable 'if_condition_277' (line 6)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 9), 'if_condition_277', if_condition_277)
            # SSA begins for if statement (line 6)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'str', 'green')
            # Assigning a type to the variable 'stypy_return_type' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'stypy_return_type', str_278)
            # SSA branch for the else part of an if statement (line 6)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'i' (line 8)
            i_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 9), 'i')
            int_280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 14), 'int')
            # Applying the binary operator '==' (line 8)
            result_eq_281 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 9), '==', i_279, int_280)
            
            # Testing if the type of an if condition is none (line 8)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 8, 9), result_eq_281):
                
                # Getting the type of 'i' (line 10)
                i_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'i')
                int_285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 14), 'int')
                # Applying the binary operator '==' (line 10)
                result_eq_286 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 9), '==', i_284, int_285)
                
                # Testing if the type of an if condition is none (line 10)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 10, 9), result_eq_286):
                    
                    # Getting the type of 'i' (line 12)
                    i_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'i')
                    int_290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'int')
                    # Applying the binary operator '==' (line 12)
                    result_eq_291 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 9), '==', i_289, int_290)
                    
                    # Testing if the type of an if condition is none (line 12)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291):
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 12)
                        if_condition_292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291)
                        # Assigning a type to the variable 'if_condition_292' (line 12)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'if_condition_292', if_condition_292)
                        # SSA begins for if statement (line 12)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        str_293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'str', 'white')
                        # Assigning a type to the variable 'stypy_return_type' (line 13)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type', str_293)
                        # SSA branch for the else part of an if statement (line 12)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 12)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 10)
                    if_condition_287 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 10, 9), result_eq_286)
                    # Assigning a type to the variable 'if_condition_287' (line 10)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'if_condition_287', if_condition_287)
                    # SSA begins for if statement (line 10)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    str_288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 15), 'str', 'yellow')
                    # Assigning a type to the variable 'stypy_return_type' (line 11)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'stypy_return_type', str_288)
                    # SSA branch for the else part of an if statement (line 10)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'i' (line 12)
                    i_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'i')
                    int_290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'int')
                    # Applying the binary operator '==' (line 12)
                    result_eq_291 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 9), '==', i_289, int_290)
                    
                    # Testing if the type of an if condition is none (line 12)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291):
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 12)
                        if_condition_292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291)
                        # Assigning a type to the variable 'if_condition_292' (line 12)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'if_condition_292', if_condition_292)
                        # SSA begins for if statement (line 12)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        str_293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'str', 'white')
                        # Assigning a type to the variable 'stypy_return_type' (line 13)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type', str_293)
                        # SSA branch for the else part of an if statement (line 12)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 12)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 10)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 8)
                if_condition_282 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 8, 9), result_eq_281)
                # Assigning a type to the variable 'if_condition_282' (line 8)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 9), 'if_condition_282', if_condition_282)
                # SSA begins for if statement (line 8)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                str_283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 15), 'str', 'purple')
                # Assigning a type to the variable 'stypy_return_type' (line 9)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'stypy_return_type', str_283)
                # SSA branch for the else part of an if statement (line 8)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'i' (line 10)
                i_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'i')
                int_285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 14), 'int')
                # Applying the binary operator '==' (line 10)
                result_eq_286 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 9), '==', i_284, int_285)
                
                # Testing if the type of an if condition is none (line 10)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 10, 9), result_eq_286):
                    
                    # Getting the type of 'i' (line 12)
                    i_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'i')
                    int_290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'int')
                    # Applying the binary operator '==' (line 12)
                    result_eq_291 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 9), '==', i_289, int_290)
                    
                    # Testing if the type of an if condition is none (line 12)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291):
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 12)
                        if_condition_292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291)
                        # Assigning a type to the variable 'if_condition_292' (line 12)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'if_condition_292', if_condition_292)
                        # SSA begins for if statement (line 12)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        str_293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'str', 'white')
                        # Assigning a type to the variable 'stypy_return_type' (line 13)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type', str_293)
                        # SSA branch for the else part of an if statement (line 12)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 12)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 10)
                    if_condition_287 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 10, 9), result_eq_286)
                    # Assigning a type to the variable 'if_condition_287' (line 10)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'if_condition_287', if_condition_287)
                    # SSA begins for if statement (line 10)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    str_288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 15), 'str', 'yellow')
                    # Assigning a type to the variable 'stypy_return_type' (line 11)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'stypy_return_type', str_288)
                    # SSA branch for the else part of an if statement (line 10)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'i' (line 12)
                    i_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'i')
                    int_290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'int')
                    # Applying the binary operator '==' (line 12)
                    result_eq_291 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 9), '==', i_289, int_290)
                    
                    # Testing if the type of an if condition is none (line 12)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291):
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 12)
                        if_condition_292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 12, 9), result_eq_291)
                        # Assigning a type to the variable 'if_condition_292' (line 12)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'if_condition_292', if_condition_292)
                        # SSA begins for if statement (line 12)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        str_293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'str', 'white')
                        # Assigning a type to the variable 'stypy_return_type' (line 13)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type', str_293)
                        # SSA branch for the else part of an if statement (line 12)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'i' (line 14)
                        i_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'i')
                        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
                        # Applying the binary operator '==' (line 14)
                        result_eq_296 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '==', i_294, int_295)
                        
                        # Testing if the type of an if condition is none (line 14)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 14)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 9), result_eq_296)
                            # Assigning a type to the variable 'if_condition_297' (line 14)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 14)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            str_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'black')
                            # Assigning a type to the variable 'stypy_return_type' (line 15)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_298)
                            # SSA join for if statement (line 14)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 12)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 10)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 8)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 6)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 4)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'getColourName(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getColourName' in the type store
    # Getting the type of 'stypy_return_type' (line 3)
    stypy_return_type_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_299)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getColourName'
    return stypy_return_type_299

# Assigning a type to the variable 'getColourName' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'getColourName', getColourName)
# Declaration of the 'Colours' class

class Colours:

    @norecursion
    def getNumberOfColours(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getNumberOfColours'
        module_type_store = module_type_store.open_function_context('getNumberOfColours', 26, 4, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Colours.getNumberOfColours.__dict__.__setitem__('stypy_localization', localization)
        Colours.getNumberOfColours.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Colours.getNumberOfColours.__dict__.__setitem__('stypy_type_store', module_type_store)
        Colours.getNumberOfColours.__dict__.__setitem__('stypy_function_name', 'Colours.getNumberOfColours')
        Colours.getNumberOfColours.__dict__.__setitem__('stypy_param_names_list', [])
        Colours.getNumberOfColours.__dict__.__setitem__('stypy_varargs_param_name', None)
        Colours.getNumberOfColours.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Colours.getNumberOfColours.__dict__.__setitem__('stypy_call_defaults', defaults)
        Colours.getNumberOfColours.__dict__.__setitem__('stypy_call_varargs', varargs)
        Colours.getNumberOfColours.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Colours.getNumberOfColours.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Colours.getNumberOfColours', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getNumberOfColours', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getNumberOfColours(...)' code ##################

        # Getting the type of 'self' (line 27)
        self_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 15), 'self')
        # Obtaining the member 'numberOfColours' of a type (line 27)
        numberOfColours_301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 15), self_300, 'numberOfColours')
        # Assigning a type to the variable 'stypy_return_type' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'stypy_return_type', numberOfColours_301)
        
        # ################# End of 'getNumberOfColours(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getNumberOfColours' in the type store
        # Getting the type of 'stypy_return_type' (line 26)
        stypy_return_type_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_302)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getNumberOfColours'
        return stypy_return_type_302


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 17, 0, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Colours.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Colours' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'Colours', Colours)

# Assigning a Num to a Name (line 18):
int_303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 22), 'int')
# Getting the type of 'Colours'
Colours_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Colours')
# Setting the type of the member 'numberOfColours' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Colours_304, 'numberOfColours', int_303)

# Assigning a Num to a Name (line 19):
int_305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 10), 'int')
# Getting the type of 'Colours'
Colours_306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Colours')
# Setting the type of the member 'red' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Colours_306, 'red', int_305)

# Assigning a Num to a Name (line 20):
int_307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 12), 'int')
# Getting the type of 'Colours'
Colours_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Colours')
# Setting the type of the member 'green' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Colours_308, 'green', int_307)

# Assigning a Num to a Name (line 21):
int_309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 13), 'int')
# Getting the type of 'Colours'
Colours_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Colours')
# Setting the type of the member 'purple' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Colours_310, 'purple', int_309)

# Assigning a Num to a Name (line 22):
int_311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 13), 'int')
# Getting the type of 'Colours'
Colours_312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Colours')
# Setting the type of the member 'yellow' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Colours_312, 'yellow', int_311)

# Assigning a Num to a Name (line 23):
int_313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 12), 'int')
# Getting the type of 'Colours'
Colours_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Colours')
# Setting the type of the member 'white' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Colours_314, 'white', int_313)

# Assigning a Num to a Name (line 24):
int_315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 12), 'int')
# Getting the type of 'Colours'
Colours_316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Colours')
# Setting the type of the member 'black' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Colours_316, 'black', int_315)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

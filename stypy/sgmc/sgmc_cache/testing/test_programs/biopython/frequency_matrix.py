
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://hplgit.github.io/bioinf-py/doc/pub/html/main_bioinf.html#basic-bioinformatics-examples-in-python
2: 
3: # Computing Frequencies
4: #
5: # Your genetic code is essentially the same from you are born until you die, and the same in your blood and your brain.
6: # Which genes that are turned on and off make the difference between the cells. This regulation of genes is orchestrated
7: # by an immensely complex mechanism, which we have only started to understand. A central part of this mechanism consists
8: #  of molecules called transcription factors that float around in the cell and attach to DNA, and in doing so turn
9: # nearby genes on or off. These molecules bind preferentially to specific DNA sequences, and this binding preference
10: # pattern can be represented by a table of frequencies of given symbols at each position of the pattern. More precisely,
11: # each row in the table corresponds to the bases A, C, G, and T, while column j reflects how many times the base appears
12: # in position j in the DNA sequence.
13: 
14: import random
15: 
16: import numpy as np
17: 
18: 
19: def generate_string(N, alphabet='ACGT'):
20:     return ''.join([random.choice(alphabet) for i in xrange(N)])
21: 
22: 
23: def freq_numpy(dna_list):
24:     frequency_matrix = np.zeros((4, len(dna_list[0])), dtype=np.int)
25:     base2index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
26:     for dna in dna_list:
27:         for index, base in enumerate(dna):
28:             frequency_matrix[base2index[base]][index] += 1
29: 
30:     return frequency_matrix
31: 
32: 
33: dna = generate_string(600000)
34: r = freq_numpy(dna)
35: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import random' statement (line 14)
import random

import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'random', random, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import numpy' statement (line 16)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/biopython/')
import_1 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy')

if (type(import_1) is not StypyTypeError):

    if (import_1 != 'pyd_module'):
        __import__(import_1)
        sys_modules_2 = sys.modules[import_1]
        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'np', sys_modules_2.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy', import_1)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/biopython/')


@norecursion
def generate_string(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 32), 'str', 'ACGT')
    defaults = [str_3]
    # Create a new context for function 'generate_string'
    module_type_store = module_type_store.open_function_context('generate_string', 19, 0, False)
    
    # Passed parameters checking function
    generate_string.stypy_localization = localization
    generate_string.stypy_type_of_self = None
    generate_string.stypy_type_store = module_type_store
    generate_string.stypy_function_name = 'generate_string'
    generate_string.stypy_param_names_list = ['N', 'alphabet']
    generate_string.stypy_varargs_param_name = None
    generate_string.stypy_kwargs_param_name = None
    generate_string.stypy_call_defaults = defaults
    generate_string.stypy_call_varargs = varargs
    generate_string.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'generate_string', ['N', 'alphabet'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'generate_string', localization, ['N', 'alphabet'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'generate_string(...)' code ##################

    
    # Call to join(...): (line 20)
    # Processing the call arguments (line 20)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to xrange(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'N' (line 20)
    N_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 60), 'N', False)
    # Processing the call keyword arguments (line 20)
    kwargs_13 = {}
    # Getting the type of 'xrange' (line 20)
    xrange_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 53), 'xrange', False)
    # Calling xrange(args, kwargs) (line 20)
    xrange_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 20, 53), xrange_11, *[N_12], **kwargs_13)
    
    comprehension_15 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 20), xrange_call_result_14)
    # Assigning a type to the variable 'i' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 20), 'i', comprehension_15)
    
    # Call to choice(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'alphabet' (line 20)
    alphabet_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 34), 'alphabet', False)
    # Processing the call keyword arguments (line 20)
    kwargs_9 = {}
    # Getting the type of 'random' (line 20)
    random_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 20), 'random', False)
    # Obtaining the member 'choice' of a type (line 20)
    choice_7 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 20), random_6, 'choice')
    # Calling choice(args, kwargs) (line 20)
    choice_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 20, 20), choice_7, *[alphabet_8], **kwargs_9)
    
    list_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 20), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 20), list_16, choice_call_result_10)
    # Processing the call keyword arguments (line 20)
    kwargs_17 = {}
    str_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 11), 'str', '')
    # Obtaining the member 'join' of a type (line 20)
    join_5 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 11), str_4, 'join')
    # Calling join(args, kwargs) (line 20)
    join_call_result_18 = invoke(stypy.reporting.localization.Localization(__file__, 20, 11), join_5, *[list_16], **kwargs_17)
    
    # Assigning a type to the variable 'stypy_return_type' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type', join_call_result_18)
    
    # ################# End of 'generate_string(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'generate_string' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_19)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'generate_string'
    return stypy_return_type_19

# Assigning a type to the variable 'generate_string' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'generate_string', generate_string)

@norecursion
def freq_numpy(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'freq_numpy'
    module_type_store = module_type_store.open_function_context('freq_numpy', 23, 0, False)
    
    # Passed parameters checking function
    freq_numpy.stypy_localization = localization
    freq_numpy.stypy_type_of_self = None
    freq_numpy.stypy_type_store = module_type_store
    freq_numpy.stypy_function_name = 'freq_numpy'
    freq_numpy.stypy_param_names_list = ['dna_list']
    freq_numpy.stypy_varargs_param_name = None
    freq_numpy.stypy_kwargs_param_name = None
    freq_numpy.stypy_call_defaults = defaults
    freq_numpy.stypy_call_varargs = varargs
    freq_numpy.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'freq_numpy', ['dna_list'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'freq_numpy', localization, ['dna_list'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'freq_numpy(...)' code ##################

    
    # Assigning a Call to a Name (line 24):
    
    # Call to zeros(...): (line 24)
    # Processing the call arguments (line 24)
    
    # Obtaining an instance of the builtin type 'tuple' (line 24)
    tuple_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 24)
    # Adding element type (line 24)
    int_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 33), tuple_22, int_23)
    # Adding element type (line 24)
    
    # Call to len(...): (line 24)
    # Processing the call arguments (line 24)
    
    # Obtaining the type of the subscript
    int_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 49), 'int')
    # Getting the type of 'dna_list' (line 24)
    dna_list_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 40), 'dna_list', False)
    # Obtaining the member '__getitem__' of a type (line 24)
    getitem___27 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 40), dna_list_26, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 24)
    subscript_call_result_28 = invoke(stypy.reporting.localization.Localization(__file__, 24, 40), getitem___27, int_25)
    
    # Processing the call keyword arguments (line 24)
    kwargs_29 = {}
    # Getting the type of 'len' (line 24)
    len_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 36), 'len', False)
    # Calling len(args, kwargs) (line 24)
    len_call_result_30 = invoke(stypy.reporting.localization.Localization(__file__, 24, 36), len_24, *[subscript_call_result_28], **kwargs_29)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 33), tuple_22, len_call_result_30)
    
    # Processing the call keyword arguments (line 24)
    # Getting the type of 'np' (line 24)
    np_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 61), 'np', False)
    # Obtaining the member 'int' of a type (line 24)
    int_32 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 61), np_31, 'int')
    keyword_33 = int_32
    kwargs_34 = {'dtype': keyword_33}
    # Getting the type of 'np' (line 24)
    np_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 23), 'np', False)
    # Obtaining the member 'zeros' of a type (line 24)
    zeros_21 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 23), np_20, 'zeros')
    # Calling zeros(args, kwargs) (line 24)
    zeros_call_result_35 = invoke(stypy.reporting.localization.Localization(__file__, 24, 23), zeros_21, *[tuple_22], **kwargs_34)
    
    # Assigning a type to the variable 'frequency_matrix' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'frequency_matrix', zeros_call_result_35)
    
    # Assigning a Dict to a Name (line 25):
    
    # Obtaining an instance of the builtin type 'dict' (line 25)
    dict_36 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 17), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 25)
    # Adding element type (key, value) (line 25)
    str_37 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 18), 'str', 'A')
    int_38 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 23), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 17), dict_36, (str_37, int_38))
    # Adding element type (key, value) (line 25)
    str_39 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 26), 'str', 'C')
    int_40 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 31), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 17), dict_36, (str_39, int_40))
    # Adding element type (key, value) (line 25)
    str_41 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 34), 'str', 'G')
    int_42 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 39), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 17), dict_36, (str_41, int_42))
    # Adding element type (key, value) (line 25)
    str_43 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 42), 'str', 'T')
    int_44 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 47), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 17), dict_36, (str_43, int_44))
    
    # Assigning a type to the variable 'base2index' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'base2index', dict_36)
    
    # Getting the type of 'dna_list' (line 26)
    dna_list_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'dna_list')
    # Testing the type of a for loop iterable (line 26)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 26, 4), dna_list_45)
    # Getting the type of the for loop variable (line 26)
    for_loop_var_46 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 26, 4), dna_list_45)
    # Assigning a type to the variable 'dna' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'dna', for_loop_var_46)
    # SSA begins for a for statement (line 26)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to enumerate(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'dna' (line 27)
    dna_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 37), 'dna', False)
    # Processing the call keyword arguments (line 27)
    kwargs_49 = {}
    # Getting the type of 'enumerate' (line 27)
    enumerate_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 27), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 27)
    enumerate_call_result_50 = invoke(stypy.reporting.localization.Localization(__file__, 27, 27), enumerate_47, *[dna_48], **kwargs_49)
    
    # Testing the type of a for loop iterable (line 27)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 27, 8), enumerate_call_result_50)
    # Getting the type of the for loop variable (line 27)
    for_loop_var_51 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 27, 8), enumerate_call_result_50)
    # Assigning a type to the variable 'index' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'index', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 8), for_loop_var_51))
    # Assigning a type to the variable 'base' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'base', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 8), for_loop_var_51))
    # SSA begins for a for statement (line 27)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    # Getting the type of 'base' (line 28)
    base_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 40), 'base')
    # Getting the type of 'base2index' (line 28)
    base2index_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 29), 'base2index')
    # Obtaining the member '__getitem__' of a type (line 28)
    getitem___54 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 29), base2index_53, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 28)
    subscript_call_result_55 = invoke(stypy.reporting.localization.Localization(__file__, 28, 29), getitem___54, base_52)
    
    # Getting the type of 'frequency_matrix' (line 28)
    frequency_matrix_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'frequency_matrix')
    # Obtaining the member '__getitem__' of a type (line 28)
    getitem___57 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 12), frequency_matrix_56, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 28)
    subscript_call_result_58 = invoke(stypy.reporting.localization.Localization(__file__, 28, 12), getitem___57, subscript_call_result_55)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'index' (line 28)
    index_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 47), 'index')
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    # Getting the type of 'base' (line 28)
    base_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 40), 'base')
    # Getting the type of 'base2index' (line 28)
    base2index_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 29), 'base2index')
    # Obtaining the member '__getitem__' of a type (line 28)
    getitem___62 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 29), base2index_61, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 28)
    subscript_call_result_63 = invoke(stypy.reporting.localization.Localization(__file__, 28, 29), getitem___62, base_60)
    
    # Getting the type of 'frequency_matrix' (line 28)
    frequency_matrix_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'frequency_matrix')
    # Obtaining the member '__getitem__' of a type (line 28)
    getitem___65 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 12), frequency_matrix_64, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 28)
    subscript_call_result_66 = invoke(stypy.reporting.localization.Localization(__file__, 28, 12), getitem___65, subscript_call_result_63)
    
    # Obtaining the member '__getitem__' of a type (line 28)
    getitem___67 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 12), subscript_call_result_66, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 28)
    subscript_call_result_68 = invoke(stypy.reporting.localization.Localization(__file__, 28, 12), getitem___67, index_59)
    
    int_69 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 57), 'int')
    # Applying the binary operator '+=' (line 28)
    result_iadd_70 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 12), '+=', subscript_call_result_68, int_69)
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    # Getting the type of 'base' (line 28)
    base_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 40), 'base')
    # Getting the type of 'base2index' (line 28)
    base2index_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 29), 'base2index')
    # Obtaining the member '__getitem__' of a type (line 28)
    getitem___73 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 29), base2index_72, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 28)
    subscript_call_result_74 = invoke(stypy.reporting.localization.Localization(__file__, 28, 29), getitem___73, base_71)
    
    # Getting the type of 'frequency_matrix' (line 28)
    frequency_matrix_75 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'frequency_matrix')
    # Obtaining the member '__getitem__' of a type (line 28)
    getitem___76 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 12), frequency_matrix_75, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 28)
    subscript_call_result_77 = invoke(stypy.reporting.localization.Localization(__file__, 28, 12), getitem___76, subscript_call_result_74)
    
    # Getting the type of 'index' (line 28)
    index_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 47), 'index')
    # Storing an element on a container (line 28)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 12), subscript_call_result_77, (index_78, result_iadd_70))
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'frequency_matrix' (line 30)
    frequency_matrix_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 11), 'frequency_matrix')
    # Assigning a type to the variable 'stypy_return_type' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type', frequency_matrix_79)
    
    # ################# End of 'freq_numpy(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'freq_numpy' in the type store
    # Getting the type of 'stypy_return_type' (line 23)
    stypy_return_type_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_80)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'freq_numpy'
    return stypy_return_type_80

# Assigning a type to the variable 'freq_numpy' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'freq_numpy', freq_numpy)

# Assigning a Call to a Name (line 33):

# Call to generate_string(...): (line 33)
# Processing the call arguments (line 33)
int_82 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 22), 'int')
# Processing the call keyword arguments (line 33)
kwargs_83 = {}
# Getting the type of 'generate_string' (line 33)
generate_string_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 6), 'generate_string', False)
# Calling generate_string(args, kwargs) (line 33)
generate_string_call_result_84 = invoke(stypy.reporting.localization.Localization(__file__, 33, 6), generate_string_81, *[int_82], **kwargs_83)

# Assigning a type to the variable 'dna' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'dna', generate_string_call_result_84)

# Assigning a Call to a Name (line 34):

# Call to freq_numpy(...): (line 34)
# Processing the call arguments (line 34)
# Getting the type of 'dna' (line 34)
dna_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'dna', False)
# Processing the call keyword arguments (line 34)
kwargs_87 = {}
# Getting the type of 'freq_numpy' (line 34)
freq_numpy_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'freq_numpy', False)
# Calling freq_numpy(args, kwargs) (line 34)
freq_numpy_call_result_88 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), freq_numpy_85, *[dna_86], **kwargs_87)

# Assigning a type to the variable 'r' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'r', freq_numpy_call_result_88)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

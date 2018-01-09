
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2003-2005 Peter J. Verveer
2: #
3: # Redistribution and use in source and binary forms, with or without
4: # modification, are permitted provided that the following conditions
5: # are met:
6: #
7: # 1. Redistributions of source code must retain the above copyright
8: #    notice, this list of conditions and the following disclaimer.
9: #
10: # 2. Redistributions in binary form must reproduce the above
11: #    copyright notice, this list of conditions and the following
12: #    disclaimer in the documentation and/or other materials provided
13: #    with the distribution.
14: #
15: # 3. The name of the author may not be used to endorse or promote
16: #    products derived from this software without specific prior
17: #    written permission.
18: #
19: # THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
20: # OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
21: # WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
22: # ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
23: # DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
24: # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
25: # GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
26: # INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
27: # WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
28: # NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
29: # SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
30: 
31: from __future__ import division, print_function, absolute_import
32: 
33: import numpy
34: 
35: from scipy._lib.six import string_types
36: 
37: 
38: def _extend_mode_to_code(mode):
39:     '''Convert an extension mode to the corresponding integer code.
40:     '''
41:     if mode == 'nearest':
42:         return 0
43:     elif mode == 'wrap':
44:         return 1
45:     elif mode == 'reflect':
46:         return 2
47:     elif mode == 'mirror':
48:         return 3
49:     elif mode == 'constant':
50:         return 4
51:     else:
52:         raise RuntimeError('boundary mode not supported')
53: 
54: 
55: def _normalize_sequence(input, rank, array_type=None):
56:     '''If input is a scalar, create a sequence of length equal to the
57:     rank by duplicating the input. If input is a sequence,
58:     check if its length is equal to the length of array.
59:     '''
60:     is_str = isinstance(input, string_types)
61:     if hasattr(input, '__iter__') and not is_str:
62:         normalized = list(input)
63:         if len(normalized) != rank:
64:             err = "sequence argument must have length equal to input rank"
65:             raise RuntimeError(err)
66:     else:
67:         normalized = [input] * rank
68:     return normalized
69: 
70: 
71: def _get_output(output, input, shape=None):
72:     if shape is None:
73:         shape = input.shape
74:     if output is None:
75:         output = numpy.zeros(shape, dtype=input.dtype.name)
76:         return_value = output
77:     elif type(output) in [type(type), type(numpy.zeros((4,)).dtype)]:
78:         output = numpy.zeros(shape, dtype=output)
79:         return_value = output
80:     elif type(output) in string_types:
81:         output = numpy.typeDict[output]
82:         output = numpy.zeros(shape, dtype=output)
83:         return_value = output
84:     else:
85:         if output.shape != shape:
86:             raise RuntimeError("output shape not correct")
87:         return_value = None
88:     return output, return_value
89: 
90: 
91: def _check_axis(axis, rank):
92:     if axis < 0:
93:         axis += rank
94:     if axis < 0 or axis >= rank:
95:         raise ValueError('invalid axis')
96:     return axis
97: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 33, 0))

# 'import numpy' statement (line 33)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_126636 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy')

if (type(import_126636) is not StypyTypeError):

    if (import_126636 != 'pyd_module'):
        __import__(import_126636)
        sys_modules_126637 = sys.modules[import_126636]
        import_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy', sys_modules_126637.module_type_store, module_type_store)
    else:
        import numpy

        import_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy', import_126636)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 35, 0))

# 'from scipy._lib.six import string_types' statement (line 35)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_126638 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy._lib.six')

if (type(import_126638) is not StypyTypeError):

    if (import_126638 != 'pyd_module'):
        __import__(import_126638)
        sys_modules_126639 = sys.modules[import_126638]
        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy._lib.six', sys_modules_126639.module_type_store, module_type_store, ['string_types'])
        nest_module(stypy.reporting.localization.Localization(__file__, 35, 0), __file__, sys_modules_126639, sys_modules_126639.module_type_store, module_type_store)
    else:
        from scipy._lib.six import string_types

        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy._lib.six', None, module_type_store, ['string_types'], [string_types])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy._lib.six', import_126638)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')


@norecursion
def _extend_mode_to_code(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_extend_mode_to_code'
    module_type_store = module_type_store.open_function_context('_extend_mode_to_code', 38, 0, False)
    
    # Passed parameters checking function
    _extend_mode_to_code.stypy_localization = localization
    _extend_mode_to_code.stypy_type_of_self = None
    _extend_mode_to_code.stypy_type_store = module_type_store
    _extend_mode_to_code.stypy_function_name = '_extend_mode_to_code'
    _extend_mode_to_code.stypy_param_names_list = ['mode']
    _extend_mode_to_code.stypy_varargs_param_name = None
    _extend_mode_to_code.stypy_kwargs_param_name = None
    _extend_mode_to_code.stypy_call_defaults = defaults
    _extend_mode_to_code.stypy_call_varargs = varargs
    _extend_mode_to_code.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_extend_mode_to_code', ['mode'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_extend_mode_to_code', localization, ['mode'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_extend_mode_to_code(...)' code ##################

    str_126640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, (-1)), 'str', 'Convert an extension mode to the corresponding integer code.\n    ')
    
    
    # Getting the type of 'mode' (line 41)
    mode_126641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 7), 'mode')
    str_126642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 15), 'str', 'nearest')
    # Applying the binary operator '==' (line 41)
    result_eq_126643 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 7), '==', mode_126641, str_126642)
    
    # Testing the type of an if condition (line 41)
    if_condition_126644 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 4), result_eq_126643)
    # Assigning a type to the variable 'if_condition_126644' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'if_condition_126644', if_condition_126644)
    # SSA begins for if statement (line 41)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_126645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'stypy_return_type', int_126645)
    # SSA branch for the else part of an if statement (line 41)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'mode' (line 43)
    mode_126646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 9), 'mode')
    str_126647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 17), 'str', 'wrap')
    # Applying the binary operator '==' (line 43)
    result_eq_126648 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 9), '==', mode_126646, str_126647)
    
    # Testing the type of an if condition (line 43)
    if_condition_126649 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 9), result_eq_126648)
    # Assigning a type to the variable 'if_condition_126649' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 9), 'if_condition_126649', if_condition_126649)
    # SSA begins for if statement (line 43)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_126650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'stypy_return_type', int_126650)
    # SSA branch for the else part of an if statement (line 43)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'mode' (line 45)
    mode_126651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 9), 'mode')
    str_126652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 17), 'str', 'reflect')
    # Applying the binary operator '==' (line 45)
    result_eq_126653 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 9), '==', mode_126651, str_126652)
    
    # Testing the type of an if condition (line 45)
    if_condition_126654 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 9), result_eq_126653)
    # Assigning a type to the variable 'if_condition_126654' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 9), 'if_condition_126654', if_condition_126654)
    # SSA begins for if statement (line 45)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_126655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'stypy_return_type', int_126655)
    # SSA branch for the else part of an if statement (line 45)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'mode' (line 47)
    mode_126656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 9), 'mode')
    str_126657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 17), 'str', 'mirror')
    # Applying the binary operator '==' (line 47)
    result_eq_126658 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 9), '==', mode_126656, str_126657)
    
    # Testing the type of an if condition (line 47)
    if_condition_126659 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 9), result_eq_126658)
    # Assigning a type to the variable 'if_condition_126659' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 9), 'if_condition_126659', if_condition_126659)
    # SSA begins for if statement (line 47)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_126660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'stypy_return_type', int_126660)
    # SSA branch for the else part of an if statement (line 47)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'mode' (line 49)
    mode_126661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 9), 'mode')
    str_126662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 17), 'str', 'constant')
    # Applying the binary operator '==' (line 49)
    result_eq_126663 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 9), '==', mode_126661, str_126662)
    
    # Testing the type of an if condition (line 49)
    if_condition_126664 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 9), result_eq_126663)
    # Assigning a type to the variable 'if_condition_126664' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 9), 'if_condition_126664', if_condition_126664)
    # SSA begins for if statement (line 49)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_126665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'stypy_return_type', int_126665)
    # SSA branch for the else part of an if statement (line 49)
    module_type_store.open_ssa_branch('else')
    
    # Call to RuntimeError(...): (line 52)
    # Processing the call arguments (line 52)
    str_126667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 27), 'str', 'boundary mode not supported')
    # Processing the call keyword arguments (line 52)
    kwargs_126668 = {}
    # Getting the type of 'RuntimeError' (line 52)
    RuntimeError_126666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 52)
    RuntimeError_call_result_126669 = invoke(stypy.reporting.localization.Localization(__file__, 52, 14), RuntimeError_126666, *[str_126667], **kwargs_126668)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 52, 8), RuntimeError_call_result_126669, 'raise parameter', BaseException)
    # SSA join for if statement (line 49)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 47)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 45)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 43)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 41)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_extend_mode_to_code(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_extend_mode_to_code' in the type store
    # Getting the type of 'stypy_return_type' (line 38)
    stypy_return_type_126670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126670)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_extend_mode_to_code'
    return stypy_return_type_126670

# Assigning a type to the variable '_extend_mode_to_code' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), '_extend_mode_to_code', _extend_mode_to_code)

@norecursion
def _normalize_sequence(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 55)
    None_126671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 48), 'None')
    defaults = [None_126671]
    # Create a new context for function '_normalize_sequence'
    module_type_store = module_type_store.open_function_context('_normalize_sequence', 55, 0, False)
    
    # Passed parameters checking function
    _normalize_sequence.stypy_localization = localization
    _normalize_sequence.stypy_type_of_self = None
    _normalize_sequence.stypy_type_store = module_type_store
    _normalize_sequence.stypy_function_name = '_normalize_sequence'
    _normalize_sequence.stypy_param_names_list = ['input', 'rank', 'array_type']
    _normalize_sequence.stypy_varargs_param_name = None
    _normalize_sequence.stypy_kwargs_param_name = None
    _normalize_sequence.stypy_call_defaults = defaults
    _normalize_sequence.stypy_call_varargs = varargs
    _normalize_sequence.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_normalize_sequence', ['input', 'rank', 'array_type'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_normalize_sequence', localization, ['input', 'rank', 'array_type'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_normalize_sequence(...)' code ##################

    str_126672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, (-1)), 'str', 'If input is a scalar, create a sequence of length equal to the\n    rank by duplicating the input. If input is a sequence,\n    check if its length is equal to the length of array.\n    ')
    
    # Assigning a Call to a Name (line 60):
    
    # Call to isinstance(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'input' (line 60)
    input_126674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 24), 'input', False)
    # Getting the type of 'string_types' (line 60)
    string_types_126675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 31), 'string_types', False)
    # Processing the call keyword arguments (line 60)
    kwargs_126676 = {}
    # Getting the type of 'isinstance' (line 60)
    isinstance_126673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 13), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 60)
    isinstance_call_result_126677 = invoke(stypy.reporting.localization.Localization(__file__, 60, 13), isinstance_126673, *[input_126674, string_types_126675], **kwargs_126676)
    
    # Assigning a type to the variable 'is_str' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'is_str', isinstance_call_result_126677)
    
    
    # Evaluating a boolean operation
    
    # Call to hasattr(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'input' (line 61)
    input_126679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 15), 'input', False)
    str_126680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 22), 'str', '__iter__')
    # Processing the call keyword arguments (line 61)
    kwargs_126681 = {}
    # Getting the type of 'hasattr' (line 61)
    hasattr_126678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 7), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 61)
    hasattr_call_result_126682 = invoke(stypy.reporting.localization.Localization(__file__, 61, 7), hasattr_126678, *[input_126679, str_126680], **kwargs_126681)
    
    
    # Getting the type of 'is_str' (line 61)
    is_str_126683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 42), 'is_str')
    # Applying the 'not' unary operator (line 61)
    result_not__126684 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 38), 'not', is_str_126683)
    
    # Applying the binary operator 'and' (line 61)
    result_and_keyword_126685 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 7), 'and', hasattr_call_result_126682, result_not__126684)
    
    # Testing the type of an if condition (line 61)
    if_condition_126686 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 4), result_and_keyword_126685)
    # Assigning a type to the variable 'if_condition_126686' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'if_condition_126686', if_condition_126686)
    # SSA begins for if statement (line 61)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 62):
    
    # Call to list(...): (line 62)
    # Processing the call arguments (line 62)
    # Getting the type of 'input' (line 62)
    input_126688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 26), 'input', False)
    # Processing the call keyword arguments (line 62)
    kwargs_126689 = {}
    # Getting the type of 'list' (line 62)
    list_126687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 21), 'list', False)
    # Calling list(args, kwargs) (line 62)
    list_call_result_126690 = invoke(stypy.reporting.localization.Localization(__file__, 62, 21), list_126687, *[input_126688], **kwargs_126689)
    
    # Assigning a type to the variable 'normalized' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'normalized', list_call_result_126690)
    
    
    
    # Call to len(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'normalized' (line 63)
    normalized_126692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 15), 'normalized', False)
    # Processing the call keyword arguments (line 63)
    kwargs_126693 = {}
    # Getting the type of 'len' (line 63)
    len_126691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 11), 'len', False)
    # Calling len(args, kwargs) (line 63)
    len_call_result_126694 = invoke(stypy.reporting.localization.Localization(__file__, 63, 11), len_126691, *[normalized_126692], **kwargs_126693)
    
    # Getting the type of 'rank' (line 63)
    rank_126695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 30), 'rank')
    # Applying the binary operator '!=' (line 63)
    result_ne_126696 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 11), '!=', len_call_result_126694, rank_126695)
    
    # Testing the type of an if condition (line 63)
    if_condition_126697 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 63, 8), result_ne_126696)
    # Assigning a type to the variable 'if_condition_126697' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'if_condition_126697', if_condition_126697)
    # SSA begins for if statement (line 63)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 64):
    str_126698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 18), 'str', 'sequence argument must have length equal to input rank')
    # Assigning a type to the variable 'err' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'err', str_126698)
    
    # Call to RuntimeError(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'err' (line 65)
    err_126700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 31), 'err', False)
    # Processing the call keyword arguments (line 65)
    kwargs_126701 = {}
    # Getting the type of 'RuntimeError' (line 65)
    RuntimeError_126699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 18), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 65)
    RuntimeError_call_result_126702 = invoke(stypy.reporting.localization.Localization(__file__, 65, 18), RuntimeError_126699, *[err_126700], **kwargs_126701)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 65, 12), RuntimeError_call_result_126702, 'raise parameter', BaseException)
    # SSA join for if statement (line 63)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 61)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 67):
    
    # Obtaining an instance of the builtin type 'list' (line 67)
    list_126703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 67)
    # Adding element type (line 67)
    # Getting the type of 'input' (line 67)
    input_126704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 22), 'input')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 21), list_126703, input_126704)
    
    # Getting the type of 'rank' (line 67)
    rank_126705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 31), 'rank')
    # Applying the binary operator '*' (line 67)
    result_mul_126706 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 21), '*', list_126703, rank_126705)
    
    # Assigning a type to the variable 'normalized' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'normalized', result_mul_126706)
    # SSA join for if statement (line 61)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'normalized' (line 68)
    normalized_126707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'normalized')
    # Assigning a type to the variable 'stypy_return_type' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type', normalized_126707)
    
    # ################# End of '_normalize_sequence(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_normalize_sequence' in the type store
    # Getting the type of 'stypy_return_type' (line 55)
    stypy_return_type_126708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126708)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_normalize_sequence'
    return stypy_return_type_126708

# Assigning a type to the variable '_normalize_sequence' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), '_normalize_sequence', _normalize_sequence)

@norecursion
def _get_output(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 71)
    None_126709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 37), 'None')
    defaults = [None_126709]
    # Create a new context for function '_get_output'
    module_type_store = module_type_store.open_function_context('_get_output', 71, 0, False)
    
    # Passed parameters checking function
    _get_output.stypy_localization = localization
    _get_output.stypy_type_of_self = None
    _get_output.stypy_type_store = module_type_store
    _get_output.stypy_function_name = '_get_output'
    _get_output.stypy_param_names_list = ['output', 'input', 'shape']
    _get_output.stypy_varargs_param_name = None
    _get_output.stypy_kwargs_param_name = None
    _get_output.stypy_call_defaults = defaults
    _get_output.stypy_call_varargs = varargs
    _get_output.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_output', ['output', 'input', 'shape'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_output', localization, ['output', 'input', 'shape'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_output(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 72)
    # Getting the type of 'shape' (line 72)
    shape_126710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 7), 'shape')
    # Getting the type of 'None' (line 72)
    None_126711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'None')
    
    (may_be_126712, more_types_in_union_126713) = may_be_none(shape_126710, None_126711)

    if may_be_126712:

        if more_types_in_union_126713:
            # Runtime conditional SSA (line 72)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 73):
        # Getting the type of 'input' (line 73)
        input_126714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'input')
        # Obtaining the member 'shape' of a type (line 73)
        shape_126715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 16), input_126714, 'shape')
        # Assigning a type to the variable 'shape' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'shape', shape_126715)

        if more_types_in_union_126713:
            # SSA join for if statement (line 72)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 74)
    # Getting the type of 'output' (line 74)
    output_126716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 7), 'output')
    # Getting the type of 'None' (line 74)
    None_126717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 17), 'None')
    
    (may_be_126718, more_types_in_union_126719) = may_be_none(output_126716, None_126717)

    if may_be_126718:

        if more_types_in_union_126719:
            # Runtime conditional SSA (line 74)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 75):
        
        # Call to zeros(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'shape' (line 75)
        shape_126722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 29), 'shape', False)
        # Processing the call keyword arguments (line 75)
        # Getting the type of 'input' (line 75)
        input_126723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 42), 'input', False)
        # Obtaining the member 'dtype' of a type (line 75)
        dtype_126724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 42), input_126723, 'dtype')
        # Obtaining the member 'name' of a type (line 75)
        name_126725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 42), dtype_126724, 'name')
        keyword_126726 = name_126725
        kwargs_126727 = {'dtype': keyword_126726}
        # Getting the type of 'numpy' (line 75)
        numpy_126720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 17), 'numpy', False)
        # Obtaining the member 'zeros' of a type (line 75)
        zeros_126721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 17), numpy_126720, 'zeros')
        # Calling zeros(args, kwargs) (line 75)
        zeros_call_result_126728 = invoke(stypy.reporting.localization.Localization(__file__, 75, 17), zeros_126721, *[shape_126722], **kwargs_126727)
        
        # Assigning a type to the variable 'output' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'output', zeros_call_result_126728)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'output' (line 76)
        output_126729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'output')
        # Assigning a type to the variable 'return_value' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'return_value', output_126729)

        if more_types_in_union_126719:
            # Runtime conditional SSA for else branch (line 74)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_126718) or more_types_in_union_126719):
        
        
        
        # Call to type(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'output' (line 77)
        output_126731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 14), 'output', False)
        # Processing the call keyword arguments (line 77)
        kwargs_126732 = {}
        # Getting the type of 'type' (line 77)
        type_126730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 9), 'type', False)
        # Calling type(args, kwargs) (line 77)
        type_call_result_126733 = invoke(stypy.reporting.localization.Localization(__file__, 77, 9), type_126730, *[output_126731], **kwargs_126732)
        
        
        # Obtaining an instance of the builtin type 'list' (line 77)
        list_126734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 77)
        # Adding element type (line 77)
        
        # Call to type(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'type' (line 77)
        type_126736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 31), 'type', False)
        # Processing the call keyword arguments (line 77)
        kwargs_126737 = {}
        # Getting the type of 'type' (line 77)
        type_126735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 26), 'type', False)
        # Calling type(args, kwargs) (line 77)
        type_call_result_126738 = invoke(stypy.reporting.localization.Localization(__file__, 77, 26), type_126735, *[type_126736], **kwargs_126737)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 25), list_126734, type_call_result_126738)
        # Adding element type (line 77)
        
        # Call to type(...): (line 77)
        # Processing the call arguments (line 77)
        
        # Call to zeros(...): (line 77)
        # Processing the call arguments (line 77)
        
        # Obtaining an instance of the builtin type 'tuple' (line 77)
        tuple_126742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 77)
        # Adding element type (line 77)
        int_126743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 56), tuple_126742, int_126743)
        
        # Processing the call keyword arguments (line 77)
        kwargs_126744 = {}
        # Getting the type of 'numpy' (line 77)
        numpy_126740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 43), 'numpy', False)
        # Obtaining the member 'zeros' of a type (line 77)
        zeros_126741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 43), numpy_126740, 'zeros')
        # Calling zeros(args, kwargs) (line 77)
        zeros_call_result_126745 = invoke(stypy.reporting.localization.Localization(__file__, 77, 43), zeros_126741, *[tuple_126742], **kwargs_126744)
        
        # Obtaining the member 'dtype' of a type (line 77)
        dtype_126746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 43), zeros_call_result_126745, 'dtype')
        # Processing the call keyword arguments (line 77)
        kwargs_126747 = {}
        # Getting the type of 'type' (line 77)
        type_126739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 38), 'type', False)
        # Calling type(args, kwargs) (line 77)
        type_call_result_126748 = invoke(stypy.reporting.localization.Localization(__file__, 77, 38), type_126739, *[dtype_126746], **kwargs_126747)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 25), list_126734, type_call_result_126748)
        
        # Applying the binary operator 'in' (line 77)
        result_contains_126749 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 9), 'in', type_call_result_126733, list_126734)
        
        # Testing the type of an if condition (line 77)
        if_condition_126750 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 9), result_contains_126749)
        # Assigning a type to the variable 'if_condition_126750' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 9), 'if_condition_126750', if_condition_126750)
        # SSA begins for if statement (line 77)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 78):
        
        # Call to zeros(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'shape' (line 78)
        shape_126753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 29), 'shape', False)
        # Processing the call keyword arguments (line 78)
        # Getting the type of 'output' (line 78)
        output_126754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 42), 'output', False)
        keyword_126755 = output_126754
        kwargs_126756 = {'dtype': keyword_126755}
        # Getting the type of 'numpy' (line 78)
        numpy_126751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 17), 'numpy', False)
        # Obtaining the member 'zeros' of a type (line 78)
        zeros_126752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 17), numpy_126751, 'zeros')
        # Calling zeros(args, kwargs) (line 78)
        zeros_call_result_126757 = invoke(stypy.reporting.localization.Localization(__file__, 78, 17), zeros_126752, *[shape_126753], **kwargs_126756)
        
        # Assigning a type to the variable 'output' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'output', zeros_call_result_126757)
        
        # Assigning a Name to a Name (line 79):
        # Getting the type of 'output' (line 79)
        output_126758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 23), 'output')
        # Assigning a type to the variable 'return_value' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'return_value', output_126758)
        # SSA branch for the else part of an if statement (line 77)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to type(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'output' (line 80)
        output_126760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 14), 'output', False)
        # Processing the call keyword arguments (line 80)
        kwargs_126761 = {}
        # Getting the type of 'type' (line 80)
        type_126759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 9), 'type', False)
        # Calling type(args, kwargs) (line 80)
        type_call_result_126762 = invoke(stypy.reporting.localization.Localization(__file__, 80, 9), type_126759, *[output_126760], **kwargs_126761)
        
        # Getting the type of 'string_types' (line 80)
        string_types_126763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 25), 'string_types')
        # Applying the binary operator 'in' (line 80)
        result_contains_126764 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 9), 'in', type_call_result_126762, string_types_126763)
        
        # Testing the type of an if condition (line 80)
        if_condition_126765 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 9), result_contains_126764)
        # Assigning a type to the variable 'if_condition_126765' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 9), 'if_condition_126765', if_condition_126765)
        # SSA begins for if statement (line 80)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 81):
        
        # Obtaining the type of the subscript
        # Getting the type of 'output' (line 81)
        output_126766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 32), 'output')
        # Getting the type of 'numpy' (line 81)
        numpy_126767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 17), 'numpy')
        # Obtaining the member 'typeDict' of a type (line 81)
        typeDict_126768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 17), numpy_126767, 'typeDict')
        # Obtaining the member '__getitem__' of a type (line 81)
        getitem___126769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 17), typeDict_126768, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 81)
        subscript_call_result_126770 = invoke(stypy.reporting.localization.Localization(__file__, 81, 17), getitem___126769, output_126766)
        
        # Assigning a type to the variable 'output' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'output', subscript_call_result_126770)
        
        # Assigning a Call to a Name (line 82):
        
        # Call to zeros(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'shape' (line 82)
        shape_126773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 29), 'shape', False)
        # Processing the call keyword arguments (line 82)
        # Getting the type of 'output' (line 82)
        output_126774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 42), 'output', False)
        keyword_126775 = output_126774
        kwargs_126776 = {'dtype': keyword_126775}
        # Getting the type of 'numpy' (line 82)
        numpy_126771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 17), 'numpy', False)
        # Obtaining the member 'zeros' of a type (line 82)
        zeros_126772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 17), numpy_126771, 'zeros')
        # Calling zeros(args, kwargs) (line 82)
        zeros_call_result_126777 = invoke(stypy.reporting.localization.Localization(__file__, 82, 17), zeros_126772, *[shape_126773], **kwargs_126776)
        
        # Assigning a type to the variable 'output' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'output', zeros_call_result_126777)
        
        # Assigning a Name to a Name (line 83):
        # Getting the type of 'output' (line 83)
        output_126778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 23), 'output')
        # Assigning a type to the variable 'return_value' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'return_value', output_126778)
        # SSA branch for the else part of an if statement (line 80)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'output' (line 85)
        output_126779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 11), 'output')
        # Obtaining the member 'shape' of a type (line 85)
        shape_126780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 11), output_126779, 'shape')
        # Getting the type of 'shape' (line 85)
        shape_126781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 27), 'shape')
        # Applying the binary operator '!=' (line 85)
        result_ne_126782 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 11), '!=', shape_126780, shape_126781)
        
        # Testing the type of an if condition (line 85)
        if_condition_126783 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 8), result_ne_126782)
        # Assigning a type to the variable 'if_condition_126783' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'if_condition_126783', if_condition_126783)
        # SSA begins for if statement (line 85)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to RuntimeError(...): (line 86)
        # Processing the call arguments (line 86)
        str_126785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 31), 'str', 'output shape not correct')
        # Processing the call keyword arguments (line 86)
        kwargs_126786 = {}
        # Getting the type of 'RuntimeError' (line 86)
        RuntimeError_126784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 18), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 86)
        RuntimeError_call_result_126787 = invoke(stypy.reporting.localization.Localization(__file__, 86, 18), RuntimeError_126784, *[str_126785], **kwargs_126786)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 86, 12), RuntimeError_call_result_126787, 'raise parameter', BaseException)
        # SSA join for if statement (line 85)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 87):
        # Getting the type of 'None' (line 87)
        None_126788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 23), 'None')
        # Assigning a type to the variable 'return_value' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'return_value', None_126788)
        # SSA join for if statement (line 80)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 77)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_126718 and more_types_in_union_126719):
            # SSA join for if statement (line 74)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Obtaining an instance of the builtin type 'tuple' (line 88)
    tuple_126789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 88)
    # Adding element type (line 88)
    # Getting the type of 'output' (line 88)
    output_126790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 11), 'output')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 11), tuple_126789, output_126790)
    # Adding element type (line 88)
    # Getting the type of 'return_value' (line 88)
    return_value_126791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 19), 'return_value')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 11), tuple_126789, return_value_126791)
    
    # Assigning a type to the variable 'stypy_return_type' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type', tuple_126789)
    
    # ################# End of '_get_output(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_output' in the type store
    # Getting the type of 'stypy_return_type' (line 71)
    stypy_return_type_126792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126792)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_output'
    return stypy_return_type_126792

# Assigning a type to the variable '_get_output' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), '_get_output', _get_output)

@norecursion
def _check_axis(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_check_axis'
    module_type_store = module_type_store.open_function_context('_check_axis', 91, 0, False)
    
    # Passed parameters checking function
    _check_axis.stypy_localization = localization
    _check_axis.stypy_type_of_self = None
    _check_axis.stypy_type_store = module_type_store
    _check_axis.stypy_function_name = '_check_axis'
    _check_axis.stypy_param_names_list = ['axis', 'rank']
    _check_axis.stypy_varargs_param_name = None
    _check_axis.stypy_kwargs_param_name = None
    _check_axis.stypy_call_defaults = defaults
    _check_axis.stypy_call_varargs = varargs
    _check_axis.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_check_axis', ['axis', 'rank'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_check_axis', localization, ['axis', 'rank'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_check_axis(...)' code ##################

    
    
    # Getting the type of 'axis' (line 92)
    axis_126793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 7), 'axis')
    int_126794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 14), 'int')
    # Applying the binary operator '<' (line 92)
    result_lt_126795 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 7), '<', axis_126793, int_126794)
    
    # Testing the type of an if condition (line 92)
    if_condition_126796 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 4), result_lt_126795)
    # Assigning a type to the variable 'if_condition_126796' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'if_condition_126796', if_condition_126796)
    # SSA begins for if statement (line 92)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'axis' (line 93)
    axis_126797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'axis')
    # Getting the type of 'rank' (line 93)
    rank_126798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'rank')
    # Applying the binary operator '+=' (line 93)
    result_iadd_126799 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 8), '+=', axis_126797, rank_126798)
    # Assigning a type to the variable 'axis' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'axis', result_iadd_126799)
    
    # SSA join for if statement (line 92)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'axis' (line 94)
    axis_126800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 7), 'axis')
    int_126801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 14), 'int')
    # Applying the binary operator '<' (line 94)
    result_lt_126802 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 7), '<', axis_126800, int_126801)
    
    
    # Getting the type of 'axis' (line 94)
    axis_126803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 19), 'axis')
    # Getting the type of 'rank' (line 94)
    rank_126804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 27), 'rank')
    # Applying the binary operator '>=' (line 94)
    result_ge_126805 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 19), '>=', axis_126803, rank_126804)
    
    # Applying the binary operator 'or' (line 94)
    result_or_keyword_126806 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 7), 'or', result_lt_126802, result_ge_126805)
    
    # Testing the type of an if condition (line 94)
    if_condition_126807 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 4), result_or_keyword_126806)
    # Assigning a type to the variable 'if_condition_126807' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'if_condition_126807', if_condition_126807)
    # SSA begins for if statement (line 94)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 95)
    # Processing the call arguments (line 95)
    str_126809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 25), 'str', 'invalid axis')
    # Processing the call keyword arguments (line 95)
    kwargs_126810 = {}
    # Getting the type of 'ValueError' (line 95)
    ValueError_126808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 95)
    ValueError_call_result_126811 = invoke(stypy.reporting.localization.Localization(__file__, 95, 14), ValueError_126808, *[str_126809], **kwargs_126810)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 95, 8), ValueError_call_result_126811, 'raise parameter', BaseException)
    # SSA join for if statement (line 94)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'axis' (line 96)
    axis_126812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 11), 'axis')
    # Assigning a type to the variable 'stypy_return_type' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'stypy_return_type', axis_126812)
    
    # ################# End of '_check_axis(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_check_axis' in the type store
    # Getting the type of 'stypy_return_type' (line 91)
    stypy_return_type_126813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126813)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_check_axis'
    return stypy_return_type_126813

# Assigning a type to the variable '_check_axis' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), '_check_axis', _check_axis)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

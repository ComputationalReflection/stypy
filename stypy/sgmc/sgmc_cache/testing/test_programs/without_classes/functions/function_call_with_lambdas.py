
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: global_var = "hi"
2: 
3: def joinseq(seq):
4:     if len(seq) == 1:
5:         return '(' + seq[0] + ',)'
6:     else:
7:         return '(' + ', '.join(seq) + ')'
8: 
9: 
10: def strseq(object, convert, join=joinseq):
11:     if type(object) in [list, tuple]:
12:         return join([strseq(_o, convert, join) for _o in object])
13:     else:
14:         return convert(object)
15: 
16: 
17: def foo(par):
18:     return '*' + par
19: 
20: def foo2():
21:     par = None
22:     return '*' + par
23: 
24: def formatargspec(args, varargs=None, varkw=None, defaults=None,
25:                  formatarg=str,
26:                  formatvarargs=lambda name: '*' + name,
27:                  formatvarkw=lambda name: '**' + name,
28:                  formatvalue=lambda value: '=' + repr(value),
29:                  join=joinseq):
30:     specs = []
31:     if defaults:
32:         firstdefault = len(args) - len(defaults)
33:         for i in range(len(args)):
34:             spec = strseq(args[i], formatarg, join)
35:             if defaults and i >= firstdefault:
36:                 spec = spec + formatvalue(defaults[i - firstdefault])
37:             specs.append(spec)
38:             if varargs is not None:
39:                 specs.append(formatvarargs(varargs))
40:                 foo(varargs)
41:                 foo2()
42:             if varkw is not None:
43:                 specs.append(formatvarkw(varkw))
44:         return '(' + ', '.join(specs) + ')'
45: 
46: 
47: r = formatargspec(('a', 'b'), None, None, (3, 4))
48: print r
49: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 1):
str_795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 13), 'str', 'hi')
# Assigning a type to the variable 'global_var' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'global_var', str_795)

@norecursion
def joinseq(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'joinseq'
    module_type_store = module_type_store.open_function_context('joinseq', 3, 0, False)
    
    # Passed parameters checking function
    joinseq.stypy_localization = localization
    joinseq.stypy_type_of_self = None
    joinseq.stypy_type_store = module_type_store
    joinseq.stypy_function_name = 'joinseq'
    joinseq.stypy_param_names_list = ['seq']
    joinseq.stypy_varargs_param_name = None
    joinseq.stypy_kwargs_param_name = None
    joinseq.stypy_call_defaults = defaults
    joinseq.stypy_call_varargs = varargs
    joinseq.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'joinseq', ['seq'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'joinseq', localization, ['seq'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'joinseq(...)' code ##################

    
    
    
    # Call to len(...): (line 4)
    # Processing the call arguments (line 4)
    # Getting the type of 'seq' (line 4)
    seq_797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 11), 'seq', False)
    # Processing the call keyword arguments (line 4)
    kwargs_798 = {}
    # Getting the type of 'len' (line 4)
    len_796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 7), 'len', False)
    # Calling len(args, kwargs) (line 4)
    len_call_result_799 = invoke(stypy.reporting.localization.Localization(__file__, 4, 7), len_796, *[seq_797], **kwargs_798)
    
    int_800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 19), 'int')
    # Applying the binary operator '==' (line 4)
    result_eq_801 = python_operator(stypy.reporting.localization.Localization(__file__, 4, 7), '==', len_call_result_799, int_800)
    
    # Testing the type of an if condition (line 4)
    if_condition_802 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 4, 4), result_eq_801)
    # Assigning a type to the variable 'if_condition_802' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'if_condition_802', if_condition_802)
    # SSA begins for if statement (line 4)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'str', '(')
    
    # Obtaining the type of the subscript
    int_804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 25), 'int')
    # Getting the type of 'seq' (line 5)
    seq_805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 21), 'seq')
    # Obtaining the member '__getitem__' of a type (line 5)
    getitem___806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 21), seq_805, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 5)
    subscript_call_result_807 = invoke(stypy.reporting.localization.Localization(__file__, 5, 21), getitem___806, int_804)
    
    # Applying the binary operator '+' (line 5)
    result_add_808 = python_operator(stypy.reporting.localization.Localization(__file__, 5, 15), '+', str_803, subscript_call_result_807)
    
    str_809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 30), 'str', ',)')
    # Applying the binary operator '+' (line 5)
    result_add_810 = python_operator(stypy.reporting.localization.Localization(__file__, 5, 28), '+', result_add_808, str_809)
    
    # Assigning a type to the variable 'stypy_return_type' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 8), 'stypy_return_type', result_add_810)
    # SSA branch for the else part of an if statement (line 4)
    module_type_store.open_ssa_branch('else')
    str_811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'str', '(')
    
    # Call to join(...): (line 7)
    # Processing the call arguments (line 7)
    # Getting the type of 'seq' (line 7)
    seq_814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 31), 'seq', False)
    # Processing the call keyword arguments (line 7)
    kwargs_815 = {}
    str_812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 21), 'str', ', ')
    # Obtaining the member 'join' of a type (line 7)
    join_813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 21), str_812, 'join')
    # Calling join(args, kwargs) (line 7)
    join_call_result_816 = invoke(stypy.reporting.localization.Localization(__file__, 7, 21), join_813, *[seq_814], **kwargs_815)
    
    # Applying the binary operator '+' (line 7)
    result_add_817 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 15), '+', str_811, join_call_result_816)
    
    str_818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 38), 'str', ')')
    # Applying the binary operator '+' (line 7)
    result_add_819 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 36), '+', result_add_817, str_818)
    
    # Assigning a type to the variable 'stypy_return_type' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'stypy_return_type', result_add_819)
    # SSA join for if statement (line 4)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'joinseq(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'joinseq' in the type store
    # Getting the type of 'stypy_return_type' (line 3)
    stypy_return_type_820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_820)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'joinseq'
    return stypy_return_type_820

# Assigning a type to the variable 'joinseq' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'joinseq', joinseq)

@norecursion
def strseq(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'joinseq' (line 10)
    joinseq_821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 33), 'joinseq')
    defaults = [joinseq_821]
    # Create a new context for function 'strseq'
    module_type_store = module_type_store.open_function_context('strseq', 10, 0, False)
    
    # Passed parameters checking function
    strseq.stypy_localization = localization
    strseq.stypy_type_of_self = None
    strseq.stypy_type_store = module_type_store
    strseq.stypy_function_name = 'strseq'
    strseq.stypy_param_names_list = ['object', 'convert', 'join']
    strseq.stypy_varargs_param_name = None
    strseq.stypy_kwargs_param_name = None
    strseq.stypy_call_defaults = defaults
    strseq.stypy_call_varargs = varargs
    strseq.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'strseq', ['object', 'convert', 'join'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'strseq', localization, ['object', 'convert', 'join'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'strseq(...)' code ##################

    
    
    
    # Call to type(...): (line 11)
    # Processing the call arguments (line 11)
    # Getting the type of 'object' (line 11)
    object_823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'object', False)
    # Processing the call keyword arguments (line 11)
    kwargs_824 = {}
    # Getting the type of 'type' (line 11)
    type_822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 7), 'type', False)
    # Calling type(args, kwargs) (line 11)
    type_call_result_825 = invoke(stypy.reporting.localization.Localization(__file__, 11, 7), type_822, *[object_823], **kwargs_824)
    
    
    # Obtaining an instance of the builtin type 'list' (line 11)
    list_826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 11)
    # Adding element type (line 11)
    # Getting the type of 'list' (line 11)
    list_827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 24), 'list')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 23), list_826, list_827)
    # Adding element type (line 11)
    # Getting the type of 'tuple' (line 11)
    tuple_828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 30), 'tuple')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 23), list_826, tuple_828)
    
    # Applying the binary operator 'in' (line 11)
    result_contains_829 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 7), 'in', type_call_result_825, list_826)
    
    # Testing the type of an if condition (line 11)
    if_condition_830 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 11, 4), result_contains_829)
    # Assigning a type to the variable 'if_condition_830' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'if_condition_830', if_condition_830)
    # SSA begins for if statement (line 11)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to join(...): (line 12)
    # Processing the call arguments (line 12)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'object' (line 12)
    object_838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 57), 'object', False)
    comprehension_839 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 21), object_838)
    # Assigning a type to the variable '_o' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 21), '_o', comprehension_839)
    
    # Call to strseq(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of '_o' (line 12)
    _o_833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 28), '_o', False)
    # Getting the type of 'convert' (line 12)
    convert_834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 32), 'convert', False)
    # Getting the type of 'join' (line 12)
    join_835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 41), 'join', False)
    # Processing the call keyword arguments (line 12)
    kwargs_836 = {}
    # Getting the type of 'strseq' (line 12)
    strseq_832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 21), 'strseq', False)
    # Calling strseq(args, kwargs) (line 12)
    strseq_call_result_837 = invoke(stypy.reporting.localization.Localization(__file__, 12, 21), strseq_832, *[_o_833, convert_834, join_835], **kwargs_836)
    
    list_840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 21), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 21), list_840, strseq_call_result_837)
    # Processing the call keyword arguments (line 12)
    kwargs_841 = {}
    # Getting the type of 'join' (line 12)
    join_831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 15), 'join', False)
    # Calling join(args, kwargs) (line 12)
    join_call_result_842 = invoke(stypy.reporting.localization.Localization(__file__, 12, 15), join_831, *[list_840], **kwargs_841)
    
    # Assigning a type to the variable 'stypy_return_type' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'stypy_return_type', join_call_result_842)
    # SSA branch for the else part of an if statement (line 11)
    module_type_store.open_ssa_branch('else')
    
    # Call to convert(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'object' (line 14)
    object_844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 23), 'object', False)
    # Processing the call keyword arguments (line 14)
    kwargs_845 = {}
    # Getting the type of 'convert' (line 14)
    convert_843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 15), 'convert', False)
    # Calling convert(args, kwargs) (line 14)
    convert_call_result_846 = invoke(stypy.reporting.localization.Localization(__file__, 14, 15), convert_843, *[object_844], **kwargs_845)
    
    # Assigning a type to the variable 'stypy_return_type' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'stypy_return_type', convert_call_result_846)
    # SSA join for if statement (line 11)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'strseq(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'strseq' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_847)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'strseq'
    return stypy_return_type_847

# Assigning a type to the variable 'strseq' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'strseq', strseq)

@norecursion
def foo(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'foo'
    module_type_store = module_type_store.open_function_context('foo', 17, 0, False)
    
    # Passed parameters checking function
    foo.stypy_localization = localization
    foo.stypy_type_of_self = None
    foo.stypy_type_store = module_type_store
    foo.stypy_function_name = 'foo'
    foo.stypy_param_names_list = ['par']
    foo.stypy_varargs_param_name = None
    foo.stypy_kwargs_param_name = None
    foo.stypy_call_defaults = defaults
    foo.stypy_call_varargs = varargs
    foo.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'foo', ['par'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'foo', localization, ['par'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'foo(...)' code ##################

    str_848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 11), 'str', '*')
    # Getting the type of 'par' (line 18)
    par_849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 17), 'par')
    # Applying the binary operator '+' (line 18)
    result_add_850 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 11), '+', str_848, par_849)
    
    # Assigning a type to the variable 'stypy_return_type' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type', result_add_850)
    
    # ################# End of 'foo(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'foo' in the type store
    # Getting the type of 'stypy_return_type' (line 17)
    stypy_return_type_851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_851)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'foo'
    return stypy_return_type_851

# Assigning a type to the variable 'foo' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'foo', foo)

@norecursion
def foo2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'foo2'
    module_type_store = module_type_store.open_function_context('foo2', 20, 0, False)
    
    # Passed parameters checking function
    foo2.stypy_localization = localization
    foo2.stypy_type_of_self = None
    foo2.stypy_type_store = module_type_store
    foo2.stypy_function_name = 'foo2'
    foo2.stypy_param_names_list = []
    foo2.stypy_varargs_param_name = None
    foo2.stypy_kwargs_param_name = None
    foo2.stypy_call_defaults = defaults
    foo2.stypy_call_varargs = varargs
    foo2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'foo2', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'foo2', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'foo2(...)' code ##################

    
    # Assigning a Name to a Name (line 21):
    # Getting the type of 'None' (line 21)
    None_852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'None')
    # Assigning a type to the variable 'par' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'par', None_852)
    str_853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 11), 'str', '*')
    # Getting the type of 'par' (line 22)
    par_854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'par')
    # Applying the binary operator '+' (line 22)
    result_add_855 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 11), '+', str_853, par_854)
    
    # Assigning a type to the variable 'stypy_return_type' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type', result_add_855)
    
    # ################# End of 'foo2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'foo2' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_856)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'foo2'
    return stypy_return_type_856

# Assigning a type to the variable 'foo2' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'foo2', foo2)

@norecursion
def formatargspec(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 24)
    None_857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 32), 'None')
    # Getting the type of 'None' (line 24)
    None_858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 44), 'None')
    # Getting the type of 'None' (line 24)
    None_859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 59), 'None')
    # Getting the type of 'str' (line 25)
    str_860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 27), 'str')

    @norecursion
    def _stypy_temp_lambda_3(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_3'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_3', 26, 31, True)
        # Passed parameters checking function
        _stypy_temp_lambda_3.stypy_localization = localization
        _stypy_temp_lambda_3.stypy_type_of_self = None
        _stypy_temp_lambda_3.stypy_type_store = module_type_store
        _stypy_temp_lambda_3.stypy_function_name = '_stypy_temp_lambda_3'
        _stypy_temp_lambda_3.stypy_param_names_list = ['name']
        _stypy_temp_lambda_3.stypy_varargs_param_name = None
        _stypy_temp_lambda_3.stypy_kwargs_param_name = None
        _stypy_temp_lambda_3.stypy_call_defaults = defaults
        _stypy_temp_lambda_3.stypy_call_varargs = varargs
        _stypy_temp_lambda_3.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_3', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_3', ['name'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        str_861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 44), 'str', '*')
        # Getting the type of 'name' (line 26)
        name_862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 50), 'name')
        # Applying the binary operator '+' (line 26)
        result_add_863 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 44), '+', str_861, name_862)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 31), 'stypy_return_type', result_add_863)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_3' in the type store
        # Getting the type of 'stypy_return_type' (line 26)
        stypy_return_type_864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 31), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_864)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_3'
        return stypy_return_type_864

    # Assigning a type to the variable '_stypy_temp_lambda_3' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 31), '_stypy_temp_lambda_3', _stypy_temp_lambda_3)
    # Getting the type of '_stypy_temp_lambda_3' (line 26)
    _stypy_temp_lambda_3_865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 31), '_stypy_temp_lambda_3')

    @norecursion
    def _stypy_temp_lambda_4(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_4'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_4', 27, 29, True)
        # Passed parameters checking function
        _stypy_temp_lambda_4.stypy_localization = localization
        _stypy_temp_lambda_4.stypy_type_of_self = None
        _stypy_temp_lambda_4.stypy_type_store = module_type_store
        _stypy_temp_lambda_4.stypy_function_name = '_stypy_temp_lambda_4'
        _stypy_temp_lambda_4.stypy_param_names_list = ['name']
        _stypy_temp_lambda_4.stypy_varargs_param_name = None
        _stypy_temp_lambda_4.stypy_kwargs_param_name = None
        _stypy_temp_lambda_4.stypy_call_defaults = defaults
        _stypy_temp_lambda_4.stypy_call_varargs = varargs
        _stypy_temp_lambda_4.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_4', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_4', ['name'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        str_866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 42), 'str', '**')
        # Getting the type of 'name' (line 27)
        name_867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 49), 'name')
        # Applying the binary operator '+' (line 27)
        result_add_868 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 42), '+', str_866, name_867)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 29), 'stypy_return_type', result_add_868)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_4' in the type store
        # Getting the type of 'stypy_return_type' (line 27)
        stypy_return_type_869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 29), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_869)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_4'
        return stypy_return_type_869

    # Assigning a type to the variable '_stypy_temp_lambda_4' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 29), '_stypy_temp_lambda_4', _stypy_temp_lambda_4)
    # Getting the type of '_stypy_temp_lambda_4' (line 27)
    _stypy_temp_lambda_4_870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 29), '_stypy_temp_lambda_4')

    @norecursion
    def _stypy_temp_lambda_5(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_5'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_5', 28, 29, True)
        # Passed parameters checking function
        _stypy_temp_lambda_5.stypy_localization = localization
        _stypy_temp_lambda_5.stypy_type_of_self = None
        _stypy_temp_lambda_5.stypy_type_store = module_type_store
        _stypy_temp_lambda_5.stypy_function_name = '_stypy_temp_lambda_5'
        _stypy_temp_lambda_5.stypy_param_names_list = ['value']
        _stypy_temp_lambda_5.stypy_varargs_param_name = None
        _stypy_temp_lambda_5.stypy_kwargs_param_name = None
        _stypy_temp_lambda_5.stypy_call_defaults = defaults
        _stypy_temp_lambda_5.stypy_call_varargs = varargs
        _stypy_temp_lambda_5.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_5', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_5', ['value'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        str_871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 43), 'str', '=')
        
        # Call to repr(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'value' (line 28)
        value_873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 54), 'value', False)
        # Processing the call keyword arguments (line 28)
        kwargs_874 = {}
        # Getting the type of 'repr' (line 28)
        repr_872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 49), 'repr', False)
        # Calling repr(args, kwargs) (line 28)
        repr_call_result_875 = invoke(stypy.reporting.localization.Localization(__file__, 28, 49), repr_872, *[value_873], **kwargs_874)
        
        # Applying the binary operator '+' (line 28)
        result_add_876 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 43), '+', str_871, repr_call_result_875)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 29), 'stypy_return_type', result_add_876)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_5' in the type store
        # Getting the type of 'stypy_return_type' (line 28)
        stypy_return_type_877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 29), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_877)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_5'
        return stypy_return_type_877

    # Assigning a type to the variable '_stypy_temp_lambda_5' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 29), '_stypy_temp_lambda_5', _stypy_temp_lambda_5)
    # Getting the type of '_stypy_temp_lambda_5' (line 28)
    _stypy_temp_lambda_5_878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 29), '_stypy_temp_lambda_5')
    # Getting the type of 'joinseq' (line 29)
    joinseq_879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 22), 'joinseq')
    defaults = [None_857, None_858, None_859, str_860, _stypy_temp_lambda_3_865, _stypy_temp_lambda_4_870, _stypy_temp_lambda_5_878, joinseq_879]
    # Create a new context for function 'formatargspec'
    module_type_store = module_type_store.open_function_context('formatargspec', 24, 0, False)
    
    # Passed parameters checking function
    formatargspec.stypy_localization = localization
    formatargspec.stypy_type_of_self = None
    formatargspec.stypy_type_store = module_type_store
    formatargspec.stypy_function_name = 'formatargspec'
    formatargspec.stypy_param_names_list = ['args', 'varargs', 'varkw', 'defaults', 'formatarg', 'formatvarargs', 'formatvarkw', 'formatvalue', 'join']
    formatargspec.stypy_varargs_param_name = None
    formatargspec.stypy_kwargs_param_name = None
    formatargspec.stypy_call_defaults = defaults
    formatargspec.stypy_call_varargs = varargs
    formatargspec.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'formatargspec', ['args', 'varargs', 'varkw', 'defaults', 'formatarg', 'formatvarargs', 'formatvarkw', 'formatvalue', 'join'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'formatargspec', localization, ['args', 'varargs', 'varkw', 'defaults', 'formatarg', 'formatvarargs', 'formatvarkw', 'formatvalue', 'join'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'formatargspec(...)' code ##################

    
    # Assigning a List to a Name (line 30):
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    
    # Assigning a type to the variable 'specs' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'specs', list_880)
    
    # Getting the type of 'defaults' (line 31)
    defaults_881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 7), 'defaults')
    # Testing the type of an if condition (line 31)
    if_condition_882 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 31, 4), defaults_881)
    # Assigning a type to the variable 'if_condition_882' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'if_condition_882', if_condition_882)
    # SSA begins for if statement (line 31)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 32):
    
    # Call to len(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'args' (line 32)
    args_884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 27), 'args', False)
    # Processing the call keyword arguments (line 32)
    kwargs_885 = {}
    # Getting the type of 'len' (line 32)
    len_883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 23), 'len', False)
    # Calling len(args, kwargs) (line 32)
    len_call_result_886 = invoke(stypy.reporting.localization.Localization(__file__, 32, 23), len_883, *[args_884], **kwargs_885)
    
    
    # Call to len(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'defaults' (line 32)
    defaults_888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 39), 'defaults', False)
    # Processing the call keyword arguments (line 32)
    kwargs_889 = {}
    # Getting the type of 'len' (line 32)
    len_887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 35), 'len', False)
    # Calling len(args, kwargs) (line 32)
    len_call_result_890 = invoke(stypy.reporting.localization.Localization(__file__, 32, 35), len_887, *[defaults_888], **kwargs_889)
    
    # Applying the binary operator '-' (line 32)
    result_sub_891 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 23), '-', len_call_result_886, len_call_result_890)
    
    # Assigning a type to the variable 'firstdefault' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'firstdefault', result_sub_891)
    
    
    # Call to range(...): (line 33)
    # Processing the call arguments (line 33)
    
    # Call to len(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'args' (line 33)
    args_894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 27), 'args', False)
    # Processing the call keyword arguments (line 33)
    kwargs_895 = {}
    # Getting the type of 'len' (line 33)
    len_893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 23), 'len', False)
    # Calling len(args, kwargs) (line 33)
    len_call_result_896 = invoke(stypy.reporting.localization.Localization(__file__, 33, 23), len_893, *[args_894], **kwargs_895)
    
    # Processing the call keyword arguments (line 33)
    kwargs_897 = {}
    # Getting the type of 'range' (line 33)
    range_892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 17), 'range', False)
    # Calling range(args, kwargs) (line 33)
    range_call_result_898 = invoke(stypy.reporting.localization.Localization(__file__, 33, 17), range_892, *[len_call_result_896], **kwargs_897)
    
    # Testing the type of a for loop iterable (line 33)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 33, 8), range_call_result_898)
    # Getting the type of the for loop variable (line 33)
    for_loop_var_899 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 33, 8), range_call_result_898)
    # Assigning a type to the variable 'i' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'i', for_loop_var_899)
    # SSA begins for a for statement (line 33)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 34):
    
    # Call to strseq(...): (line 34)
    # Processing the call arguments (line 34)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 34)
    i_901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 31), 'i', False)
    # Getting the type of 'args' (line 34)
    args_902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 26), 'args', False)
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 26), args_902, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_904 = invoke(stypy.reporting.localization.Localization(__file__, 34, 26), getitem___903, i_901)
    
    # Getting the type of 'formatarg' (line 34)
    formatarg_905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 35), 'formatarg', False)
    # Getting the type of 'join' (line 34)
    join_906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 46), 'join', False)
    # Processing the call keyword arguments (line 34)
    kwargs_907 = {}
    # Getting the type of 'strseq' (line 34)
    strseq_900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'strseq', False)
    # Calling strseq(args, kwargs) (line 34)
    strseq_call_result_908 = invoke(stypy.reporting.localization.Localization(__file__, 34, 19), strseq_900, *[subscript_call_result_904, formatarg_905, join_906], **kwargs_907)
    
    # Assigning a type to the variable 'spec' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'spec', strseq_call_result_908)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'defaults' (line 35)
    defaults_909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'defaults')
    
    # Getting the type of 'i' (line 35)
    i_910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 28), 'i')
    # Getting the type of 'firstdefault' (line 35)
    firstdefault_911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 33), 'firstdefault')
    # Applying the binary operator '>=' (line 35)
    result_ge_912 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 28), '>=', i_910, firstdefault_911)
    
    # Applying the binary operator 'and' (line 35)
    result_and_keyword_913 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 15), 'and', defaults_909, result_ge_912)
    
    # Testing the type of an if condition (line 35)
    if_condition_914 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 12), result_and_keyword_913)
    # Assigning a type to the variable 'if_condition_914' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'if_condition_914', if_condition_914)
    # SSA begins for if statement (line 35)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 36):
    # Getting the type of 'spec' (line 36)
    spec_915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 23), 'spec')
    
    # Call to formatvalue(...): (line 36)
    # Processing the call arguments (line 36)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 36)
    i_917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 51), 'i', False)
    # Getting the type of 'firstdefault' (line 36)
    firstdefault_918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 55), 'firstdefault', False)
    # Applying the binary operator '-' (line 36)
    result_sub_919 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 51), '-', i_917, firstdefault_918)
    
    # Getting the type of 'defaults' (line 36)
    defaults_920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 42), 'defaults', False)
    # Obtaining the member '__getitem__' of a type (line 36)
    getitem___921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 42), defaults_920, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 36)
    subscript_call_result_922 = invoke(stypy.reporting.localization.Localization(__file__, 36, 42), getitem___921, result_sub_919)
    
    # Processing the call keyword arguments (line 36)
    kwargs_923 = {}
    # Getting the type of 'formatvalue' (line 36)
    formatvalue_916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 30), 'formatvalue', False)
    # Calling formatvalue(args, kwargs) (line 36)
    formatvalue_call_result_924 = invoke(stypy.reporting.localization.Localization(__file__, 36, 30), formatvalue_916, *[subscript_call_result_922], **kwargs_923)
    
    # Applying the binary operator '+' (line 36)
    result_add_925 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 23), '+', spec_915, formatvalue_call_result_924)
    
    # Assigning a type to the variable 'spec' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'spec', result_add_925)
    # SSA join for if statement (line 35)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'spec' (line 37)
    spec_928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 25), 'spec', False)
    # Processing the call keyword arguments (line 37)
    kwargs_929 = {}
    # Getting the type of 'specs' (line 37)
    specs_926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'specs', False)
    # Obtaining the member 'append' of a type (line 37)
    append_927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 12), specs_926, 'append')
    # Calling append(args, kwargs) (line 37)
    append_call_result_930 = invoke(stypy.reporting.localization.Localization(__file__, 37, 12), append_927, *[spec_928], **kwargs_929)
    
    
    # Type idiom detected: calculating its left and rigth part (line 38)
    # Getting the type of 'varargs' (line 38)
    varargs_931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'varargs')
    # Getting the type of 'None' (line 38)
    None_932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 30), 'None')
    
    (may_be_933, more_types_in_union_934) = may_not_be_none(varargs_931, None_932)

    if may_be_933:

        if more_types_in_union_934:
            # Runtime conditional SSA (line 38)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to append(...): (line 39)
        # Processing the call arguments (line 39)
        
        # Call to formatvarargs(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'varargs' (line 39)
        varargs_938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 43), 'varargs', False)
        # Processing the call keyword arguments (line 39)
        kwargs_939 = {}
        # Getting the type of 'formatvarargs' (line 39)
        formatvarargs_937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 29), 'formatvarargs', False)
        # Calling formatvarargs(args, kwargs) (line 39)
        formatvarargs_call_result_940 = invoke(stypy.reporting.localization.Localization(__file__, 39, 29), formatvarargs_937, *[varargs_938], **kwargs_939)
        
        # Processing the call keyword arguments (line 39)
        kwargs_941 = {}
        # Getting the type of 'specs' (line 39)
        specs_935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'specs', False)
        # Obtaining the member 'append' of a type (line 39)
        append_936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 16), specs_935, 'append')
        # Calling append(args, kwargs) (line 39)
        append_call_result_942 = invoke(stypy.reporting.localization.Localization(__file__, 39, 16), append_936, *[formatvarargs_call_result_940], **kwargs_941)
        
        
        # Call to foo(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'varargs' (line 40)
        varargs_944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 20), 'varargs', False)
        # Processing the call keyword arguments (line 40)
        kwargs_945 = {}
        # Getting the type of 'foo' (line 40)
        foo_943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'foo', False)
        # Calling foo(args, kwargs) (line 40)
        foo_call_result_946 = invoke(stypy.reporting.localization.Localization(__file__, 40, 16), foo_943, *[varargs_944], **kwargs_945)
        
        
        # Call to foo2(...): (line 41)
        # Processing the call keyword arguments (line 41)
        kwargs_948 = {}
        # Getting the type of 'foo2' (line 41)
        foo2_947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'foo2', False)
        # Calling foo2(args, kwargs) (line 41)
        foo2_call_result_949 = invoke(stypy.reporting.localization.Localization(__file__, 41, 16), foo2_947, *[], **kwargs_948)
        

        if more_types_in_union_934:
            # SSA join for if statement (line 38)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 42)
    # Getting the type of 'varkw' (line 42)
    varkw_950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'varkw')
    # Getting the type of 'None' (line 42)
    None_951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 28), 'None')
    
    (may_be_952, more_types_in_union_953) = may_not_be_none(varkw_950, None_951)

    if may_be_952:

        if more_types_in_union_953:
            # Runtime conditional SSA (line 42)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to append(...): (line 43)
        # Processing the call arguments (line 43)
        
        # Call to formatvarkw(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'varkw' (line 43)
        varkw_957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 41), 'varkw', False)
        # Processing the call keyword arguments (line 43)
        kwargs_958 = {}
        # Getting the type of 'formatvarkw' (line 43)
        formatvarkw_956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 29), 'formatvarkw', False)
        # Calling formatvarkw(args, kwargs) (line 43)
        formatvarkw_call_result_959 = invoke(stypy.reporting.localization.Localization(__file__, 43, 29), formatvarkw_956, *[varkw_957], **kwargs_958)
        
        # Processing the call keyword arguments (line 43)
        kwargs_960 = {}
        # Getting the type of 'specs' (line 43)
        specs_954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 16), 'specs', False)
        # Obtaining the member 'append' of a type (line 43)
        append_955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 16), specs_954, 'append')
        # Calling append(args, kwargs) (line 43)
        append_call_result_961 = invoke(stypy.reporting.localization.Localization(__file__, 43, 16), append_955, *[formatvarkw_call_result_959], **kwargs_960)
        

        if more_types_in_union_953:
            # SSA join for if statement (line 42)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    str_962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 15), 'str', '(')
    
    # Call to join(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'specs' (line 44)
    specs_965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 31), 'specs', False)
    # Processing the call keyword arguments (line 44)
    kwargs_966 = {}
    str_963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 21), 'str', ', ')
    # Obtaining the member 'join' of a type (line 44)
    join_964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 21), str_963, 'join')
    # Calling join(args, kwargs) (line 44)
    join_call_result_967 = invoke(stypy.reporting.localization.Localization(__file__, 44, 21), join_964, *[specs_965], **kwargs_966)
    
    # Applying the binary operator '+' (line 44)
    result_add_968 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 15), '+', str_962, join_call_result_967)
    
    str_969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 40), 'str', ')')
    # Applying the binary operator '+' (line 44)
    result_add_970 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 38), '+', result_add_968, str_969)
    
    # Assigning a type to the variable 'stypy_return_type' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'stypy_return_type', result_add_970)
    # SSA join for if statement (line 31)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'formatargspec(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'formatargspec' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_971)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'formatargspec'
    return stypy_return_type_971

# Assigning a type to the variable 'formatargspec' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'formatargspec', formatargspec)

# Assigning a Call to a Name (line 47):

# Call to formatargspec(...): (line 47)
# Processing the call arguments (line 47)

# Obtaining an instance of the builtin type 'tuple' (line 47)
tuple_973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 47)
# Adding element type (line 47)
str_974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 19), 'str', 'a')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 19), tuple_973, str_974)
# Adding element type (line 47)
str_975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 24), 'str', 'b')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 19), tuple_973, str_975)

# Getting the type of 'None' (line 47)
None_976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 30), 'None', False)
# Getting the type of 'None' (line 47)
None_977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 36), 'None', False)

# Obtaining an instance of the builtin type 'tuple' (line 47)
tuple_978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 43), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 47)
# Adding element type (line 47)
int_979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 43), tuple_978, int_979)
# Adding element type (line 47)
int_980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 46), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 43), tuple_978, int_980)

# Processing the call keyword arguments (line 47)
kwargs_981 = {}
# Getting the type of 'formatargspec' (line 47)
formatargspec_972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'formatargspec', False)
# Calling formatargspec(args, kwargs) (line 47)
formatargspec_call_result_982 = invoke(stypy.reporting.localization.Localization(__file__, 47, 4), formatargspec_972, *[tuple_973, None_976, None_977, tuple_978], **kwargs_981)

# Assigning a type to the variable 'r' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'r', formatargspec_call_result_982)
# Getting the type of 'r' (line 48)
r_983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 6), 'r')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

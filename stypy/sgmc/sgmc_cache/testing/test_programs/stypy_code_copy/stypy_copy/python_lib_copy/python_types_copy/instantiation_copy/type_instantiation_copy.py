
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import time
2: import operator
3: 
4: from known_python_types_copy import known_python_type_typename_samplevalues, Foo
5: 
6: # Predefined instances for some types that do not have a non-parameter constructor of an alternative way to create
7: # an instance. This is used when needing fake values for types
8: __known_instances = {
9:     UnicodeEncodeError: UnicodeEncodeError("a", u"b", 1, 2, "e"),
10:     UnicodeDecodeError: UnicodeDecodeError("a", "b", 1, 2, "e"),
11:     UnicodeTranslateError: UnicodeTranslateError(u'0', 1, 2, '3'),
12:     type(time.gmtime()): time.gmtime(),
13:     operator.attrgetter: operator.attrgetter(Foo.qux),
14:     operator.methodcaller: operator.methodcaller(Foo.bar),
15: }
16: 
17: 
18: # TODO: This needs to be completed
19: def get_type_sample_value(type_):
20:     try:
21:         if type_ in __known_instances:
22:             return __known_instances[type_]
23: 
24:         if type_ in known_python_type_typename_samplevalues:
25:             return known_python_type_typename_samplevalues[type_][1]
26: 
27:         return type_()
28:     except TypeError as t:
29:         pass
30: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import time' statement (line 1)
import time

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'time', time, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import operator' statement (line 2)
import operator

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'operator', operator, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from known_python_types_copy import known_python_type_typename_samplevalues, Foo' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/instantiation_copy/')
import_9904 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'known_python_types_copy')

if (type(import_9904) is not StypyTypeError):

    if (import_9904 != 'pyd_module'):
        __import__(import_9904)
        sys_modules_9905 = sys.modules[import_9904]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'known_python_types_copy', sys_modules_9905.module_type_store, module_type_store, ['known_python_type_typename_samplevalues', 'Foo'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_9905, sys_modules_9905.module_type_store, module_type_store)
    else:
        from known_python_types_copy import known_python_type_typename_samplevalues, Foo

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'known_python_types_copy', None, module_type_store, ['known_python_type_typename_samplevalues', 'Foo'], [known_python_type_typename_samplevalues, Foo])

else:
    # Assigning a type to the variable 'known_python_types_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'known_python_types_copy', import_9904)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/instantiation_copy/')


# Assigning a Dict to a Name (line 8):

# Obtaining an instance of the builtin type 'dict' (line 8)
dict_9906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 20), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 8)
# Adding element type (key, value) (line 8)
# Getting the type of 'UnicodeEncodeError' (line 9)
UnicodeEncodeError_9907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'UnicodeEncodeError')

# Call to UnicodeEncodeError(...): (line 9)
# Processing the call arguments (line 9)
str_9909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 43), 'str', 'a')
unicode_9910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 48), 'unicode', u'b')
int_9911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 54), 'int')
int_9912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 57), 'int')
str_9913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 60), 'str', 'e')
# Processing the call keyword arguments (line 9)
kwargs_9914 = {}
# Getting the type of 'UnicodeEncodeError' (line 9)
UnicodeEncodeError_9908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 24), 'UnicodeEncodeError', False)
# Calling UnicodeEncodeError(args, kwargs) (line 9)
UnicodeEncodeError_call_result_9915 = invoke(stypy.reporting.localization.Localization(__file__, 9, 24), UnicodeEncodeError_9908, *[str_9909, unicode_9910, int_9911, int_9912, str_9913], **kwargs_9914)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 20), dict_9906, (UnicodeEncodeError_9907, UnicodeEncodeError_call_result_9915))
# Adding element type (key, value) (line 8)
# Getting the type of 'UnicodeDecodeError' (line 10)
UnicodeDecodeError_9916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'UnicodeDecodeError')

# Call to UnicodeDecodeError(...): (line 10)
# Processing the call arguments (line 10)
str_9918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 43), 'str', 'a')
str_9919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 48), 'str', 'b')
int_9920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 53), 'int')
int_9921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 56), 'int')
str_9922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 59), 'str', 'e')
# Processing the call keyword arguments (line 10)
kwargs_9923 = {}
# Getting the type of 'UnicodeDecodeError' (line 10)
UnicodeDecodeError_9917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 24), 'UnicodeDecodeError', False)
# Calling UnicodeDecodeError(args, kwargs) (line 10)
UnicodeDecodeError_call_result_9924 = invoke(stypy.reporting.localization.Localization(__file__, 10, 24), UnicodeDecodeError_9917, *[str_9918, str_9919, int_9920, int_9921, str_9922], **kwargs_9923)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 20), dict_9906, (UnicodeDecodeError_9916, UnicodeDecodeError_call_result_9924))
# Adding element type (key, value) (line 8)
# Getting the type of 'UnicodeTranslateError' (line 11)
UnicodeTranslateError_9925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'UnicodeTranslateError')

# Call to UnicodeTranslateError(...): (line 11)
# Processing the call arguments (line 11)
unicode_9927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 49), 'unicode', u'0')
int_9928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 55), 'int')
int_9929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 58), 'int')
str_9930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 61), 'str', '3')
# Processing the call keyword arguments (line 11)
kwargs_9931 = {}
# Getting the type of 'UnicodeTranslateError' (line 11)
UnicodeTranslateError_9926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 27), 'UnicodeTranslateError', False)
# Calling UnicodeTranslateError(args, kwargs) (line 11)
UnicodeTranslateError_call_result_9932 = invoke(stypy.reporting.localization.Localization(__file__, 11, 27), UnicodeTranslateError_9926, *[unicode_9927, int_9928, int_9929, str_9930], **kwargs_9931)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 20), dict_9906, (UnicodeTranslateError_9925, UnicodeTranslateError_call_result_9932))
# Adding element type (key, value) (line 8)

# Call to type(...): (line 12)
# Processing the call arguments (line 12)

# Call to gmtime(...): (line 12)
# Processing the call keyword arguments (line 12)
kwargs_9936 = {}
# Getting the type of 'time' (line 12)
time_9934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'time', False)
# Obtaining the member 'gmtime' of a type (line 12)
gmtime_9935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 9), time_9934, 'gmtime')
# Calling gmtime(args, kwargs) (line 12)
gmtime_call_result_9937 = invoke(stypy.reporting.localization.Localization(__file__, 12, 9), gmtime_9935, *[], **kwargs_9936)

# Processing the call keyword arguments (line 12)
kwargs_9938 = {}
# Getting the type of 'type' (line 12)
type_9933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'type', False)
# Calling type(args, kwargs) (line 12)
type_call_result_9939 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), type_9933, *[gmtime_call_result_9937], **kwargs_9938)


# Call to gmtime(...): (line 12)
# Processing the call keyword arguments (line 12)
kwargs_9942 = {}
# Getting the type of 'time' (line 12)
time_9940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 25), 'time', False)
# Obtaining the member 'gmtime' of a type (line 12)
gmtime_9941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 25), time_9940, 'gmtime')
# Calling gmtime(args, kwargs) (line 12)
gmtime_call_result_9943 = invoke(stypy.reporting.localization.Localization(__file__, 12, 25), gmtime_9941, *[], **kwargs_9942)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 20), dict_9906, (type_call_result_9939, gmtime_call_result_9943))
# Adding element type (key, value) (line 8)
# Getting the type of 'operator' (line 13)
operator_9944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'operator')
# Obtaining the member 'attrgetter' of a type (line 13)
attrgetter_9945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 4), operator_9944, 'attrgetter')

# Call to attrgetter(...): (line 13)
# Processing the call arguments (line 13)
# Getting the type of 'Foo' (line 13)
Foo_9948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 45), 'Foo', False)
# Obtaining the member 'qux' of a type (line 13)
qux_9949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 45), Foo_9948, 'qux')
# Processing the call keyword arguments (line 13)
kwargs_9950 = {}
# Getting the type of 'operator' (line 13)
operator_9946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 25), 'operator', False)
# Obtaining the member 'attrgetter' of a type (line 13)
attrgetter_9947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 25), operator_9946, 'attrgetter')
# Calling attrgetter(args, kwargs) (line 13)
attrgetter_call_result_9951 = invoke(stypy.reporting.localization.Localization(__file__, 13, 25), attrgetter_9947, *[qux_9949], **kwargs_9950)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 20), dict_9906, (attrgetter_9945, attrgetter_call_result_9951))
# Adding element type (key, value) (line 8)
# Getting the type of 'operator' (line 14)
operator_9952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'operator')
# Obtaining the member 'methodcaller' of a type (line 14)
methodcaller_9953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 4), operator_9952, 'methodcaller')

# Call to methodcaller(...): (line 14)
# Processing the call arguments (line 14)
# Getting the type of 'Foo' (line 14)
Foo_9956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 49), 'Foo', False)
# Obtaining the member 'bar' of a type (line 14)
bar_9957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 49), Foo_9956, 'bar')
# Processing the call keyword arguments (line 14)
kwargs_9958 = {}
# Getting the type of 'operator' (line 14)
operator_9954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 27), 'operator', False)
# Obtaining the member 'methodcaller' of a type (line 14)
methodcaller_9955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 27), operator_9954, 'methodcaller')
# Calling methodcaller(args, kwargs) (line 14)
methodcaller_call_result_9959 = invoke(stypy.reporting.localization.Localization(__file__, 14, 27), methodcaller_9955, *[bar_9957], **kwargs_9958)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 20), dict_9906, (methodcaller_9953, methodcaller_call_result_9959))

# Assigning a type to the variable '__known_instances' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '__known_instances', dict_9906)

@norecursion
def get_type_sample_value(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_type_sample_value'
    module_type_store = module_type_store.open_function_context('get_type_sample_value', 19, 0, False)
    
    # Passed parameters checking function
    get_type_sample_value.stypy_localization = localization
    get_type_sample_value.stypy_type_of_self = None
    get_type_sample_value.stypy_type_store = module_type_store
    get_type_sample_value.stypy_function_name = 'get_type_sample_value'
    get_type_sample_value.stypy_param_names_list = ['type_']
    get_type_sample_value.stypy_varargs_param_name = None
    get_type_sample_value.stypy_kwargs_param_name = None
    get_type_sample_value.stypy_call_defaults = defaults
    get_type_sample_value.stypy_call_varargs = varargs
    get_type_sample_value.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_type_sample_value', ['type_'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_type_sample_value', localization, ['type_'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_type_sample_value(...)' code ##################

    
    
    # SSA begins for try-except statement (line 20)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Getting the type of 'type_' (line 21)
    type__9960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 11), 'type_')
    # Getting the type of '__known_instances' (line 21)
    known_instances_9961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 20), '__known_instances')
    # Applying the binary operator 'in' (line 21)
    result_contains_9962 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 11), 'in', type__9960, known_instances_9961)
    
    # Testing if the type of an if condition is none (line 21)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 21, 8), result_contains_9962):
        pass
    else:
        
        # Testing the type of an if condition (line 21)
        if_condition_9963 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 21, 8), result_contains_9962)
        # Assigning a type to the variable 'if_condition_9963' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'if_condition_9963', if_condition_9963)
        # SSA begins for if statement (line 21)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining the type of the subscript
        # Getting the type of 'type_' (line 22)
        type__9964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 37), 'type_')
        # Getting the type of '__known_instances' (line 22)
        known_instances_9965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 19), '__known_instances')
        # Obtaining the member '__getitem__' of a type (line 22)
        getitem___9966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 19), known_instances_9965, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 22)
        subscript_call_result_9967 = invoke(stypy.reporting.localization.Localization(__file__, 22, 19), getitem___9966, type__9964)
        
        # Assigning a type to the variable 'stypy_return_type' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'stypy_return_type', subscript_call_result_9967)
        # SSA join for if statement (line 21)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'type_' (line 24)
    type__9968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 11), 'type_')
    # Getting the type of 'known_python_type_typename_samplevalues' (line 24)
    known_python_type_typename_samplevalues_9969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 20), 'known_python_type_typename_samplevalues')
    # Applying the binary operator 'in' (line 24)
    result_contains_9970 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 11), 'in', type__9968, known_python_type_typename_samplevalues_9969)
    
    # Testing if the type of an if condition is none (line 24)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 24, 8), result_contains_9970):
        pass
    else:
        
        # Testing the type of an if condition (line 24)
        if_condition_9971 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 24, 8), result_contains_9970)
        # Assigning a type to the variable 'if_condition_9971' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'if_condition_9971', if_condition_9971)
        # SSA begins for if statement (line 24)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining the type of the subscript
        int_9972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 66), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'type_' (line 25)
        type__9973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 59), 'type_')
        # Getting the type of 'known_python_type_typename_samplevalues' (line 25)
        known_python_type_typename_samplevalues_9974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 19), 'known_python_type_typename_samplevalues')
        # Obtaining the member '__getitem__' of a type (line 25)
        getitem___9975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 19), known_python_type_typename_samplevalues_9974, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 25)
        subscript_call_result_9976 = invoke(stypy.reporting.localization.Localization(__file__, 25, 19), getitem___9975, type__9973)
        
        # Obtaining the member '__getitem__' of a type (line 25)
        getitem___9977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 19), subscript_call_result_9976, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 25)
        subscript_call_result_9978 = invoke(stypy.reporting.localization.Localization(__file__, 25, 19), getitem___9977, int_9972)
        
        # Assigning a type to the variable 'stypy_return_type' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'stypy_return_type', subscript_call_result_9978)
        # SSA join for if statement (line 24)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to type_(...): (line 27)
    # Processing the call keyword arguments (line 27)
    kwargs_9980 = {}
    # Getting the type of 'type_' (line 27)
    type__9979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 15), 'type_', False)
    # Calling type_(args, kwargs) (line 27)
    type__call_result_9981 = invoke(stypy.reporting.localization.Localization(__file__, 27, 15), type__9979, *[], **kwargs_9980)
    
    # Assigning a type to the variable 'stypy_return_type' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'stypy_return_type', type__call_result_9981)
    # SSA branch for the except part of a try statement (line 20)
    # SSA branch for the except 'TypeError' branch of a try statement (line 20)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'TypeError' (line 28)
    TypeError_9982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 11), 'TypeError')
    # Assigning a type to the variable 't' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 't', TypeError_9982)
    pass
    # SSA join for try-except statement (line 20)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'get_type_sample_value(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_type_sample_value' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_9983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9983)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_type_sample_value'
    return stypy_return_type_9983

# Assigning a type to the variable 'get_type_sample_value' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'get_type_sample_value', get_type_sample_value)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

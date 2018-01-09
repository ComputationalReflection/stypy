
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Here we perform some symbolic computations required for the N-D
3: interpolation routines in `interpnd.pyx`.
4: 
5: '''
6: from __future__ import division, print_function, absolute_import
7: 
8: from sympy import symbols, binomial, Matrix
9: 
10: 
11: def _estimate_gradients_2d_global():
12: 
13:     #
14:     # Compute
15:     #
16:     #
17: 
18:     f1, f2, df1, df2, x = symbols(['f1', 'f2', 'df1', 'df2', 'x'])
19:     c = [f1, (df1 + 3*f1)/3, (df2 + 3*f2)/3, f2]
20: 
21:     w = 0
22:     for k in range(4):
23:         w += binomial(3, k) * c[k] * x**k*(1-x)**(3-k)
24: 
25:     wpp = w.diff(x, 2).expand()
26:     intwpp2 = (wpp**2).integrate((x, 0, 1)).expand()
27: 
28:     A = Matrix([[intwpp2.coeff(df1**2), intwpp2.coeff(df1*df2)/2],
29:                 [intwpp2.coeff(df1*df2)/2, intwpp2.coeff(df2**2)]])
30: 
31:     B = Matrix([[intwpp2.coeff(df1).subs(df2, 0)],
32:                 [intwpp2.coeff(df2).subs(df1, 0)]]) / 2
33: 
34:     print("A")
35:     print(A)
36:     print("B")
37:     print(B)
38:     print("solution")
39:     print(A.inv() * B)
40: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_63356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'str', '\nHere we perform some symbolic computations required for the N-D\ninterpolation routines in `interpnd.pyx`.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from sympy import symbols, binomial, Matrix' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_63357 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'sympy')

if (type(import_63357) is not StypyTypeError):

    if (import_63357 != 'pyd_module'):
        __import__(import_63357)
        sys_modules_63358 = sys.modules[import_63357]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'sympy', sys_modules_63358.module_type_store, module_type_store, ['symbols', 'binomial', 'Matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_63358, sys_modules_63358.module_type_store, module_type_store)
    else:
        from sympy import symbols, binomial, Matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'sympy', None, module_type_store, ['symbols', 'binomial', 'Matrix'], [symbols, binomial, Matrix])

else:
    # Assigning a type to the variable 'sympy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'sympy', import_63357)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')


@norecursion
def _estimate_gradients_2d_global(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_estimate_gradients_2d_global'
    module_type_store = module_type_store.open_function_context('_estimate_gradients_2d_global', 11, 0, False)
    
    # Passed parameters checking function
    _estimate_gradients_2d_global.stypy_localization = localization
    _estimate_gradients_2d_global.stypy_type_of_self = None
    _estimate_gradients_2d_global.stypy_type_store = module_type_store
    _estimate_gradients_2d_global.stypy_function_name = '_estimate_gradients_2d_global'
    _estimate_gradients_2d_global.stypy_param_names_list = []
    _estimate_gradients_2d_global.stypy_varargs_param_name = None
    _estimate_gradients_2d_global.stypy_kwargs_param_name = None
    _estimate_gradients_2d_global.stypy_call_defaults = defaults
    _estimate_gradients_2d_global.stypy_call_varargs = varargs
    _estimate_gradients_2d_global.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_estimate_gradients_2d_global', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_estimate_gradients_2d_global', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_estimate_gradients_2d_global(...)' code ##################

    
    # Assigning a Call to a Tuple (line 18):
    
    # Assigning a Subscript to a Name (line 18):
    
    # Obtaining the type of the subscript
    int_63359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 4), 'int')
    
    # Call to symbols(...): (line 18)
    # Processing the call arguments (line 18)
    
    # Obtaining an instance of the builtin type 'list' (line 18)
    list_63361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 18)
    # Adding element type (line 18)
    str_63362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 35), 'str', 'f1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 34), list_63361, str_63362)
    # Adding element type (line 18)
    str_63363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 41), 'str', 'f2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 34), list_63361, str_63363)
    # Adding element type (line 18)
    str_63364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 47), 'str', 'df1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 34), list_63361, str_63364)
    # Adding element type (line 18)
    str_63365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 54), 'str', 'df2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 34), list_63361, str_63365)
    # Adding element type (line 18)
    str_63366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 61), 'str', 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 34), list_63361, str_63366)
    
    # Processing the call keyword arguments (line 18)
    kwargs_63367 = {}
    # Getting the type of 'symbols' (line 18)
    symbols_63360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 26), 'symbols', False)
    # Calling symbols(args, kwargs) (line 18)
    symbols_call_result_63368 = invoke(stypy.reporting.localization.Localization(__file__, 18, 26), symbols_63360, *[list_63361], **kwargs_63367)
    
    # Obtaining the member '__getitem__' of a type (line 18)
    getitem___63369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 4), symbols_call_result_63368, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 18)
    subscript_call_result_63370 = invoke(stypy.reporting.localization.Localization(__file__, 18, 4), getitem___63369, int_63359)
    
    # Assigning a type to the variable 'tuple_var_assignment_63351' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'tuple_var_assignment_63351', subscript_call_result_63370)
    
    # Assigning a Subscript to a Name (line 18):
    
    # Obtaining the type of the subscript
    int_63371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 4), 'int')
    
    # Call to symbols(...): (line 18)
    # Processing the call arguments (line 18)
    
    # Obtaining an instance of the builtin type 'list' (line 18)
    list_63373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 18)
    # Adding element type (line 18)
    str_63374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 35), 'str', 'f1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 34), list_63373, str_63374)
    # Adding element type (line 18)
    str_63375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 41), 'str', 'f2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 34), list_63373, str_63375)
    # Adding element type (line 18)
    str_63376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 47), 'str', 'df1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 34), list_63373, str_63376)
    # Adding element type (line 18)
    str_63377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 54), 'str', 'df2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 34), list_63373, str_63377)
    # Adding element type (line 18)
    str_63378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 61), 'str', 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 34), list_63373, str_63378)
    
    # Processing the call keyword arguments (line 18)
    kwargs_63379 = {}
    # Getting the type of 'symbols' (line 18)
    symbols_63372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 26), 'symbols', False)
    # Calling symbols(args, kwargs) (line 18)
    symbols_call_result_63380 = invoke(stypy.reporting.localization.Localization(__file__, 18, 26), symbols_63372, *[list_63373], **kwargs_63379)
    
    # Obtaining the member '__getitem__' of a type (line 18)
    getitem___63381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 4), symbols_call_result_63380, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 18)
    subscript_call_result_63382 = invoke(stypy.reporting.localization.Localization(__file__, 18, 4), getitem___63381, int_63371)
    
    # Assigning a type to the variable 'tuple_var_assignment_63352' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'tuple_var_assignment_63352', subscript_call_result_63382)
    
    # Assigning a Subscript to a Name (line 18):
    
    # Obtaining the type of the subscript
    int_63383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 4), 'int')
    
    # Call to symbols(...): (line 18)
    # Processing the call arguments (line 18)
    
    # Obtaining an instance of the builtin type 'list' (line 18)
    list_63385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 18)
    # Adding element type (line 18)
    str_63386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 35), 'str', 'f1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 34), list_63385, str_63386)
    # Adding element type (line 18)
    str_63387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 41), 'str', 'f2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 34), list_63385, str_63387)
    # Adding element type (line 18)
    str_63388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 47), 'str', 'df1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 34), list_63385, str_63388)
    # Adding element type (line 18)
    str_63389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 54), 'str', 'df2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 34), list_63385, str_63389)
    # Adding element type (line 18)
    str_63390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 61), 'str', 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 34), list_63385, str_63390)
    
    # Processing the call keyword arguments (line 18)
    kwargs_63391 = {}
    # Getting the type of 'symbols' (line 18)
    symbols_63384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 26), 'symbols', False)
    # Calling symbols(args, kwargs) (line 18)
    symbols_call_result_63392 = invoke(stypy.reporting.localization.Localization(__file__, 18, 26), symbols_63384, *[list_63385], **kwargs_63391)
    
    # Obtaining the member '__getitem__' of a type (line 18)
    getitem___63393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 4), symbols_call_result_63392, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 18)
    subscript_call_result_63394 = invoke(stypy.reporting.localization.Localization(__file__, 18, 4), getitem___63393, int_63383)
    
    # Assigning a type to the variable 'tuple_var_assignment_63353' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'tuple_var_assignment_63353', subscript_call_result_63394)
    
    # Assigning a Subscript to a Name (line 18):
    
    # Obtaining the type of the subscript
    int_63395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 4), 'int')
    
    # Call to symbols(...): (line 18)
    # Processing the call arguments (line 18)
    
    # Obtaining an instance of the builtin type 'list' (line 18)
    list_63397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 18)
    # Adding element type (line 18)
    str_63398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 35), 'str', 'f1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 34), list_63397, str_63398)
    # Adding element type (line 18)
    str_63399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 41), 'str', 'f2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 34), list_63397, str_63399)
    # Adding element type (line 18)
    str_63400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 47), 'str', 'df1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 34), list_63397, str_63400)
    # Adding element type (line 18)
    str_63401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 54), 'str', 'df2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 34), list_63397, str_63401)
    # Adding element type (line 18)
    str_63402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 61), 'str', 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 34), list_63397, str_63402)
    
    # Processing the call keyword arguments (line 18)
    kwargs_63403 = {}
    # Getting the type of 'symbols' (line 18)
    symbols_63396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 26), 'symbols', False)
    # Calling symbols(args, kwargs) (line 18)
    symbols_call_result_63404 = invoke(stypy.reporting.localization.Localization(__file__, 18, 26), symbols_63396, *[list_63397], **kwargs_63403)
    
    # Obtaining the member '__getitem__' of a type (line 18)
    getitem___63405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 4), symbols_call_result_63404, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 18)
    subscript_call_result_63406 = invoke(stypy.reporting.localization.Localization(__file__, 18, 4), getitem___63405, int_63395)
    
    # Assigning a type to the variable 'tuple_var_assignment_63354' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'tuple_var_assignment_63354', subscript_call_result_63406)
    
    # Assigning a Subscript to a Name (line 18):
    
    # Obtaining the type of the subscript
    int_63407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 4), 'int')
    
    # Call to symbols(...): (line 18)
    # Processing the call arguments (line 18)
    
    # Obtaining an instance of the builtin type 'list' (line 18)
    list_63409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 18)
    # Adding element type (line 18)
    str_63410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 35), 'str', 'f1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 34), list_63409, str_63410)
    # Adding element type (line 18)
    str_63411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 41), 'str', 'f2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 34), list_63409, str_63411)
    # Adding element type (line 18)
    str_63412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 47), 'str', 'df1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 34), list_63409, str_63412)
    # Adding element type (line 18)
    str_63413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 54), 'str', 'df2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 34), list_63409, str_63413)
    # Adding element type (line 18)
    str_63414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 61), 'str', 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 34), list_63409, str_63414)
    
    # Processing the call keyword arguments (line 18)
    kwargs_63415 = {}
    # Getting the type of 'symbols' (line 18)
    symbols_63408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 26), 'symbols', False)
    # Calling symbols(args, kwargs) (line 18)
    symbols_call_result_63416 = invoke(stypy.reporting.localization.Localization(__file__, 18, 26), symbols_63408, *[list_63409], **kwargs_63415)
    
    # Obtaining the member '__getitem__' of a type (line 18)
    getitem___63417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 4), symbols_call_result_63416, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 18)
    subscript_call_result_63418 = invoke(stypy.reporting.localization.Localization(__file__, 18, 4), getitem___63417, int_63407)
    
    # Assigning a type to the variable 'tuple_var_assignment_63355' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'tuple_var_assignment_63355', subscript_call_result_63418)
    
    # Assigning a Name to a Name (line 18):
    # Getting the type of 'tuple_var_assignment_63351' (line 18)
    tuple_var_assignment_63351_63419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'tuple_var_assignment_63351')
    # Assigning a type to the variable 'f1' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'f1', tuple_var_assignment_63351_63419)
    
    # Assigning a Name to a Name (line 18):
    # Getting the type of 'tuple_var_assignment_63352' (line 18)
    tuple_var_assignment_63352_63420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'tuple_var_assignment_63352')
    # Assigning a type to the variable 'f2' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'f2', tuple_var_assignment_63352_63420)
    
    # Assigning a Name to a Name (line 18):
    # Getting the type of 'tuple_var_assignment_63353' (line 18)
    tuple_var_assignment_63353_63421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'tuple_var_assignment_63353')
    # Assigning a type to the variable 'df1' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 12), 'df1', tuple_var_assignment_63353_63421)
    
    # Assigning a Name to a Name (line 18):
    # Getting the type of 'tuple_var_assignment_63354' (line 18)
    tuple_var_assignment_63354_63422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'tuple_var_assignment_63354')
    # Assigning a type to the variable 'df2' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 17), 'df2', tuple_var_assignment_63354_63422)
    
    # Assigning a Name to a Name (line 18):
    # Getting the type of 'tuple_var_assignment_63355' (line 18)
    tuple_var_assignment_63355_63423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'tuple_var_assignment_63355')
    # Assigning a type to the variable 'x' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 22), 'x', tuple_var_assignment_63355_63423)
    
    # Assigning a List to a Name (line 19):
    
    # Assigning a List to a Name (line 19):
    
    # Obtaining an instance of the builtin type 'list' (line 19)
    list_63424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 19)
    # Adding element type (line 19)
    # Getting the type of 'f1' (line 19)
    f1_63425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 9), 'f1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 8), list_63424, f1_63425)
    # Adding element type (line 19)
    # Getting the type of 'df1' (line 19)
    df1_63426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 14), 'df1')
    int_63427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 20), 'int')
    # Getting the type of 'f1' (line 19)
    f1_63428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 22), 'f1')
    # Applying the binary operator '*' (line 19)
    result_mul_63429 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 20), '*', int_63427, f1_63428)
    
    # Applying the binary operator '+' (line 19)
    result_add_63430 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 14), '+', df1_63426, result_mul_63429)
    
    int_63431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 26), 'int')
    # Applying the binary operator 'div' (line 19)
    result_div_63432 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 13), 'div', result_add_63430, int_63431)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 8), list_63424, result_div_63432)
    # Adding element type (line 19)
    # Getting the type of 'df2' (line 19)
    df2_63433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 30), 'df2')
    int_63434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 36), 'int')
    # Getting the type of 'f2' (line 19)
    f2_63435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 38), 'f2')
    # Applying the binary operator '*' (line 19)
    result_mul_63436 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 36), '*', int_63434, f2_63435)
    
    # Applying the binary operator '+' (line 19)
    result_add_63437 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 30), '+', df2_63433, result_mul_63436)
    
    int_63438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 42), 'int')
    # Applying the binary operator 'div' (line 19)
    result_div_63439 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 29), 'div', result_add_63437, int_63438)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 8), list_63424, result_div_63439)
    # Adding element type (line 19)
    # Getting the type of 'f2' (line 19)
    f2_63440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 45), 'f2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 8), list_63424, f2_63440)
    
    # Assigning a type to the variable 'c' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'c', list_63424)
    
    # Assigning a Num to a Name (line 21):
    
    # Assigning a Num to a Name (line 21):
    int_63441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 8), 'int')
    # Assigning a type to the variable 'w' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'w', int_63441)
    
    
    # Call to range(...): (line 22)
    # Processing the call arguments (line 22)
    int_63443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 19), 'int')
    # Processing the call keyword arguments (line 22)
    kwargs_63444 = {}
    # Getting the type of 'range' (line 22)
    range_63442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 13), 'range', False)
    # Calling range(args, kwargs) (line 22)
    range_call_result_63445 = invoke(stypy.reporting.localization.Localization(__file__, 22, 13), range_63442, *[int_63443], **kwargs_63444)
    
    # Testing the type of a for loop iterable (line 22)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 22, 4), range_call_result_63445)
    # Getting the type of the for loop variable (line 22)
    for_loop_var_63446 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 22, 4), range_call_result_63445)
    # Assigning a type to the variable 'k' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'k', for_loop_var_63446)
    # SSA begins for a for statement (line 22)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'w' (line 23)
    w_63447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'w')
    
    # Call to binomial(...): (line 23)
    # Processing the call arguments (line 23)
    int_63449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 22), 'int')
    # Getting the type of 'k' (line 23)
    k_63450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 25), 'k', False)
    # Processing the call keyword arguments (line 23)
    kwargs_63451 = {}
    # Getting the type of 'binomial' (line 23)
    binomial_63448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 13), 'binomial', False)
    # Calling binomial(args, kwargs) (line 23)
    binomial_call_result_63452 = invoke(stypy.reporting.localization.Localization(__file__, 23, 13), binomial_63448, *[int_63449, k_63450], **kwargs_63451)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 23)
    k_63453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 32), 'k')
    # Getting the type of 'c' (line 23)
    c_63454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 30), 'c')
    # Obtaining the member '__getitem__' of a type (line 23)
    getitem___63455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 30), c_63454, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 23)
    subscript_call_result_63456 = invoke(stypy.reporting.localization.Localization(__file__, 23, 30), getitem___63455, k_63453)
    
    # Applying the binary operator '*' (line 23)
    result_mul_63457 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 13), '*', binomial_call_result_63452, subscript_call_result_63456)
    
    # Getting the type of 'x' (line 23)
    x_63458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 37), 'x')
    # Getting the type of 'k' (line 23)
    k_63459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 40), 'k')
    # Applying the binary operator '**' (line 23)
    result_pow_63460 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 37), '**', x_63458, k_63459)
    
    # Applying the binary operator '*' (line 23)
    result_mul_63461 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 35), '*', result_mul_63457, result_pow_63460)
    
    int_63462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 43), 'int')
    # Getting the type of 'x' (line 23)
    x_63463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 45), 'x')
    # Applying the binary operator '-' (line 23)
    result_sub_63464 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 43), '-', int_63462, x_63463)
    
    int_63465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 50), 'int')
    # Getting the type of 'k' (line 23)
    k_63466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 52), 'k')
    # Applying the binary operator '-' (line 23)
    result_sub_63467 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 50), '-', int_63465, k_63466)
    
    # Applying the binary operator '**' (line 23)
    result_pow_63468 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 42), '**', result_sub_63464, result_sub_63467)
    
    # Applying the binary operator '*' (line 23)
    result_mul_63469 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 41), '*', result_mul_63461, result_pow_63468)
    
    # Applying the binary operator '+=' (line 23)
    result_iadd_63470 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 8), '+=', w_63447, result_mul_63469)
    # Assigning a type to the variable 'w' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'w', result_iadd_63470)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 25):
    
    # Assigning a Call to a Name (line 25):
    
    # Call to expand(...): (line 25)
    # Processing the call keyword arguments (line 25)
    kwargs_63478 = {}
    
    # Call to diff(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'x' (line 25)
    x_63473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 17), 'x', False)
    int_63474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 20), 'int')
    # Processing the call keyword arguments (line 25)
    kwargs_63475 = {}
    # Getting the type of 'w' (line 25)
    w_63471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 10), 'w', False)
    # Obtaining the member 'diff' of a type (line 25)
    diff_63472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 10), w_63471, 'diff')
    # Calling diff(args, kwargs) (line 25)
    diff_call_result_63476 = invoke(stypy.reporting.localization.Localization(__file__, 25, 10), diff_63472, *[x_63473, int_63474], **kwargs_63475)
    
    # Obtaining the member 'expand' of a type (line 25)
    expand_63477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 10), diff_call_result_63476, 'expand')
    # Calling expand(args, kwargs) (line 25)
    expand_call_result_63479 = invoke(stypy.reporting.localization.Localization(__file__, 25, 10), expand_63477, *[], **kwargs_63478)
    
    # Assigning a type to the variable 'wpp' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'wpp', expand_call_result_63479)
    
    # Assigning a Call to a Name (line 26):
    
    # Assigning a Call to a Name (line 26):
    
    # Call to expand(...): (line 26)
    # Processing the call keyword arguments (line 26)
    kwargs_63491 = {}
    
    # Call to integrate(...): (line 26)
    # Processing the call arguments (line 26)
    
    # Obtaining an instance of the builtin type 'tuple' (line 26)
    tuple_63484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 26)
    # Adding element type (line 26)
    # Getting the type of 'x' (line 26)
    x_63485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 34), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 34), tuple_63484, x_63485)
    # Adding element type (line 26)
    int_63486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 34), tuple_63484, int_63486)
    # Adding element type (line 26)
    int_63487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 34), tuple_63484, int_63487)
    
    # Processing the call keyword arguments (line 26)
    kwargs_63488 = {}
    # Getting the type of 'wpp' (line 26)
    wpp_63480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'wpp', False)
    int_63481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 20), 'int')
    # Applying the binary operator '**' (line 26)
    result_pow_63482 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 15), '**', wpp_63480, int_63481)
    
    # Obtaining the member 'integrate' of a type (line 26)
    integrate_63483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 15), result_pow_63482, 'integrate')
    # Calling integrate(args, kwargs) (line 26)
    integrate_call_result_63489 = invoke(stypy.reporting.localization.Localization(__file__, 26, 15), integrate_63483, *[tuple_63484], **kwargs_63488)
    
    # Obtaining the member 'expand' of a type (line 26)
    expand_63490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 15), integrate_call_result_63489, 'expand')
    # Calling expand(args, kwargs) (line 26)
    expand_call_result_63492 = invoke(stypy.reporting.localization.Localization(__file__, 26, 15), expand_63490, *[], **kwargs_63491)
    
    # Assigning a type to the variable 'intwpp2' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'intwpp2', expand_call_result_63492)
    
    # Assigning a Call to a Name (line 28):
    
    # Assigning a Call to a Name (line 28):
    
    # Call to Matrix(...): (line 28)
    # Processing the call arguments (line 28)
    
    # Obtaining an instance of the builtin type 'list' (line 28)
    list_63494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 28)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'list' (line 28)
    list_63495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 28)
    # Adding element type (line 28)
    
    # Call to coeff(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'df1' (line 28)
    df1_63498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 31), 'df1', False)
    int_63499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 36), 'int')
    # Applying the binary operator '**' (line 28)
    result_pow_63500 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 31), '**', df1_63498, int_63499)
    
    # Processing the call keyword arguments (line 28)
    kwargs_63501 = {}
    # Getting the type of 'intwpp2' (line 28)
    intwpp2_63496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 17), 'intwpp2', False)
    # Obtaining the member 'coeff' of a type (line 28)
    coeff_63497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 17), intwpp2_63496, 'coeff')
    # Calling coeff(args, kwargs) (line 28)
    coeff_call_result_63502 = invoke(stypy.reporting.localization.Localization(__file__, 28, 17), coeff_63497, *[result_pow_63500], **kwargs_63501)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 16), list_63495, coeff_call_result_63502)
    # Adding element type (line 28)
    
    # Call to coeff(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'df1' (line 28)
    df1_63505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 54), 'df1', False)
    # Getting the type of 'df2' (line 28)
    df2_63506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 58), 'df2', False)
    # Applying the binary operator '*' (line 28)
    result_mul_63507 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 54), '*', df1_63505, df2_63506)
    
    # Processing the call keyword arguments (line 28)
    kwargs_63508 = {}
    # Getting the type of 'intwpp2' (line 28)
    intwpp2_63503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 40), 'intwpp2', False)
    # Obtaining the member 'coeff' of a type (line 28)
    coeff_63504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 40), intwpp2_63503, 'coeff')
    # Calling coeff(args, kwargs) (line 28)
    coeff_call_result_63509 = invoke(stypy.reporting.localization.Localization(__file__, 28, 40), coeff_63504, *[result_mul_63507], **kwargs_63508)
    
    int_63510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 63), 'int')
    # Applying the binary operator 'div' (line 28)
    result_div_63511 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 40), 'div', coeff_call_result_63509, int_63510)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 16), list_63495, result_div_63511)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 15), list_63494, list_63495)
    # Adding element type (line 28)
    
    # Obtaining an instance of the builtin type 'list' (line 29)
    list_63512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 29)
    # Adding element type (line 29)
    
    # Call to coeff(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'df1' (line 29)
    df1_63515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 31), 'df1', False)
    # Getting the type of 'df2' (line 29)
    df2_63516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 35), 'df2', False)
    # Applying the binary operator '*' (line 29)
    result_mul_63517 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 31), '*', df1_63515, df2_63516)
    
    # Processing the call keyword arguments (line 29)
    kwargs_63518 = {}
    # Getting the type of 'intwpp2' (line 29)
    intwpp2_63513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 17), 'intwpp2', False)
    # Obtaining the member 'coeff' of a type (line 29)
    coeff_63514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 17), intwpp2_63513, 'coeff')
    # Calling coeff(args, kwargs) (line 29)
    coeff_call_result_63519 = invoke(stypy.reporting.localization.Localization(__file__, 29, 17), coeff_63514, *[result_mul_63517], **kwargs_63518)
    
    int_63520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 40), 'int')
    # Applying the binary operator 'div' (line 29)
    result_div_63521 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 17), 'div', coeff_call_result_63519, int_63520)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 16), list_63512, result_div_63521)
    # Adding element type (line 29)
    
    # Call to coeff(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'df2' (line 29)
    df2_63524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 57), 'df2', False)
    int_63525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 62), 'int')
    # Applying the binary operator '**' (line 29)
    result_pow_63526 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 57), '**', df2_63524, int_63525)
    
    # Processing the call keyword arguments (line 29)
    kwargs_63527 = {}
    # Getting the type of 'intwpp2' (line 29)
    intwpp2_63522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 43), 'intwpp2', False)
    # Obtaining the member 'coeff' of a type (line 29)
    coeff_63523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 43), intwpp2_63522, 'coeff')
    # Calling coeff(args, kwargs) (line 29)
    coeff_call_result_63528 = invoke(stypy.reporting.localization.Localization(__file__, 29, 43), coeff_63523, *[result_pow_63526], **kwargs_63527)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 16), list_63512, coeff_call_result_63528)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 15), list_63494, list_63512)
    
    # Processing the call keyword arguments (line 28)
    kwargs_63529 = {}
    # Getting the type of 'Matrix' (line 28)
    Matrix_63493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'Matrix', False)
    # Calling Matrix(args, kwargs) (line 28)
    Matrix_call_result_63530 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), Matrix_63493, *[list_63494], **kwargs_63529)
    
    # Assigning a type to the variable 'A' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'A', Matrix_call_result_63530)
    
    # Assigning a BinOp to a Name (line 31):
    
    # Assigning a BinOp to a Name (line 31):
    
    # Call to Matrix(...): (line 31)
    # Processing the call arguments (line 31)
    
    # Obtaining an instance of the builtin type 'list' (line 31)
    list_63532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 31)
    # Adding element type (line 31)
    
    # Obtaining an instance of the builtin type 'list' (line 31)
    list_63533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 31)
    # Adding element type (line 31)
    
    # Call to subs(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'df2' (line 31)
    df2_63540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 41), 'df2', False)
    int_63541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 46), 'int')
    # Processing the call keyword arguments (line 31)
    kwargs_63542 = {}
    
    # Call to coeff(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'df1' (line 31)
    df1_63536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 31), 'df1', False)
    # Processing the call keyword arguments (line 31)
    kwargs_63537 = {}
    # Getting the type of 'intwpp2' (line 31)
    intwpp2_63534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 17), 'intwpp2', False)
    # Obtaining the member 'coeff' of a type (line 31)
    coeff_63535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 17), intwpp2_63534, 'coeff')
    # Calling coeff(args, kwargs) (line 31)
    coeff_call_result_63538 = invoke(stypy.reporting.localization.Localization(__file__, 31, 17), coeff_63535, *[df1_63536], **kwargs_63537)
    
    # Obtaining the member 'subs' of a type (line 31)
    subs_63539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 17), coeff_call_result_63538, 'subs')
    # Calling subs(args, kwargs) (line 31)
    subs_call_result_63543 = invoke(stypy.reporting.localization.Localization(__file__, 31, 17), subs_63539, *[df2_63540, int_63541], **kwargs_63542)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 16), list_63533, subs_call_result_63543)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 15), list_63532, list_63533)
    # Adding element type (line 31)
    
    # Obtaining an instance of the builtin type 'list' (line 32)
    list_63544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 32)
    # Adding element type (line 32)
    
    # Call to subs(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'df1' (line 32)
    df1_63551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 41), 'df1', False)
    int_63552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 46), 'int')
    # Processing the call keyword arguments (line 32)
    kwargs_63553 = {}
    
    # Call to coeff(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'df2' (line 32)
    df2_63547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 31), 'df2', False)
    # Processing the call keyword arguments (line 32)
    kwargs_63548 = {}
    # Getting the type of 'intwpp2' (line 32)
    intwpp2_63545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 17), 'intwpp2', False)
    # Obtaining the member 'coeff' of a type (line 32)
    coeff_63546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 17), intwpp2_63545, 'coeff')
    # Calling coeff(args, kwargs) (line 32)
    coeff_call_result_63549 = invoke(stypy.reporting.localization.Localization(__file__, 32, 17), coeff_63546, *[df2_63547], **kwargs_63548)
    
    # Obtaining the member 'subs' of a type (line 32)
    subs_63550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 17), coeff_call_result_63549, 'subs')
    # Calling subs(args, kwargs) (line 32)
    subs_call_result_63554 = invoke(stypy.reporting.localization.Localization(__file__, 32, 17), subs_63550, *[df1_63551, int_63552], **kwargs_63553)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 16), list_63544, subs_call_result_63554)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 15), list_63532, list_63544)
    
    # Processing the call keyword arguments (line 31)
    kwargs_63555 = {}
    # Getting the type of 'Matrix' (line 31)
    Matrix_63531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'Matrix', False)
    # Calling Matrix(args, kwargs) (line 31)
    Matrix_call_result_63556 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), Matrix_63531, *[list_63532], **kwargs_63555)
    
    int_63557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 54), 'int')
    # Applying the binary operator 'div' (line 31)
    result_div_63558 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 8), 'div', Matrix_call_result_63556, int_63557)
    
    # Assigning a type to the variable 'B' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'B', result_div_63558)
    
    # Call to print(...): (line 34)
    # Processing the call arguments (line 34)
    str_63560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 10), 'str', 'A')
    # Processing the call keyword arguments (line 34)
    kwargs_63561 = {}
    # Getting the type of 'print' (line 34)
    print_63559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'print', False)
    # Calling print(args, kwargs) (line 34)
    print_call_result_63562 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), print_63559, *[str_63560], **kwargs_63561)
    
    
    # Call to print(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'A' (line 35)
    A_63564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 10), 'A', False)
    # Processing the call keyword arguments (line 35)
    kwargs_63565 = {}
    # Getting the type of 'print' (line 35)
    print_63563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'print', False)
    # Calling print(args, kwargs) (line 35)
    print_call_result_63566 = invoke(stypy.reporting.localization.Localization(__file__, 35, 4), print_63563, *[A_63564], **kwargs_63565)
    
    
    # Call to print(...): (line 36)
    # Processing the call arguments (line 36)
    str_63568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 10), 'str', 'B')
    # Processing the call keyword arguments (line 36)
    kwargs_63569 = {}
    # Getting the type of 'print' (line 36)
    print_63567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'print', False)
    # Calling print(args, kwargs) (line 36)
    print_call_result_63570 = invoke(stypy.reporting.localization.Localization(__file__, 36, 4), print_63567, *[str_63568], **kwargs_63569)
    
    
    # Call to print(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'B' (line 37)
    B_63572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 10), 'B', False)
    # Processing the call keyword arguments (line 37)
    kwargs_63573 = {}
    # Getting the type of 'print' (line 37)
    print_63571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'print', False)
    # Calling print(args, kwargs) (line 37)
    print_call_result_63574 = invoke(stypy.reporting.localization.Localization(__file__, 37, 4), print_63571, *[B_63572], **kwargs_63573)
    
    
    # Call to print(...): (line 38)
    # Processing the call arguments (line 38)
    str_63576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 10), 'str', 'solution')
    # Processing the call keyword arguments (line 38)
    kwargs_63577 = {}
    # Getting the type of 'print' (line 38)
    print_63575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'print', False)
    # Calling print(args, kwargs) (line 38)
    print_call_result_63578 = invoke(stypy.reporting.localization.Localization(__file__, 38, 4), print_63575, *[str_63576], **kwargs_63577)
    
    
    # Call to print(...): (line 39)
    # Processing the call arguments (line 39)
    
    # Call to inv(...): (line 39)
    # Processing the call keyword arguments (line 39)
    kwargs_63582 = {}
    # Getting the type of 'A' (line 39)
    A_63580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 10), 'A', False)
    # Obtaining the member 'inv' of a type (line 39)
    inv_63581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 10), A_63580, 'inv')
    # Calling inv(args, kwargs) (line 39)
    inv_call_result_63583 = invoke(stypy.reporting.localization.Localization(__file__, 39, 10), inv_63581, *[], **kwargs_63582)
    
    # Getting the type of 'B' (line 39)
    B_63584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 'B', False)
    # Applying the binary operator '*' (line 39)
    result_mul_63585 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 10), '*', inv_call_result_63583, B_63584)
    
    # Processing the call keyword arguments (line 39)
    kwargs_63586 = {}
    # Getting the type of 'print' (line 39)
    print_63579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'print', False)
    # Calling print(args, kwargs) (line 39)
    print_call_result_63587 = invoke(stypy.reporting.localization.Localization(__file__, 39, 4), print_63579, *[result_mul_63585], **kwargs_63586)
    
    
    # ################# End of '_estimate_gradients_2d_global(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_estimate_gradients_2d_global' in the type store
    # Getting the type of 'stypy_return_type' (line 11)
    stypy_return_type_63588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_63588)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_estimate_gradients_2d_global'
    return stypy_return_type_63588

# Assigning a type to the variable '_estimate_gradients_2d_global' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), '_estimate_gradients_2d_global', _estimate_gradients_2d_global)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

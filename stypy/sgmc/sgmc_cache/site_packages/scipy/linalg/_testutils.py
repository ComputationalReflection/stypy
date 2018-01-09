
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: 
5: 
6: class _FakeMatrix(object):
7:     def __init__(self, data):
8:         self._data = data
9:         self.__array_interface__ = data.__array_interface__
10: 
11: 
12: class _FakeMatrix2(object):
13:     def __init__(self, data):
14:         self._data = data
15: 
16:     def __array__(self):
17:         return self._data
18: 
19: 
20: def _get_array(shape, dtype):
21:     '''
22:     Get a test array of given shape and data type.
23:     Returned NxN matrices are posdef, and 2xN are banded-posdef.
24: 
25:     '''
26:     if len(shape) == 2 and shape[0] == 2:
27:         # yield a banded positive definite one
28:         x = np.zeros(shape, dtype=dtype)
29:         x[0, 1:] = -1
30:         x[1] = 2
31:         return x
32:     elif len(shape) == 2 and shape[0] == shape[1]:
33:         # always yield a positive definite matrix
34:         x = np.zeros(shape, dtype=dtype)
35:         j = np.arange(shape[0])
36:         x[j, j] = 2
37:         x[j[:-1], j[:-1]+1] = -1
38:         x[j[:-1]+1, j[:-1]] = -1
39:         return x
40:     else:
41:         np.random.seed(1234)
42:         return np.random.randn(*shape).astype(dtype)
43: 
44: 
45: def _id(x):
46:     return x
47: 
48: 
49: def assert_no_overwrite(call, shapes, dtypes=None):
50:     '''
51:     Test that a call does not overwrite its input arguments
52:     '''
53: 
54:     if dtypes is None:
55:         dtypes = [np.float32, np.float64, np.complex64, np.complex128]
56: 
57:     for dtype in dtypes:
58:         for order in ["C", "F"]:
59:             for faker in [_id, _FakeMatrix, _FakeMatrix2]:
60:                 orig_inputs = [_get_array(s, dtype) for s in shapes]
61:                 inputs = [faker(x.copy(order)) for x in orig_inputs]
62:                 call(*inputs)
63:                 msg = "call modified inputs [%r, %r]" % (dtype, faker)
64:                 for a, b in zip(inputs, orig_inputs):
65:                     np.testing.assert_equal(a, b, err_msg=msg)
66: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_38522 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_38522) is not StypyTypeError):

    if (import_38522 != 'pyd_module'):
        __import__(import_38522)
        sys_modules_38523 = sys.modules[import_38522]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_38523.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_38522)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

# Declaration of the '_FakeMatrix' class

class _FakeMatrix(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 7, 4, False)
        # Assigning a type to the variable 'self' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_FakeMatrix.__init__', ['data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 8):
        # Getting the type of 'data' (line 8)
        data_38524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 21), 'data')
        # Getting the type of 'self' (line 8)
        self_38525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'self')
        # Setting the type of the member '_data' of a type (line 8)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 8), self_38525, '_data', data_38524)
        
        # Assigning a Attribute to a Attribute (line 9):
        # Getting the type of 'data' (line 9)
        data_38526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 35), 'data')
        # Obtaining the member '__array_interface__' of a type (line 9)
        array_interface___38527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 35), data_38526, '__array_interface__')
        # Getting the type of 'self' (line 9)
        self_38528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'self')
        # Setting the type of the member '__array_interface__' of a type (line 9)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 8), self_38528, '__array_interface__', array_interface___38527)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable '_FakeMatrix' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '_FakeMatrix', _FakeMatrix)
# Declaration of the '_FakeMatrix2' class

class _FakeMatrix2(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 13, 4, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_FakeMatrix2.__init__', ['data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 14):
        # Getting the type of 'data' (line 14)
        data_38529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 21), 'data')
        # Getting the type of 'self' (line 14)
        self_38530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'self')
        # Setting the type of the member '_data' of a type (line 14)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), self_38530, '_data', data_38529)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __array__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__array__'
        module_type_store = module_type_store.open_function_context('__array__', 16, 4, False)
        # Assigning a type to the variable 'self' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _FakeMatrix2.__array__.__dict__.__setitem__('stypy_localization', localization)
        _FakeMatrix2.__array__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _FakeMatrix2.__array__.__dict__.__setitem__('stypy_type_store', module_type_store)
        _FakeMatrix2.__array__.__dict__.__setitem__('stypy_function_name', '_FakeMatrix2.__array__')
        _FakeMatrix2.__array__.__dict__.__setitem__('stypy_param_names_list', [])
        _FakeMatrix2.__array__.__dict__.__setitem__('stypy_varargs_param_name', None)
        _FakeMatrix2.__array__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _FakeMatrix2.__array__.__dict__.__setitem__('stypy_call_defaults', defaults)
        _FakeMatrix2.__array__.__dict__.__setitem__('stypy_call_varargs', varargs)
        _FakeMatrix2.__array__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _FakeMatrix2.__array__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_FakeMatrix2.__array__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__array__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__array__(...)' code ##################

        # Getting the type of 'self' (line 17)
        self_38531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 15), 'self')
        # Obtaining the member '_data' of a type (line 17)
        _data_38532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 15), self_38531, '_data')
        # Assigning a type to the variable 'stypy_return_type' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'stypy_return_type', _data_38532)
        
        # ################# End of '__array__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__array__' in the type store
        # Getting the type of 'stypy_return_type' (line 16)
        stypy_return_type_38533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_38533)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__array__'
        return stypy_return_type_38533


# Assigning a type to the variable '_FakeMatrix2' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), '_FakeMatrix2', _FakeMatrix2)

@norecursion
def _get_array(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_array'
    module_type_store = module_type_store.open_function_context('_get_array', 20, 0, False)
    
    # Passed parameters checking function
    _get_array.stypy_localization = localization
    _get_array.stypy_type_of_self = None
    _get_array.stypy_type_store = module_type_store
    _get_array.stypy_function_name = '_get_array'
    _get_array.stypy_param_names_list = ['shape', 'dtype']
    _get_array.stypy_varargs_param_name = None
    _get_array.stypy_kwargs_param_name = None
    _get_array.stypy_call_defaults = defaults
    _get_array.stypy_call_varargs = varargs
    _get_array.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_array', ['shape', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_array', localization, ['shape', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_array(...)' code ##################

    str_38534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, (-1)), 'str', '\n    Get a test array of given shape and data type.\n    Returned NxN matrices are posdef, and 2xN are banded-posdef.\n\n    ')
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'shape' (line 26)
    shape_38536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'shape', False)
    # Processing the call keyword arguments (line 26)
    kwargs_38537 = {}
    # Getting the type of 'len' (line 26)
    len_38535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 7), 'len', False)
    # Calling len(args, kwargs) (line 26)
    len_call_result_38538 = invoke(stypy.reporting.localization.Localization(__file__, 26, 7), len_38535, *[shape_38536], **kwargs_38537)
    
    int_38539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 21), 'int')
    # Applying the binary operator '==' (line 26)
    result_eq_38540 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 7), '==', len_call_result_38538, int_38539)
    
    
    
    # Obtaining the type of the subscript
    int_38541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 33), 'int')
    # Getting the type of 'shape' (line 26)
    shape_38542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 27), 'shape')
    # Obtaining the member '__getitem__' of a type (line 26)
    getitem___38543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 27), shape_38542, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 26)
    subscript_call_result_38544 = invoke(stypy.reporting.localization.Localization(__file__, 26, 27), getitem___38543, int_38541)
    
    int_38545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 39), 'int')
    # Applying the binary operator '==' (line 26)
    result_eq_38546 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 27), '==', subscript_call_result_38544, int_38545)
    
    # Applying the binary operator 'and' (line 26)
    result_and_keyword_38547 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 7), 'and', result_eq_38540, result_eq_38546)
    
    # Testing the type of an if condition (line 26)
    if_condition_38548 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 26, 4), result_and_keyword_38547)
    # Assigning a type to the variable 'if_condition_38548' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'if_condition_38548', if_condition_38548)
    # SSA begins for if statement (line 26)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 28):
    
    # Call to zeros(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'shape' (line 28)
    shape_38551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 21), 'shape', False)
    # Processing the call keyword arguments (line 28)
    # Getting the type of 'dtype' (line 28)
    dtype_38552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 34), 'dtype', False)
    keyword_38553 = dtype_38552
    kwargs_38554 = {'dtype': keyword_38553}
    # Getting the type of 'np' (line 28)
    np_38549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'np', False)
    # Obtaining the member 'zeros' of a type (line 28)
    zeros_38550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 12), np_38549, 'zeros')
    # Calling zeros(args, kwargs) (line 28)
    zeros_call_result_38555 = invoke(stypy.reporting.localization.Localization(__file__, 28, 12), zeros_38550, *[shape_38551], **kwargs_38554)
    
    # Assigning a type to the variable 'x' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'x', zeros_call_result_38555)
    
    # Assigning a Num to a Subscript (line 29):
    int_38556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 19), 'int')
    # Getting the type of 'x' (line 29)
    x_38557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'x')
    int_38558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 10), 'int')
    int_38559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 13), 'int')
    slice_38560 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 29, 8), int_38559, None, None)
    # Storing an element on a container (line 29)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 8), x_38557, ((int_38558, slice_38560), int_38556))
    
    # Assigning a Num to a Subscript (line 30):
    int_38561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 15), 'int')
    # Getting the type of 'x' (line 30)
    x_38562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'x')
    int_38563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 10), 'int')
    # Storing an element on a container (line 30)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 8), x_38562, (int_38563, int_38561))
    # Getting the type of 'x' (line 31)
    x_38564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'stypy_return_type', x_38564)
    # SSA branch for the else part of an if statement (line 26)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'shape' (line 32)
    shape_38566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 13), 'shape', False)
    # Processing the call keyword arguments (line 32)
    kwargs_38567 = {}
    # Getting the type of 'len' (line 32)
    len_38565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 9), 'len', False)
    # Calling len(args, kwargs) (line 32)
    len_call_result_38568 = invoke(stypy.reporting.localization.Localization(__file__, 32, 9), len_38565, *[shape_38566], **kwargs_38567)
    
    int_38569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 23), 'int')
    # Applying the binary operator '==' (line 32)
    result_eq_38570 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 9), '==', len_call_result_38568, int_38569)
    
    
    
    # Obtaining the type of the subscript
    int_38571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 35), 'int')
    # Getting the type of 'shape' (line 32)
    shape_38572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 29), 'shape')
    # Obtaining the member '__getitem__' of a type (line 32)
    getitem___38573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 29), shape_38572, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 32)
    subscript_call_result_38574 = invoke(stypy.reporting.localization.Localization(__file__, 32, 29), getitem___38573, int_38571)
    
    
    # Obtaining the type of the subscript
    int_38575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 47), 'int')
    # Getting the type of 'shape' (line 32)
    shape_38576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 41), 'shape')
    # Obtaining the member '__getitem__' of a type (line 32)
    getitem___38577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 41), shape_38576, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 32)
    subscript_call_result_38578 = invoke(stypy.reporting.localization.Localization(__file__, 32, 41), getitem___38577, int_38575)
    
    # Applying the binary operator '==' (line 32)
    result_eq_38579 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 29), '==', subscript_call_result_38574, subscript_call_result_38578)
    
    # Applying the binary operator 'and' (line 32)
    result_and_keyword_38580 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 9), 'and', result_eq_38570, result_eq_38579)
    
    # Testing the type of an if condition (line 32)
    if_condition_38581 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 32, 9), result_and_keyword_38580)
    # Assigning a type to the variable 'if_condition_38581' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 9), 'if_condition_38581', if_condition_38581)
    # SSA begins for if statement (line 32)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 34):
    
    # Call to zeros(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'shape' (line 34)
    shape_38584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 21), 'shape', False)
    # Processing the call keyword arguments (line 34)
    # Getting the type of 'dtype' (line 34)
    dtype_38585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 34), 'dtype', False)
    keyword_38586 = dtype_38585
    kwargs_38587 = {'dtype': keyword_38586}
    # Getting the type of 'np' (line 34)
    np_38582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'np', False)
    # Obtaining the member 'zeros' of a type (line 34)
    zeros_38583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), np_38582, 'zeros')
    # Calling zeros(args, kwargs) (line 34)
    zeros_call_result_38588 = invoke(stypy.reporting.localization.Localization(__file__, 34, 12), zeros_38583, *[shape_38584], **kwargs_38587)
    
    # Assigning a type to the variable 'x' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'x', zeros_call_result_38588)
    
    # Assigning a Call to a Name (line 35):
    
    # Call to arange(...): (line 35)
    # Processing the call arguments (line 35)
    
    # Obtaining the type of the subscript
    int_38591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 28), 'int')
    # Getting the type of 'shape' (line 35)
    shape_38592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 22), 'shape', False)
    # Obtaining the member '__getitem__' of a type (line 35)
    getitem___38593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 22), shape_38592, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 35)
    subscript_call_result_38594 = invoke(stypy.reporting.localization.Localization(__file__, 35, 22), getitem___38593, int_38591)
    
    # Processing the call keyword arguments (line 35)
    kwargs_38595 = {}
    # Getting the type of 'np' (line 35)
    np_38589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'np', False)
    # Obtaining the member 'arange' of a type (line 35)
    arange_38590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 12), np_38589, 'arange')
    # Calling arange(args, kwargs) (line 35)
    arange_call_result_38596 = invoke(stypy.reporting.localization.Localization(__file__, 35, 12), arange_38590, *[subscript_call_result_38594], **kwargs_38595)
    
    # Assigning a type to the variable 'j' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'j', arange_call_result_38596)
    
    # Assigning a Num to a Subscript (line 36):
    int_38597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 18), 'int')
    # Getting the type of 'x' (line 36)
    x_38598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'x')
    
    # Obtaining an instance of the builtin type 'tuple' (line 36)
    tuple_38599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 36)
    # Adding element type (line 36)
    # Getting the type of 'j' (line 36)
    j_38600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 10), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 10), tuple_38599, j_38600)
    # Adding element type (line 36)
    # Getting the type of 'j' (line 36)
    j_38601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 13), 'j')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 10), tuple_38599, j_38601)
    
    # Storing an element on a container (line 36)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 8), x_38598, (tuple_38599, int_38597))
    
    # Assigning a Num to a Subscript (line 37):
    int_38602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 30), 'int')
    # Getting the type of 'x' (line 37)
    x_38603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'x')
    
    # Obtaining an instance of the builtin type 'tuple' (line 37)
    tuple_38604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 37)
    # Adding element type (line 37)
    
    # Obtaining the type of the subscript
    int_38605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 13), 'int')
    slice_38606 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 37, 10), None, int_38605, None)
    # Getting the type of 'j' (line 37)
    j_38607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 10), 'j')
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___38608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 10), j_38607, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 37)
    subscript_call_result_38609 = invoke(stypy.reporting.localization.Localization(__file__, 37, 10), getitem___38608, slice_38606)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 10), tuple_38604, subscript_call_result_38609)
    # Adding element type (line 37)
    
    # Obtaining the type of the subscript
    int_38610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 21), 'int')
    slice_38611 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 37, 18), None, int_38610, None)
    # Getting the type of 'j' (line 37)
    j_38612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 18), 'j')
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___38613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 18), j_38612, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 37)
    subscript_call_result_38614 = invoke(stypy.reporting.localization.Localization(__file__, 37, 18), getitem___38613, slice_38611)
    
    int_38615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 25), 'int')
    # Applying the binary operator '+' (line 37)
    result_add_38616 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 18), '+', subscript_call_result_38614, int_38615)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 10), tuple_38604, result_add_38616)
    
    # Storing an element on a container (line 37)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 8), x_38603, (tuple_38604, int_38602))
    
    # Assigning a Num to a Subscript (line 38):
    int_38617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 30), 'int')
    # Getting the type of 'x' (line 38)
    x_38618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'x')
    
    # Obtaining an instance of the builtin type 'tuple' (line 38)
    tuple_38619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 38)
    # Adding element type (line 38)
    
    # Obtaining the type of the subscript
    int_38620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 13), 'int')
    slice_38621 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 38, 10), None, int_38620, None)
    # Getting the type of 'j' (line 38)
    j_38622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 10), 'j')
    # Obtaining the member '__getitem__' of a type (line 38)
    getitem___38623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 10), j_38622, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 38)
    subscript_call_result_38624 = invoke(stypy.reporting.localization.Localization(__file__, 38, 10), getitem___38623, slice_38621)
    
    int_38625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 17), 'int')
    # Applying the binary operator '+' (line 38)
    result_add_38626 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 10), '+', subscript_call_result_38624, int_38625)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 10), tuple_38619, result_add_38626)
    # Adding element type (line 38)
    
    # Obtaining the type of the subscript
    int_38627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 23), 'int')
    slice_38628 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 38, 20), None, int_38627, None)
    # Getting the type of 'j' (line 38)
    j_38629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 20), 'j')
    # Obtaining the member '__getitem__' of a type (line 38)
    getitem___38630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 20), j_38629, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 38)
    subscript_call_result_38631 = invoke(stypy.reporting.localization.Localization(__file__, 38, 20), getitem___38630, slice_38628)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 10), tuple_38619, subscript_call_result_38631)
    
    # Storing an element on a container (line 38)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 8), x_38618, (tuple_38619, int_38617))
    # Getting the type of 'x' (line 39)
    x_38632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'stypy_return_type', x_38632)
    # SSA branch for the else part of an if statement (line 32)
    module_type_store.open_ssa_branch('else')
    
    # Call to seed(...): (line 41)
    # Processing the call arguments (line 41)
    int_38636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 23), 'int')
    # Processing the call keyword arguments (line 41)
    kwargs_38637 = {}
    # Getting the type of 'np' (line 41)
    np_38633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 41)
    random_38634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), np_38633, 'random')
    # Obtaining the member 'seed' of a type (line 41)
    seed_38635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), random_38634, 'seed')
    # Calling seed(args, kwargs) (line 41)
    seed_call_result_38638 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), seed_38635, *[int_38636], **kwargs_38637)
    
    
    # Call to astype(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'dtype' (line 42)
    dtype_38646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 46), 'dtype', False)
    # Processing the call keyword arguments (line 42)
    kwargs_38647 = {}
    
    # Call to randn(...): (line 42)
    # Getting the type of 'shape' (line 42)
    shape_38642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 32), 'shape', False)
    # Processing the call keyword arguments (line 42)
    kwargs_38643 = {}
    # Getting the type of 'np' (line 42)
    np_38639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 15), 'np', False)
    # Obtaining the member 'random' of a type (line 42)
    random_38640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 15), np_38639, 'random')
    # Obtaining the member 'randn' of a type (line 42)
    randn_38641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 15), random_38640, 'randn')
    # Calling randn(args, kwargs) (line 42)
    randn_call_result_38644 = invoke(stypy.reporting.localization.Localization(__file__, 42, 15), randn_38641, *[shape_38642], **kwargs_38643)
    
    # Obtaining the member 'astype' of a type (line 42)
    astype_38645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 15), randn_call_result_38644, 'astype')
    # Calling astype(args, kwargs) (line 42)
    astype_call_result_38648 = invoke(stypy.reporting.localization.Localization(__file__, 42, 15), astype_38645, *[dtype_38646], **kwargs_38647)
    
    # Assigning a type to the variable 'stypy_return_type' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'stypy_return_type', astype_call_result_38648)
    # SSA join for if statement (line 32)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 26)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_get_array(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_array' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_38649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38649)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_array'
    return stypy_return_type_38649

# Assigning a type to the variable '_get_array' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), '_get_array', _get_array)

@norecursion
def _id(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_id'
    module_type_store = module_type_store.open_function_context('_id', 45, 0, False)
    
    # Passed parameters checking function
    _id.stypy_localization = localization
    _id.stypy_type_of_self = None
    _id.stypy_type_store = module_type_store
    _id.stypy_function_name = '_id'
    _id.stypy_param_names_list = ['x']
    _id.stypy_varargs_param_name = None
    _id.stypy_kwargs_param_name = None
    _id.stypy_call_defaults = defaults
    _id.stypy_call_varargs = varargs
    _id.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_id', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_id', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_id(...)' code ##################

    # Getting the type of 'x' (line 46)
    x_38650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type', x_38650)
    
    # ################# End of '_id(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_id' in the type store
    # Getting the type of 'stypy_return_type' (line 45)
    stypy_return_type_38651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38651)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_id'
    return stypy_return_type_38651

# Assigning a type to the variable '_id' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), '_id', _id)

@norecursion
def assert_no_overwrite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 49)
    None_38652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 45), 'None')
    defaults = [None_38652]
    # Create a new context for function 'assert_no_overwrite'
    module_type_store = module_type_store.open_function_context('assert_no_overwrite', 49, 0, False)
    
    # Passed parameters checking function
    assert_no_overwrite.stypy_localization = localization
    assert_no_overwrite.stypy_type_of_self = None
    assert_no_overwrite.stypy_type_store = module_type_store
    assert_no_overwrite.stypy_function_name = 'assert_no_overwrite'
    assert_no_overwrite.stypy_param_names_list = ['call', 'shapes', 'dtypes']
    assert_no_overwrite.stypy_varargs_param_name = None
    assert_no_overwrite.stypy_kwargs_param_name = None
    assert_no_overwrite.stypy_call_defaults = defaults
    assert_no_overwrite.stypy_call_varargs = varargs
    assert_no_overwrite.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assert_no_overwrite', ['call', 'shapes', 'dtypes'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assert_no_overwrite', localization, ['call', 'shapes', 'dtypes'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assert_no_overwrite(...)' code ##################

    str_38653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, (-1)), 'str', '\n    Test that a call does not overwrite its input arguments\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 54)
    # Getting the type of 'dtypes' (line 54)
    dtypes_38654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 7), 'dtypes')
    # Getting the type of 'None' (line 54)
    None_38655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 17), 'None')
    
    (may_be_38656, more_types_in_union_38657) = may_be_none(dtypes_38654, None_38655)

    if may_be_38656:

        if more_types_in_union_38657:
            # Runtime conditional SSA (line 54)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a List to a Name (line 55):
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_38658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        # Adding element type (line 55)
        # Getting the type of 'np' (line 55)
        np_38659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 18), 'np')
        # Obtaining the member 'float32' of a type (line 55)
        float32_38660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 18), np_38659, 'float32')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 17), list_38658, float32_38660)
        # Adding element type (line 55)
        # Getting the type of 'np' (line 55)
        np_38661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 30), 'np')
        # Obtaining the member 'float64' of a type (line 55)
        float64_38662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 30), np_38661, 'float64')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 17), list_38658, float64_38662)
        # Adding element type (line 55)
        # Getting the type of 'np' (line 55)
        np_38663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 42), 'np')
        # Obtaining the member 'complex64' of a type (line 55)
        complex64_38664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 42), np_38663, 'complex64')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 17), list_38658, complex64_38664)
        # Adding element type (line 55)
        # Getting the type of 'np' (line 55)
        np_38665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 56), 'np')
        # Obtaining the member 'complex128' of a type (line 55)
        complex128_38666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 56), np_38665, 'complex128')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 17), list_38658, complex128_38666)
        
        # Assigning a type to the variable 'dtypes' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'dtypes', list_38658)

        if more_types_in_union_38657:
            # SSA join for if statement (line 54)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'dtypes' (line 57)
    dtypes_38667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 17), 'dtypes')
    # Testing the type of a for loop iterable (line 57)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 57, 4), dtypes_38667)
    # Getting the type of the for loop variable (line 57)
    for_loop_var_38668 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 57, 4), dtypes_38667)
    # Assigning a type to the variable 'dtype' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'dtype', for_loop_var_38668)
    # SSA begins for a for statement (line 57)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining an instance of the builtin type 'list' (line 58)
    list_38669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 58)
    # Adding element type (line 58)
    str_38670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 22), 'str', 'C')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 21), list_38669, str_38670)
    # Adding element type (line 58)
    str_38671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 27), 'str', 'F')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 21), list_38669, str_38671)
    
    # Testing the type of a for loop iterable (line 58)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 58, 8), list_38669)
    # Getting the type of the for loop variable (line 58)
    for_loop_var_38672 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 58, 8), list_38669)
    # Assigning a type to the variable 'order' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'order', for_loop_var_38672)
    # SSA begins for a for statement (line 58)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining an instance of the builtin type 'list' (line 59)
    list_38673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 59)
    # Adding element type (line 59)
    # Getting the type of '_id' (line 59)
    _id_38674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 26), '_id')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 25), list_38673, _id_38674)
    # Adding element type (line 59)
    # Getting the type of '_FakeMatrix' (line 59)
    _FakeMatrix_38675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 31), '_FakeMatrix')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 25), list_38673, _FakeMatrix_38675)
    # Adding element type (line 59)
    # Getting the type of '_FakeMatrix2' (line 59)
    _FakeMatrix2_38676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 44), '_FakeMatrix2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 25), list_38673, _FakeMatrix2_38676)
    
    # Testing the type of a for loop iterable (line 59)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 59, 12), list_38673)
    # Getting the type of the for loop variable (line 59)
    for_loop_var_38677 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 59, 12), list_38673)
    # Assigning a type to the variable 'faker' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'faker', for_loop_var_38677)
    # SSA begins for a for statement (line 59)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a ListComp to a Name (line 60):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'shapes' (line 60)
    shapes_38683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 61), 'shapes')
    comprehension_38684 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 31), shapes_38683)
    # Assigning a type to the variable 's' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 31), 's', comprehension_38684)
    
    # Call to _get_array(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 's' (line 60)
    s_38679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 42), 's', False)
    # Getting the type of 'dtype' (line 60)
    dtype_38680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 45), 'dtype', False)
    # Processing the call keyword arguments (line 60)
    kwargs_38681 = {}
    # Getting the type of '_get_array' (line 60)
    _get_array_38678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 31), '_get_array', False)
    # Calling _get_array(args, kwargs) (line 60)
    _get_array_call_result_38682 = invoke(stypy.reporting.localization.Localization(__file__, 60, 31), _get_array_38678, *[s_38679, dtype_38680], **kwargs_38681)
    
    list_38685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 31), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 31), list_38685, _get_array_call_result_38682)
    # Assigning a type to the variable 'orig_inputs' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'orig_inputs', list_38685)
    
    # Assigning a ListComp to a Name (line 61):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'orig_inputs' (line 61)
    orig_inputs_38694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 56), 'orig_inputs')
    comprehension_38695 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 26), orig_inputs_38694)
    # Assigning a type to the variable 'x' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 26), 'x', comprehension_38695)
    
    # Call to faker(...): (line 61)
    # Processing the call arguments (line 61)
    
    # Call to copy(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'order' (line 61)
    order_38689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 39), 'order', False)
    # Processing the call keyword arguments (line 61)
    kwargs_38690 = {}
    # Getting the type of 'x' (line 61)
    x_38687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 32), 'x', False)
    # Obtaining the member 'copy' of a type (line 61)
    copy_38688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 32), x_38687, 'copy')
    # Calling copy(args, kwargs) (line 61)
    copy_call_result_38691 = invoke(stypy.reporting.localization.Localization(__file__, 61, 32), copy_38688, *[order_38689], **kwargs_38690)
    
    # Processing the call keyword arguments (line 61)
    kwargs_38692 = {}
    # Getting the type of 'faker' (line 61)
    faker_38686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 26), 'faker', False)
    # Calling faker(args, kwargs) (line 61)
    faker_call_result_38693 = invoke(stypy.reporting.localization.Localization(__file__, 61, 26), faker_38686, *[copy_call_result_38691], **kwargs_38692)
    
    list_38696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 26), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 26), list_38696, faker_call_result_38693)
    # Assigning a type to the variable 'inputs' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'inputs', list_38696)
    
    # Call to call(...): (line 62)
    # Getting the type of 'inputs' (line 62)
    inputs_38698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 22), 'inputs', False)
    # Processing the call keyword arguments (line 62)
    kwargs_38699 = {}
    # Getting the type of 'call' (line 62)
    call_38697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'call', False)
    # Calling call(args, kwargs) (line 62)
    call_call_result_38700 = invoke(stypy.reporting.localization.Localization(__file__, 62, 16), call_38697, *[inputs_38698], **kwargs_38699)
    
    
    # Assigning a BinOp to a Name (line 63):
    str_38701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 22), 'str', 'call modified inputs [%r, %r]')
    
    # Obtaining an instance of the builtin type 'tuple' (line 63)
    tuple_38702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 57), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 63)
    # Adding element type (line 63)
    # Getting the type of 'dtype' (line 63)
    dtype_38703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 57), 'dtype')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 57), tuple_38702, dtype_38703)
    # Adding element type (line 63)
    # Getting the type of 'faker' (line 63)
    faker_38704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 64), 'faker')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 57), tuple_38702, faker_38704)
    
    # Applying the binary operator '%' (line 63)
    result_mod_38705 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 22), '%', str_38701, tuple_38702)
    
    # Assigning a type to the variable 'msg' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 16), 'msg', result_mod_38705)
    
    
    # Call to zip(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'inputs' (line 64)
    inputs_38707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 32), 'inputs', False)
    # Getting the type of 'orig_inputs' (line 64)
    orig_inputs_38708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 40), 'orig_inputs', False)
    # Processing the call keyword arguments (line 64)
    kwargs_38709 = {}
    # Getting the type of 'zip' (line 64)
    zip_38706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 28), 'zip', False)
    # Calling zip(args, kwargs) (line 64)
    zip_call_result_38710 = invoke(stypy.reporting.localization.Localization(__file__, 64, 28), zip_38706, *[inputs_38707, orig_inputs_38708], **kwargs_38709)
    
    # Testing the type of a for loop iterable (line 64)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 64, 16), zip_call_result_38710)
    # Getting the type of the for loop variable (line 64)
    for_loop_var_38711 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 64, 16), zip_call_result_38710)
    # Assigning a type to the variable 'a' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 16), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 16), for_loop_var_38711))
    # Assigning a type to the variable 'b' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 16), 'b', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 16), for_loop_var_38711))
    # SSA begins for a for statement (line 64)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to assert_equal(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'a' (line 65)
    a_38715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 44), 'a', False)
    # Getting the type of 'b' (line 65)
    b_38716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 47), 'b', False)
    # Processing the call keyword arguments (line 65)
    # Getting the type of 'msg' (line 65)
    msg_38717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 58), 'msg', False)
    keyword_38718 = msg_38717
    kwargs_38719 = {'err_msg': keyword_38718}
    # Getting the type of 'np' (line 65)
    np_38712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 20), 'np', False)
    # Obtaining the member 'testing' of a type (line 65)
    testing_38713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 20), np_38712, 'testing')
    # Obtaining the member 'assert_equal' of a type (line 65)
    assert_equal_38714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 20), testing_38713, 'assert_equal')
    # Calling assert_equal(args, kwargs) (line 65)
    assert_equal_call_result_38720 = invoke(stypy.reporting.localization.Localization(__file__, 65, 20), assert_equal_38714, *[a_38715, b_38716], **kwargs_38719)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'assert_no_overwrite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assert_no_overwrite' in the type store
    # Getting the type of 'stypy_return_type' (line 49)
    stypy_return_type_38721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38721)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assert_no_overwrite'
    return stypy_return_type_38721

# Assigning a type to the variable 'assert_no_overwrite' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'assert_no_overwrite', assert_no_overwrite)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

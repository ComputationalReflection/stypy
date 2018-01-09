
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import assert_equal, assert_
5: from pytest import raises as assert_raises
6: 
7: from scipy._lib._util import _aligned_zeros, check_random_state
8: 
9: 
10: def test__aligned_zeros():
11:     niter = 10
12: 
13:     def check(shape, dtype, order, align):
14:         err_msg = repr((shape, dtype, order, align))
15:         x = _aligned_zeros(shape, dtype, order, align=align)
16:         if align is None:
17:             align = np.dtype(dtype).alignment
18:         assert_equal(x.__array_interface__['data'][0] % align, 0)
19:         if hasattr(shape, '__len__'):
20:             assert_equal(x.shape, shape, err_msg)
21:         else:
22:             assert_equal(x.shape, (shape,), err_msg)
23:         assert_equal(x.dtype, dtype)
24:         if order == "C":
25:             assert_(x.flags.c_contiguous, err_msg)
26:         elif order == "F":
27:             if x.size > 0:
28:                 # Size-0 arrays get invalid flags on Numpy 1.5
29:                 assert_(x.flags.f_contiguous, err_msg)
30:         elif order is None:
31:             assert_(x.flags.c_contiguous, err_msg)
32:         else:
33:             raise ValueError()
34: 
35:     # try various alignments
36:     for align in [1, 2, 3, 4, 8, 16, 32, 64, None]:
37:         for n in [0, 1, 3, 11]:
38:             for order in ["C", "F", None]:
39:                 for dtype in [np.uint8, np.float64]:
40:                     for shape in [n, (1, 2, 3, n)]:
41:                         for j in range(niter):
42:                             check(shape, dtype, order, align)
43: 
44: 
45: def test_check_random_state():
46:     # If seed is None, return the RandomState singleton used by np.random.
47:     # If seed is an int, return a new RandomState instance seeded with seed.
48:     # If seed is already a RandomState instance, return it.
49:     # Otherwise raise ValueError.
50:     rsi = check_random_state(1)
51:     assert_equal(type(rsi), np.random.RandomState)
52:     rsi = check_random_state(rsi)
53:     assert_equal(type(rsi), np.random.RandomState)
54:     rsi = check_random_state(None)
55:     assert_equal(type(rsi), np.random.RandomState)
56:     assert_raises(ValueError, check_random_state, 'a')
57: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_712616 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_712616) is not StypyTypeError):

    if (import_712616 != 'pyd_module'):
        __import__(import_712616)
        sys_modules_712617 = sys.modules[import_712616]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_712617.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_712616)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_equal, assert_' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_712618 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_712618) is not StypyTypeError):

    if (import_712618 != 'pyd_module'):
        __import__(import_712618)
        sys_modules_712619 = sys.modules[import_712618]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_712619.module_type_store, module_type_store, ['assert_equal', 'assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_712619, sys_modules_712619.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_'], [assert_equal, assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_712618)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from pytest import assert_raises' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_712620 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest')

if (type(import_712620) is not StypyTypeError):

    if (import_712620 != 'pyd_module'):
        __import__(import_712620)
        sys_modules_712621 = sys.modules[import_712620]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', sys_modules_712621.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_712621, sys_modules_712621.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', import_712620)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy._lib._util import _aligned_zeros, check_random_state' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_712622 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._util')

if (type(import_712622) is not StypyTypeError):

    if (import_712622 != 'pyd_module'):
        __import__(import_712622)
        sys_modules_712623 = sys.modules[import_712622]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._util', sys_modules_712623.module_type_store, module_type_store, ['_aligned_zeros', 'check_random_state'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_712623, sys_modules_712623.module_type_store, module_type_store)
    else:
        from scipy._lib._util import _aligned_zeros, check_random_state

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._util', None, module_type_store, ['_aligned_zeros', 'check_random_state'], [_aligned_zeros, check_random_state])

else:
    # Assigning a type to the variable 'scipy._lib._util' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._util', import_712622)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')


@norecursion
def test__aligned_zeros(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test__aligned_zeros'
    module_type_store = module_type_store.open_function_context('test__aligned_zeros', 10, 0, False)
    
    # Passed parameters checking function
    test__aligned_zeros.stypy_localization = localization
    test__aligned_zeros.stypy_type_of_self = None
    test__aligned_zeros.stypy_type_store = module_type_store
    test__aligned_zeros.stypy_function_name = 'test__aligned_zeros'
    test__aligned_zeros.stypy_param_names_list = []
    test__aligned_zeros.stypy_varargs_param_name = None
    test__aligned_zeros.stypy_kwargs_param_name = None
    test__aligned_zeros.stypy_call_defaults = defaults
    test__aligned_zeros.stypy_call_varargs = varargs
    test__aligned_zeros.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test__aligned_zeros', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test__aligned_zeros', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test__aligned_zeros(...)' code ##################

    
    # Assigning a Num to a Name (line 11):
    int_712624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 12), 'int')
    # Assigning a type to the variable 'niter' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'niter', int_712624)

    @norecursion
    def check(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check'
        module_type_store = module_type_store.open_function_context('check', 13, 4, False)
        
        # Passed parameters checking function
        check.stypy_localization = localization
        check.stypy_type_of_self = None
        check.stypy_type_store = module_type_store
        check.stypy_function_name = 'check'
        check.stypy_param_names_list = ['shape', 'dtype', 'order', 'align']
        check.stypy_varargs_param_name = None
        check.stypy_kwargs_param_name = None
        check.stypy_call_defaults = defaults
        check.stypy_call_varargs = varargs
        check.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'check', ['shape', 'dtype', 'order', 'align'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check', localization, ['shape', 'dtype', 'order', 'align'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check(...)' code ##################

        
        # Assigning a Call to a Name (line 14):
        
        # Call to repr(...): (line 14)
        # Processing the call arguments (line 14)
        
        # Obtaining an instance of the builtin type 'tuple' (line 14)
        tuple_712626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 14)
        # Adding element type (line 14)
        # Getting the type of 'shape' (line 14)
        shape_712627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 24), 'shape', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 24), tuple_712626, shape_712627)
        # Adding element type (line 14)
        # Getting the type of 'dtype' (line 14)
        dtype_712628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 31), 'dtype', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 24), tuple_712626, dtype_712628)
        # Adding element type (line 14)
        # Getting the type of 'order' (line 14)
        order_712629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 38), 'order', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 24), tuple_712626, order_712629)
        # Adding element type (line 14)
        # Getting the type of 'align' (line 14)
        align_712630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 45), 'align', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 24), tuple_712626, align_712630)
        
        # Processing the call keyword arguments (line 14)
        kwargs_712631 = {}
        # Getting the type of 'repr' (line 14)
        repr_712625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 18), 'repr', False)
        # Calling repr(args, kwargs) (line 14)
        repr_call_result_712632 = invoke(stypy.reporting.localization.Localization(__file__, 14, 18), repr_712625, *[tuple_712626], **kwargs_712631)
        
        # Assigning a type to the variable 'err_msg' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'err_msg', repr_call_result_712632)
        
        # Assigning a Call to a Name (line 15):
        
        # Call to _aligned_zeros(...): (line 15)
        # Processing the call arguments (line 15)
        # Getting the type of 'shape' (line 15)
        shape_712634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 27), 'shape', False)
        # Getting the type of 'dtype' (line 15)
        dtype_712635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 34), 'dtype', False)
        # Getting the type of 'order' (line 15)
        order_712636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 41), 'order', False)
        # Processing the call keyword arguments (line 15)
        # Getting the type of 'align' (line 15)
        align_712637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 54), 'align', False)
        keyword_712638 = align_712637
        kwargs_712639 = {'align': keyword_712638}
        # Getting the type of '_aligned_zeros' (line 15)
        _aligned_zeros_712633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), '_aligned_zeros', False)
        # Calling _aligned_zeros(args, kwargs) (line 15)
        _aligned_zeros_call_result_712640 = invoke(stypy.reporting.localization.Localization(__file__, 15, 12), _aligned_zeros_712633, *[shape_712634, dtype_712635, order_712636], **kwargs_712639)
        
        # Assigning a type to the variable 'x' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'x', _aligned_zeros_call_result_712640)
        
        # Type idiom detected: calculating its left and rigth part (line 16)
        # Getting the type of 'align' (line 16)
        align_712641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 11), 'align')
        # Getting the type of 'None' (line 16)
        None_712642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 20), 'None')
        
        (may_be_712643, more_types_in_union_712644) = may_be_none(align_712641, None_712642)

        if may_be_712643:

            if more_types_in_union_712644:
                # Runtime conditional SSA (line 16)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 17):
            
            # Call to dtype(...): (line 17)
            # Processing the call arguments (line 17)
            # Getting the type of 'dtype' (line 17)
            dtype_712647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 29), 'dtype', False)
            # Processing the call keyword arguments (line 17)
            kwargs_712648 = {}
            # Getting the type of 'np' (line 17)
            np_712645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 20), 'np', False)
            # Obtaining the member 'dtype' of a type (line 17)
            dtype_712646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 20), np_712645, 'dtype')
            # Calling dtype(args, kwargs) (line 17)
            dtype_call_result_712649 = invoke(stypy.reporting.localization.Localization(__file__, 17, 20), dtype_712646, *[dtype_712647], **kwargs_712648)
            
            # Obtaining the member 'alignment' of a type (line 17)
            alignment_712650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 20), dtype_call_result_712649, 'alignment')
            # Assigning a type to the variable 'align' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'align', alignment_712650)

            if more_types_in_union_712644:
                # SSA join for if statement (line 16)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to assert_equal(...): (line 18)
        # Processing the call arguments (line 18)
        
        # Obtaining the type of the subscript
        int_712652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 51), 'int')
        
        # Obtaining the type of the subscript
        str_712653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 43), 'str', 'data')
        # Getting the type of 'x' (line 18)
        x_712654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 21), 'x', False)
        # Obtaining the member '__array_interface__' of a type (line 18)
        array_interface___712655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 21), x_712654, '__array_interface__')
        # Obtaining the member '__getitem__' of a type (line 18)
        getitem___712656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 21), array_interface___712655, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 18)
        subscript_call_result_712657 = invoke(stypy.reporting.localization.Localization(__file__, 18, 21), getitem___712656, str_712653)
        
        # Obtaining the member '__getitem__' of a type (line 18)
        getitem___712658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 21), subscript_call_result_712657, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 18)
        subscript_call_result_712659 = invoke(stypy.reporting.localization.Localization(__file__, 18, 21), getitem___712658, int_712652)
        
        # Getting the type of 'align' (line 18)
        align_712660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 56), 'align', False)
        # Applying the binary operator '%' (line 18)
        result_mod_712661 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 21), '%', subscript_call_result_712659, align_712660)
        
        int_712662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 63), 'int')
        # Processing the call keyword arguments (line 18)
        kwargs_712663 = {}
        # Getting the type of 'assert_equal' (line 18)
        assert_equal_712651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 18)
        assert_equal_call_result_712664 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), assert_equal_712651, *[result_mod_712661, int_712662], **kwargs_712663)
        
        
        # Type idiom detected: calculating its left and rigth part (line 19)
        str_712665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 26), 'str', '__len__')
        # Getting the type of 'shape' (line 19)
        shape_712666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 19), 'shape')
        
        (may_be_712667, more_types_in_union_712668) = may_provide_member(str_712665, shape_712666)

        if may_be_712667:

            if more_types_in_union_712668:
                # Runtime conditional SSA (line 19)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'shape' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'shape', remove_not_member_provider_from_union(shape_712666, '__len__'))
            
            # Call to assert_equal(...): (line 20)
            # Processing the call arguments (line 20)
            # Getting the type of 'x' (line 20)
            x_712670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 25), 'x', False)
            # Obtaining the member 'shape' of a type (line 20)
            shape_712671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 25), x_712670, 'shape')
            # Getting the type of 'shape' (line 20)
            shape_712672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 34), 'shape', False)
            # Getting the type of 'err_msg' (line 20)
            err_msg_712673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 41), 'err_msg', False)
            # Processing the call keyword arguments (line 20)
            kwargs_712674 = {}
            # Getting the type of 'assert_equal' (line 20)
            assert_equal_712669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'assert_equal', False)
            # Calling assert_equal(args, kwargs) (line 20)
            assert_equal_call_result_712675 = invoke(stypy.reporting.localization.Localization(__file__, 20, 12), assert_equal_712669, *[shape_712671, shape_712672, err_msg_712673], **kwargs_712674)
            

            if more_types_in_union_712668:
                # Runtime conditional SSA for else branch (line 19)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_712667) or more_types_in_union_712668):
            # Assigning a type to the variable 'shape' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'shape', remove_member_provider_from_union(shape_712666, '__len__'))
            
            # Call to assert_equal(...): (line 22)
            # Processing the call arguments (line 22)
            # Getting the type of 'x' (line 22)
            x_712677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 25), 'x', False)
            # Obtaining the member 'shape' of a type (line 22)
            shape_712678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 25), x_712677, 'shape')
            
            # Obtaining an instance of the builtin type 'tuple' (line 22)
            tuple_712679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 35), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 22)
            # Adding element type (line 22)
            # Getting the type of 'shape' (line 22)
            shape_712680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 35), 'shape', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 35), tuple_712679, shape_712680)
            
            # Getting the type of 'err_msg' (line 22)
            err_msg_712681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 44), 'err_msg', False)
            # Processing the call keyword arguments (line 22)
            kwargs_712682 = {}
            # Getting the type of 'assert_equal' (line 22)
            assert_equal_712676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'assert_equal', False)
            # Calling assert_equal(args, kwargs) (line 22)
            assert_equal_call_result_712683 = invoke(stypy.reporting.localization.Localization(__file__, 22, 12), assert_equal_712676, *[shape_712678, tuple_712679, err_msg_712681], **kwargs_712682)
            

            if (may_be_712667 and more_types_in_union_712668):
                # SSA join for if statement (line 19)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to assert_equal(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'x' (line 23)
        x_712685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 21), 'x', False)
        # Obtaining the member 'dtype' of a type (line 23)
        dtype_712686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 21), x_712685, 'dtype')
        # Getting the type of 'dtype' (line 23)
        dtype_712687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 30), 'dtype', False)
        # Processing the call keyword arguments (line 23)
        kwargs_712688 = {}
        # Getting the type of 'assert_equal' (line 23)
        assert_equal_712684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 23)
        assert_equal_call_result_712689 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), assert_equal_712684, *[dtype_712686, dtype_712687], **kwargs_712688)
        
        
        
        # Getting the type of 'order' (line 24)
        order_712690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 11), 'order')
        str_712691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 20), 'str', 'C')
        # Applying the binary operator '==' (line 24)
        result_eq_712692 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 11), '==', order_712690, str_712691)
        
        # Testing the type of an if condition (line 24)
        if_condition_712693 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 24, 8), result_eq_712692)
        # Assigning a type to the variable 'if_condition_712693' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'if_condition_712693', if_condition_712693)
        # SSA begins for if statement (line 24)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assert_(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'x' (line 25)
        x_712695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), 'x', False)
        # Obtaining the member 'flags' of a type (line 25)
        flags_712696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 20), x_712695, 'flags')
        # Obtaining the member 'c_contiguous' of a type (line 25)
        c_contiguous_712697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 20), flags_712696, 'c_contiguous')
        # Getting the type of 'err_msg' (line 25)
        err_msg_712698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 42), 'err_msg', False)
        # Processing the call keyword arguments (line 25)
        kwargs_712699 = {}
        # Getting the type of 'assert_' (line 25)
        assert__712694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 25)
        assert__call_result_712700 = invoke(stypy.reporting.localization.Localization(__file__, 25, 12), assert__712694, *[c_contiguous_712697, err_msg_712698], **kwargs_712699)
        
        # SSA branch for the else part of an if statement (line 24)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'order' (line 26)
        order_712701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 13), 'order')
        str_712702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 22), 'str', 'F')
        # Applying the binary operator '==' (line 26)
        result_eq_712703 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 13), '==', order_712701, str_712702)
        
        # Testing the type of an if condition (line 26)
        if_condition_712704 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 26, 13), result_eq_712703)
        # Assigning a type to the variable 'if_condition_712704' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 13), 'if_condition_712704', if_condition_712704)
        # SSA begins for if statement (line 26)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'x' (line 27)
        x_712705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 15), 'x')
        # Obtaining the member 'size' of a type (line 27)
        size_712706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 15), x_712705, 'size')
        int_712707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 24), 'int')
        # Applying the binary operator '>' (line 27)
        result_gt_712708 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 15), '>', size_712706, int_712707)
        
        # Testing the type of an if condition (line 27)
        if_condition_712709 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 12), result_gt_712708)
        # Assigning a type to the variable 'if_condition_712709' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'if_condition_712709', if_condition_712709)
        # SSA begins for if statement (line 27)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assert_(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'x' (line 29)
        x_712711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 24), 'x', False)
        # Obtaining the member 'flags' of a type (line 29)
        flags_712712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 24), x_712711, 'flags')
        # Obtaining the member 'f_contiguous' of a type (line 29)
        f_contiguous_712713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 24), flags_712712, 'f_contiguous')
        # Getting the type of 'err_msg' (line 29)
        err_msg_712714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 46), 'err_msg', False)
        # Processing the call keyword arguments (line 29)
        kwargs_712715 = {}
        # Getting the type of 'assert_' (line 29)
        assert__712710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 16), 'assert_', False)
        # Calling assert_(args, kwargs) (line 29)
        assert__call_result_712716 = invoke(stypy.reporting.localization.Localization(__file__, 29, 16), assert__712710, *[f_contiguous_712713, err_msg_712714], **kwargs_712715)
        
        # SSA join for if statement (line 27)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 26)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 30)
        # Getting the type of 'order' (line 30)
        order_712717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 13), 'order')
        # Getting the type of 'None' (line 30)
        None_712718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 22), 'None')
        
        (may_be_712719, more_types_in_union_712720) = may_be_none(order_712717, None_712718)

        if may_be_712719:

            if more_types_in_union_712720:
                # Runtime conditional SSA (line 30)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to assert_(...): (line 31)
            # Processing the call arguments (line 31)
            # Getting the type of 'x' (line 31)
            x_712722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'x', False)
            # Obtaining the member 'flags' of a type (line 31)
            flags_712723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 20), x_712722, 'flags')
            # Obtaining the member 'c_contiguous' of a type (line 31)
            c_contiguous_712724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 20), flags_712723, 'c_contiguous')
            # Getting the type of 'err_msg' (line 31)
            err_msg_712725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 42), 'err_msg', False)
            # Processing the call keyword arguments (line 31)
            kwargs_712726 = {}
            # Getting the type of 'assert_' (line 31)
            assert__712721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'assert_', False)
            # Calling assert_(args, kwargs) (line 31)
            assert__call_result_712727 = invoke(stypy.reporting.localization.Localization(__file__, 31, 12), assert__712721, *[c_contiguous_712724, err_msg_712725], **kwargs_712726)
            

            if more_types_in_union_712720:
                # Runtime conditional SSA for else branch (line 30)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_712719) or more_types_in_union_712720):
            
            # Call to ValueError(...): (line 33)
            # Processing the call keyword arguments (line 33)
            kwargs_712729 = {}
            # Getting the type of 'ValueError' (line 33)
            ValueError_712728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 33)
            ValueError_call_result_712730 = invoke(stypy.reporting.localization.Localization(__file__, 33, 18), ValueError_712728, *[], **kwargs_712729)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 33, 12), ValueError_call_result_712730, 'raise parameter', BaseException)

            if (may_be_712719 and more_types_in_union_712720):
                # SSA join for if statement (line 30)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 26)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 24)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check' in the type store
        # Getting the type of 'stypy_return_type' (line 13)
        stypy_return_type_712731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_712731)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check'
        return stypy_return_type_712731

    # Assigning a type to the variable 'check' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'check', check)
    
    
    # Obtaining an instance of the builtin type 'list' (line 36)
    list_712732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 36)
    # Adding element type (line 36)
    int_712733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 17), list_712732, int_712733)
    # Adding element type (line 36)
    int_712734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 17), list_712732, int_712734)
    # Adding element type (line 36)
    int_712735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 17), list_712732, int_712735)
    # Adding element type (line 36)
    int_712736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 17), list_712732, int_712736)
    # Adding element type (line 36)
    int_712737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 17), list_712732, int_712737)
    # Adding element type (line 36)
    int_712738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 17), list_712732, int_712738)
    # Adding element type (line 36)
    int_712739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 17), list_712732, int_712739)
    # Adding element type (line 36)
    int_712740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 17), list_712732, int_712740)
    # Adding element type (line 36)
    # Getting the type of 'None' (line 36)
    None_712741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 45), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 17), list_712732, None_712741)
    
    # Testing the type of a for loop iterable (line 36)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 36, 4), list_712732)
    # Getting the type of the for loop variable (line 36)
    for_loop_var_712742 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 36, 4), list_712732)
    # Assigning a type to the variable 'align' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'align', for_loop_var_712742)
    # SSA begins for a for statement (line 36)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining an instance of the builtin type 'list' (line 37)
    list_712743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 37)
    # Adding element type (line 37)
    int_712744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 17), list_712743, int_712744)
    # Adding element type (line 37)
    int_712745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 17), list_712743, int_712745)
    # Adding element type (line 37)
    int_712746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 17), list_712743, int_712746)
    # Adding element type (line 37)
    int_712747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 17), list_712743, int_712747)
    
    # Testing the type of a for loop iterable (line 37)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 37, 8), list_712743)
    # Getting the type of the for loop variable (line 37)
    for_loop_var_712748 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 37, 8), list_712743)
    # Assigning a type to the variable 'n' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'n', for_loop_var_712748)
    # SSA begins for a for statement (line 37)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining an instance of the builtin type 'list' (line 38)
    list_712749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 38)
    # Adding element type (line 38)
    str_712750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 26), 'str', 'C')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 25), list_712749, str_712750)
    # Adding element type (line 38)
    str_712751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 31), 'str', 'F')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 25), list_712749, str_712751)
    # Adding element type (line 38)
    # Getting the type of 'None' (line 38)
    None_712752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 36), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 25), list_712749, None_712752)
    
    # Testing the type of a for loop iterable (line 38)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 38, 12), list_712749)
    # Getting the type of the for loop variable (line 38)
    for_loop_var_712753 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 38, 12), list_712749)
    # Assigning a type to the variable 'order' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'order', for_loop_var_712753)
    # SSA begins for a for statement (line 38)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining an instance of the builtin type 'list' (line 39)
    list_712754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 39)
    # Adding element type (line 39)
    # Getting the type of 'np' (line 39)
    np_712755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 30), 'np')
    # Obtaining the member 'uint8' of a type (line 39)
    uint8_712756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 30), np_712755, 'uint8')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 29), list_712754, uint8_712756)
    # Adding element type (line 39)
    # Getting the type of 'np' (line 39)
    np_712757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 40), 'np')
    # Obtaining the member 'float64' of a type (line 39)
    float64_712758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 40), np_712757, 'float64')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 29), list_712754, float64_712758)
    
    # Testing the type of a for loop iterable (line 39)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 39, 16), list_712754)
    # Getting the type of the for loop variable (line 39)
    for_loop_var_712759 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 39, 16), list_712754)
    # Assigning a type to the variable 'dtype' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'dtype', for_loop_var_712759)
    # SSA begins for a for statement (line 39)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining an instance of the builtin type 'list' (line 40)
    list_712760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 40)
    # Adding element type (line 40)
    # Getting the type of 'n' (line 40)
    n_712761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 34), 'n')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 33), list_712760, n_712761)
    # Adding element type (line 40)
    
    # Obtaining an instance of the builtin type 'tuple' (line 40)
    tuple_712762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 40)
    # Adding element type (line 40)
    int_712763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 38), tuple_712762, int_712763)
    # Adding element type (line 40)
    int_712764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 38), tuple_712762, int_712764)
    # Adding element type (line 40)
    int_712765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 38), tuple_712762, int_712765)
    # Adding element type (line 40)
    # Getting the type of 'n' (line 40)
    n_712766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 47), 'n')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 38), tuple_712762, n_712766)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 33), list_712760, tuple_712762)
    
    # Testing the type of a for loop iterable (line 40)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 40, 20), list_712760)
    # Getting the type of the for loop variable (line 40)
    for_loop_var_712767 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 40, 20), list_712760)
    # Assigning a type to the variable 'shape' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 20), 'shape', for_loop_var_712767)
    # SSA begins for a for statement (line 40)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to range(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'niter' (line 41)
    niter_712769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 39), 'niter', False)
    # Processing the call keyword arguments (line 41)
    kwargs_712770 = {}
    # Getting the type of 'range' (line 41)
    range_712768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 33), 'range', False)
    # Calling range(args, kwargs) (line 41)
    range_call_result_712771 = invoke(stypy.reporting.localization.Localization(__file__, 41, 33), range_712768, *[niter_712769], **kwargs_712770)
    
    # Testing the type of a for loop iterable (line 41)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 41, 24), range_call_result_712771)
    # Getting the type of the for loop variable (line 41)
    for_loop_var_712772 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 41, 24), range_call_result_712771)
    # Assigning a type to the variable 'j' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 24), 'j', for_loop_var_712772)
    # SSA begins for a for statement (line 41)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to check(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'shape' (line 42)
    shape_712774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 34), 'shape', False)
    # Getting the type of 'dtype' (line 42)
    dtype_712775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 41), 'dtype', False)
    # Getting the type of 'order' (line 42)
    order_712776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 48), 'order', False)
    # Getting the type of 'align' (line 42)
    align_712777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 55), 'align', False)
    # Processing the call keyword arguments (line 42)
    kwargs_712778 = {}
    # Getting the type of 'check' (line 42)
    check_712773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 28), 'check', False)
    # Calling check(args, kwargs) (line 42)
    check_call_result_712779 = invoke(stypy.reporting.localization.Localization(__file__, 42, 28), check_712773, *[shape_712774, dtype_712775, order_712776, align_712777], **kwargs_712778)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test__aligned_zeros(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test__aligned_zeros' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_712780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_712780)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test__aligned_zeros'
    return stypy_return_type_712780

# Assigning a type to the variable 'test__aligned_zeros' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'test__aligned_zeros', test__aligned_zeros)

@norecursion
def test_check_random_state(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_check_random_state'
    module_type_store = module_type_store.open_function_context('test_check_random_state', 45, 0, False)
    
    # Passed parameters checking function
    test_check_random_state.stypy_localization = localization
    test_check_random_state.stypy_type_of_self = None
    test_check_random_state.stypy_type_store = module_type_store
    test_check_random_state.stypy_function_name = 'test_check_random_state'
    test_check_random_state.stypy_param_names_list = []
    test_check_random_state.stypy_varargs_param_name = None
    test_check_random_state.stypy_kwargs_param_name = None
    test_check_random_state.stypy_call_defaults = defaults
    test_check_random_state.stypy_call_varargs = varargs
    test_check_random_state.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_check_random_state', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_check_random_state', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_check_random_state(...)' code ##################

    
    # Assigning a Call to a Name (line 50):
    
    # Call to check_random_state(...): (line 50)
    # Processing the call arguments (line 50)
    int_712782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 29), 'int')
    # Processing the call keyword arguments (line 50)
    kwargs_712783 = {}
    # Getting the type of 'check_random_state' (line 50)
    check_random_state_712781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 10), 'check_random_state', False)
    # Calling check_random_state(args, kwargs) (line 50)
    check_random_state_call_result_712784 = invoke(stypy.reporting.localization.Localization(__file__, 50, 10), check_random_state_712781, *[int_712782], **kwargs_712783)
    
    # Assigning a type to the variable 'rsi' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'rsi', check_random_state_call_result_712784)
    
    # Call to assert_equal(...): (line 51)
    # Processing the call arguments (line 51)
    
    # Call to type(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'rsi' (line 51)
    rsi_712787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 22), 'rsi', False)
    # Processing the call keyword arguments (line 51)
    kwargs_712788 = {}
    # Getting the type of 'type' (line 51)
    type_712786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'type', False)
    # Calling type(args, kwargs) (line 51)
    type_call_result_712789 = invoke(stypy.reporting.localization.Localization(__file__, 51, 17), type_712786, *[rsi_712787], **kwargs_712788)
    
    # Getting the type of 'np' (line 51)
    np_712790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 28), 'np', False)
    # Obtaining the member 'random' of a type (line 51)
    random_712791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 28), np_712790, 'random')
    # Obtaining the member 'RandomState' of a type (line 51)
    RandomState_712792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 28), random_712791, 'RandomState')
    # Processing the call keyword arguments (line 51)
    kwargs_712793 = {}
    # Getting the type of 'assert_equal' (line 51)
    assert_equal_712785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 51)
    assert_equal_call_result_712794 = invoke(stypy.reporting.localization.Localization(__file__, 51, 4), assert_equal_712785, *[type_call_result_712789, RandomState_712792], **kwargs_712793)
    
    
    # Assigning a Call to a Name (line 52):
    
    # Call to check_random_state(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'rsi' (line 52)
    rsi_712796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 29), 'rsi', False)
    # Processing the call keyword arguments (line 52)
    kwargs_712797 = {}
    # Getting the type of 'check_random_state' (line 52)
    check_random_state_712795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 10), 'check_random_state', False)
    # Calling check_random_state(args, kwargs) (line 52)
    check_random_state_call_result_712798 = invoke(stypy.reporting.localization.Localization(__file__, 52, 10), check_random_state_712795, *[rsi_712796], **kwargs_712797)
    
    # Assigning a type to the variable 'rsi' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'rsi', check_random_state_call_result_712798)
    
    # Call to assert_equal(...): (line 53)
    # Processing the call arguments (line 53)
    
    # Call to type(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'rsi' (line 53)
    rsi_712801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 22), 'rsi', False)
    # Processing the call keyword arguments (line 53)
    kwargs_712802 = {}
    # Getting the type of 'type' (line 53)
    type_712800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 17), 'type', False)
    # Calling type(args, kwargs) (line 53)
    type_call_result_712803 = invoke(stypy.reporting.localization.Localization(__file__, 53, 17), type_712800, *[rsi_712801], **kwargs_712802)
    
    # Getting the type of 'np' (line 53)
    np_712804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 28), 'np', False)
    # Obtaining the member 'random' of a type (line 53)
    random_712805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 28), np_712804, 'random')
    # Obtaining the member 'RandomState' of a type (line 53)
    RandomState_712806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 28), random_712805, 'RandomState')
    # Processing the call keyword arguments (line 53)
    kwargs_712807 = {}
    # Getting the type of 'assert_equal' (line 53)
    assert_equal_712799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 53)
    assert_equal_call_result_712808 = invoke(stypy.reporting.localization.Localization(__file__, 53, 4), assert_equal_712799, *[type_call_result_712803, RandomState_712806], **kwargs_712807)
    
    
    # Assigning a Call to a Name (line 54):
    
    # Call to check_random_state(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'None' (line 54)
    None_712810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 29), 'None', False)
    # Processing the call keyword arguments (line 54)
    kwargs_712811 = {}
    # Getting the type of 'check_random_state' (line 54)
    check_random_state_712809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 10), 'check_random_state', False)
    # Calling check_random_state(args, kwargs) (line 54)
    check_random_state_call_result_712812 = invoke(stypy.reporting.localization.Localization(__file__, 54, 10), check_random_state_712809, *[None_712810], **kwargs_712811)
    
    # Assigning a type to the variable 'rsi' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'rsi', check_random_state_call_result_712812)
    
    # Call to assert_equal(...): (line 55)
    # Processing the call arguments (line 55)
    
    # Call to type(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'rsi' (line 55)
    rsi_712815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 22), 'rsi', False)
    # Processing the call keyword arguments (line 55)
    kwargs_712816 = {}
    # Getting the type of 'type' (line 55)
    type_712814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 17), 'type', False)
    # Calling type(args, kwargs) (line 55)
    type_call_result_712817 = invoke(stypy.reporting.localization.Localization(__file__, 55, 17), type_712814, *[rsi_712815], **kwargs_712816)
    
    # Getting the type of 'np' (line 55)
    np_712818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 28), 'np', False)
    # Obtaining the member 'random' of a type (line 55)
    random_712819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 28), np_712818, 'random')
    # Obtaining the member 'RandomState' of a type (line 55)
    RandomState_712820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 28), random_712819, 'RandomState')
    # Processing the call keyword arguments (line 55)
    kwargs_712821 = {}
    # Getting the type of 'assert_equal' (line 55)
    assert_equal_712813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 55)
    assert_equal_call_result_712822 = invoke(stypy.reporting.localization.Localization(__file__, 55, 4), assert_equal_712813, *[type_call_result_712817, RandomState_712820], **kwargs_712821)
    
    
    # Call to assert_raises(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'ValueError' (line 56)
    ValueError_712824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 18), 'ValueError', False)
    # Getting the type of 'check_random_state' (line 56)
    check_random_state_712825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 30), 'check_random_state', False)
    str_712826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 50), 'str', 'a')
    # Processing the call keyword arguments (line 56)
    kwargs_712827 = {}
    # Getting the type of 'assert_raises' (line 56)
    assert_raises_712823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 56)
    assert_raises_call_result_712828 = invoke(stypy.reporting.localization.Localization(__file__, 56, 4), assert_raises_712823, *[ValueError_712824, check_random_state_712825, str_712826], **kwargs_712827)
    
    
    # ################# End of 'test_check_random_state(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_check_random_state' in the type store
    # Getting the type of 'stypy_return_type' (line 45)
    stypy_return_type_712829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_712829)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_check_random_state'
    return stypy_return_type_712829

# Assigning a type to the variable 'test_check_random_state' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'test_check_random_state', test_check_random_state)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import assert_allclose, assert_array_equal
5: from pytest import raises as assert_raises
6: 
7: from numpy.fft import fft, ifft
8: 
9: from scipy.signal import max_len_seq
10: 
11: 
12: class TestMLS(object):
13: 
14:     def test_mls_inputs(self):
15:         # can't all be zero state
16:         assert_raises(ValueError, max_len_seq,
17:                       10, state=np.zeros(10))
18:         # wrong size state
19:         assert_raises(ValueError, max_len_seq, 10,
20:                       state=np.ones(3))
21:         # wrong length
22:         assert_raises(ValueError, max_len_seq, 10, length=-1)
23:         assert_array_equal(max_len_seq(10, length=0)[0], [])
24:         # unknown taps
25:         assert_raises(ValueError, max_len_seq, 64)
26:         # bad taps
27:         assert_raises(ValueError, max_len_seq, 10, taps=[-1, 1])
28: 
29:     def test_mls_output(self):
30:         # define some alternate working taps
31:         alt_taps = {2: [1], 3: [2], 4: [3], 5: [4, 3, 2], 6: [5, 4, 1], 7: [4],
32:                     8: [7, 5, 3]}
33:         # assume the other bit levels work, too slow to test higher orders...
34:         for nbits in range(2, 8):
35:             for state in [None, np.round(np.random.rand(nbits))]:
36:                 for taps in [None, alt_taps[nbits]]:
37:                     if state is not None and np.all(state == 0):
38:                         state[0] = 1  # they can't all be zero
39:                     orig_m = max_len_seq(nbits, state=state,
40:                                          taps=taps)[0]
41:                     m = 2. * orig_m - 1.  # convert to +/- 1 representation
42:                     # First, make sure we got all 1's or -1
43:                     err_msg = "mls had non binary terms"
44:                     assert_array_equal(np.abs(m), np.ones_like(m),
45:                                        err_msg=err_msg)
46:                     # Test via circular cross-correlation, which is just mult.
47:                     # in the frequency domain with one signal conjugated
48:                     tester = np.real(ifft(fft(m) * np.conj(fft(m))))
49:                     out_len = 2**nbits - 1
50:                     # impulse amplitude == test_len
51:                     err_msg = "mls impulse has incorrect value"
52:                     assert_allclose(tester[0], out_len, err_msg=err_msg)
53:                     # steady-state is -1
54:                     err_msg = "mls steady-state has incorrect value"
55:                     assert_allclose(tester[1:], -1 * np.ones(out_len - 1),
56:                                     err_msg=err_msg)
57:                     # let's do the split thing using a couple options
58:                     for n in (1, 2**(nbits - 1)):
59:                         m1, s1 = max_len_seq(nbits, state=state, taps=taps,
60:                                              length=n)
61:                         m2, s2 = max_len_seq(nbits, state=s1, taps=taps,
62:                                              length=1)
63:                         m3, s3 = max_len_seq(nbits, state=s2, taps=taps,
64:                                              length=out_len - n - 1)
65:                         new_m = np.concatenate((m1, m2, m3))
66:                         assert_array_equal(orig_m, new_m)
67: 
68: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_324610 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_324610) is not StypyTypeError):

    if (import_324610 != 'pyd_module'):
        __import__(import_324610)
        sys_modules_324611 = sys.modules[import_324610]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_324611.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_324610)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_allclose, assert_array_equal' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_324612 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_324612) is not StypyTypeError):

    if (import_324612 != 'pyd_module'):
        __import__(import_324612)
        sys_modules_324613 = sys.modules[import_324612]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_324613.module_type_store, module_type_store, ['assert_allclose', 'assert_array_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_324613, sys_modules_324613.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_allclose, assert_array_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_allclose', 'assert_array_equal'], [assert_allclose, assert_array_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_324612)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from pytest import assert_raises' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_324614 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest')

if (type(import_324614) is not StypyTypeError):

    if (import_324614 != 'pyd_module'):
        __import__(import_324614)
        sys_modules_324615 = sys.modules[import_324614]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', sys_modules_324615.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_324615, sys_modules_324615.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', import_324614)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.fft import fft, ifft' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_324616 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.fft')

if (type(import_324616) is not StypyTypeError):

    if (import_324616 != 'pyd_module'):
        __import__(import_324616)
        sys_modules_324617 = sys.modules[import_324616]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.fft', sys_modules_324617.module_type_store, module_type_store, ['fft', 'ifft'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_324617, sys_modules_324617.module_type_store, module_type_store)
    else:
        from numpy.fft import fft, ifft

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.fft', None, module_type_store, ['fft', 'ifft'], [fft, ifft])

else:
    # Assigning a type to the variable 'numpy.fft' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.fft', import_324616)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.signal import max_len_seq' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_324618 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.signal')

if (type(import_324618) is not StypyTypeError):

    if (import_324618 != 'pyd_module'):
        __import__(import_324618)
        sys_modules_324619 = sys.modules[import_324618]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.signal', sys_modules_324619.module_type_store, module_type_store, ['max_len_seq'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_324619, sys_modules_324619.module_type_store, module_type_store)
    else:
        from scipy.signal import max_len_seq

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.signal', None, module_type_store, ['max_len_seq'], [max_len_seq])

else:
    # Assigning a type to the variable 'scipy.signal' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.signal', import_324618)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

# Declaration of the 'TestMLS' class

class TestMLS(object, ):

    @norecursion
    def test_mls_inputs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_mls_inputs'
        module_type_store = module_type_store.open_function_context('test_mls_inputs', 14, 4, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMLS.test_mls_inputs.__dict__.__setitem__('stypy_localization', localization)
        TestMLS.test_mls_inputs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMLS.test_mls_inputs.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMLS.test_mls_inputs.__dict__.__setitem__('stypy_function_name', 'TestMLS.test_mls_inputs')
        TestMLS.test_mls_inputs.__dict__.__setitem__('stypy_param_names_list', [])
        TestMLS.test_mls_inputs.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMLS.test_mls_inputs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMLS.test_mls_inputs.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMLS.test_mls_inputs.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMLS.test_mls_inputs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMLS.test_mls_inputs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMLS.test_mls_inputs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_mls_inputs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_mls_inputs(...)' code ##################

        
        # Call to assert_raises(...): (line 16)
        # Processing the call arguments (line 16)
        # Getting the type of 'ValueError' (line 16)
        ValueError_324621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 22), 'ValueError', False)
        # Getting the type of 'max_len_seq' (line 16)
        max_len_seq_324622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 34), 'max_len_seq', False)
        int_324623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 22), 'int')
        # Processing the call keyword arguments (line 16)
        
        # Call to zeros(...): (line 17)
        # Processing the call arguments (line 17)
        int_324626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 41), 'int')
        # Processing the call keyword arguments (line 17)
        kwargs_324627 = {}
        # Getting the type of 'np' (line 17)
        np_324624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 32), 'np', False)
        # Obtaining the member 'zeros' of a type (line 17)
        zeros_324625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 32), np_324624, 'zeros')
        # Calling zeros(args, kwargs) (line 17)
        zeros_call_result_324628 = invoke(stypy.reporting.localization.Localization(__file__, 17, 32), zeros_324625, *[int_324626], **kwargs_324627)
        
        keyword_324629 = zeros_call_result_324628
        kwargs_324630 = {'state': keyword_324629}
        # Getting the type of 'assert_raises' (line 16)
        assert_raises_324620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 16)
        assert_raises_call_result_324631 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), assert_raises_324620, *[ValueError_324621, max_len_seq_324622, int_324623], **kwargs_324630)
        
        
        # Call to assert_raises(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'ValueError' (line 19)
        ValueError_324633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 22), 'ValueError', False)
        # Getting the type of 'max_len_seq' (line 19)
        max_len_seq_324634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 34), 'max_len_seq', False)
        int_324635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 47), 'int')
        # Processing the call keyword arguments (line 19)
        
        # Call to ones(...): (line 20)
        # Processing the call arguments (line 20)
        int_324638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 36), 'int')
        # Processing the call keyword arguments (line 20)
        kwargs_324639 = {}
        # Getting the type of 'np' (line 20)
        np_324636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 28), 'np', False)
        # Obtaining the member 'ones' of a type (line 20)
        ones_324637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 28), np_324636, 'ones')
        # Calling ones(args, kwargs) (line 20)
        ones_call_result_324640 = invoke(stypy.reporting.localization.Localization(__file__, 20, 28), ones_324637, *[int_324638], **kwargs_324639)
        
        keyword_324641 = ones_call_result_324640
        kwargs_324642 = {'state': keyword_324641}
        # Getting the type of 'assert_raises' (line 19)
        assert_raises_324632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 19)
        assert_raises_call_result_324643 = invoke(stypy.reporting.localization.Localization(__file__, 19, 8), assert_raises_324632, *[ValueError_324633, max_len_seq_324634, int_324635], **kwargs_324642)
        
        
        # Call to assert_raises(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'ValueError' (line 22)
        ValueError_324645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 22), 'ValueError', False)
        # Getting the type of 'max_len_seq' (line 22)
        max_len_seq_324646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 34), 'max_len_seq', False)
        int_324647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 47), 'int')
        # Processing the call keyword arguments (line 22)
        int_324648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 58), 'int')
        keyword_324649 = int_324648
        kwargs_324650 = {'length': keyword_324649}
        # Getting the type of 'assert_raises' (line 22)
        assert_raises_324644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 22)
        assert_raises_call_result_324651 = invoke(stypy.reporting.localization.Localization(__file__, 22, 8), assert_raises_324644, *[ValueError_324645, max_len_seq_324646, int_324647], **kwargs_324650)
        
        
        # Call to assert_array_equal(...): (line 23)
        # Processing the call arguments (line 23)
        
        # Obtaining the type of the subscript
        int_324653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 53), 'int')
        
        # Call to max_len_seq(...): (line 23)
        # Processing the call arguments (line 23)
        int_324655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 39), 'int')
        # Processing the call keyword arguments (line 23)
        int_324656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 50), 'int')
        keyword_324657 = int_324656
        kwargs_324658 = {'length': keyword_324657}
        # Getting the type of 'max_len_seq' (line 23)
        max_len_seq_324654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 27), 'max_len_seq', False)
        # Calling max_len_seq(args, kwargs) (line 23)
        max_len_seq_call_result_324659 = invoke(stypy.reporting.localization.Localization(__file__, 23, 27), max_len_seq_324654, *[int_324655], **kwargs_324658)
        
        # Obtaining the member '__getitem__' of a type (line 23)
        getitem___324660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 27), max_len_seq_call_result_324659, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 23)
        subscript_call_result_324661 = invoke(stypy.reporting.localization.Localization(__file__, 23, 27), getitem___324660, int_324653)
        
        
        # Obtaining an instance of the builtin type 'list' (line 23)
        list_324662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 23)
        
        # Processing the call keyword arguments (line 23)
        kwargs_324663 = {}
        # Getting the type of 'assert_array_equal' (line 23)
        assert_array_equal_324652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 23)
        assert_array_equal_call_result_324664 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), assert_array_equal_324652, *[subscript_call_result_324661, list_324662], **kwargs_324663)
        
        
        # Call to assert_raises(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'ValueError' (line 25)
        ValueError_324666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 22), 'ValueError', False)
        # Getting the type of 'max_len_seq' (line 25)
        max_len_seq_324667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 34), 'max_len_seq', False)
        int_324668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 47), 'int')
        # Processing the call keyword arguments (line 25)
        kwargs_324669 = {}
        # Getting the type of 'assert_raises' (line 25)
        assert_raises_324665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 25)
        assert_raises_call_result_324670 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), assert_raises_324665, *[ValueError_324666, max_len_seq_324667, int_324668], **kwargs_324669)
        
        
        # Call to assert_raises(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'ValueError' (line 27)
        ValueError_324672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 22), 'ValueError', False)
        # Getting the type of 'max_len_seq' (line 27)
        max_len_seq_324673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 34), 'max_len_seq', False)
        int_324674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 47), 'int')
        # Processing the call keyword arguments (line 27)
        
        # Obtaining an instance of the builtin type 'list' (line 27)
        list_324675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 27)
        # Adding element type (line 27)
        int_324676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 56), list_324675, int_324676)
        # Adding element type (line 27)
        int_324677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 56), list_324675, int_324677)
        
        keyword_324678 = list_324675
        kwargs_324679 = {'taps': keyword_324678}
        # Getting the type of 'assert_raises' (line 27)
        assert_raises_324671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 27)
        assert_raises_call_result_324680 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), assert_raises_324671, *[ValueError_324672, max_len_seq_324673, int_324674], **kwargs_324679)
        
        
        # ################# End of 'test_mls_inputs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_mls_inputs' in the type store
        # Getting the type of 'stypy_return_type' (line 14)
        stypy_return_type_324681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_324681)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_mls_inputs'
        return stypy_return_type_324681


    @norecursion
    def test_mls_output(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_mls_output'
        module_type_store = module_type_store.open_function_context('test_mls_output', 29, 4, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMLS.test_mls_output.__dict__.__setitem__('stypy_localization', localization)
        TestMLS.test_mls_output.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMLS.test_mls_output.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMLS.test_mls_output.__dict__.__setitem__('stypy_function_name', 'TestMLS.test_mls_output')
        TestMLS.test_mls_output.__dict__.__setitem__('stypy_param_names_list', [])
        TestMLS.test_mls_output.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMLS.test_mls_output.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMLS.test_mls_output.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMLS.test_mls_output.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMLS.test_mls_output.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMLS.test_mls_output.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMLS.test_mls_output', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_mls_output', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_mls_output(...)' code ##################

        
        # Assigning a Dict to a Name (line 31):
        
        # Assigning a Dict to a Name (line 31):
        
        # Obtaining an instance of the builtin type 'dict' (line 31)
        dict_324682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 19), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 31)
        # Adding element type (key, value) (line 31)
        int_324683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 20), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 31)
        list_324684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 31)
        # Adding element type (line 31)
        int_324685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 23), list_324684, int_324685)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), dict_324682, (int_324683, list_324684))
        # Adding element type (key, value) (line 31)
        int_324686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 28), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 31)
        list_324687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 31)
        # Adding element type (line 31)
        int_324688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 31), list_324687, int_324688)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), dict_324682, (int_324686, list_324687))
        # Adding element type (key, value) (line 31)
        int_324689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 36), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 31)
        list_324690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 31)
        # Adding element type (line 31)
        int_324691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 39), list_324690, int_324691)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), dict_324682, (int_324689, list_324690))
        # Adding element type (key, value) (line 31)
        int_324692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 44), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 31)
        list_324693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 31)
        # Adding element type (line 31)
        int_324694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 47), list_324693, int_324694)
        # Adding element type (line 31)
        int_324695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 47), list_324693, int_324695)
        # Adding element type (line 31)
        int_324696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 47), list_324693, int_324696)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), dict_324682, (int_324692, list_324693))
        # Adding element type (key, value) (line 31)
        int_324697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 58), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 31)
        list_324698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 61), 'list')
        # Adding type elements to the builtin type 'list' instance (line 31)
        # Adding element type (line 31)
        int_324699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 61), list_324698, int_324699)
        # Adding element type (line 31)
        int_324700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 61), list_324698, int_324700)
        # Adding element type (line 31)
        int_324701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 68), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 61), list_324698, int_324701)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), dict_324682, (int_324697, list_324698))
        # Adding element type (key, value) (line 31)
        int_324702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 72), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 31)
        list_324703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 75), 'list')
        # Adding type elements to the builtin type 'list' instance (line 31)
        # Adding element type (line 31)
        int_324704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 76), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 75), list_324703, int_324704)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), dict_324682, (int_324702, list_324703))
        # Adding element type (key, value) (line 31)
        int_324705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 20), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 32)
        list_324706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 32)
        # Adding element type (line 32)
        int_324707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 23), list_324706, int_324707)
        # Adding element type (line 32)
        int_324708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 23), list_324706, int_324708)
        # Adding element type (line 32)
        int_324709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 23), list_324706, int_324709)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), dict_324682, (int_324705, list_324706))
        
        # Assigning a type to the variable 'alt_taps' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'alt_taps', dict_324682)
        
        
        # Call to range(...): (line 34)
        # Processing the call arguments (line 34)
        int_324711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 27), 'int')
        int_324712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 30), 'int')
        # Processing the call keyword arguments (line 34)
        kwargs_324713 = {}
        # Getting the type of 'range' (line 34)
        range_324710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 21), 'range', False)
        # Calling range(args, kwargs) (line 34)
        range_call_result_324714 = invoke(stypy.reporting.localization.Localization(__file__, 34, 21), range_324710, *[int_324711, int_324712], **kwargs_324713)
        
        # Testing the type of a for loop iterable (line 34)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 34, 8), range_call_result_324714)
        # Getting the type of the for loop variable (line 34)
        for_loop_var_324715 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 34, 8), range_call_result_324714)
        # Assigning a type to the variable 'nbits' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'nbits', for_loop_var_324715)
        # SSA begins for a for statement (line 34)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'list' (line 35)
        list_324716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 35)
        # Adding element type (line 35)
        # Getting the type of 'None' (line 35)
        None_324717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 26), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 25), list_324716, None_324717)
        # Adding element type (line 35)
        
        # Call to round(...): (line 35)
        # Processing the call arguments (line 35)
        
        # Call to rand(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'nbits' (line 35)
        nbits_324723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 56), 'nbits', False)
        # Processing the call keyword arguments (line 35)
        kwargs_324724 = {}
        # Getting the type of 'np' (line 35)
        np_324720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 41), 'np', False)
        # Obtaining the member 'random' of a type (line 35)
        random_324721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 41), np_324720, 'random')
        # Obtaining the member 'rand' of a type (line 35)
        rand_324722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 41), random_324721, 'rand')
        # Calling rand(args, kwargs) (line 35)
        rand_call_result_324725 = invoke(stypy.reporting.localization.Localization(__file__, 35, 41), rand_324722, *[nbits_324723], **kwargs_324724)
        
        # Processing the call keyword arguments (line 35)
        kwargs_324726 = {}
        # Getting the type of 'np' (line 35)
        np_324718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 32), 'np', False)
        # Obtaining the member 'round' of a type (line 35)
        round_324719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 32), np_324718, 'round')
        # Calling round(args, kwargs) (line 35)
        round_call_result_324727 = invoke(stypy.reporting.localization.Localization(__file__, 35, 32), round_324719, *[rand_call_result_324725], **kwargs_324726)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 25), list_324716, round_call_result_324727)
        
        # Testing the type of a for loop iterable (line 35)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 35, 12), list_324716)
        # Getting the type of the for loop variable (line 35)
        for_loop_var_324728 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 35, 12), list_324716)
        # Assigning a type to the variable 'state' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'state', for_loop_var_324728)
        # SSA begins for a for statement (line 35)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'list' (line 36)
        list_324729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 36)
        # Adding element type (line 36)
        # Getting the type of 'None' (line 36)
        None_324730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 29), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 28), list_324729, None_324730)
        # Adding element type (line 36)
        
        # Obtaining the type of the subscript
        # Getting the type of 'nbits' (line 36)
        nbits_324731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 44), 'nbits')
        # Getting the type of 'alt_taps' (line 36)
        alt_taps_324732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 35), 'alt_taps')
        # Obtaining the member '__getitem__' of a type (line 36)
        getitem___324733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 35), alt_taps_324732, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 36)
        subscript_call_result_324734 = invoke(stypy.reporting.localization.Localization(__file__, 36, 35), getitem___324733, nbits_324731)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 28), list_324729, subscript_call_result_324734)
        
        # Testing the type of a for loop iterable (line 36)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 36, 16), list_324729)
        # Getting the type of the for loop variable (line 36)
        for_loop_var_324735 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 36, 16), list_324729)
        # Assigning a type to the variable 'taps' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'taps', for_loop_var_324735)
        # SSA begins for a for statement (line 36)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'state' (line 37)
        state_324736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 23), 'state')
        # Getting the type of 'None' (line 37)
        None_324737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 36), 'None')
        # Applying the binary operator 'isnot' (line 37)
        result_is_not_324738 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 23), 'isnot', state_324736, None_324737)
        
        
        # Call to all(...): (line 37)
        # Processing the call arguments (line 37)
        
        # Getting the type of 'state' (line 37)
        state_324741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 52), 'state', False)
        int_324742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 61), 'int')
        # Applying the binary operator '==' (line 37)
        result_eq_324743 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 52), '==', state_324741, int_324742)
        
        # Processing the call keyword arguments (line 37)
        kwargs_324744 = {}
        # Getting the type of 'np' (line 37)
        np_324739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 45), 'np', False)
        # Obtaining the member 'all' of a type (line 37)
        all_324740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 45), np_324739, 'all')
        # Calling all(args, kwargs) (line 37)
        all_call_result_324745 = invoke(stypy.reporting.localization.Localization(__file__, 37, 45), all_324740, *[result_eq_324743], **kwargs_324744)
        
        # Applying the binary operator 'and' (line 37)
        result_and_keyword_324746 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 23), 'and', result_is_not_324738, all_call_result_324745)
        
        # Testing the type of an if condition (line 37)
        if_condition_324747 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 37, 20), result_and_keyword_324746)
        # Assigning a type to the variable 'if_condition_324747' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 20), 'if_condition_324747', if_condition_324747)
        # SSA begins for if statement (line 37)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Subscript (line 38):
        
        # Assigning a Num to a Subscript (line 38):
        int_324748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 35), 'int')
        # Getting the type of 'state' (line 38)
        state_324749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 24), 'state')
        int_324750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 30), 'int')
        # Storing an element on a container (line 38)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 24), state_324749, (int_324750, int_324748))
        # SSA join for if statement (line 37)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 39):
        
        # Assigning a Subscript to a Name (line 39):
        
        # Obtaining the type of the subscript
        int_324751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 52), 'int')
        
        # Call to max_len_seq(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'nbits' (line 39)
        nbits_324753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 41), 'nbits', False)
        # Processing the call keyword arguments (line 39)
        # Getting the type of 'state' (line 39)
        state_324754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 54), 'state', False)
        keyword_324755 = state_324754
        # Getting the type of 'taps' (line 40)
        taps_324756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 46), 'taps', False)
        keyword_324757 = taps_324756
        kwargs_324758 = {'state': keyword_324755, 'taps': keyword_324757}
        # Getting the type of 'max_len_seq' (line 39)
        max_len_seq_324752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 29), 'max_len_seq', False)
        # Calling max_len_seq(args, kwargs) (line 39)
        max_len_seq_call_result_324759 = invoke(stypy.reporting.localization.Localization(__file__, 39, 29), max_len_seq_324752, *[nbits_324753], **kwargs_324758)
        
        # Obtaining the member '__getitem__' of a type (line 39)
        getitem___324760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 29), max_len_seq_call_result_324759, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 39)
        subscript_call_result_324761 = invoke(stypy.reporting.localization.Localization(__file__, 39, 29), getitem___324760, int_324751)
        
        # Assigning a type to the variable 'orig_m' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 'orig_m', subscript_call_result_324761)
        
        # Assigning a BinOp to a Name (line 41):
        
        # Assigning a BinOp to a Name (line 41):
        float_324762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 24), 'float')
        # Getting the type of 'orig_m' (line 41)
        orig_m_324763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 29), 'orig_m')
        # Applying the binary operator '*' (line 41)
        result_mul_324764 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 24), '*', float_324762, orig_m_324763)
        
        float_324765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 38), 'float')
        # Applying the binary operator '-' (line 41)
        result_sub_324766 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 24), '-', result_mul_324764, float_324765)
        
        # Assigning a type to the variable 'm' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 20), 'm', result_sub_324766)
        
        # Assigning a Str to a Name (line 43):
        
        # Assigning a Str to a Name (line 43):
        str_324767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 30), 'str', 'mls had non binary terms')
        # Assigning a type to the variable 'err_msg' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 20), 'err_msg', str_324767)
        
        # Call to assert_array_equal(...): (line 44)
        # Processing the call arguments (line 44)
        
        # Call to abs(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'm' (line 44)
        m_324771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 46), 'm', False)
        # Processing the call keyword arguments (line 44)
        kwargs_324772 = {}
        # Getting the type of 'np' (line 44)
        np_324769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 39), 'np', False)
        # Obtaining the member 'abs' of a type (line 44)
        abs_324770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 39), np_324769, 'abs')
        # Calling abs(args, kwargs) (line 44)
        abs_call_result_324773 = invoke(stypy.reporting.localization.Localization(__file__, 44, 39), abs_324770, *[m_324771], **kwargs_324772)
        
        
        # Call to ones_like(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'm' (line 44)
        m_324776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 63), 'm', False)
        # Processing the call keyword arguments (line 44)
        kwargs_324777 = {}
        # Getting the type of 'np' (line 44)
        np_324774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 50), 'np', False)
        # Obtaining the member 'ones_like' of a type (line 44)
        ones_like_324775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 50), np_324774, 'ones_like')
        # Calling ones_like(args, kwargs) (line 44)
        ones_like_call_result_324778 = invoke(stypy.reporting.localization.Localization(__file__, 44, 50), ones_like_324775, *[m_324776], **kwargs_324777)
        
        # Processing the call keyword arguments (line 44)
        # Getting the type of 'err_msg' (line 45)
        err_msg_324779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 47), 'err_msg', False)
        keyword_324780 = err_msg_324779
        kwargs_324781 = {'err_msg': keyword_324780}
        # Getting the type of 'assert_array_equal' (line 44)
        assert_array_equal_324768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 20), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 44)
        assert_array_equal_call_result_324782 = invoke(stypy.reporting.localization.Localization(__file__, 44, 20), assert_array_equal_324768, *[abs_call_result_324773, ones_like_call_result_324778], **kwargs_324781)
        
        
        # Assigning a Call to a Name (line 48):
        
        # Assigning a Call to a Name (line 48):
        
        # Call to real(...): (line 48)
        # Processing the call arguments (line 48)
        
        # Call to ifft(...): (line 48)
        # Processing the call arguments (line 48)
        
        # Call to fft(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'm' (line 48)
        m_324787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 46), 'm', False)
        # Processing the call keyword arguments (line 48)
        kwargs_324788 = {}
        # Getting the type of 'fft' (line 48)
        fft_324786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 42), 'fft', False)
        # Calling fft(args, kwargs) (line 48)
        fft_call_result_324789 = invoke(stypy.reporting.localization.Localization(__file__, 48, 42), fft_324786, *[m_324787], **kwargs_324788)
        
        
        # Call to conj(...): (line 48)
        # Processing the call arguments (line 48)
        
        # Call to fft(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'm' (line 48)
        m_324793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 63), 'm', False)
        # Processing the call keyword arguments (line 48)
        kwargs_324794 = {}
        # Getting the type of 'fft' (line 48)
        fft_324792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 59), 'fft', False)
        # Calling fft(args, kwargs) (line 48)
        fft_call_result_324795 = invoke(stypy.reporting.localization.Localization(__file__, 48, 59), fft_324792, *[m_324793], **kwargs_324794)
        
        # Processing the call keyword arguments (line 48)
        kwargs_324796 = {}
        # Getting the type of 'np' (line 48)
        np_324790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 51), 'np', False)
        # Obtaining the member 'conj' of a type (line 48)
        conj_324791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 51), np_324790, 'conj')
        # Calling conj(args, kwargs) (line 48)
        conj_call_result_324797 = invoke(stypy.reporting.localization.Localization(__file__, 48, 51), conj_324791, *[fft_call_result_324795], **kwargs_324796)
        
        # Applying the binary operator '*' (line 48)
        result_mul_324798 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 42), '*', fft_call_result_324789, conj_call_result_324797)
        
        # Processing the call keyword arguments (line 48)
        kwargs_324799 = {}
        # Getting the type of 'ifft' (line 48)
        ifft_324785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 37), 'ifft', False)
        # Calling ifft(args, kwargs) (line 48)
        ifft_call_result_324800 = invoke(stypy.reporting.localization.Localization(__file__, 48, 37), ifft_324785, *[result_mul_324798], **kwargs_324799)
        
        # Processing the call keyword arguments (line 48)
        kwargs_324801 = {}
        # Getting the type of 'np' (line 48)
        np_324783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 29), 'np', False)
        # Obtaining the member 'real' of a type (line 48)
        real_324784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 29), np_324783, 'real')
        # Calling real(args, kwargs) (line 48)
        real_call_result_324802 = invoke(stypy.reporting.localization.Localization(__file__, 48, 29), real_324784, *[ifft_call_result_324800], **kwargs_324801)
        
        # Assigning a type to the variable 'tester' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 20), 'tester', real_call_result_324802)
        
        # Assigning a BinOp to a Name (line 49):
        
        # Assigning a BinOp to a Name (line 49):
        int_324803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 30), 'int')
        # Getting the type of 'nbits' (line 49)
        nbits_324804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 33), 'nbits')
        # Applying the binary operator '**' (line 49)
        result_pow_324805 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 30), '**', int_324803, nbits_324804)
        
        int_324806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 41), 'int')
        # Applying the binary operator '-' (line 49)
        result_sub_324807 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 30), '-', result_pow_324805, int_324806)
        
        # Assigning a type to the variable 'out_len' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 20), 'out_len', result_sub_324807)
        
        # Assigning a Str to a Name (line 51):
        
        # Assigning a Str to a Name (line 51):
        str_324808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 30), 'str', 'mls impulse has incorrect value')
        # Assigning a type to the variable 'err_msg' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 20), 'err_msg', str_324808)
        
        # Call to assert_allclose(...): (line 52)
        # Processing the call arguments (line 52)
        
        # Obtaining the type of the subscript
        int_324810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 43), 'int')
        # Getting the type of 'tester' (line 52)
        tester_324811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 36), 'tester', False)
        # Obtaining the member '__getitem__' of a type (line 52)
        getitem___324812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 36), tester_324811, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 52)
        subscript_call_result_324813 = invoke(stypy.reporting.localization.Localization(__file__, 52, 36), getitem___324812, int_324810)
        
        # Getting the type of 'out_len' (line 52)
        out_len_324814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 47), 'out_len', False)
        # Processing the call keyword arguments (line 52)
        # Getting the type of 'err_msg' (line 52)
        err_msg_324815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 64), 'err_msg', False)
        keyword_324816 = err_msg_324815
        kwargs_324817 = {'err_msg': keyword_324816}
        # Getting the type of 'assert_allclose' (line 52)
        assert_allclose_324809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 20), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 52)
        assert_allclose_call_result_324818 = invoke(stypy.reporting.localization.Localization(__file__, 52, 20), assert_allclose_324809, *[subscript_call_result_324813, out_len_324814], **kwargs_324817)
        
        
        # Assigning a Str to a Name (line 54):
        
        # Assigning a Str to a Name (line 54):
        str_324819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 30), 'str', 'mls steady-state has incorrect value')
        # Assigning a type to the variable 'err_msg' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'err_msg', str_324819)
        
        # Call to assert_allclose(...): (line 55)
        # Processing the call arguments (line 55)
        
        # Obtaining the type of the subscript
        int_324821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 43), 'int')
        slice_324822 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 55, 36), int_324821, None, None)
        # Getting the type of 'tester' (line 55)
        tester_324823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 36), 'tester', False)
        # Obtaining the member '__getitem__' of a type (line 55)
        getitem___324824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 36), tester_324823, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 55)
        subscript_call_result_324825 = invoke(stypy.reporting.localization.Localization(__file__, 55, 36), getitem___324824, slice_324822)
        
        int_324826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 48), 'int')
        
        # Call to ones(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'out_len' (line 55)
        out_len_324829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 61), 'out_len', False)
        int_324830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 71), 'int')
        # Applying the binary operator '-' (line 55)
        result_sub_324831 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 61), '-', out_len_324829, int_324830)
        
        # Processing the call keyword arguments (line 55)
        kwargs_324832 = {}
        # Getting the type of 'np' (line 55)
        np_324827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 53), 'np', False)
        # Obtaining the member 'ones' of a type (line 55)
        ones_324828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 53), np_324827, 'ones')
        # Calling ones(args, kwargs) (line 55)
        ones_call_result_324833 = invoke(stypy.reporting.localization.Localization(__file__, 55, 53), ones_324828, *[result_sub_324831], **kwargs_324832)
        
        # Applying the binary operator '*' (line 55)
        result_mul_324834 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 48), '*', int_324826, ones_call_result_324833)
        
        # Processing the call keyword arguments (line 55)
        # Getting the type of 'err_msg' (line 56)
        err_msg_324835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 44), 'err_msg', False)
        keyword_324836 = err_msg_324835
        kwargs_324837 = {'err_msg': keyword_324836}
        # Getting the type of 'assert_allclose' (line 55)
        assert_allclose_324820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 20), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 55)
        assert_allclose_call_result_324838 = invoke(stypy.reporting.localization.Localization(__file__, 55, 20), assert_allclose_324820, *[subscript_call_result_324825, result_mul_324834], **kwargs_324837)
        
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 58)
        tuple_324839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 58)
        # Adding element type (line 58)
        int_324840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 30), tuple_324839, int_324840)
        # Adding element type (line 58)
        int_324841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 33), 'int')
        # Getting the type of 'nbits' (line 58)
        nbits_324842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 37), 'nbits')
        int_324843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 45), 'int')
        # Applying the binary operator '-' (line 58)
        result_sub_324844 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 37), '-', nbits_324842, int_324843)
        
        # Applying the binary operator '**' (line 58)
        result_pow_324845 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 33), '**', int_324841, result_sub_324844)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 30), tuple_324839, result_pow_324845)
        
        # Testing the type of a for loop iterable (line 58)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 58, 20), tuple_324839)
        # Getting the type of the for loop variable (line 58)
        for_loop_var_324846 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 58, 20), tuple_324839)
        # Assigning a type to the variable 'n' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 20), 'n', for_loop_var_324846)
        # SSA begins for a for statement (line 58)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 59):
        
        # Assigning a Subscript to a Name (line 59):
        
        # Obtaining the type of the subscript
        int_324847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 24), 'int')
        
        # Call to max_len_seq(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'nbits' (line 59)
        nbits_324849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 45), 'nbits', False)
        # Processing the call keyword arguments (line 59)
        # Getting the type of 'state' (line 59)
        state_324850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 58), 'state', False)
        keyword_324851 = state_324850
        # Getting the type of 'taps' (line 59)
        taps_324852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 70), 'taps', False)
        keyword_324853 = taps_324852
        # Getting the type of 'n' (line 60)
        n_324854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 52), 'n', False)
        keyword_324855 = n_324854
        kwargs_324856 = {'length': keyword_324855, 'state': keyword_324851, 'taps': keyword_324853}
        # Getting the type of 'max_len_seq' (line 59)
        max_len_seq_324848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 33), 'max_len_seq', False)
        # Calling max_len_seq(args, kwargs) (line 59)
        max_len_seq_call_result_324857 = invoke(stypy.reporting.localization.Localization(__file__, 59, 33), max_len_seq_324848, *[nbits_324849], **kwargs_324856)
        
        # Obtaining the member '__getitem__' of a type (line 59)
        getitem___324858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 24), max_len_seq_call_result_324857, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 59)
        subscript_call_result_324859 = invoke(stypy.reporting.localization.Localization(__file__, 59, 24), getitem___324858, int_324847)
        
        # Assigning a type to the variable 'tuple_var_assignment_324604' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 24), 'tuple_var_assignment_324604', subscript_call_result_324859)
        
        # Assigning a Subscript to a Name (line 59):
        
        # Obtaining the type of the subscript
        int_324860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 24), 'int')
        
        # Call to max_len_seq(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'nbits' (line 59)
        nbits_324862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 45), 'nbits', False)
        # Processing the call keyword arguments (line 59)
        # Getting the type of 'state' (line 59)
        state_324863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 58), 'state', False)
        keyword_324864 = state_324863
        # Getting the type of 'taps' (line 59)
        taps_324865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 70), 'taps', False)
        keyword_324866 = taps_324865
        # Getting the type of 'n' (line 60)
        n_324867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 52), 'n', False)
        keyword_324868 = n_324867
        kwargs_324869 = {'length': keyword_324868, 'state': keyword_324864, 'taps': keyword_324866}
        # Getting the type of 'max_len_seq' (line 59)
        max_len_seq_324861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 33), 'max_len_seq', False)
        # Calling max_len_seq(args, kwargs) (line 59)
        max_len_seq_call_result_324870 = invoke(stypy.reporting.localization.Localization(__file__, 59, 33), max_len_seq_324861, *[nbits_324862], **kwargs_324869)
        
        # Obtaining the member '__getitem__' of a type (line 59)
        getitem___324871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 24), max_len_seq_call_result_324870, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 59)
        subscript_call_result_324872 = invoke(stypy.reporting.localization.Localization(__file__, 59, 24), getitem___324871, int_324860)
        
        # Assigning a type to the variable 'tuple_var_assignment_324605' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 24), 'tuple_var_assignment_324605', subscript_call_result_324872)
        
        # Assigning a Name to a Name (line 59):
        # Getting the type of 'tuple_var_assignment_324604' (line 59)
        tuple_var_assignment_324604_324873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 24), 'tuple_var_assignment_324604')
        # Assigning a type to the variable 'm1' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 24), 'm1', tuple_var_assignment_324604_324873)
        
        # Assigning a Name to a Name (line 59):
        # Getting the type of 'tuple_var_assignment_324605' (line 59)
        tuple_var_assignment_324605_324874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 24), 'tuple_var_assignment_324605')
        # Assigning a type to the variable 's1' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 28), 's1', tuple_var_assignment_324605_324874)
        
        # Assigning a Call to a Tuple (line 61):
        
        # Assigning a Subscript to a Name (line 61):
        
        # Obtaining the type of the subscript
        int_324875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 24), 'int')
        
        # Call to max_len_seq(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'nbits' (line 61)
        nbits_324877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 45), 'nbits', False)
        # Processing the call keyword arguments (line 61)
        # Getting the type of 's1' (line 61)
        s1_324878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 58), 's1', False)
        keyword_324879 = s1_324878
        # Getting the type of 'taps' (line 61)
        taps_324880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 67), 'taps', False)
        keyword_324881 = taps_324880
        int_324882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 52), 'int')
        keyword_324883 = int_324882
        kwargs_324884 = {'length': keyword_324883, 'state': keyword_324879, 'taps': keyword_324881}
        # Getting the type of 'max_len_seq' (line 61)
        max_len_seq_324876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 33), 'max_len_seq', False)
        # Calling max_len_seq(args, kwargs) (line 61)
        max_len_seq_call_result_324885 = invoke(stypy.reporting.localization.Localization(__file__, 61, 33), max_len_seq_324876, *[nbits_324877], **kwargs_324884)
        
        # Obtaining the member '__getitem__' of a type (line 61)
        getitem___324886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 24), max_len_seq_call_result_324885, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 61)
        subscript_call_result_324887 = invoke(stypy.reporting.localization.Localization(__file__, 61, 24), getitem___324886, int_324875)
        
        # Assigning a type to the variable 'tuple_var_assignment_324606' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'tuple_var_assignment_324606', subscript_call_result_324887)
        
        # Assigning a Subscript to a Name (line 61):
        
        # Obtaining the type of the subscript
        int_324888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 24), 'int')
        
        # Call to max_len_seq(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'nbits' (line 61)
        nbits_324890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 45), 'nbits', False)
        # Processing the call keyword arguments (line 61)
        # Getting the type of 's1' (line 61)
        s1_324891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 58), 's1', False)
        keyword_324892 = s1_324891
        # Getting the type of 'taps' (line 61)
        taps_324893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 67), 'taps', False)
        keyword_324894 = taps_324893
        int_324895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 52), 'int')
        keyword_324896 = int_324895
        kwargs_324897 = {'length': keyword_324896, 'state': keyword_324892, 'taps': keyword_324894}
        # Getting the type of 'max_len_seq' (line 61)
        max_len_seq_324889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 33), 'max_len_seq', False)
        # Calling max_len_seq(args, kwargs) (line 61)
        max_len_seq_call_result_324898 = invoke(stypy.reporting.localization.Localization(__file__, 61, 33), max_len_seq_324889, *[nbits_324890], **kwargs_324897)
        
        # Obtaining the member '__getitem__' of a type (line 61)
        getitem___324899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 24), max_len_seq_call_result_324898, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 61)
        subscript_call_result_324900 = invoke(stypy.reporting.localization.Localization(__file__, 61, 24), getitem___324899, int_324888)
        
        # Assigning a type to the variable 'tuple_var_assignment_324607' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'tuple_var_assignment_324607', subscript_call_result_324900)
        
        # Assigning a Name to a Name (line 61):
        # Getting the type of 'tuple_var_assignment_324606' (line 61)
        tuple_var_assignment_324606_324901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'tuple_var_assignment_324606')
        # Assigning a type to the variable 'm2' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'm2', tuple_var_assignment_324606_324901)
        
        # Assigning a Name to a Name (line 61):
        # Getting the type of 'tuple_var_assignment_324607' (line 61)
        tuple_var_assignment_324607_324902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'tuple_var_assignment_324607')
        # Assigning a type to the variable 's2' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 28), 's2', tuple_var_assignment_324607_324902)
        
        # Assigning a Call to a Tuple (line 63):
        
        # Assigning a Subscript to a Name (line 63):
        
        # Obtaining the type of the subscript
        int_324903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 24), 'int')
        
        # Call to max_len_seq(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'nbits' (line 63)
        nbits_324905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 45), 'nbits', False)
        # Processing the call keyword arguments (line 63)
        # Getting the type of 's2' (line 63)
        s2_324906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 58), 's2', False)
        keyword_324907 = s2_324906
        # Getting the type of 'taps' (line 63)
        taps_324908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 67), 'taps', False)
        keyword_324909 = taps_324908
        # Getting the type of 'out_len' (line 64)
        out_len_324910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 52), 'out_len', False)
        # Getting the type of 'n' (line 64)
        n_324911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 62), 'n', False)
        # Applying the binary operator '-' (line 64)
        result_sub_324912 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 52), '-', out_len_324910, n_324911)
        
        int_324913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 66), 'int')
        # Applying the binary operator '-' (line 64)
        result_sub_324914 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 64), '-', result_sub_324912, int_324913)
        
        keyword_324915 = result_sub_324914
        kwargs_324916 = {'length': keyword_324915, 'state': keyword_324907, 'taps': keyword_324909}
        # Getting the type of 'max_len_seq' (line 63)
        max_len_seq_324904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 33), 'max_len_seq', False)
        # Calling max_len_seq(args, kwargs) (line 63)
        max_len_seq_call_result_324917 = invoke(stypy.reporting.localization.Localization(__file__, 63, 33), max_len_seq_324904, *[nbits_324905], **kwargs_324916)
        
        # Obtaining the member '__getitem__' of a type (line 63)
        getitem___324918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 24), max_len_seq_call_result_324917, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 63)
        subscript_call_result_324919 = invoke(stypy.reporting.localization.Localization(__file__, 63, 24), getitem___324918, int_324903)
        
        # Assigning a type to the variable 'tuple_var_assignment_324608' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 24), 'tuple_var_assignment_324608', subscript_call_result_324919)
        
        # Assigning a Subscript to a Name (line 63):
        
        # Obtaining the type of the subscript
        int_324920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 24), 'int')
        
        # Call to max_len_seq(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'nbits' (line 63)
        nbits_324922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 45), 'nbits', False)
        # Processing the call keyword arguments (line 63)
        # Getting the type of 's2' (line 63)
        s2_324923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 58), 's2', False)
        keyword_324924 = s2_324923
        # Getting the type of 'taps' (line 63)
        taps_324925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 67), 'taps', False)
        keyword_324926 = taps_324925
        # Getting the type of 'out_len' (line 64)
        out_len_324927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 52), 'out_len', False)
        # Getting the type of 'n' (line 64)
        n_324928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 62), 'n', False)
        # Applying the binary operator '-' (line 64)
        result_sub_324929 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 52), '-', out_len_324927, n_324928)
        
        int_324930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 66), 'int')
        # Applying the binary operator '-' (line 64)
        result_sub_324931 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 64), '-', result_sub_324929, int_324930)
        
        keyword_324932 = result_sub_324931
        kwargs_324933 = {'length': keyword_324932, 'state': keyword_324924, 'taps': keyword_324926}
        # Getting the type of 'max_len_seq' (line 63)
        max_len_seq_324921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 33), 'max_len_seq', False)
        # Calling max_len_seq(args, kwargs) (line 63)
        max_len_seq_call_result_324934 = invoke(stypy.reporting.localization.Localization(__file__, 63, 33), max_len_seq_324921, *[nbits_324922], **kwargs_324933)
        
        # Obtaining the member '__getitem__' of a type (line 63)
        getitem___324935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 24), max_len_seq_call_result_324934, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 63)
        subscript_call_result_324936 = invoke(stypy.reporting.localization.Localization(__file__, 63, 24), getitem___324935, int_324920)
        
        # Assigning a type to the variable 'tuple_var_assignment_324609' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 24), 'tuple_var_assignment_324609', subscript_call_result_324936)
        
        # Assigning a Name to a Name (line 63):
        # Getting the type of 'tuple_var_assignment_324608' (line 63)
        tuple_var_assignment_324608_324937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 24), 'tuple_var_assignment_324608')
        # Assigning a type to the variable 'm3' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 24), 'm3', tuple_var_assignment_324608_324937)
        
        # Assigning a Name to a Name (line 63):
        # Getting the type of 'tuple_var_assignment_324609' (line 63)
        tuple_var_assignment_324609_324938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 24), 'tuple_var_assignment_324609')
        # Assigning a type to the variable 's3' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 28), 's3', tuple_var_assignment_324609_324938)
        
        # Assigning a Call to a Name (line 65):
        
        # Assigning a Call to a Name (line 65):
        
        # Call to concatenate(...): (line 65)
        # Processing the call arguments (line 65)
        
        # Obtaining an instance of the builtin type 'tuple' (line 65)
        tuple_324941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 48), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 65)
        # Adding element type (line 65)
        # Getting the type of 'm1' (line 65)
        m1_324942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 48), 'm1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 48), tuple_324941, m1_324942)
        # Adding element type (line 65)
        # Getting the type of 'm2' (line 65)
        m2_324943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 52), 'm2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 48), tuple_324941, m2_324943)
        # Adding element type (line 65)
        # Getting the type of 'm3' (line 65)
        m3_324944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 56), 'm3', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 48), tuple_324941, m3_324944)
        
        # Processing the call keyword arguments (line 65)
        kwargs_324945 = {}
        # Getting the type of 'np' (line 65)
        np_324939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 32), 'np', False)
        # Obtaining the member 'concatenate' of a type (line 65)
        concatenate_324940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 32), np_324939, 'concatenate')
        # Calling concatenate(args, kwargs) (line 65)
        concatenate_call_result_324946 = invoke(stypy.reporting.localization.Localization(__file__, 65, 32), concatenate_324940, *[tuple_324941], **kwargs_324945)
        
        # Assigning a type to the variable 'new_m' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 24), 'new_m', concatenate_call_result_324946)
        
        # Call to assert_array_equal(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'orig_m' (line 66)
        orig_m_324948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 43), 'orig_m', False)
        # Getting the type of 'new_m' (line 66)
        new_m_324949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 51), 'new_m', False)
        # Processing the call keyword arguments (line 66)
        kwargs_324950 = {}
        # Getting the type of 'assert_array_equal' (line 66)
        assert_array_equal_324947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 24), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 66)
        assert_array_equal_call_result_324951 = invoke(stypy.reporting.localization.Localization(__file__, 66, 24), assert_array_equal_324947, *[orig_m_324948, new_m_324949], **kwargs_324950)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_mls_output(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_mls_output' in the type store
        # Getting the type of 'stypy_return_type' (line 29)
        stypy_return_type_324952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_324952)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_mls_output'
        return stypy_return_type_324952


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 12, 0, False)
        # Assigning a type to the variable 'self' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMLS.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestMLS' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'TestMLS', TestMLS)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

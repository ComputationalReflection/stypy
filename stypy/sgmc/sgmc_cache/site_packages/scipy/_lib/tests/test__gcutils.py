
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Test for assert_deallocated context manager and gc utilities
2: '''
3: from __future__ import division, print_function, absolute_import
4: 
5: import gc
6: 
7: from scipy._lib._gcutils import set_gc_state, gc_state, assert_deallocated, ReferenceError
8: 
9: from numpy.testing import assert_equal
10: 
11: import pytest
12: 
13: def test_set_gc_state():
14:     gc_status = gc.isenabled()
15:     try:
16:         for state in (True, False):
17:             gc.enable()
18:             set_gc_state(state)
19:             assert_equal(gc.isenabled(), state)
20:             gc.disable()
21:             set_gc_state(state)
22:             assert_equal(gc.isenabled(), state)
23:     finally:
24:         if gc_status:
25:             gc.enable()
26: 
27: 
28: def test_gc_state():
29:     # Test gc_state context manager
30:     gc_status = gc.isenabled()
31:     try:
32:         for pre_state in (True, False):
33:             set_gc_state(pre_state)
34:             for with_state in (True, False):
35:                 # Check the gc state is with_state in with block
36:                 with gc_state(with_state):
37:                     assert_equal(gc.isenabled(), with_state)
38:                 # And returns to previous state outside block
39:                 assert_equal(gc.isenabled(), pre_state)
40:                 # Even if the gc state is set explicitly within the block
41:                 with gc_state(with_state):
42:                     assert_equal(gc.isenabled(), with_state)
43:                     set_gc_state(not with_state)
44:                 assert_equal(gc.isenabled(), pre_state)
45:     finally:
46:         if gc_status:
47:             gc.enable()
48: 
49: 
50: def test_assert_deallocated():
51:     # Ordinary use
52:     class C(object):
53:         def __init__(self, arg0, arg1, name='myname'):
54:             self.name = name
55:     for gc_current in (True, False):
56:         with gc_state(gc_current):
57:             # We are deleting from with-block context, so that's OK
58:             with assert_deallocated(C, 0, 2, 'another name') as c:
59:                 assert_equal(c.name, 'another name')
60:                 del c
61:             # Or not using the thing in with-block context, also OK
62:             with assert_deallocated(C, 0, 2, name='third name'):
63:                 pass
64:             assert_equal(gc.isenabled(), gc_current)
65: 
66: 
67: def test_assert_deallocated_nodel():
68:     class C(object):
69:         pass
70:     with pytest.raises(ReferenceError):
71:         # Need to delete after using if in with-block context
72:         with assert_deallocated(C) as c:
73:             pass
74: 
75: 
76: def test_assert_deallocated_circular():
77:     class C(object):
78:         def __init__(self):
79:             self._circular = self
80:     with pytest.raises(ReferenceError):
81:         # Circular reference, no automatic garbage collection
82:         with assert_deallocated(C) as c:
83:             del c
84: 
85: 
86: def test_assert_deallocated_circular2():
87:     class C(object):
88:         def __init__(self):
89:             self._circular = self
90:     with pytest.raises(ReferenceError):
91:         # Still circular reference, no automatic garbage collection
92:         with assert_deallocated(C):
93:             pass
94: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_712170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', ' Test for assert_deallocated context manager and gc utilities\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import gc' statement (line 5)
import gc

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'gc', gc, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy._lib._gcutils import set_gc_state, gc_state, assert_deallocated, ReferenceError' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_712171 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._gcutils')

if (type(import_712171) is not StypyTypeError):

    if (import_712171 != 'pyd_module'):
        __import__(import_712171)
        sys_modules_712172 = sys.modules[import_712171]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._gcutils', sys_modules_712172.module_type_store, module_type_store, ['set_gc_state', 'gc_state', 'assert_deallocated', 'ReferenceError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_712172, sys_modules_712172.module_type_store, module_type_store)
    else:
        from scipy._lib._gcutils import set_gc_state, gc_state, assert_deallocated, ReferenceError

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._gcutils', None, module_type_store, ['set_gc_state', 'gc_state', 'assert_deallocated', 'ReferenceError'], [set_gc_state, gc_state, assert_deallocated, ReferenceError])

else:
    # Assigning a type to the variable 'scipy._lib._gcutils' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._gcutils', import_712171)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy.testing import assert_equal' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_712173 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing')

if (type(import_712173) is not StypyTypeError):

    if (import_712173 != 'pyd_module'):
        __import__(import_712173)
        sys_modules_712174 = sys.modules[import_712173]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', sys_modules_712174.module_type_store, module_type_store, ['assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_712174, sys_modules_712174.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', None, module_type_store, ['assert_equal'], [assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', import_712173)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import pytest' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_712175 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'pytest')

if (type(import_712175) is not StypyTypeError):

    if (import_712175 != 'pyd_module'):
        __import__(import_712175)
        sys_modules_712176 = sys.modules[import_712175]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'pytest', sys_modules_712176.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'pytest', import_712175)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')


@norecursion
def test_set_gc_state(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_set_gc_state'
    module_type_store = module_type_store.open_function_context('test_set_gc_state', 13, 0, False)
    
    # Passed parameters checking function
    test_set_gc_state.stypy_localization = localization
    test_set_gc_state.stypy_type_of_self = None
    test_set_gc_state.stypy_type_store = module_type_store
    test_set_gc_state.stypy_function_name = 'test_set_gc_state'
    test_set_gc_state.stypy_param_names_list = []
    test_set_gc_state.stypy_varargs_param_name = None
    test_set_gc_state.stypy_kwargs_param_name = None
    test_set_gc_state.stypy_call_defaults = defaults
    test_set_gc_state.stypy_call_varargs = varargs
    test_set_gc_state.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_set_gc_state', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_set_gc_state', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_set_gc_state(...)' code ##################

    
    # Assigning a Call to a Name (line 14):
    
    # Call to isenabled(...): (line 14)
    # Processing the call keyword arguments (line 14)
    kwargs_712179 = {}
    # Getting the type of 'gc' (line 14)
    gc_712177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 16), 'gc', False)
    # Obtaining the member 'isenabled' of a type (line 14)
    isenabled_712178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 16), gc_712177, 'isenabled')
    # Calling isenabled(args, kwargs) (line 14)
    isenabled_call_result_712180 = invoke(stypy.reporting.localization.Localization(__file__, 14, 16), isenabled_712178, *[], **kwargs_712179)
    
    # Assigning a type to the variable 'gc_status' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'gc_status', isenabled_call_result_712180)
    
    # Try-finally block (line 15)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 16)
    tuple_712181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 16)
    # Adding element type (line 16)
    # Getting the type of 'True' (line 16)
    True_712182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 22), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 22), tuple_712181, True_712182)
    # Adding element type (line 16)
    # Getting the type of 'False' (line 16)
    False_712183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 28), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 22), tuple_712181, False_712183)
    
    # Testing the type of a for loop iterable (line 16)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 16, 8), tuple_712181)
    # Getting the type of the for loop variable (line 16)
    for_loop_var_712184 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 16, 8), tuple_712181)
    # Assigning a type to the variable 'state' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'state', for_loop_var_712184)
    # SSA begins for a for statement (line 16)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to enable(...): (line 17)
    # Processing the call keyword arguments (line 17)
    kwargs_712187 = {}
    # Getting the type of 'gc' (line 17)
    gc_712185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'gc', False)
    # Obtaining the member 'enable' of a type (line 17)
    enable_712186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 12), gc_712185, 'enable')
    # Calling enable(args, kwargs) (line 17)
    enable_call_result_712188 = invoke(stypy.reporting.localization.Localization(__file__, 17, 12), enable_712186, *[], **kwargs_712187)
    
    
    # Call to set_gc_state(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'state' (line 18)
    state_712190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 25), 'state', False)
    # Processing the call keyword arguments (line 18)
    kwargs_712191 = {}
    # Getting the type of 'set_gc_state' (line 18)
    set_gc_state_712189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 12), 'set_gc_state', False)
    # Calling set_gc_state(args, kwargs) (line 18)
    set_gc_state_call_result_712192 = invoke(stypy.reporting.localization.Localization(__file__, 18, 12), set_gc_state_712189, *[state_712190], **kwargs_712191)
    
    
    # Call to assert_equal(...): (line 19)
    # Processing the call arguments (line 19)
    
    # Call to isenabled(...): (line 19)
    # Processing the call keyword arguments (line 19)
    kwargs_712196 = {}
    # Getting the type of 'gc' (line 19)
    gc_712194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 25), 'gc', False)
    # Obtaining the member 'isenabled' of a type (line 19)
    isenabled_712195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 25), gc_712194, 'isenabled')
    # Calling isenabled(args, kwargs) (line 19)
    isenabled_call_result_712197 = invoke(stypy.reporting.localization.Localization(__file__, 19, 25), isenabled_712195, *[], **kwargs_712196)
    
    # Getting the type of 'state' (line 19)
    state_712198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 41), 'state', False)
    # Processing the call keyword arguments (line 19)
    kwargs_712199 = {}
    # Getting the type of 'assert_equal' (line 19)
    assert_equal_712193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 19)
    assert_equal_call_result_712200 = invoke(stypy.reporting.localization.Localization(__file__, 19, 12), assert_equal_712193, *[isenabled_call_result_712197, state_712198], **kwargs_712199)
    
    
    # Call to disable(...): (line 20)
    # Processing the call keyword arguments (line 20)
    kwargs_712203 = {}
    # Getting the type of 'gc' (line 20)
    gc_712201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'gc', False)
    # Obtaining the member 'disable' of a type (line 20)
    disable_712202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 12), gc_712201, 'disable')
    # Calling disable(args, kwargs) (line 20)
    disable_call_result_712204 = invoke(stypy.reporting.localization.Localization(__file__, 20, 12), disable_712202, *[], **kwargs_712203)
    
    
    # Call to set_gc_state(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'state' (line 21)
    state_712206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 25), 'state', False)
    # Processing the call keyword arguments (line 21)
    kwargs_712207 = {}
    # Getting the type of 'set_gc_state' (line 21)
    set_gc_state_712205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'set_gc_state', False)
    # Calling set_gc_state(args, kwargs) (line 21)
    set_gc_state_call_result_712208 = invoke(stypy.reporting.localization.Localization(__file__, 21, 12), set_gc_state_712205, *[state_712206], **kwargs_712207)
    
    
    # Call to assert_equal(...): (line 22)
    # Processing the call arguments (line 22)
    
    # Call to isenabled(...): (line 22)
    # Processing the call keyword arguments (line 22)
    kwargs_712212 = {}
    # Getting the type of 'gc' (line 22)
    gc_712210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 25), 'gc', False)
    # Obtaining the member 'isenabled' of a type (line 22)
    isenabled_712211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 25), gc_712210, 'isenabled')
    # Calling isenabled(args, kwargs) (line 22)
    isenabled_call_result_712213 = invoke(stypy.reporting.localization.Localization(__file__, 22, 25), isenabled_712211, *[], **kwargs_712212)
    
    # Getting the type of 'state' (line 22)
    state_712214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 41), 'state', False)
    # Processing the call keyword arguments (line 22)
    kwargs_712215 = {}
    # Getting the type of 'assert_equal' (line 22)
    assert_equal_712209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 22)
    assert_equal_call_result_712216 = invoke(stypy.reporting.localization.Localization(__file__, 22, 12), assert_equal_712209, *[isenabled_call_result_712213, state_712214], **kwargs_712215)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # finally branch of the try-finally block (line 15)
    
    # Getting the type of 'gc_status' (line 24)
    gc_status_712217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 11), 'gc_status')
    # Testing the type of an if condition (line 24)
    if_condition_712218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 24, 8), gc_status_712217)
    # Assigning a type to the variable 'if_condition_712218' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'if_condition_712218', if_condition_712218)
    # SSA begins for if statement (line 24)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to enable(...): (line 25)
    # Processing the call keyword arguments (line 25)
    kwargs_712221 = {}
    # Getting the type of 'gc' (line 25)
    gc_712219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'gc', False)
    # Obtaining the member 'enable' of a type (line 25)
    enable_712220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 12), gc_712219, 'enable')
    # Calling enable(args, kwargs) (line 25)
    enable_call_result_712222 = invoke(stypy.reporting.localization.Localization(__file__, 25, 12), enable_712220, *[], **kwargs_712221)
    
    # SSA join for if statement (line 24)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # ################# End of 'test_set_gc_state(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_set_gc_state' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_712223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_712223)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_set_gc_state'
    return stypy_return_type_712223

# Assigning a type to the variable 'test_set_gc_state' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'test_set_gc_state', test_set_gc_state)

@norecursion
def test_gc_state(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_gc_state'
    module_type_store = module_type_store.open_function_context('test_gc_state', 28, 0, False)
    
    # Passed parameters checking function
    test_gc_state.stypy_localization = localization
    test_gc_state.stypy_type_of_self = None
    test_gc_state.stypy_type_store = module_type_store
    test_gc_state.stypy_function_name = 'test_gc_state'
    test_gc_state.stypy_param_names_list = []
    test_gc_state.stypy_varargs_param_name = None
    test_gc_state.stypy_kwargs_param_name = None
    test_gc_state.stypy_call_defaults = defaults
    test_gc_state.stypy_call_varargs = varargs
    test_gc_state.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_gc_state', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_gc_state', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_gc_state(...)' code ##################

    
    # Assigning a Call to a Name (line 30):
    
    # Call to isenabled(...): (line 30)
    # Processing the call keyword arguments (line 30)
    kwargs_712226 = {}
    # Getting the type of 'gc' (line 30)
    gc_712224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'gc', False)
    # Obtaining the member 'isenabled' of a type (line 30)
    isenabled_712225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 16), gc_712224, 'isenabled')
    # Calling isenabled(args, kwargs) (line 30)
    isenabled_call_result_712227 = invoke(stypy.reporting.localization.Localization(__file__, 30, 16), isenabled_712225, *[], **kwargs_712226)
    
    # Assigning a type to the variable 'gc_status' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'gc_status', isenabled_call_result_712227)
    
    # Try-finally block (line 31)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 32)
    tuple_712228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 32)
    # Adding element type (line 32)
    # Getting the type of 'True' (line 32)
    True_712229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 26), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 26), tuple_712228, True_712229)
    # Adding element type (line 32)
    # Getting the type of 'False' (line 32)
    False_712230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 32), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 26), tuple_712228, False_712230)
    
    # Testing the type of a for loop iterable (line 32)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 32, 8), tuple_712228)
    # Getting the type of the for loop variable (line 32)
    for_loop_var_712231 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 32, 8), tuple_712228)
    # Assigning a type to the variable 'pre_state' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'pre_state', for_loop_var_712231)
    # SSA begins for a for statement (line 32)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to set_gc_state(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'pre_state' (line 33)
    pre_state_712233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 25), 'pre_state', False)
    # Processing the call keyword arguments (line 33)
    kwargs_712234 = {}
    # Getting the type of 'set_gc_state' (line 33)
    set_gc_state_712232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'set_gc_state', False)
    # Calling set_gc_state(args, kwargs) (line 33)
    set_gc_state_call_result_712235 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), set_gc_state_712232, *[pre_state_712233], **kwargs_712234)
    
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 34)
    tuple_712236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 34)
    # Adding element type (line 34)
    # Getting the type of 'True' (line 34)
    True_712237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 31), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 31), tuple_712236, True_712237)
    # Adding element type (line 34)
    # Getting the type of 'False' (line 34)
    False_712238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 37), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 31), tuple_712236, False_712238)
    
    # Testing the type of a for loop iterable (line 34)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 34, 12), tuple_712236)
    # Getting the type of the for loop variable (line 34)
    for_loop_var_712239 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 34, 12), tuple_712236)
    # Assigning a type to the variable 'with_state' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'with_state', for_loop_var_712239)
    # SSA begins for a for statement (line 34)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to gc_state(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'with_state' (line 36)
    with_state_712241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 30), 'with_state', False)
    # Processing the call keyword arguments (line 36)
    kwargs_712242 = {}
    # Getting the type of 'gc_state' (line 36)
    gc_state_712240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 21), 'gc_state', False)
    # Calling gc_state(args, kwargs) (line 36)
    gc_state_call_result_712243 = invoke(stypy.reporting.localization.Localization(__file__, 36, 21), gc_state_712240, *[with_state_712241], **kwargs_712242)
    
    with_712244 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 36, 21), gc_state_call_result_712243, 'with parameter', '__enter__', '__exit__')

    if with_712244:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 36)
        enter___712245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 21), gc_state_call_result_712243, '__enter__')
        with_enter_712246 = invoke(stypy.reporting.localization.Localization(__file__, 36, 21), enter___712245)
        
        # Call to assert_equal(...): (line 37)
        # Processing the call arguments (line 37)
        
        # Call to isenabled(...): (line 37)
        # Processing the call keyword arguments (line 37)
        kwargs_712250 = {}
        # Getting the type of 'gc' (line 37)
        gc_712248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 33), 'gc', False)
        # Obtaining the member 'isenabled' of a type (line 37)
        isenabled_712249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 33), gc_712248, 'isenabled')
        # Calling isenabled(args, kwargs) (line 37)
        isenabled_call_result_712251 = invoke(stypy.reporting.localization.Localization(__file__, 37, 33), isenabled_712249, *[], **kwargs_712250)
        
        # Getting the type of 'with_state' (line 37)
        with_state_712252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 49), 'with_state', False)
        # Processing the call keyword arguments (line 37)
        kwargs_712253 = {}
        # Getting the type of 'assert_equal' (line 37)
        assert_equal_712247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 20), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 37)
        assert_equal_call_result_712254 = invoke(stypy.reporting.localization.Localization(__file__, 37, 20), assert_equal_712247, *[isenabled_call_result_712251, with_state_712252], **kwargs_712253)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 36)
        exit___712255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 21), gc_state_call_result_712243, '__exit__')
        with_exit_712256 = invoke(stypy.reporting.localization.Localization(__file__, 36, 21), exit___712255, None, None, None)

    
    # Call to assert_equal(...): (line 39)
    # Processing the call arguments (line 39)
    
    # Call to isenabled(...): (line 39)
    # Processing the call keyword arguments (line 39)
    kwargs_712260 = {}
    # Getting the type of 'gc' (line 39)
    gc_712258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 29), 'gc', False)
    # Obtaining the member 'isenabled' of a type (line 39)
    isenabled_712259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 29), gc_712258, 'isenabled')
    # Calling isenabled(args, kwargs) (line 39)
    isenabled_call_result_712261 = invoke(stypy.reporting.localization.Localization(__file__, 39, 29), isenabled_712259, *[], **kwargs_712260)
    
    # Getting the type of 'pre_state' (line 39)
    pre_state_712262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 45), 'pre_state', False)
    # Processing the call keyword arguments (line 39)
    kwargs_712263 = {}
    # Getting the type of 'assert_equal' (line 39)
    assert_equal_712257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 39)
    assert_equal_call_result_712264 = invoke(stypy.reporting.localization.Localization(__file__, 39, 16), assert_equal_712257, *[isenabled_call_result_712261, pre_state_712262], **kwargs_712263)
    
    
    # Call to gc_state(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'with_state' (line 41)
    with_state_712266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 30), 'with_state', False)
    # Processing the call keyword arguments (line 41)
    kwargs_712267 = {}
    # Getting the type of 'gc_state' (line 41)
    gc_state_712265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 21), 'gc_state', False)
    # Calling gc_state(args, kwargs) (line 41)
    gc_state_call_result_712268 = invoke(stypy.reporting.localization.Localization(__file__, 41, 21), gc_state_712265, *[with_state_712266], **kwargs_712267)
    
    with_712269 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 41, 21), gc_state_call_result_712268, 'with parameter', '__enter__', '__exit__')

    if with_712269:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 41)
        enter___712270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 21), gc_state_call_result_712268, '__enter__')
        with_enter_712271 = invoke(stypy.reporting.localization.Localization(__file__, 41, 21), enter___712270)
        
        # Call to assert_equal(...): (line 42)
        # Processing the call arguments (line 42)
        
        # Call to isenabled(...): (line 42)
        # Processing the call keyword arguments (line 42)
        kwargs_712275 = {}
        # Getting the type of 'gc' (line 42)
        gc_712273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 33), 'gc', False)
        # Obtaining the member 'isenabled' of a type (line 42)
        isenabled_712274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 33), gc_712273, 'isenabled')
        # Calling isenabled(args, kwargs) (line 42)
        isenabled_call_result_712276 = invoke(stypy.reporting.localization.Localization(__file__, 42, 33), isenabled_712274, *[], **kwargs_712275)
        
        # Getting the type of 'with_state' (line 42)
        with_state_712277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 49), 'with_state', False)
        # Processing the call keyword arguments (line 42)
        kwargs_712278 = {}
        # Getting the type of 'assert_equal' (line 42)
        assert_equal_712272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 20), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 42)
        assert_equal_call_result_712279 = invoke(stypy.reporting.localization.Localization(__file__, 42, 20), assert_equal_712272, *[isenabled_call_result_712276, with_state_712277], **kwargs_712278)
        
        
        # Call to set_gc_state(...): (line 43)
        # Processing the call arguments (line 43)
        
        # Getting the type of 'with_state' (line 43)
        with_state_712281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 37), 'with_state', False)
        # Applying the 'not' unary operator (line 43)
        result_not__712282 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 33), 'not', with_state_712281)
        
        # Processing the call keyword arguments (line 43)
        kwargs_712283 = {}
        # Getting the type of 'set_gc_state' (line 43)
        set_gc_state_712280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 20), 'set_gc_state', False)
        # Calling set_gc_state(args, kwargs) (line 43)
        set_gc_state_call_result_712284 = invoke(stypy.reporting.localization.Localization(__file__, 43, 20), set_gc_state_712280, *[result_not__712282], **kwargs_712283)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 41)
        exit___712285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 21), gc_state_call_result_712268, '__exit__')
        with_exit_712286 = invoke(stypy.reporting.localization.Localization(__file__, 41, 21), exit___712285, None, None, None)

    
    # Call to assert_equal(...): (line 44)
    # Processing the call arguments (line 44)
    
    # Call to isenabled(...): (line 44)
    # Processing the call keyword arguments (line 44)
    kwargs_712290 = {}
    # Getting the type of 'gc' (line 44)
    gc_712288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 29), 'gc', False)
    # Obtaining the member 'isenabled' of a type (line 44)
    isenabled_712289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 29), gc_712288, 'isenabled')
    # Calling isenabled(args, kwargs) (line 44)
    isenabled_call_result_712291 = invoke(stypy.reporting.localization.Localization(__file__, 44, 29), isenabled_712289, *[], **kwargs_712290)
    
    # Getting the type of 'pre_state' (line 44)
    pre_state_712292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 45), 'pre_state', False)
    # Processing the call keyword arguments (line 44)
    kwargs_712293 = {}
    # Getting the type of 'assert_equal' (line 44)
    assert_equal_712287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 44)
    assert_equal_call_result_712294 = invoke(stypy.reporting.localization.Localization(__file__, 44, 16), assert_equal_712287, *[isenabled_call_result_712291, pre_state_712292], **kwargs_712293)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # finally branch of the try-finally block (line 31)
    
    # Getting the type of 'gc_status' (line 46)
    gc_status_712295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 11), 'gc_status')
    # Testing the type of an if condition (line 46)
    if_condition_712296 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 46, 8), gc_status_712295)
    # Assigning a type to the variable 'if_condition_712296' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'if_condition_712296', if_condition_712296)
    # SSA begins for if statement (line 46)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to enable(...): (line 47)
    # Processing the call keyword arguments (line 47)
    kwargs_712299 = {}
    # Getting the type of 'gc' (line 47)
    gc_712297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'gc', False)
    # Obtaining the member 'enable' of a type (line 47)
    enable_712298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), gc_712297, 'enable')
    # Calling enable(args, kwargs) (line 47)
    enable_call_result_712300 = invoke(stypy.reporting.localization.Localization(__file__, 47, 12), enable_712298, *[], **kwargs_712299)
    
    # SSA join for if statement (line 46)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # ################# End of 'test_gc_state(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_gc_state' in the type store
    # Getting the type of 'stypy_return_type' (line 28)
    stypy_return_type_712301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_712301)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_gc_state'
    return stypy_return_type_712301

# Assigning a type to the variable 'test_gc_state' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'test_gc_state', test_gc_state)

@norecursion
def test_assert_deallocated(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_assert_deallocated'
    module_type_store = module_type_store.open_function_context('test_assert_deallocated', 50, 0, False)
    
    # Passed parameters checking function
    test_assert_deallocated.stypy_localization = localization
    test_assert_deallocated.stypy_type_of_self = None
    test_assert_deallocated.stypy_type_store = module_type_store
    test_assert_deallocated.stypy_function_name = 'test_assert_deallocated'
    test_assert_deallocated.stypy_param_names_list = []
    test_assert_deallocated.stypy_varargs_param_name = None
    test_assert_deallocated.stypy_kwargs_param_name = None
    test_assert_deallocated.stypy_call_defaults = defaults
    test_assert_deallocated.stypy_call_varargs = varargs
    test_assert_deallocated.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_assert_deallocated', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_assert_deallocated', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_assert_deallocated(...)' code ##################

    # Declaration of the 'C' class

    class C(object, ):

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            str_712302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 44), 'str', 'myname')
            defaults = [str_712302]
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 53, 8, False)
            # Assigning a type to the variable 'self' (line 54)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'C.__init__', ['arg0', 'arg1', 'name'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return

            # Initialize method data
            init_call_information(module_type_store, '__init__', localization, ['arg0', 'arg1', 'name'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__init__(...)' code ##################

            
            # Assigning a Name to a Attribute (line 54):
            # Getting the type of 'name' (line 54)
            name_712303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'name')
            # Getting the type of 'self' (line 54)
            self_712304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'self')
            # Setting the type of the member 'name' of a type (line 54)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), self_712304, 'name', name_712303)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()

    
    # Assigning a type to the variable 'C' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'C', C)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 55)
    tuple_712305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 55)
    # Adding element type (line 55)
    # Getting the type of 'True' (line 55)
    True_712306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 23), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 23), tuple_712305, True_712306)
    # Adding element type (line 55)
    # Getting the type of 'False' (line 55)
    False_712307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 29), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 23), tuple_712305, False_712307)
    
    # Testing the type of a for loop iterable (line 55)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 55, 4), tuple_712305)
    # Getting the type of the for loop variable (line 55)
    for_loop_var_712308 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 55, 4), tuple_712305)
    # Assigning a type to the variable 'gc_current' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'gc_current', for_loop_var_712308)
    # SSA begins for a for statement (line 55)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to gc_state(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'gc_current' (line 56)
    gc_current_712310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 22), 'gc_current', False)
    # Processing the call keyword arguments (line 56)
    kwargs_712311 = {}
    # Getting the type of 'gc_state' (line 56)
    gc_state_712309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 13), 'gc_state', False)
    # Calling gc_state(args, kwargs) (line 56)
    gc_state_call_result_712312 = invoke(stypy.reporting.localization.Localization(__file__, 56, 13), gc_state_712309, *[gc_current_712310], **kwargs_712311)
    
    with_712313 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 56, 13), gc_state_call_result_712312, 'with parameter', '__enter__', '__exit__')

    if with_712313:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 56)
        enter___712314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 13), gc_state_call_result_712312, '__enter__')
        with_enter_712315 = invoke(stypy.reporting.localization.Localization(__file__, 56, 13), enter___712314)
        
        # Call to assert_deallocated(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'C' (line 58)
        C_712317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 36), 'C', False)
        int_712318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 39), 'int')
        int_712319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 42), 'int')
        str_712320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 45), 'str', 'another name')
        # Processing the call keyword arguments (line 58)
        kwargs_712321 = {}
        # Getting the type of 'assert_deallocated' (line 58)
        assert_deallocated_712316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 17), 'assert_deallocated', False)
        # Calling assert_deallocated(args, kwargs) (line 58)
        assert_deallocated_call_result_712322 = invoke(stypy.reporting.localization.Localization(__file__, 58, 17), assert_deallocated_712316, *[C_712317, int_712318, int_712319, str_712320], **kwargs_712321)
        
        with_712323 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 58, 17), assert_deallocated_call_result_712322, 'with parameter', '__enter__', '__exit__')

        if with_712323:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 58)
            enter___712324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 17), assert_deallocated_call_result_712322, '__enter__')
            with_enter_712325 = invoke(stypy.reporting.localization.Localization(__file__, 58, 17), enter___712324)
            # Assigning a type to the variable 'c' (line 58)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 17), 'c', with_enter_712325)
            
            # Call to assert_equal(...): (line 59)
            # Processing the call arguments (line 59)
            # Getting the type of 'c' (line 59)
            c_712327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 29), 'c', False)
            # Obtaining the member 'name' of a type (line 59)
            name_712328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 29), c_712327, 'name')
            str_712329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 37), 'str', 'another name')
            # Processing the call keyword arguments (line 59)
            kwargs_712330 = {}
            # Getting the type of 'assert_equal' (line 59)
            assert_equal_712326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'assert_equal', False)
            # Calling assert_equal(args, kwargs) (line 59)
            assert_equal_call_result_712331 = invoke(stypy.reporting.localization.Localization(__file__, 59, 16), assert_equal_712326, *[name_712328, str_712329], **kwargs_712330)
            
            # Deleting a member
            module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 60, 16), module_type_store, 'c')
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 58)
            exit___712332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 17), assert_deallocated_call_result_712322, '__exit__')
            with_exit_712333 = invoke(stypy.reporting.localization.Localization(__file__, 58, 17), exit___712332, None, None, None)

        
        # Call to assert_deallocated(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'C' (line 62)
        C_712335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 36), 'C', False)
        int_712336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 39), 'int')
        int_712337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 42), 'int')
        # Processing the call keyword arguments (line 62)
        str_712338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 50), 'str', 'third name')
        keyword_712339 = str_712338
        kwargs_712340 = {'name': keyword_712339}
        # Getting the type of 'assert_deallocated' (line 62)
        assert_deallocated_712334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 17), 'assert_deallocated', False)
        # Calling assert_deallocated(args, kwargs) (line 62)
        assert_deallocated_call_result_712341 = invoke(stypy.reporting.localization.Localization(__file__, 62, 17), assert_deallocated_712334, *[C_712335, int_712336, int_712337], **kwargs_712340)
        
        with_712342 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 62, 17), assert_deallocated_call_result_712341, 'with parameter', '__enter__', '__exit__')

        if with_712342:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 62)
            enter___712343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 17), assert_deallocated_call_result_712341, '__enter__')
            with_enter_712344 = invoke(stypy.reporting.localization.Localization(__file__, 62, 17), enter___712343)
            pass
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 62)
            exit___712345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 17), assert_deallocated_call_result_712341, '__exit__')
            with_exit_712346 = invoke(stypy.reporting.localization.Localization(__file__, 62, 17), exit___712345, None, None, None)

        
        # Call to assert_equal(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Call to isenabled(...): (line 64)
        # Processing the call keyword arguments (line 64)
        kwargs_712350 = {}
        # Getting the type of 'gc' (line 64)
        gc_712348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 25), 'gc', False)
        # Obtaining the member 'isenabled' of a type (line 64)
        isenabled_712349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 25), gc_712348, 'isenabled')
        # Calling isenabled(args, kwargs) (line 64)
        isenabled_call_result_712351 = invoke(stypy.reporting.localization.Localization(__file__, 64, 25), isenabled_712349, *[], **kwargs_712350)
        
        # Getting the type of 'gc_current' (line 64)
        gc_current_712352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 41), 'gc_current', False)
        # Processing the call keyword arguments (line 64)
        kwargs_712353 = {}
        # Getting the type of 'assert_equal' (line 64)
        assert_equal_712347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 64)
        assert_equal_call_result_712354 = invoke(stypy.reporting.localization.Localization(__file__, 64, 12), assert_equal_712347, *[isenabled_call_result_712351, gc_current_712352], **kwargs_712353)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 56)
        exit___712355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 13), gc_state_call_result_712312, '__exit__')
        with_exit_712356 = invoke(stypy.reporting.localization.Localization(__file__, 56, 13), exit___712355, None, None, None)

    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_assert_deallocated(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_assert_deallocated' in the type store
    # Getting the type of 'stypy_return_type' (line 50)
    stypy_return_type_712357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_712357)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_assert_deallocated'
    return stypy_return_type_712357

# Assigning a type to the variable 'test_assert_deallocated' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'test_assert_deallocated', test_assert_deallocated)

@norecursion
def test_assert_deallocated_nodel(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_assert_deallocated_nodel'
    module_type_store = module_type_store.open_function_context('test_assert_deallocated_nodel', 67, 0, False)
    
    # Passed parameters checking function
    test_assert_deallocated_nodel.stypy_localization = localization
    test_assert_deallocated_nodel.stypy_type_of_self = None
    test_assert_deallocated_nodel.stypy_type_store = module_type_store
    test_assert_deallocated_nodel.stypy_function_name = 'test_assert_deallocated_nodel'
    test_assert_deallocated_nodel.stypy_param_names_list = []
    test_assert_deallocated_nodel.stypy_varargs_param_name = None
    test_assert_deallocated_nodel.stypy_kwargs_param_name = None
    test_assert_deallocated_nodel.stypy_call_defaults = defaults
    test_assert_deallocated_nodel.stypy_call_varargs = varargs
    test_assert_deallocated_nodel.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_assert_deallocated_nodel', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_assert_deallocated_nodel', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_assert_deallocated_nodel(...)' code ##################

    # Declaration of the 'C' class

    class C(object, ):
        pass

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 68, 4, False)
            # Assigning a type to the variable 'self' (line 69)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'C.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'C' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'C', C)
    
    # Call to raises(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'ReferenceError' (line 70)
    ReferenceError_712360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 23), 'ReferenceError', False)
    # Processing the call keyword arguments (line 70)
    kwargs_712361 = {}
    # Getting the type of 'pytest' (line 70)
    pytest_712358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 9), 'pytest', False)
    # Obtaining the member 'raises' of a type (line 70)
    raises_712359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 9), pytest_712358, 'raises')
    # Calling raises(args, kwargs) (line 70)
    raises_call_result_712362 = invoke(stypy.reporting.localization.Localization(__file__, 70, 9), raises_712359, *[ReferenceError_712360], **kwargs_712361)
    
    with_712363 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 70, 9), raises_call_result_712362, 'with parameter', '__enter__', '__exit__')

    if with_712363:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 70)
        enter___712364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 9), raises_call_result_712362, '__enter__')
        with_enter_712365 = invoke(stypy.reporting.localization.Localization(__file__, 70, 9), enter___712364)
        
        # Call to assert_deallocated(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'C' (line 72)
        C_712367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 32), 'C', False)
        # Processing the call keyword arguments (line 72)
        kwargs_712368 = {}
        # Getting the type of 'assert_deallocated' (line 72)
        assert_deallocated_712366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'assert_deallocated', False)
        # Calling assert_deallocated(args, kwargs) (line 72)
        assert_deallocated_call_result_712369 = invoke(stypy.reporting.localization.Localization(__file__, 72, 13), assert_deallocated_712366, *[C_712367], **kwargs_712368)
        
        with_712370 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 72, 13), assert_deallocated_call_result_712369, 'with parameter', '__enter__', '__exit__')

        if with_712370:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 72)
            enter___712371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 13), assert_deallocated_call_result_712369, '__enter__')
            with_enter_712372 = invoke(stypy.reporting.localization.Localization(__file__, 72, 13), enter___712371)
            # Assigning a type to the variable 'c' (line 72)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'c', with_enter_712372)
            pass
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 72)
            exit___712373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 13), assert_deallocated_call_result_712369, '__exit__')
            with_exit_712374 = invoke(stypy.reporting.localization.Localization(__file__, 72, 13), exit___712373, None, None, None)

        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 70)
        exit___712375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 9), raises_call_result_712362, '__exit__')
        with_exit_712376 = invoke(stypy.reporting.localization.Localization(__file__, 70, 9), exit___712375, None, None, None)

    
    # ################# End of 'test_assert_deallocated_nodel(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_assert_deallocated_nodel' in the type store
    # Getting the type of 'stypy_return_type' (line 67)
    stypy_return_type_712377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_712377)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_assert_deallocated_nodel'
    return stypy_return_type_712377

# Assigning a type to the variable 'test_assert_deallocated_nodel' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'test_assert_deallocated_nodel', test_assert_deallocated_nodel)

@norecursion
def test_assert_deallocated_circular(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_assert_deallocated_circular'
    module_type_store = module_type_store.open_function_context('test_assert_deallocated_circular', 76, 0, False)
    
    # Passed parameters checking function
    test_assert_deallocated_circular.stypy_localization = localization
    test_assert_deallocated_circular.stypy_type_of_self = None
    test_assert_deallocated_circular.stypy_type_store = module_type_store
    test_assert_deallocated_circular.stypy_function_name = 'test_assert_deallocated_circular'
    test_assert_deallocated_circular.stypy_param_names_list = []
    test_assert_deallocated_circular.stypy_varargs_param_name = None
    test_assert_deallocated_circular.stypy_kwargs_param_name = None
    test_assert_deallocated_circular.stypy_call_defaults = defaults
    test_assert_deallocated_circular.stypy_call_varargs = varargs
    test_assert_deallocated_circular.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_assert_deallocated_circular', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_assert_deallocated_circular', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_assert_deallocated_circular(...)' code ##################

    # Declaration of the 'C' class

    class C(object, ):

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 78, 8, False)
            # Assigning a type to the variable 'self' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'C.__init__', [], None, None, defaults, varargs, kwargs)

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

            
            # Assigning a Name to a Attribute (line 79):
            # Getting the type of 'self' (line 79)
            self_712378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 29), 'self')
            # Getting the type of 'self' (line 79)
            self_712379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'self')
            # Setting the type of the member '_circular' of a type (line 79)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), self_712379, '_circular', self_712378)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()

    
    # Assigning a type to the variable 'C' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'C', C)
    
    # Call to raises(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'ReferenceError' (line 80)
    ReferenceError_712382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 23), 'ReferenceError', False)
    # Processing the call keyword arguments (line 80)
    kwargs_712383 = {}
    # Getting the type of 'pytest' (line 80)
    pytest_712380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 9), 'pytest', False)
    # Obtaining the member 'raises' of a type (line 80)
    raises_712381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 9), pytest_712380, 'raises')
    # Calling raises(args, kwargs) (line 80)
    raises_call_result_712384 = invoke(stypy.reporting.localization.Localization(__file__, 80, 9), raises_712381, *[ReferenceError_712382], **kwargs_712383)
    
    with_712385 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 80, 9), raises_call_result_712384, 'with parameter', '__enter__', '__exit__')

    if with_712385:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 80)
        enter___712386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 9), raises_call_result_712384, '__enter__')
        with_enter_712387 = invoke(stypy.reporting.localization.Localization(__file__, 80, 9), enter___712386)
        
        # Call to assert_deallocated(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'C' (line 82)
        C_712389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 32), 'C', False)
        # Processing the call keyword arguments (line 82)
        kwargs_712390 = {}
        # Getting the type of 'assert_deallocated' (line 82)
        assert_deallocated_712388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 13), 'assert_deallocated', False)
        # Calling assert_deallocated(args, kwargs) (line 82)
        assert_deallocated_call_result_712391 = invoke(stypy.reporting.localization.Localization(__file__, 82, 13), assert_deallocated_712388, *[C_712389], **kwargs_712390)
        
        with_712392 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 82, 13), assert_deallocated_call_result_712391, 'with parameter', '__enter__', '__exit__')

        if with_712392:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 82)
            enter___712393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 13), assert_deallocated_call_result_712391, '__enter__')
            with_enter_712394 = invoke(stypy.reporting.localization.Localization(__file__, 82, 13), enter___712393)
            # Assigning a type to the variable 'c' (line 82)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 13), 'c', with_enter_712394)
            # Deleting a member
            module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 83, 12), module_type_store, 'c')
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 82)
            exit___712395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 13), assert_deallocated_call_result_712391, '__exit__')
            with_exit_712396 = invoke(stypy.reporting.localization.Localization(__file__, 82, 13), exit___712395, None, None, None)

        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 80)
        exit___712397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 9), raises_call_result_712384, '__exit__')
        with_exit_712398 = invoke(stypy.reporting.localization.Localization(__file__, 80, 9), exit___712397, None, None, None)

    
    # ################# End of 'test_assert_deallocated_circular(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_assert_deallocated_circular' in the type store
    # Getting the type of 'stypy_return_type' (line 76)
    stypy_return_type_712399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_712399)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_assert_deallocated_circular'
    return stypy_return_type_712399

# Assigning a type to the variable 'test_assert_deallocated_circular' (line 76)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'test_assert_deallocated_circular', test_assert_deallocated_circular)

@norecursion
def test_assert_deallocated_circular2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_assert_deallocated_circular2'
    module_type_store = module_type_store.open_function_context('test_assert_deallocated_circular2', 86, 0, False)
    
    # Passed parameters checking function
    test_assert_deallocated_circular2.stypy_localization = localization
    test_assert_deallocated_circular2.stypy_type_of_self = None
    test_assert_deallocated_circular2.stypy_type_store = module_type_store
    test_assert_deallocated_circular2.stypy_function_name = 'test_assert_deallocated_circular2'
    test_assert_deallocated_circular2.stypy_param_names_list = []
    test_assert_deallocated_circular2.stypy_varargs_param_name = None
    test_assert_deallocated_circular2.stypy_kwargs_param_name = None
    test_assert_deallocated_circular2.stypy_call_defaults = defaults
    test_assert_deallocated_circular2.stypy_call_varargs = varargs
    test_assert_deallocated_circular2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_assert_deallocated_circular2', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_assert_deallocated_circular2', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_assert_deallocated_circular2(...)' code ##################

    # Declaration of the 'C' class

    class C(object, ):

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 88, 8, False)
            # Assigning a type to the variable 'self' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'C.__init__', [], None, None, defaults, varargs, kwargs)

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

            
            # Assigning a Name to a Attribute (line 89):
            # Getting the type of 'self' (line 89)
            self_712400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 29), 'self')
            # Getting the type of 'self' (line 89)
            self_712401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'self')
            # Setting the type of the member '_circular' of a type (line 89)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 12), self_712401, '_circular', self_712400)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()

    
    # Assigning a type to the variable 'C' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'C', C)
    
    # Call to raises(...): (line 90)
    # Processing the call arguments (line 90)
    # Getting the type of 'ReferenceError' (line 90)
    ReferenceError_712404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 23), 'ReferenceError', False)
    # Processing the call keyword arguments (line 90)
    kwargs_712405 = {}
    # Getting the type of 'pytest' (line 90)
    pytest_712402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 9), 'pytest', False)
    # Obtaining the member 'raises' of a type (line 90)
    raises_712403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 9), pytest_712402, 'raises')
    # Calling raises(args, kwargs) (line 90)
    raises_call_result_712406 = invoke(stypy.reporting.localization.Localization(__file__, 90, 9), raises_712403, *[ReferenceError_712404], **kwargs_712405)
    
    with_712407 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 90, 9), raises_call_result_712406, 'with parameter', '__enter__', '__exit__')

    if with_712407:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 90)
        enter___712408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 9), raises_call_result_712406, '__enter__')
        with_enter_712409 = invoke(stypy.reporting.localization.Localization(__file__, 90, 9), enter___712408)
        
        # Call to assert_deallocated(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'C' (line 92)
        C_712411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 32), 'C', False)
        # Processing the call keyword arguments (line 92)
        kwargs_712412 = {}
        # Getting the type of 'assert_deallocated' (line 92)
        assert_deallocated_712410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 13), 'assert_deallocated', False)
        # Calling assert_deallocated(args, kwargs) (line 92)
        assert_deallocated_call_result_712413 = invoke(stypy.reporting.localization.Localization(__file__, 92, 13), assert_deallocated_712410, *[C_712411], **kwargs_712412)
        
        with_712414 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 92, 13), assert_deallocated_call_result_712413, 'with parameter', '__enter__', '__exit__')

        if with_712414:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 92)
            enter___712415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 13), assert_deallocated_call_result_712413, '__enter__')
            with_enter_712416 = invoke(stypy.reporting.localization.Localization(__file__, 92, 13), enter___712415)
            pass
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 92)
            exit___712417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 13), assert_deallocated_call_result_712413, '__exit__')
            with_exit_712418 = invoke(stypy.reporting.localization.Localization(__file__, 92, 13), exit___712417, None, None, None)

        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 90)
        exit___712419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 9), raises_call_result_712406, '__exit__')
        with_exit_712420 = invoke(stypy.reporting.localization.Localization(__file__, 90, 9), exit___712419, None, None, None)

    
    # ################# End of 'test_assert_deallocated_circular2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_assert_deallocated_circular2' in the type store
    # Getting the type of 'stypy_return_type' (line 86)
    stypy_return_type_712421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_712421)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_assert_deallocated_circular2'
    return stypy_return_type_712421

# Assigning a type to the variable 'test_assert_deallocated_circular2' (line 86)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'test_assert_deallocated_circular2', test_assert_deallocated_circular2)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import pytest
4: from numpy.testing import assert_array_equal
5: from scipy._lib._numpy_compat import suppress_warnings
6: import scipy.ndimage as ndi
7: 
8: import os
9: 
10: try:
11:     from PIL import Image
12:     pil_missing = False
13: except ImportError:
14:     pil_missing = True
15: 
16: 
17: @pytest.mark.skipif(pil_missing, reason="The Python Image Library could not be found.")
18: def test_imread():
19:     lp = os.path.join(os.path.dirname(__file__), 'dots.png')
20:     with suppress_warnings() as sup:
21:         # PIL causes a Py3k ResourceWarning
22:         sup.filter(message="unclosed file")
23:         sup.filter(DeprecationWarning)
24:         img = ndi.imread(lp, mode="RGB")
25:     assert_array_equal(img.shape, (300, 420, 3))
26: 
27:     with suppress_warnings() as sup:
28:         # PIL causes a Py3k ResourceWarning
29:         sup.filter(message="unclosed file")
30:         sup.filter(DeprecationWarning)
31:         img = ndi.imread(lp, flatten=True)
32:     assert_array_equal(img.shape, (300, 420))
33: 
34:     with open(lp, 'rb') as fobj:
35:         with suppress_warnings() as sup:
36:             sup.filter(DeprecationWarning)
37:             img = ndi.imread(fobj, mode="RGB")
38:         assert_array_equal(img.shape, (300, 420, 3))
39: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import pytest' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/tests/')
import_129468 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'pytest')

if (type(import_129468) is not StypyTypeError):

    if (import_129468 != 'pyd_module'):
        __import__(import_129468)
        sys_modules_129469 = sys.modules[import_129468]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'pytest', sys_modules_129469.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'pytest', import_129468)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_array_equal' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/tests/')
import_129470 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_129470) is not StypyTypeError):

    if (import_129470 != 'pyd_module'):
        __import__(import_129470)
        sys_modules_129471 = sys.modules[import_129470]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_129471.module_type_store, module_type_store, ['assert_array_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_129471, sys_modules_129471.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_array_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_array_equal'], [assert_array_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_129470)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy._lib._numpy_compat import suppress_warnings' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/tests/')
import_129472 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib._numpy_compat')

if (type(import_129472) is not StypyTypeError):

    if (import_129472 != 'pyd_module'):
        __import__(import_129472)
        sys_modules_129473 = sys.modules[import_129472]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib._numpy_compat', sys_modules_129473.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_129473, sys_modules_129473.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib._numpy_compat', import_129472)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import scipy.ndimage' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/tests/')
import_129474 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.ndimage')

if (type(import_129474) is not StypyTypeError):

    if (import_129474 != 'pyd_module'):
        __import__(import_129474)
        sys_modules_129475 = sys.modules[import_129474]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'ndi', sys_modules_129475.module_type_store, module_type_store)
    else:
        import scipy.ndimage as ndi

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'ndi', scipy.ndimage, module_type_store)

else:
    # Assigning a type to the variable 'scipy.ndimage' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.ndimage', import_129474)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import os' statement (line 8)
import os

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'os', os, module_type_store)



# SSA begins for try-except statement (line 10)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 4))

# 'from PIL import Image' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/tests/')
import_129476 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'PIL')

if (type(import_129476) is not StypyTypeError):

    if (import_129476 != 'pyd_module'):
        __import__(import_129476)
        sys_modules_129477 = sys.modules[import_129476]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'PIL', sys_modules_129477.module_type_store, module_type_store, ['Image'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 4), __file__, sys_modules_129477, sys_modules_129477.module_type_store, module_type_store)
    else:
        from PIL import Image

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'PIL', None, module_type_store, ['Image'], [Image])

else:
    # Assigning a type to the variable 'PIL' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'PIL', import_129476)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/tests/')


# Assigning a Name to a Name (line 12):
# Getting the type of 'False' (line 12)
False_129478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 18), 'False')
# Assigning a type to the variable 'pil_missing' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'pil_missing', False_129478)
# SSA branch for the except part of a try statement (line 10)
# SSA branch for the except 'ImportError' branch of a try statement (line 10)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 14):
# Getting the type of 'True' (line 14)
True_129479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 18), 'True')
# Assigning a type to the variable 'pil_missing' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'pil_missing', True_129479)
# SSA join for try-except statement (line 10)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def test_imread(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_imread'
    module_type_store = module_type_store.open_function_context('test_imread', 17, 0, False)
    
    # Passed parameters checking function
    test_imread.stypy_localization = localization
    test_imread.stypy_type_of_self = None
    test_imread.stypy_type_store = module_type_store
    test_imread.stypy_function_name = 'test_imread'
    test_imread.stypy_param_names_list = []
    test_imread.stypy_varargs_param_name = None
    test_imread.stypy_kwargs_param_name = None
    test_imread.stypy_call_defaults = defaults
    test_imread.stypy_call_varargs = varargs
    test_imread.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_imread', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_imread', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_imread(...)' code ##################

    
    # Assigning a Call to a Name (line 19):
    
    # Call to join(...): (line 19)
    # Processing the call arguments (line 19)
    
    # Call to dirname(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of '__file__' (line 19)
    file___129486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 38), '__file__', False)
    # Processing the call keyword arguments (line 19)
    kwargs_129487 = {}
    # Getting the type of 'os' (line 19)
    os_129483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 22), 'os', False)
    # Obtaining the member 'path' of a type (line 19)
    path_129484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 22), os_129483, 'path')
    # Obtaining the member 'dirname' of a type (line 19)
    dirname_129485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 22), path_129484, 'dirname')
    # Calling dirname(args, kwargs) (line 19)
    dirname_call_result_129488 = invoke(stypy.reporting.localization.Localization(__file__, 19, 22), dirname_129485, *[file___129486], **kwargs_129487)
    
    str_129489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 49), 'str', 'dots.png')
    # Processing the call keyword arguments (line 19)
    kwargs_129490 = {}
    # Getting the type of 'os' (line 19)
    os_129480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 9), 'os', False)
    # Obtaining the member 'path' of a type (line 19)
    path_129481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 9), os_129480, 'path')
    # Obtaining the member 'join' of a type (line 19)
    join_129482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 9), path_129481, 'join')
    # Calling join(args, kwargs) (line 19)
    join_call_result_129491 = invoke(stypy.reporting.localization.Localization(__file__, 19, 9), join_129482, *[dirname_call_result_129488, str_129489], **kwargs_129490)
    
    # Assigning a type to the variable 'lp' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'lp', join_call_result_129491)
    
    # Call to suppress_warnings(...): (line 20)
    # Processing the call keyword arguments (line 20)
    kwargs_129493 = {}
    # Getting the type of 'suppress_warnings' (line 20)
    suppress_warnings_129492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 9), 'suppress_warnings', False)
    # Calling suppress_warnings(args, kwargs) (line 20)
    suppress_warnings_call_result_129494 = invoke(stypy.reporting.localization.Localization(__file__, 20, 9), suppress_warnings_129492, *[], **kwargs_129493)
    
    with_129495 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 20, 9), suppress_warnings_call_result_129494, 'with parameter', '__enter__', '__exit__')

    if with_129495:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 20)
        enter___129496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 9), suppress_warnings_call_result_129494, '__enter__')
        with_enter_129497 = invoke(stypy.reporting.localization.Localization(__file__, 20, 9), enter___129496)
        # Assigning a type to the variable 'sup' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 9), 'sup', with_enter_129497)
        
        # Call to filter(...): (line 22)
        # Processing the call keyword arguments (line 22)
        str_129500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 27), 'str', 'unclosed file')
        keyword_129501 = str_129500
        kwargs_129502 = {'message': keyword_129501}
        # Getting the type of 'sup' (line 22)
        sup_129498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'sup', False)
        # Obtaining the member 'filter' of a type (line 22)
        filter_129499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), sup_129498, 'filter')
        # Calling filter(args, kwargs) (line 22)
        filter_call_result_129503 = invoke(stypy.reporting.localization.Localization(__file__, 22, 8), filter_129499, *[], **kwargs_129502)
        
        
        # Call to filter(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'DeprecationWarning' (line 23)
        DeprecationWarning_129506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 19), 'DeprecationWarning', False)
        # Processing the call keyword arguments (line 23)
        kwargs_129507 = {}
        # Getting the type of 'sup' (line 23)
        sup_129504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'sup', False)
        # Obtaining the member 'filter' of a type (line 23)
        filter_129505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), sup_129504, 'filter')
        # Calling filter(args, kwargs) (line 23)
        filter_call_result_129508 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), filter_129505, *[DeprecationWarning_129506], **kwargs_129507)
        
        
        # Assigning a Call to a Name (line 24):
        
        # Call to imread(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'lp' (line 24)
        lp_129511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 25), 'lp', False)
        # Processing the call keyword arguments (line 24)
        str_129512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 34), 'str', 'RGB')
        keyword_129513 = str_129512
        kwargs_129514 = {'mode': keyword_129513}
        # Getting the type of 'ndi' (line 24)
        ndi_129509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 14), 'ndi', False)
        # Obtaining the member 'imread' of a type (line 24)
        imread_129510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 14), ndi_129509, 'imread')
        # Calling imread(args, kwargs) (line 24)
        imread_call_result_129515 = invoke(stypy.reporting.localization.Localization(__file__, 24, 14), imread_129510, *[lp_129511], **kwargs_129514)
        
        # Assigning a type to the variable 'img' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'img', imread_call_result_129515)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 20)
        exit___129516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 9), suppress_warnings_call_result_129494, '__exit__')
        with_exit_129517 = invoke(stypy.reporting.localization.Localization(__file__, 20, 9), exit___129516, None, None, None)

    
    # Call to assert_array_equal(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'img' (line 25)
    img_129519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 23), 'img', False)
    # Obtaining the member 'shape' of a type (line 25)
    shape_129520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 23), img_129519, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 25)
    tuple_129521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 25)
    # Adding element type (line 25)
    int_129522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 35), tuple_129521, int_129522)
    # Adding element type (line 25)
    int_129523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 35), tuple_129521, int_129523)
    # Adding element type (line 25)
    int_129524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 35), tuple_129521, int_129524)
    
    # Processing the call keyword arguments (line 25)
    kwargs_129525 = {}
    # Getting the type of 'assert_array_equal' (line 25)
    assert_array_equal_129518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 25)
    assert_array_equal_call_result_129526 = invoke(stypy.reporting.localization.Localization(__file__, 25, 4), assert_array_equal_129518, *[shape_129520, tuple_129521], **kwargs_129525)
    
    
    # Call to suppress_warnings(...): (line 27)
    # Processing the call keyword arguments (line 27)
    kwargs_129528 = {}
    # Getting the type of 'suppress_warnings' (line 27)
    suppress_warnings_129527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 9), 'suppress_warnings', False)
    # Calling suppress_warnings(args, kwargs) (line 27)
    suppress_warnings_call_result_129529 = invoke(stypy.reporting.localization.Localization(__file__, 27, 9), suppress_warnings_129527, *[], **kwargs_129528)
    
    with_129530 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 27, 9), suppress_warnings_call_result_129529, 'with parameter', '__enter__', '__exit__')

    if with_129530:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 27)
        enter___129531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 9), suppress_warnings_call_result_129529, '__enter__')
        with_enter_129532 = invoke(stypy.reporting.localization.Localization(__file__, 27, 9), enter___129531)
        # Assigning a type to the variable 'sup' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 9), 'sup', with_enter_129532)
        
        # Call to filter(...): (line 29)
        # Processing the call keyword arguments (line 29)
        str_129535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 27), 'str', 'unclosed file')
        keyword_129536 = str_129535
        kwargs_129537 = {'message': keyword_129536}
        # Getting the type of 'sup' (line 29)
        sup_129533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'sup', False)
        # Obtaining the member 'filter' of a type (line 29)
        filter_129534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), sup_129533, 'filter')
        # Calling filter(args, kwargs) (line 29)
        filter_call_result_129538 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), filter_129534, *[], **kwargs_129537)
        
        
        # Call to filter(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'DeprecationWarning' (line 30)
        DeprecationWarning_129541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 19), 'DeprecationWarning', False)
        # Processing the call keyword arguments (line 30)
        kwargs_129542 = {}
        # Getting the type of 'sup' (line 30)
        sup_129539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'sup', False)
        # Obtaining the member 'filter' of a type (line 30)
        filter_129540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), sup_129539, 'filter')
        # Calling filter(args, kwargs) (line 30)
        filter_call_result_129543 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), filter_129540, *[DeprecationWarning_129541], **kwargs_129542)
        
        
        # Assigning a Call to a Name (line 31):
        
        # Call to imread(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'lp' (line 31)
        lp_129546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 25), 'lp', False)
        # Processing the call keyword arguments (line 31)
        # Getting the type of 'True' (line 31)
        True_129547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 37), 'True', False)
        keyword_129548 = True_129547
        kwargs_129549 = {'flatten': keyword_129548}
        # Getting the type of 'ndi' (line 31)
        ndi_129544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'ndi', False)
        # Obtaining the member 'imread' of a type (line 31)
        imread_129545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 14), ndi_129544, 'imread')
        # Calling imread(args, kwargs) (line 31)
        imread_call_result_129550 = invoke(stypy.reporting.localization.Localization(__file__, 31, 14), imread_129545, *[lp_129546], **kwargs_129549)
        
        # Assigning a type to the variable 'img' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'img', imread_call_result_129550)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 27)
        exit___129551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 9), suppress_warnings_call_result_129529, '__exit__')
        with_exit_129552 = invoke(stypy.reporting.localization.Localization(__file__, 27, 9), exit___129551, None, None, None)

    
    # Call to assert_array_equal(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'img' (line 32)
    img_129554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 23), 'img', False)
    # Obtaining the member 'shape' of a type (line 32)
    shape_129555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 23), img_129554, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 32)
    tuple_129556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 32)
    # Adding element type (line 32)
    int_129557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 35), tuple_129556, int_129557)
    # Adding element type (line 32)
    int_129558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 35), tuple_129556, int_129558)
    
    # Processing the call keyword arguments (line 32)
    kwargs_129559 = {}
    # Getting the type of 'assert_array_equal' (line 32)
    assert_array_equal_129553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 32)
    assert_array_equal_call_result_129560 = invoke(stypy.reporting.localization.Localization(__file__, 32, 4), assert_array_equal_129553, *[shape_129555, tuple_129556], **kwargs_129559)
    
    
    # Call to open(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'lp' (line 34)
    lp_129562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 14), 'lp', False)
    str_129563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 18), 'str', 'rb')
    # Processing the call keyword arguments (line 34)
    kwargs_129564 = {}
    # Getting the type of 'open' (line 34)
    open_129561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 9), 'open', False)
    # Calling open(args, kwargs) (line 34)
    open_call_result_129565 = invoke(stypy.reporting.localization.Localization(__file__, 34, 9), open_129561, *[lp_129562, str_129563], **kwargs_129564)
    
    with_129566 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 34, 9), open_call_result_129565, 'with parameter', '__enter__', '__exit__')

    if with_129566:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 34)
        enter___129567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 9), open_call_result_129565, '__enter__')
        with_enter_129568 = invoke(stypy.reporting.localization.Localization(__file__, 34, 9), enter___129567)
        # Assigning a type to the variable 'fobj' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 9), 'fobj', with_enter_129568)
        
        # Call to suppress_warnings(...): (line 35)
        # Processing the call keyword arguments (line 35)
        kwargs_129570 = {}
        # Getting the type of 'suppress_warnings' (line 35)
        suppress_warnings_129569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 35)
        suppress_warnings_call_result_129571 = invoke(stypy.reporting.localization.Localization(__file__, 35, 13), suppress_warnings_129569, *[], **kwargs_129570)
        
        with_129572 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 35, 13), suppress_warnings_call_result_129571, 'with parameter', '__enter__', '__exit__')

        if with_129572:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 35)
            enter___129573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 13), suppress_warnings_call_result_129571, '__enter__')
            with_enter_129574 = invoke(stypy.reporting.localization.Localization(__file__, 35, 13), enter___129573)
            # Assigning a type to the variable 'sup' (line 35)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 13), 'sup', with_enter_129574)
            
            # Call to filter(...): (line 36)
            # Processing the call arguments (line 36)
            # Getting the type of 'DeprecationWarning' (line 36)
            DeprecationWarning_129577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 23), 'DeprecationWarning', False)
            # Processing the call keyword arguments (line 36)
            kwargs_129578 = {}
            # Getting the type of 'sup' (line 36)
            sup_129575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 36)
            filter_129576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), sup_129575, 'filter')
            # Calling filter(args, kwargs) (line 36)
            filter_call_result_129579 = invoke(stypy.reporting.localization.Localization(__file__, 36, 12), filter_129576, *[DeprecationWarning_129577], **kwargs_129578)
            
            
            # Assigning a Call to a Name (line 37):
            
            # Call to imread(...): (line 37)
            # Processing the call arguments (line 37)
            # Getting the type of 'fobj' (line 37)
            fobj_129582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 29), 'fobj', False)
            # Processing the call keyword arguments (line 37)
            str_129583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 40), 'str', 'RGB')
            keyword_129584 = str_129583
            kwargs_129585 = {'mode': keyword_129584}
            # Getting the type of 'ndi' (line 37)
            ndi_129580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 18), 'ndi', False)
            # Obtaining the member 'imread' of a type (line 37)
            imread_129581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 18), ndi_129580, 'imread')
            # Calling imread(args, kwargs) (line 37)
            imread_call_result_129586 = invoke(stypy.reporting.localization.Localization(__file__, 37, 18), imread_129581, *[fobj_129582], **kwargs_129585)
            
            # Assigning a type to the variable 'img' (line 37)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'img', imread_call_result_129586)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 35)
            exit___129587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 13), suppress_warnings_call_result_129571, '__exit__')
            with_exit_129588 = invoke(stypy.reporting.localization.Localization(__file__, 35, 13), exit___129587, None, None, None)

        
        # Call to assert_array_equal(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'img' (line 38)
        img_129590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 27), 'img', False)
        # Obtaining the member 'shape' of a type (line 38)
        shape_129591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 27), img_129590, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 38)
        tuple_129592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 38)
        # Adding element type (line 38)
        int_129593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 39), tuple_129592, int_129593)
        # Adding element type (line 38)
        int_129594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 39), tuple_129592, int_129594)
        # Adding element type (line 38)
        int_129595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 39), tuple_129592, int_129595)
        
        # Processing the call keyword arguments (line 38)
        kwargs_129596 = {}
        # Getting the type of 'assert_array_equal' (line 38)
        assert_array_equal_129589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 38)
        assert_array_equal_call_result_129597 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), assert_array_equal_129589, *[shape_129591, tuple_129592], **kwargs_129596)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 34)
        exit___129598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 9), open_call_result_129565, '__exit__')
        with_exit_129599 = invoke(stypy.reporting.localization.Localization(__file__, 34, 9), exit___129598, None, None, None)

    
    # ################# End of 'test_imread(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_imread' in the type store
    # Getting the type of 'stypy_return_type' (line 17)
    stypy_return_type_129600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_129600)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_imread'
    return stypy_return_type_129600

# Assigning a type to the variable 'test_imread' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'test_imread', test_imread)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import pytest
4: from numpy.testing import assert_, assert_array_equal
5: from scipy._lib._numpy_compat import suppress_warnings
6: 
7: try:
8:     import matplotlib
9:     matplotlib.rcParams['backend'] = 'Agg'
10:     import matplotlib.pyplot as plt
11:     from matplotlib.collections import LineCollection
12:     from matplotlib import MatplotlibDeprecationWarning
13:     has_matplotlib = True
14: except:
15:     has_matplotlib = False
16: 
17: from scipy.spatial import \
18:      delaunay_plot_2d, voronoi_plot_2d, convex_hull_plot_2d, \
19:      Delaunay, Voronoi, ConvexHull
20: 
21: 
22: @pytest.mark.skipif(not has_matplotlib, reason="Matplotlib not available")
23: class TestPlotting:
24:     points = [(0,0), (0,1), (1,0), (1,1)]
25: 
26:     def test_delaunay(self):
27:         # Smoke test
28:         fig = plt.figure()
29:         obj = Delaunay(self.points)
30:         s_before = obj.simplices.copy()
31:         with suppress_warnings as sup:
32:             # filter can be removed when matplotlib 1.x is dropped
33:             sup.filter(message="The ishold function was deprecated in version")
34:             r = delaunay_plot_2d(obj, ax=fig.gca())
35:         assert_array_equal(obj.simplices, s_before)  # shouldn't modify
36:         assert_(r is fig)
37:         delaunay_plot_2d(obj, ax=fig.gca())
38: 
39:     def test_voronoi(self):
40:         # Smoke test
41:         fig = plt.figure()
42:         obj = Voronoi(self.points)
43:         with suppress_warnings as sup:
44:             # filter can be removed when matplotlib 1.x is dropped
45:             sup.filter(message="The ishold function was deprecated in version")
46:             r = voronoi_plot_2d(obj, ax=fig.gca())
47:         assert_(r is fig)
48:         voronoi_plot_2d(obj)
49:         voronoi_plot_2d(obj, show_vertices=False)
50: 
51:     def test_convex_hull(self):
52:         # Smoke test
53:         fig = plt.figure()
54:         tri = ConvexHull(self.points)
55:         with suppress_warnings as sup:
56:             # filter can be removed when matplotlib 1.x is dropped
57:             sup.filter(message="The ishold function was deprecated in version")
58:             r = convex_hull_plot_2d(tri, ax=fig.gca())
59:         assert_(r is fig)
60:         convex_hull_plot_2d(tri)
61: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import pytest' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/tests/')
import_492388 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'pytest')

if (type(import_492388) is not StypyTypeError):

    if (import_492388 != 'pyd_module'):
        __import__(import_492388)
        sys_modules_492389 = sys.modules[import_492388]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'pytest', sys_modules_492389.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'pytest', import_492388)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_, assert_array_equal' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/tests/')
import_492390 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_492390) is not StypyTypeError):

    if (import_492390 != 'pyd_module'):
        __import__(import_492390)
        sys_modules_492391 = sys.modules[import_492390]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_492391.module_type_store, module_type_store, ['assert_', 'assert_array_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_492391, sys_modules_492391.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_array_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_array_equal'], [assert_, assert_array_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_492390)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy._lib._numpy_compat import suppress_warnings' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/tests/')
import_492392 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib._numpy_compat')

if (type(import_492392) is not StypyTypeError):

    if (import_492392 != 'pyd_module'):
        __import__(import_492392)
        sys_modules_492393 = sys.modules[import_492392]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib._numpy_compat', sys_modules_492393.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_492393, sys_modules_492393.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib._numpy_compat', import_492392)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/tests/')



# SSA begins for try-except statement (line 7)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 4))

# 'import matplotlib' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/tests/')
import_492394 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'matplotlib')

if (type(import_492394) is not StypyTypeError):

    if (import_492394 != 'pyd_module'):
        __import__(import_492394)
        sys_modules_492395 = sys.modules[import_492394]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'matplotlib', sys_modules_492395.module_type_store, module_type_store)
    else:
        import matplotlib

        import_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'matplotlib', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'matplotlib', import_492394)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/tests/')


# Assigning a Str to a Subscript (line 9):
str_492396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 37), 'str', 'Agg')
# Getting the type of 'matplotlib' (line 9)
matplotlib_492397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'matplotlib')
# Obtaining the member 'rcParams' of a type (line 9)
rcParams_492398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 4), matplotlib_492397, 'rcParams')
str_492399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 24), 'str', 'backend')
# Storing an element on a container (line 9)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 4), rcParams_492398, (str_492399, str_492396))
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 4))

# 'import matplotlib.pyplot' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/tests/')
import_492400 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'matplotlib.pyplot')

if (type(import_492400) is not StypyTypeError):

    if (import_492400 != 'pyd_module'):
        __import__(import_492400)
        sys_modules_492401 = sys.modules[import_492400]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'plt', sys_modules_492401.module_type_store, module_type_store)
    else:
        import matplotlib.pyplot as plt

        import_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'plt', matplotlib.pyplot, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.pyplot' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'matplotlib.pyplot', import_492400)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 4))

# 'from matplotlib.collections import LineCollection' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/tests/')
import_492402 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'matplotlib.collections')

if (type(import_492402) is not StypyTypeError):

    if (import_492402 != 'pyd_module'):
        __import__(import_492402)
        sys_modules_492403 = sys.modules[import_492402]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'matplotlib.collections', sys_modules_492403.module_type_store, module_type_store, ['LineCollection'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 4), __file__, sys_modules_492403, sys_modules_492403.module_type_store, module_type_store)
    else:
        from matplotlib.collections import LineCollection

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'matplotlib.collections', None, module_type_store, ['LineCollection'], [LineCollection])

else:
    # Assigning a type to the variable 'matplotlib.collections' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'matplotlib.collections', import_492402)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 4))

# 'from matplotlib import MatplotlibDeprecationWarning' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/tests/')
import_492404 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 4), 'matplotlib')

if (type(import_492404) is not StypyTypeError):

    if (import_492404 != 'pyd_module'):
        __import__(import_492404)
        sys_modules_492405 = sys.modules[import_492404]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 4), 'matplotlib', sys_modules_492405.module_type_store, module_type_store, ['MatplotlibDeprecationWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 4), __file__, sys_modules_492405, sys_modules_492405.module_type_store, module_type_store)
    else:
        from matplotlib import MatplotlibDeprecationWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 4), 'matplotlib', None, module_type_store, ['MatplotlibDeprecationWarning'], [MatplotlibDeprecationWarning])

else:
    # Assigning a type to the variable 'matplotlib' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'matplotlib', import_492404)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/tests/')


# Assigning a Name to a Name (line 13):
# Getting the type of 'True' (line 13)
True_492406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 21), 'True')
# Assigning a type to the variable 'has_matplotlib' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'has_matplotlib', True_492406)
# SSA branch for the except part of a try statement (line 7)
# SSA branch for the except '<any exception>' branch of a try statement (line 7)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 15):
# Getting the type of 'False' (line 15)
False_492407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 21), 'False')
# Assigning a type to the variable 'has_matplotlib' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'has_matplotlib', False_492407)
# SSA join for try-except statement (line 7)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from scipy.spatial import delaunay_plot_2d, voronoi_plot_2d, convex_hull_plot_2d, Delaunay, Voronoi, ConvexHull' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/tests/')
import_492408 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.spatial')

if (type(import_492408) is not StypyTypeError):

    if (import_492408 != 'pyd_module'):
        __import__(import_492408)
        sys_modules_492409 = sys.modules[import_492408]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.spatial', sys_modules_492409.module_type_store, module_type_store, ['delaunay_plot_2d', 'voronoi_plot_2d', 'convex_hull_plot_2d', 'Delaunay', 'Voronoi', 'ConvexHull'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_492409, sys_modules_492409.module_type_store, module_type_store)
    else:
        from scipy.spatial import delaunay_plot_2d, voronoi_plot_2d, convex_hull_plot_2d, Delaunay, Voronoi, ConvexHull

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.spatial', None, module_type_store, ['delaunay_plot_2d', 'voronoi_plot_2d', 'convex_hull_plot_2d', 'Delaunay', 'Voronoi', 'ConvexHull'], [delaunay_plot_2d, voronoi_plot_2d, convex_hull_plot_2d, Delaunay, Voronoi, ConvexHull])

else:
    # Assigning a type to the variable 'scipy.spatial' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.spatial', import_492408)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/tests/')

# Declaration of the 'TestPlotting' class

class TestPlotting:

    @norecursion
    def test_delaunay(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_delaunay'
        module_type_store = module_type_store.open_function_context('test_delaunay', 26, 4, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPlotting.test_delaunay.__dict__.__setitem__('stypy_localization', localization)
        TestPlotting.test_delaunay.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPlotting.test_delaunay.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPlotting.test_delaunay.__dict__.__setitem__('stypy_function_name', 'TestPlotting.test_delaunay')
        TestPlotting.test_delaunay.__dict__.__setitem__('stypy_param_names_list', [])
        TestPlotting.test_delaunay.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPlotting.test_delaunay.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPlotting.test_delaunay.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPlotting.test_delaunay.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPlotting.test_delaunay.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPlotting.test_delaunay.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPlotting.test_delaunay', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_delaunay', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_delaunay(...)' code ##################

        
        # Assigning a Call to a Name (line 28):
        
        # Call to figure(...): (line 28)
        # Processing the call keyword arguments (line 28)
        kwargs_492412 = {}
        # Getting the type of 'plt' (line 28)
        plt_492410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 14), 'plt', False)
        # Obtaining the member 'figure' of a type (line 28)
        figure_492411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 14), plt_492410, 'figure')
        # Calling figure(args, kwargs) (line 28)
        figure_call_result_492413 = invoke(stypy.reporting.localization.Localization(__file__, 28, 14), figure_492411, *[], **kwargs_492412)
        
        # Assigning a type to the variable 'fig' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'fig', figure_call_result_492413)
        
        # Assigning a Call to a Name (line 29):
        
        # Call to Delaunay(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'self' (line 29)
        self_492415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 23), 'self', False)
        # Obtaining the member 'points' of a type (line 29)
        points_492416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 23), self_492415, 'points')
        # Processing the call keyword arguments (line 29)
        kwargs_492417 = {}
        # Getting the type of 'Delaunay' (line 29)
        Delaunay_492414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 14), 'Delaunay', False)
        # Calling Delaunay(args, kwargs) (line 29)
        Delaunay_call_result_492418 = invoke(stypy.reporting.localization.Localization(__file__, 29, 14), Delaunay_492414, *[points_492416], **kwargs_492417)
        
        # Assigning a type to the variable 'obj' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'obj', Delaunay_call_result_492418)
        
        # Assigning a Call to a Name (line 30):
        
        # Call to copy(...): (line 30)
        # Processing the call keyword arguments (line 30)
        kwargs_492422 = {}
        # Getting the type of 'obj' (line 30)
        obj_492419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 19), 'obj', False)
        # Obtaining the member 'simplices' of a type (line 30)
        simplices_492420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 19), obj_492419, 'simplices')
        # Obtaining the member 'copy' of a type (line 30)
        copy_492421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 19), simplices_492420, 'copy')
        # Calling copy(args, kwargs) (line 30)
        copy_call_result_492423 = invoke(stypy.reporting.localization.Localization(__file__, 30, 19), copy_492421, *[], **kwargs_492422)
        
        # Assigning a type to the variable 's_before' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 's_before', copy_call_result_492423)
        # Getting the type of 'suppress_warnings' (line 31)
        suppress_warnings_492424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 13), 'suppress_warnings')
        with_492425 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 31, 13), suppress_warnings_492424, 'with parameter', '__enter__', '__exit__')

        if with_492425:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 31)
            enter___492426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 13), suppress_warnings_492424, '__enter__')
            with_enter_492427 = invoke(stypy.reporting.localization.Localization(__file__, 31, 13), enter___492426)
            # Assigning a type to the variable 'sup' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 13), 'sup', with_enter_492427)
            
            # Call to filter(...): (line 33)
            # Processing the call keyword arguments (line 33)
            str_492430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 31), 'str', 'The ishold function was deprecated in version')
            keyword_492431 = str_492430
            kwargs_492432 = {'message': keyword_492431}
            # Getting the type of 'sup' (line 33)
            sup_492428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 33)
            filter_492429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), sup_492428, 'filter')
            # Calling filter(args, kwargs) (line 33)
            filter_call_result_492433 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), filter_492429, *[], **kwargs_492432)
            
            
            # Assigning a Call to a Name (line 34):
            
            # Call to delaunay_plot_2d(...): (line 34)
            # Processing the call arguments (line 34)
            # Getting the type of 'obj' (line 34)
            obj_492435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 33), 'obj', False)
            # Processing the call keyword arguments (line 34)
            
            # Call to gca(...): (line 34)
            # Processing the call keyword arguments (line 34)
            kwargs_492438 = {}
            # Getting the type of 'fig' (line 34)
            fig_492436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 41), 'fig', False)
            # Obtaining the member 'gca' of a type (line 34)
            gca_492437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 41), fig_492436, 'gca')
            # Calling gca(args, kwargs) (line 34)
            gca_call_result_492439 = invoke(stypy.reporting.localization.Localization(__file__, 34, 41), gca_492437, *[], **kwargs_492438)
            
            keyword_492440 = gca_call_result_492439
            kwargs_492441 = {'ax': keyword_492440}
            # Getting the type of 'delaunay_plot_2d' (line 34)
            delaunay_plot_2d_492434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'delaunay_plot_2d', False)
            # Calling delaunay_plot_2d(args, kwargs) (line 34)
            delaunay_plot_2d_call_result_492442 = invoke(stypy.reporting.localization.Localization(__file__, 34, 16), delaunay_plot_2d_492434, *[obj_492435], **kwargs_492441)
            
            # Assigning a type to the variable 'r' (line 34)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'r', delaunay_plot_2d_call_result_492442)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 31)
            exit___492443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 13), suppress_warnings_492424, '__exit__')
            with_exit_492444 = invoke(stypy.reporting.localization.Localization(__file__, 31, 13), exit___492443, None, None, None)

        
        # Call to assert_array_equal(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'obj' (line 35)
        obj_492446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 27), 'obj', False)
        # Obtaining the member 'simplices' of a type (line 35)
        simplices_492447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 27), obj_492446, 'simplices')
        # Getting the type of 's_before' (line 35)
        s_before_492448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 42), 's_before', False)
        # Processing the call keyword arguments (line 35)
        kwargs_492449 = {}
        # Getting the type of 'assert_array_equal' (line 35)
        assert_array_equal_492445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 35)
        assert_array_equal_call_result_492450 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), assert_array_equal_492445, *[simplices_492447, s_before_492448], **kwargs_492449)
        
        
        # Call to assert_(...): (line 36)
        # Processing the call arguments (line 36)
        
        # Getting the type of 'r' (line 36)
        r_492452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'r', False)
        # Getting the type of 'fig' (line 36)
        fig_492453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 21), 'fig', False)
        # Applying the binary operator 'is' (line 36)
        result_is__492454 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 16), 'is', r_492452, fig_492453)
        
        # Processing the call keyword arguments (line 36)
        kwargs_492455 = {}
        # Getting the type of 'assert_' (line 36)
        assert__492451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 36)
        assert__call_result_492456 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), assert__492451, *[result_is__492454], **kwargs_492455)
        
        
        # Call to delaunay_plot_2d(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'obj' (line 37)
        obj_492458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 25), 'obj', False)
        # Processing the call keyword arguments (line 37)
        
        # Call to gca(...): (line 37)
        # Processing the call keyword arguments (line 37)
        kwargs_492461 = {}
        # Getting the type of 'fig' (line 37)
        fig_492459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 33), 'fig', False)
        # Obtaining the member 'gca' of a type (line 37)
        gca_492460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 33), fig_492459, 'gca')
        # Calling gca(args, kwargs) (line 37)
        gca_call_result_492462 = invoke(stypy.reporting.localization.Localization(__file__, 37, 33), gca_492460, *[], **kwargs_492461)
        
        keyword_492463 = gca_call_result_492462
        kwargs_492464 = {'ax': keyword_492463}
        # Getting the type of 'delaunay_plot_2d' (line 37)
        delaunay_plot_2d_492457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'delaunay_plot_2d', False)
        # Calling delaunay_plot_2d(args, kwargs) (line 37)
        delaunay_plot_2d_call_result_492465 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), delaunay_plot_2d_492457, *[obj_492458], **kwargs_492464)
        
        
        # ################# End of 'test_delaunay(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_delaunay' in the type store
        # Getting the type of 'stypy_return_type' (line 26)
        stypy_return_type_492466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_492466)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_delaunay'
        return stypy_return_type_492466


    @norecursion
    def test_voronoi(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_voronoi'
        module_type_store = module_type_store.open_function_context('test_voronoi', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPlotting.test_voronoi.__dict__.__setitem__('stypy_localization', localization)
        TestPlotting.test_voronoi.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPlotting.test_voronoi.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPlotting.test_voronoi.__dict__.__setitem__('stypy_function_name', 'TestPlotting.test_voronoi')
        TestPlotting.test_voronoi.__dict__.__setitem__('stypy_param_names_list', [])
        TestPlotting.test_voronoi.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPlotting.test_voronoi.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPlotting.test_voronoi.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPlotting.test_voronoi.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPlotting.test_voronoi.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPlotting.test_voronoi.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPlotting.test_voronoi', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_voronoi', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_voronoi(...)' code ##################

        
        # Assigning a Call to a Name (line 41):
        
        # Call to figure(...): (line 41)
        # Processing the call keyword arguments (line 41)
        kwargs_492469 = {}
        # Getting the type of 'plt' (line 41)
        plt_492467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 14), 'plt', False)
        # Obtaining the member 'figure' of a type (line 41)
        figure_492468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 14), plt_492467, 'figure')
        # Calling figure(args, kwargs) (line 41)
        figure_call_result_492470 = invoke(stypy.reporting.localization.Localization(__file__, 41, 14), figure_492468, *[], **kwargs_492469)
        
        # Assigning a type to the variable 'fig' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'fig', figure_call_result_492470)
        
        # Assigning a Call to a Name (line 42):
        
        # Call to Voronoi(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'self' (line 42)
        self_492472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 22), 'self', False)
        # Obtaining the member 'points' of a type (line 42)
        points_492473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 22), self_492472, 'points')
        # Processing the call keyword arguments (line 42)
        kwargs_492474 = {}
        # Getting the type of 'Voronoi' (line 42)
        Voronoi_492471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 14), 'Voronoi', False)
        # Calling Voronoi(args, kwargs) (line 42)
        Voronoi_call_result_492475 = invoke(stypy.reporting.localization.Localization(__file__, 42, 14), Voronoi_492471, *[points_492473], **kwargs_492474)
        
        # Assigning a type to the variable 'obj' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'obj', Voronoi_call_result_492475)
        # Getting the type of 'suppress_warnings' (line 43)
        suppress_warnings_492476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 13), 'suppress_warnings')
        with_492477 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 43, 13), suppress_warnings_492476, 'with parameter', '__enter__', '__exit__')

        if with_492477:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 43)
            enter___492478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 13), suppress_warnings_492476, '__enter__')
            with_enter_492479 = invoke(stypy.reporting.localization.Localization(__file__, 43, 13), enter___492478)
            # Assigning a type to the variable 'sup' (line 43)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 13), 'sup', with_enter_492479)
            
            # Call to filter(...): (line 45)
            # Processing the call keyword arguments (line 45)
            str_492482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 31), 'str', 'The ishold function was deprecated in version')
            keyword_492483 = str_492482
            kwargs_492484 = {'message': keyword_492483}
            # Getting the type of 'sup' (line 45)
            sup_492480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 45)
            filter_492481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 12), sup_492480, 'filter')
            # Calling filter(args, kwargs) (line 45)
            filter_call_result_492485 = invoke(stypy.reporting.localization.Localization(__file__, 45, 12), filter_492481, *[], **kwargs_492484)
            
            
            # Assigning a Call to a Name (line 46):
            
            # Call to voronoi_plot_2d(...): (line 46)
            # Processing the call arguments (line 46)
            # Getting the type of 'obj' (line 46)
            obj_492487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 32), 'obj', False)
            # Processing the call keyword arguments (line 46)
            
            # Call to gca(...): (line 46)
            # Processing the call keyword arguments (line 46)
            kwargs_492490 = {}
            # Getting the type of 'fig' (line 46)
            fig_492488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 40), 'fig', False)
            # Obtaining the member 'gca' of a type (line 46)
            gca_492489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 40), fig_492488, 'gca')
            # Calling gca(args, kwargs) (line 46)
            gca_call_result_492491 = invoke(stypy.reporting.localization.Localization(__file__, 46, 40), gca_492489, *[], **kwargs_492490)
            
            keyword_492492 = gca_call_result_492491
            kwargs_492493 = {'ax': keyword_492492}
            # Getting the type of 'voronoi_plot_2d' (line 46)
            voronoi_plot_2d_492486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 16), 'voronoi_plot_2d', False)
            # Calling voronoi_plot_2d(args, kwargs) (line 46)
            voronoi_plot_2d_call_result_492494 = invoke(stypy.reporting.localization.Localization(__file__, 46, 16), voronoi_plot_2d_492486, *[obj_492487], **kwargs_492493)
            
            # Assigning a type to the variable 'r' (line 46)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'r', voronoi_plot_2d_call_result_492494)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 43)
            exit___492495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 13), suppress_warnings_492476, '__exit__')
            with_exit_492496 = invoke(stypy.reporting.localization.Localization(__file__, 43, 13), exit___492495, None, None, None)

        
        # Call to assert_(...): (line 47)
        # Processing the call arguments (line 47)
        
        # Getting the type of 'r' (line 47)
        r_492498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'r', False)
        # Getting the type of 'fig' (line 47)
        fig_492499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 21), 'fig', False)
        # Applying the binary operator 'is' (line 47)
        result_is__492500 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 16), 'is', r_492498, fig_492499)
        
        # Processing the call keyword arguments (line 47)
        kwargs_492501 = {}
        # Getting the type of 'assert_' (line 47)
        assert__492497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 47)
        assert__call_result_492502 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), assert__492497, *[result_is__492500], **kwargs_492501)
        
        
        # Call to voronoi_plot_2d(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'obj' (line 48)
        obj_492504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 24), 'obj', False)
        # Processing the call keyword arguments (line 48)
        kwargs_492505 = {}
        # Getting the type of 'voronoi_plot_2d' (line 48)
        voronoi_plot_2d_492503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'voronoi_plot_2d', False)
        # Calling voronoi_plot_2d(args, kwargs) (line 48)
        voronoi_plot_2d_call_result_492506 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), voronoi_plot_2d_492503, *[obj_492504], **kwargs_492505)
        
        
        # Call to voronoi_plot_2d(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'obj' (line 49)
        obj_492508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 24), 'obj', False)
        # Processing the call keyword arguments (line 49)
        # Getting the type of 'False' (line 49)
        False_492509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 43), 'False', False)
        keyword_492510 = False_492509
        kwargs_492511 = {'show_vertices': keyword_492510}
        # Getting the type of 'voronoi_plot_2d' (line 49)
        voronoi_plot_2d_492507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'voronoi_plot_2d', False)
        # Calling voronoi_plot_2d(args, kwargs) (line 49)
        voronoi_plot_2d_call_result_492512 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), voronoi_plot_2d_492507, *[obj_492508], **kwargs_492511)
        
        
        # ################# End of 'test_voronoi(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_voronoi' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_492513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_492513)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_voronoi'
        return stypy_return_type_492513


    @norecursion
    def test_convex_hull(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_convex_hull'
        module_type_store = module_type_store.open_function_context('test_convex_hull', 51, 4, False)
        # Assigning a type to the variable 'self' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPlotting.test_convex_hull.__dict__.__setitem__('stypy_localization', localization)
        TestPlotting.test_convex_hull.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPlotting.test_convex_hull.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPlotting.test_convex_hull.__dict__.__setitem__('stypy_function_name', 'TestPlotting.test_convex_hull')
        TestPlotting.test_convex_hull.__dict__.__setitem__('stypy_param_names_list', [])
        TestPlotting.test_convex_hull.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPlotting.test_convex_hull.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPlotting.test_convex_hull.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPlotting.test_convex_hull.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPlotting.test_convex_hull.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPlotting.test_convex_hull.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPlotting.test_convex_hull', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_convex_hull', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_convex_hull(...)' code ##################

        
        # Assigning a Call to a Name (line 53):
        
        # Call to figure(...): (line 53)
        # Processing the call keyword arguments (line 53)
        kwargs_492516 = {}
        # Getting the type of 'plt' (line 53)
        plt_492514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 14), 'plt', False)
        # Obtaining the member 'figure' of a type (line 53)
        figure_492515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 14), plt_492514, 'figure')
        # Calling figure(args, kwargs) (line 53)
        figure_call_result_492517 = invoke(stypy.reporting.localization.Localization(__file__, 53, 14), figure_492515, *[], **kwargs_492516)
        
        # Assigning a type to the variable 'fig' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'fig', figure_call_result_492517)
        
        # Assigning a Call to a Name (line 54):
        
        # Call to ConvexHull(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'self' (line 54)
        self_492519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 25), 'self', False)
        # Obtaining the member 'points' of a type (line 54)
        points_492520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 25), self_492519, 'points')
        # Processing the call keyword arguments (line 54)
        kwargs_492521 = {}
        # Getting the type of 'ConvexHull' (line 54)
        ConvexHull_492518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 14), 'ConvexHull', False)
        # Calling ConvexHull(args, kwargs) (line 54)
        ConvexHull_call_result_492522 = invoke(stypy.reporting.localization.Localization(__file__, 54, 14), ConvexHull_492518, *[points_492520], **kwargs_492521)
        
        # Assigning a type to the variable 'tri' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'tri', ConvexHull_call_result_492522)
        # Getting the type of 'suppress_warnings' (line 55)
        suppress_warnings_492523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 13), 'suppress_warnings')
        with_492524 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 55, 13), suppress_warnings_492523, 'with parameter', '__enter__', '__exit__')

        if with_492524:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 55)
            enter___492525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 13), suppress_warnings_492523, '__enter__')
            with_enter_492526 = invoke(stypy.reporting.localization.Localization(__file__, 55, 13), enter___492525)
            # Assigning a type to the variable 'sup' (line 55)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 13), 'sup', with_enter_492526)
            
            # Call to filter(...): (line 57)
            # Processing the call keyword arguments (line 57)
            str_492529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 31), 'str', 'The ishold function was deprecated in version')
            keyword_492530 = str_492529
            kwargs_492531 = {'message': keyword_492530}
            # Getting the type of 'sup' (line 57)
            sup_492527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 57)
            filter_492528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), sup_492527, 'filter')
            # Calling filter(args, kwargs) (line 57)
            filter_call_result_492532 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), filter_492528, *[], **kwargs_492531)
            
            
            # Assigning a Call to a Name (line 58):
            
            # Call to convex_hull_plot_2d(...): (line 58)
            # Processing the call arguments (line 58)
            # Getting the type of 'tri' (line 58)
            tri_492534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 36), 'tri', False)
            # Processing the call keyword arguments (line 58)
            
            # Call to gca(...): (line 58)
            # Processing the call keyword arguments (line 58)
            kwargs_492537 = {}
            # Getting the type of 'fig' (line 58)
            fig_492535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 44), 'fig', False)
            # Obtaining the member 'gca' of a type (line 58)
            gca_492536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 44), fig_492535, 'gca')
            # Calling gca(args, kwargs) (line 58)
            gca_call_result_492538 = invoke(stypy.reporting.localization.Localization(__file__, 58, 44), gca_492536, *[], **kwargs_492537)
            
            keyword_492539 = gca_call_result_492538
            kwargs_492540 = {'ax': keyword_492539}
            # Getting the type of 'convex_hull_plot_2d' (line 58)
            convex_hull_plot_2d_492533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'convex_hull_plot_2d', False)
            # Calling convex_hull_plot_2d(args, kwargs) (line 58)
            convex_hull_plot_2d_call_result_492541 = invoke(stypy.reporting.localization.Localization(__file__, 58, 16), convex_hull_plot_2d_492533, *[tri_492534], **kwargs_492540)
            
            # Assigning a type to the variable 'r' (line 58)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'r', convex_hull_plot_2d_call_result_492541)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 55)
            exit___492542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 13), suppress_warnings_492523, '__exit__')
            with_exit_492543 = invoke(stypy.reporting.localization.Localization(__file__, 55, 13), exit___492542, None, None, None)

        
        # Call to assert_(...): (line 59)
        # Processing the call arguments (line 59)
        
        # Getting the type of 'r' (line 59)
        r_492545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'r', False)
        # Getting the type of 'fig' (line 59)
        fig_492546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 21), 'fig', False)
        # Applying the binary operator 'is' (line 59)
        result_is__492547 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 16), 'is', r_492545, fig_492546)
        
        # Processing the call keyword arguments (line 59)
        kwargs_492548 = {}
        # Getting the type of 'assert_' (line 59)
        assert__492544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 59)
        assert__call_result_492549 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), assert__492544, *[result_is__492547], **kwargs_492548)
        
        
        # Call to convex_hull_plot_2d(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'tri' (line 60)
        tri_492551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 28), 'tri', False)
        # Processing the call keyword arguments (line 60)
        kwargs_492552 = {}
        # Getting the type of 'convex_hull_plot_2d' (line 60)
        convex_hull_plot_2d_492550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'convex_hull_plot_2d', False)
        # Calling convex_hull_plot_2d(args, kwargs) (line 60)
        convex_hull_plot_2d_call_result_492553 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), convex_hull_plot_2d_492550, *[tri_492551], **kwargs_492552)
        
        
        # ################# End of 'test_convex_hull(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_convex_hull' in the type store
        # Getting the type of 'stypy_return_type' (line 51)
        stypy_return_type_492554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_492554)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_convex_hull'
        return stypy_return_type_492554


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 22, 0, False)
        # Assigning a type to the variable 'self' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPlotting.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestPlotting' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'TestPlotting', TestPlotting)

# Assigning a List to a Name (line 24):

# Obtaining an instance of the builtin type 'list' (line 24)
list_492555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 24)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 24)
tuple_492556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 24)
# Adding element type (line 24)
int_492557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 15), tuple_492556, int_492557)
# Adding element type (line 24)
int_492558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 15), tuple_492556, int_492558)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 13), list_492555, tuple_492556)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 24)
tuple_492559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 24)
# Adding element type (line 24)
int_492560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 22), tuple_492559, int_492560)
# Adding element type (line 24)
int_492561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 22), tuple_492559, int_492561)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 13), list_492555, tuple_492559)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 24)
tuple_492562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 29), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 24)
# Adding element type (line 24)
int_492563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 29), tuple_492562, int_492563)
# Adding element type (line 24)
int_492564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 29), tuple_492562, int_492564)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 13), list_492555, tuple_492562)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 24)
tuple_492565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 36), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 24)
# Adding element type (line 24)
int_492566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 36), tuple_492565, int_492566)
# Adding element type (line 24)
int_492567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 38), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 36), tuple_492565, int_492567)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 13), list_492555, tuple_492565)

# Getting the type of 'TestPlotting'
TestPlotting_492568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestPlotting')
# Setting the type of the member 'points' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestPlotting_492568, 'points', list_492555)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import six
5: 
6: import matplotlib
7: import inspect
8: import traceback
9: import warnings
10: 
11: 
12: backend = matplotlib.get_backend()
13: _backend_loading_tb = "".join(
14:     line for line in traceback.format_stack()
15:     # Filter out line noise from importlib line.
16:     if not line.startswith('  File "<frozen importlib._bootstrap'))
17: 
18: 
19: def pylab_setup(name=None):
20:     '''return new_figure_manager, draw_if_interactive and show for pyplot
21: 
22:     This provides the backend-specific functions that are used by
23:     pyplot to abstract away the difference between interactive backends.
24: 
25:     Parameters
26:     ----------
27:     name : str, optional
28:         The name of the backend to use.  If `None`, falls back to
29:         ``matplotlib.get_backend()`` (which return ``rcParams['backend']``)
30: 
31:     Returns
32:     -------
33:     backend_mod : module
34:         The module which contains the backend of choice
35: 
36:     new_figure_manager : function
37:         Create a new figure manager (roughly maps to GUI window)
38: 
39:     draw_if_interactive : function
40:         Redraw the current figure if pyplot is interactive
41: 
42:     show : function
43:         Show (and possibly block) any unshown figures.
44: 
45:     '''
46:     # Import the requested backend into a generic module object
47:     if name is None:
48:         # validates, to match all_backends
49:         name = matplotlib.get_backend()
50:     if name.startswith('module://'):
51:         backend_name = name[9:]
52:     else:
53:         backend_name = 'backend_' + name
54:         backend_name = backend_name.lower()  # until we banish mixed case
55:         backend_name = 'matplotlib.backends.%s' % backend_name.lower()
56: 
57:     # the last argument is specifies whether to use absolute or relative
58:     # imports. 0 means only perform absolute imports.
59:     backend_mod = __import__(backend_name, globals(), locals(),
60:                              [backend_name], 0)
61: 
62:     # Things we pull in from all backends
63:     new_figure_manager = backend_mod.new_figure_manager
64: 
65:     # image backends like pdf, agg or svg do not need to do anything
66:     # for "show" or "draw_if_interactive", so if they are not defined
67:     # by the backend, just do nothing
68:     def do_nothing_show(*args, **kwargs):
69:         frame = inspect.currentframe()
70:         fname = frame.f_back.f_code.co_filename
71:         if fname in ('<stdin>', '<ipython console>'):
72:             warnings.warn('''
73: Your currently selected backend, '%s' does not support show().
74: Please select a GUI backend in your matplotlibrc file ('%s')
75: or with matplotlib.use()''' %
76:                           (name, matplotlib.matplotlib_fname()))
77: 
78:     def do_nothing(*args, **kwargs):
79:         pass
80: 
81:     backend_version = getattr(backend_mod, 'backend_version', 'unknown')
82: 
83:     show = getattr(backend_mod, 'show', do_nothing_show)
84: 
85:     draw_if_interactive = getattr(backend_mod, 'draw_if_interactive',
86:                                   do_nothing)
87: 
88:     matplotlib.verbose.report('backend %s version %s' %
89:                               (name, backend_version))
90: 
91:     # need to keep a global reference to the backend for compatibility
92:     # reasons. See https://github.com/matplotlib/matplotlib/issues/6092
93:     global backend
94:     backend = name
95:     return backend_mod, new_figure_manager, draw_if_interactive, show
96: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import six' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_269835 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six')

if (type(import_269835) is not StypyTypeError):

    if (import_269835 != 'pyd_module'):
        __import__(import_269835)
        sys_modules_269836 = sys.modules[import_269835]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', sys_modules_269836.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', import_269835)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import matplotlib' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_269837 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib')

if (type(import_269837) is not StypyTypeError):

    if (import_269837 != 'pyd_module'):
        __import__(import_269837)
        sys_modules_269838 = sys.modules[import_269837]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib', sys_modules_269838.module_type_store, module_type_store)
    else:
        import matplotlib

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib', import_269837)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import inspect' statement (line 7)
import inspect

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'inspect', inspect, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import traceback' statement (line 8)
import traceback

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'traceback', traceback, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import warnings' statement (line 9)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'warnings', warnings, module_type_store)


# Assigning a Call to a Name (line 12):

# Call to get_backend(...): (line 12)
# Processing the call keyword arguments (line 12)
kwargs_269841 = {}
# Getting the type of 'matplotlib' (line 12)
matplotlib_269839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'matplotlib', False)
# Obtaining the member 'get_backend' of a type (line 12)
get_backend_269840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 10), matplotlib_269839, 'get_backend')
# Calling get_backend(args, kwargs) (line 12)
get_backend_call_result_269842 = invoke(stypy.reporting.localization.Localization(__file__, 12, 10), get_backend_269840, *[], **kwargs_269841)

# Assigning a type to the variable 'backend' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'backend', get_backend_call_result_269842)

# Assigning a Call to a Name (line 13):

# Call to join(...): (line 13)
# Processing the call arguments (line 13)
# Calculating generator expression
module_type_store = module_type_store.open_function_context('list comprehension expression', 14, 4, True)
# Calculating comprehension expression

# Call to format_stack(...): (line 14)
# Processing the call keyword arguments (line 14)
kwargs_269854 = {}
# Getting the type of 'traceback' (line 14)
traceback_269852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 21), 'traceback', False)
# Obtaining the member 'format_stack' of a type (line 14)
format_stack_269853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 21), traceback_269852, 'format_stack')
# Calling format_stack(args, kwargs) (line 14)
format_stack_call_result_269855 = invoke(stypy.reporting.localization.Localization(__file__, 14, 21), format_stack_269853, *[], **kwargs_269854)

comprehension_269856 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 4), format_stack_call_result_269855)
# Assigning a type to the variable 'line' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'line', comprehension_269856)


# Call to startswith(...): (line 16)
# Processing the call arguments (line 16)
unicode_269848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 27), 'unicode', u'  File "<frozen importlib._bootstrap')
# Processing the call keyword arguments (line 16)
kwargs_269849 = {}
# Getting the type of 'line' (line 16)
line_269846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 11), 'line', False)
# Obtaining the member 'startswith' of a type (line 16)
startswith_269847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 11), line_269846, 'startswith')
# Calling startswith(args, kwargs) (line 16)
startswith_call_result_269850 = invoke(stypy.reporting.localization.Localization(__file__, 16, 11), startswith_269847, *[unicode_269848], **kwargs_269849)

# Applying the 'not' unary operator (line 16)
result_not__269851 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 7), 'not', startswith_call_result_269850)

# Getting the type of 'line' (line 14)
line_269845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'line', False)
list_269857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 4), 'list')
# Destroy the current context
module_type_store = module_type_store.close_function_context()
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 4), list_269857, line_269845)
# Processing the call keyword arguments (line 13)
kwargs_269858 = {}
unicode_269843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 22), 'unicode', u'')
# Obtaining the member 'join' of a type (line 13)
join_269844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 22), unicode_269843, 'join')
# Calling join(args, kwargs) (line 13)
join_call_result_269859 = invoke(stypy.reporting.localization.Localization(__file__, 13, 22), join_269844, *[list_269857], **kwargs_269858)

# Assigning a type to the variable '_backend_loading_tb' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), '_backend_loading_tb', join_call_result_269859)

@norecursion
def pylab_setup(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 19)
    None_269860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 21), 'None')
    defaults = [None_269860]
    # Create a new context for function 'pylab_setup'
    module_type_store = module_type_store.open_function_context('pylab_setup', 19, 0, False)
    
    # Passed parameters checking function
    pylab_setup.stypy_localization = localization
    pylab_setup.stypy_type_of_self = None
    pylab_setup.stypy_type_store = module_type_store
    pylab_setup.stypy_function_name = 'pylab_setup'
    pylab_setup.stypy_param_names_list = ['name']
    pylab_setup.stypy_varargs_param_name = None
    pylab_setup.stypy_kwargs_param_name = None
    pylab_setup.stypy_call_defaults = defaults
    pylab_setup.stypy_call_varargs = varargs
    pylab_setup.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pylab_setup', ['name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pylab_setup', localization, ['name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pylab_setup(...)' code ##################

    unicode_269861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, (-1)), 'unicode', u"return new_figure_manager, draw_if_interactive and show for pyplot\n\n    This provides the backend-specific functions that are used by\n    pyplot to abstract away the difference between interactive backends.\n\n    Parameters\n    ----------\n    name : str, optional\n        The name of the backend to use.  If `None`, falls back to\n        ``matplotlib.get_backend()`` (which return ``rcParams['backend']``)\n\n    Returns\n    -------\n    backend_mod : module\n        The module which contains the backend of choice\n\n    new_figure_manager : function\n        Create a new figure manager (roughly maps to GUI window)\n\n    draw_if_interactive : function\n        Redraw the current figure if pyplot is interactive\n\n    show : function\n        Show (and possibly block) any unshown figures.\n\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 47)
    # Getting the type of 'name' (line 47)
    name_269862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 7), 'name')
    # Getting the type of 'None' (line 47)
    None_269863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 15), 'None')
    
    (may_be_269864, more_types_in_union_269865) = may_be_none(name_269862, None_269863)

    if may_be_269864:

        if more_types_in_union_269865:
            # Runtime conditional SSA (line 47)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 49):
        
        # Call to get_backend(...): (line 49)
        # Processing the call keyword arguments (line 49)
        kwargs_269868 = {}
        # Getting the type of 'matplotlib' (line 49)
        matplotlib_269866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'matplotlib', False)
        # Obtaining the member 'get_backend' of a type (line 49)
        get_backend_269867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 15), matplotlib_269866, 'get_backend')
        # Calling get_backend(args, kwargs) (line 49)
        get_backend_call_result_269869 = invoke(stypy.reporting.localization.Localization(__file__, 49, 15), get_backend_269867, *[], **kwargs_269868)
        
        # Assigning a type to the variable 'name' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'name', get_backend_call_result_269869)

        if more_types_in_union_269865:
            # SSA join for if statement (line 47)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to startswith(...): (line 50)
    # Processing the call arguments (line 50)
    unicode_269872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 23), 'unicode', u'module://')
    # Processing the call keyword arguments (line 50)
    kwargs_269873 = {}
    # Getting the type of 'name' (line 50)
    name_269870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 7), 'name', False)
    # Obtaining the member 'startswith' of a type (line 50)
    startswith_269871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 7), name_269870, 'startswith')
    # Calling startswith(args, kwargs) (line 50)
    startswith_call_result_269874 = invoke(stypy.reporting.localization.Localization(__file__, 50, 7), startswith_269871, *[unicode_269872], **kwargs_269873)
    
    # Testing the type of an if condition (line 50)
    if_condition_269875 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 4), startswith_call_result_269874)
    # Assigning a type to the variable 'if_condition_269875' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'if_condition_269875', if_condition_269875)
    # SSA begins for if statement (line 50)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 51):
    
    # Obtaining the type of the subscript
    int_269876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 28), 'int')
    slice_269877 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 51, 23), int_269876, None, None)
    # Getting the type of 'name' (line 51)
    name_269878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 23), 'name')
    # Obtaining the member '__getitem__' of a type (line 51)
    getitem___269879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 23), name_269878, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 51)
    subscript_call_result_269880 = invoke(stypy.reporting.localization.Localization(__file__, 51, 23), getitem___269879, slice_269877)
    
    # Assigning a type to the variable 'backend_name' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'backend_name', subscript_call_result_269880)
    # SSA branch for the else part of an if statement (line 50)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 53):
    unicode_269881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 23), 'unicode', u'backend_')
    # Getting the type of 'name' (line 53)
    name_269882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 36), 'name')
    # Applying the binary operator '+' (line 53)
    result_add_269883 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 23), '+', unicode_269881, name_269882)
    
    # Assigning a type to the variable 'backend_name' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'backend_name', result_add_269883)
    
    # Assigning a Call to a Name (line 54):
    
    # Call to lower(...): (line 54)
    # Processing the call keyword arguments (line 54)
    kwargs_269886 = {}
    # Getting the type of 'backend_name' (line 54)
    backend_name_269884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 23), 'backend_name', False)
    # Obtaining the member 'lower' of a type (line 54)
    lower_269885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 23), backend_name_269884, 'lower')
    # Calling lower(args, kwargs) (line 54)
    lower_call_result_269887 = invoke(stypy.reporting.localization.Localization(__file__, 54, 23), lower_269885, *[], **kwargs_269886)
    
    # Assigning a type to the variable 'backend_name' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'backend_name', lower_call_result_269887)
    
    # Assigning a BinOp to a Name (line 55):
    unicode_269888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 23), 'unicode', u'matplotlib.backends.%s')
    
    # Call to lower(...): (line 55)
    # Processing the call keyword arguments (line 55)
    kwargs_269891 = {}
    # Getting the type of 'backend_name' (line 55)
    backend_name_269889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 50), 'backend_name', False)
    # Obtaining the member 'lower' of a type (line 55)
    lower_269890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 50), backend_name_269889, 'lower')
    # Calling lower(args, kwargs) (line 55)
    lower_call_result_269892 = invoke(stypy.reporting.localization.Localization(__file__, 55, 50), lower_269890, *[], **kwargs_269891)
    
    # Applying the binary operator '%' (line 55)
    result_mod_269893 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 23), '%', unicode_269888, lower_call_result_269892)
    
    # Assigning a type to the variable 'backend_name' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'backend_name', result_mod_269893)
    # SSA join for if statement (line 50)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 59):
    
    # Call to __import__(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'backend_name' (line 59)
    backend_name_269895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 29), 'backend_name', False)
    
    # Call to globals(...): (line 59)
    # Processing the call keyword arguments (line 59)
    kwargs_269897 = {}
    # Getting the type of 'globals' (line 59)
    globals_269896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 43), 'globals', False)
    # Calling globals(args, kwargs) (line 59)
    globals_call_result_269898 = invoke(stypy.reporting.localization.Localization(__file__, 59, 43), globals_269896, *[], **kwargs_269897)
    
    
    # Call to locals(...): (line 59)
    # Processing the call keyword arguments (line 59)
    kwargs_269900 = {}
    # Getting the type of 'locals' (line 59)
    locals_269899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 54), 'locals', False)
    # Calling locals(args, kwargs) (line 59)
    locals_call_result_269901 = invoke(stypy.reporting.localization.Localization(__file__, 59, 54), locals_269899, *[], **kwargs_269900)
    
    
    # Obtaining an instance of the builtin type 'list' (line 60)
    list_269902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 60)
    # Adding element type (line 60)
    # Getting the type of 'backend_name' (line 60)
    backend_name_269903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 30), 'backend_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 29), list_269902, backend_name_269903)
    
    int_269904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 45), 'int')
    # Processing the call keyword arguments (line 59)
    kwargs_269905 = {}
    # Getting the type of '__import__' (line 59)
    import___269894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 18), '__import__', False)
    # Calling __import__(args, kwargs) (line 59)
    import___call_result_269906 = invoke(stypy.reporting.localization.Localization(__file__, 59, 18), import___269894, *[backend_name_269895, globals_call_result_269898, locals_call_result_269901, list_269902, int_269904], **kwargs_269905)
    
    # Assigning a type to the variable 'backend_mod' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'backend_mod', import___call_result_269906)
    
    # Assigning a Attribute to a Name (line 63):
    # Getting the type of 'backend_mod' (line 63)
    backend_mod_269907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 25), 'backend_mod')
    # Obtaining the member 'new_figure_manager' of a type (line 63)
    new_figure_manager_269908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 25), backend_mod_269907, 'new_figure_manager')
    # Assigning a type to the variable 'new_figure_manager' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'new_figure_manager', new_figure_manager_269908)

    @norecursion
    def do_nothing_show(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'do_nothing_show'
        module_type_store = module_type_store.open_function_context('do_nothing_show', 68, 4, False)
        
        # Passed parameters checking function
        do_nothing_show.stypy_localization = localization
        do_nothing_show.stypy_type_of_self = None
        do_nothing_show.stypy_type_store = module_type_store
        do_nothing_show.stypy_function_name = 'do_nothing_show'
        do_nothing_show.stypy_param_names_list = []
        do_nothing_show.stypy_varargs_param_name = 'args'
        do_nothing_show.stypy_kwargs_param_name = 'kwargs'
        do_nothing_show.stypy_call_defaults = defaults
        do_nothing_show.stypy_call_varargs = varargs
        do_nothing_show.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'do_nothing_show', [], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'do_nothing_show', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'do_nothing_show(...)' code ##################

        
        # Assigning a Call to a Name (line 69):
        
        # Call to currentframe(...): (line 69)
        # Processing the call keyword arguments (line 69)
        kwargs_269911 = {}
        # Getting the type of 'inspect' (line 69)
        inspect_269909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'inspect', False)
        # Obtaining the member 'currentframe' of a type (line 69)
        currentframe_269910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 16), inspect_269909, 'currentframe')
        # Calling currentframe(args, kwargs) (line 69)
        currentframe_call_result_269912 = invoke(stypy.reporting.localization.Localization(__file__, 69, 16), currentframe_269910, *[], **kwargs_269911)
        
        # Assigning a type to the variable 'frame' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'frame', currentframe_call_result_269912)
        
        # Assigning a Attribute to a Name (line 70):
        # Getting the type of 'frame' (line 70)
        frame_269913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'frame')
        # Obtaining the member 'f_back' of a type (line 70)
        f_back_269914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 16), frame_269913, 'f_back')
        # Obtaining the member 'f_code' of a type (line 70)
        f_code_269915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 16), f_back_269914, 'f_code')
        # Obtaining the member 'co_filename' of a type (line 70)
        co_filename_269916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 16), f_code_269915, 'co_filename')
        # Assigning a type to the variable 'fname' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'fname', co_filename_269916)
        
        
        # Getting the type of 'fname' (line 71)
        fname_269917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 11), 'fname')
        
        # Obtaining an instance of the builtin type 'tuple' (line 71)
        tuple_269918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 71)
        # Adding element type (line 71)
        unicode_269919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 21), 'unicode', u'<stdin>')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 21), tuple_269918, unicode_269919)
        # Adding element type (line 71)
        unicode_269920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 32), 'unicode', u'<ipython console>')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 21), tuple_269918, unicode_269920)
        
        # Applying the binary operator 'in' (line 71)
        result_contains_269921 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 11), 'in', fname_269917, tuple_269918)
        
        # Testing the type of an if condition (line 71)
        if_condition_269922 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 8), result_contains_269921)
        # Assigning a type to the variable 'if_condition_269922' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'if_condition_269922', if_condition_269922)
        # SSA begins for if statement (line 71)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 72)
        # Processing the call arguments (line 72)
        unicode_269925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, (-1)), 'unicode', u"\nYour currently selected backend, '%s' does not support show().\nPlease select a GUI backend in your matplotlibrc file ('%s')\nor with matplotlib.use()")
        
        # Obtaining an instance of the builtin type 'tuple' (line 76)
        tuple_269926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 76)
        # Adding element type (line 76)
        # Getting the type of 'name' (line 76)
        name_269927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 27), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 27), tuple_269926, name_269927)
        # Adding element type (line 76)
        
        # Call to matplotlib_fname(...): (line 76)
        # Processing the call keyword arguments (line 76)
        kwargs_269930 = {}
        # Getting the type of 'matplotlib' (line 76)
        matplotlib_269928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 33), 'matplotlib', False)
        # Obtaining the member 'matplotlib_fname' of a type (line 76)
        matplotlib_fname_269929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 33), matplotlib_269928, 'matplotlib_fname')
        # Calling matplotlib_fname(args, kwargs) (line 76)
        matplotlib_fname_call_result_269931 = invoke(stypy.reporting.localization.Localization(__file__, 76, 33), matplotlib_fname_269929, *[], **kwargs_269930)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 27), tuple_269926, matplotlib_fname_call_result_269931)
        
        # Applying the binary operator '%' (line 75)
        result_mod_269932 = python_operator(stypy.reporting.localization.Localization(__file__, 75, (-1)), '%', unicode_269925, tuple_269926)
        
        # Processing the call keyword arguments (line 72)
        kwargs_269933 = {}
        # Getting the type of 'warnings' (line 72)
        warnings_269923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 72)
        warn_269924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 12), warnings_269923, 'warn')
        # Calling warn(args, kwargs) (line 72)
        warn_call_result_269934 = invoke(stypy.reporting.localization.Localization(__file__, 72, 12), warn_269924, *[result_mod_269932], **kwargs_269933)
        
        # SSA join for if statement (line 71)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'do_nothing_show(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'do_nothing_show' in the type store
        # Getting the type of 'stypy_return_type' (line 68)
        stypy_return_type_269935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_269935)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'do_nothing_show'
        return stypy_return_type_269935

    # Assigning a type to the variable 'do_nothing_show' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'do_nothing_show', do_nothing_show)

    @norecursion
    def do_nothing(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'do_nothing'
        module_type_store = module_type_store.open_function_context('do_nothing', 78, 4, False)
        
        # Passed parameters checking function
        do_nothing.stypy_localization = localization
        do_nothing.stypy_type_of_self = None
        do_nothing.stypy_type_store = module_type_store
        do_nothing.stypy_function_name = 'do_nothing'
        do_nothing.stypy_param_names_list = []
        do_nothing.stypy_varargs_param_name = 'args'
        do_nothing.stypy_kwargs_param_name = 'kwargs'
        do_nothing.stypy_call_defaults = defaults
        do_nothing.stypy_call_varargs = varargs
        do_nothing.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'do_nothing', [], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'do_nothing', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'do_nothing(...)' code ##################

        pass
        
        # ################# End of 'do_nothing(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'do_nothing' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_269936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_269936)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'do_nothing'
        return stypy_return_type_269936

    # Assigning a type to the variable 'do_nothing' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'do_nothing', do_nothing)
    
    # Assigning a Call to a Name (line 81):
    
    # Call to getattr(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'backend_mod' (line 81)
    backend_mod_269938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 30), 'backend_mod', False)
    unicode_269939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 43), 'unicode', u'backend_version')
    unicode_269940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 62), 'unicode', u'unknown')
    # Processing the call keyword arguments (line 81)
    kwargs_269941 = {}
    # Getting the type of 'getattr' (line 81)
    getattr_269937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 22), 'getattr', False)
    # Calling getattr(args, kwargs) (line 81)
    getattr_call_result_269942 = invoke(stypy.reporting.localization.Localization(__file__, 81, 22), getattr_269937, *[backend_mod_269938, unicode_269939, unicode_269940], **kwargs_269941)
    
    # Assigning a type to the variable 'backend_version' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'backend_version', getattr_call_result_269942)
    
    # Assigning a Call to a Name (line 83):
    
    # Call to getattr(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'backend_mod' (line 83)
    backend_mod_269944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 19), 'backend_mod', False)
    unicode_269945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 32), 'unicode', u'show')
    # Getting the type of 'do_nothing_show' (line 83)
    do_nothing_show_269946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 40), 'do_nothing_show', False)
    # Processing the call keyword arguments (line 83)
    kwargs_269947 = {}
    # Getting the type of 'getattr' (line 83)
    getattr_269943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 11), 'getattr', False)
    # Calling getattr(args, kwargs) (line 83)
    getattr_call_result_269948 = invoke(stypy.reporting.localization.Localization(__file__, 83, 11), getattr_269943, *[backend_mod_269944, unicode_269945, do_nothing_show_269946], **kwargs_269947)
    
    # Assigning a type to the variable 'show' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'show', getattr_call_result_269948)
    
    # Assigning a Call to a Name (line 85):
    
    # Call to getattr(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'backend_mod' (line 85)
    backend_mod_269950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 34), 'backend_mod', False)
    unicode_269951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 47), 'unicode', u'draw_if_interactive')
    # Getting the type of 'do_nothing' (line 86)
    do_nothing_269952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 34), 'do_nothing', False)
    # Processing the call keyword arguments (line 85)
    kwargs_269953 = {}
    # Getting the type of 'getattr' (line 85)
    getattr_269949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 26), 'getattr', False)
    # Calling getattr(args, kwargs) (line 85)
    getattr_call_result_269954 = invoke(stypy.reporting.localization.Localization(__file__, 85, 26), getattr_269949, *[backend_mod_269950, unicode_269951, do_nothing_269952], **kwargs_269953)
    
    # Assigning a type to the variable 'draw_if_interactive' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'draw_if_interactive', getattr_call_result_269954)
    
    # Call to report(...): (line 88)
    # Processing the call arguments (line 88)
    unicode_269958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 30), 'unicode', u'backend %s version %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 89)
    tuple_269959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 89)
    # Adding element type (line 89)
    # Getting the type of 'name' (line 89)
    name_269960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 31), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 31), tuple_269959, name_269960)
    # Adding element type (line 89)
    # Getting the type of 'backend_version' (line 89)
    backend_version_269961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 37), 'backend_version', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 31), tuple_269959, backend_version_269961)
    
    # Applying the binary operator '%' (line 88)
    result_mod_269962 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 30), '%', unicode_269958, tuple_269959)
    
    # Processing the call keyword arguments (line 88)
    kwargs_269963 = {}
    # Getting the type of 'matplotlib' (line 88)
    matplotlib_269955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'matplotlib', False)
    # Obtaining the member 'verbose' of a type (line 88)
    verbose_269956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 4), matplotlib_269955, 'verbose')
    # Obtaining the member 'report' of a type (line 88)
    report_269957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 4), verbose_269956, 'report')
    # Calling report(args, kwargs) (line 88)
    report_call_result_269964 = invoke(stypy.reporting.localization.Localization(__file__, 88, 4), report_269957, *[result_mod_269962], **kwargs_269963)
    
    # Marking variables as global (line 93)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 93, 4), 'backend')
    
    # Assigning a Name to a Name (line 94):
    # Getting the type of 'name' (line 94)
    name_269965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 14), 'name')
    # Assigning a type to the variable 'backend' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'backend', name_269965)
    
    # Obtaining an instance of the builtin type 'tuple' (line 95)
    tuple_269966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 95)
    # Adding element type (line 95)
    # Getting the type of 'backend_mod' (line 95)
    backend_mod_269967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 11), 'backend_mod')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 11), tuple_269966, backend_mod_269967)
    # Adding element type (line 95)
    # Getting the type of 'new_figure_manager' (line 95)
    new_figure_manager_269968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 24), 'new_figure_manager')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 11), tuple_269966, new_figure_manager_269968)
    # Adding element type (line 95)
    # Getting the type of 'draw_if_interactive' (line 95)
    draw_if_interactive_269969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 44), 'draw_if_interactive')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 11), tuple_269966, draw_if_interactive_269969)
    # Adding element type (line 95)
    # Getting the type of 'show' (line 95)
    show_269970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 65), 'show')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 11), tuple_269966, show_269970)
    
    # Assigning a type to the variable 'stypy_return_type' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'stypy_return_type', tuple_269966)
    
    # ################# End of 'pylab_setup(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pylab_setup' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_269971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_269971)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pylab_setup'
    return stypy_return_type_269971

# Assigning a type to the variable 'pylab_setup' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'pylab_setup', pylab_setup)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

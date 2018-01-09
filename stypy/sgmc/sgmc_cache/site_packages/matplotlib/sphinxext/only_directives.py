
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #
2: # A pair of directives for inserting content that will only appear in
3: # either html or latex.
4: #
5: 
6: from __future__ import (absolute_import, division, print_function,
7:                         unicode_literals)
8: 
9: import six
10: 
11: from docutils.nodes import Body, Element
12: from docutils.parsers.rst import directives
13: 
14: class only_base(Body, Element):
15:     def dont_traverse(self, *args, **kwargs):
16:         return []
17: 
18: class html_only(only_base):
19:     pass
20: 
21: class latex_only(only_base):
22:     pass
23: 
24: def run(content, node_class, state, content_offset):
25:     text = '\n'.join(content)
26:     node = node_class(text)
27:     state.nested_parse(content, content_offset, node)
28:     return [node]
29: 
30: def html_only_directive(name, arguments, options, content, lineno,
31:                         content_offset, block_text, state, state_machine):
32:     return run(content, html_only, state, content_offset)
33: 
34: def latex_only_directive(name, arguments, options, content, lineno,
35:                          content_offset, block_text, state, state_machine):
36:     return run(content, latex_only, state, content_offset)
37: 
38: def builder_inited(app):
39:     if app.builder.name == 'html':
40:         latex_only.traverse = only_base.dont_traverse
41:     else:
42:         html_only.traverse = only_base.dont_traverse
43: 
44: 
45: def setup(app):
46:     app.add_directive('htmlonly', html_only_directive, True, (0, 0, 0))
47:     app.add_directive('latexonly', latex_only_directive, True, (0, 0, 0))
48: 
49:     # This will *really* never see the light of day As it turns out,
50:     # this results in "broken" image nodes since they never get
51:     # processed, so best not to do this.
52:     # app.connect('builder-inited', builder_inited)
53: 
54:     # Add visit/depart methods to HTML-Translator:
55:     def visit_perform(self, node):
56:         pass
57: 
58:     def depart_perform(self, node):
59:         pass
60: 
61:     def visit_ignore(self, node):
62:         node.children = []
63: 
64:     def depart_ignore(self, node):
65:         node.children = []
66: 
67:     app.add_node(html_only,
68:                  html=(visit_perform, depart_perform),
69:                  latex=(visit_ignore, depart_ignore))
70:     app.add_node(latex_only,
71:                  latex=(visit_perform, depart_perform),
72:                  html=(visit_ignore, depart_ignore))
73: 
74:     metadata = {'parallel_read_safe': True, 'parallel_write_safe': True}
75:     return metadata
76: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import six' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/sphinxext/')
import_285812 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'six')

if (type(import_285812) is not StypyTypeError):

    if (import_285812 != 'pyd_module'):
        __import__(import_285812)
        sys_modules_285813 = sys.modules[import_285812]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'six', sys_modules_285813.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'six', import_285812)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/sphinxext/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from docutils.nodes import Body, Element' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/sphinxext/')
import_285814 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'docutils.nodes')

if (type(import_285814) is not StypyTypeError):

    if (import_285814 != 'pyd_module'):
        __import__(import_285814)
        sys_modules_285815 = sys.modules[import_285814]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'docutils.nodes', sys_modules_285815.module_type_store, module_type_store, ['Body', 'Element'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_285815, sys_modules_285815.module_type_store, module_type_store)
    else:
        from docutils.nodes import Body, Element

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'docutils.nodes', None, module_type_store, ['Body', 'Element'], [Body, Element])

else:
    # Assigning a type to the variable 'docutils.nodes' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'docutils.nodes', import_285814)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/sphinxext/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from docutils.parsers.rst import directives' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/sphinxext/')
import_285816 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'docutils.parsers.rst')

if (type(import_285816) is not StypyTypeError):

    if (import_285816 != 'pyd_module'):
        __import__(import_285816)
        sys_modules_285817 = sys.modules[import_285816]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'docutils.parsers.rst', sys_modules_285817.module_type_store, module_type_store, ['directives'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_285817, sys_modules_285817.module_type_store, module_type_store)
    else:
        from docutils.parsers.rst import directives

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'docutils.parsers.rst', None, module_type_store, ['directives'], [directives])

else:
    # Assigning a type to the variable 'docutils.parsers.rst' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'docutils.parsers.rst', import_285816)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/sphinxext/')

# Declaration of the 'only_base' class
# Getting the type of 'Body' (line 14)
Body_285818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 16), 'Body')
# Getting the type of 'Element' (line 14)
Element_285819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 22), 'Element')

class only_base(Body_285818, Element_285819, ):

    @norecursion
    def dont_traverse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dont_traverse'
        module_type_store = module_type_store.open_function_context('dont_traverse', 15, 4, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        only_base.dont_traverse.__dict__.__setitem__('stypy_localization', localization)
        only_base.dont_traverse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        only_base.dont_traverse.__dict__.__setitem__('stypy_type_store', module_type_store)
        only_base.dont_traverse.__dict__.__setitem__('stypy_function_name', 'only_base.dont_traverse')
        only_base.dont_traverse.__dict__.__setitem__('stypy_param_names_list', [])
        only_base.dont_traverse.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        only_base.dont_traverse.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        only_base.dont_traverse.__dict__.__setitem__('stypy_call_defaults', defaults)
        only_base.dont_traverse.__dict__.__setitem__('stypy_call_varargs', varargs)
        only_base.dont_traverse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        only_base.dont_traverse.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'only_base.dont_traverse', [], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dont_traverse', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dont_traverse(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 16)
        list_285820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 16)
        
        # Assigning a type to the variable 'stypy_return_type' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'stypy_return_type', list_285820)
        
        # ################# End of 'dont_traverse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dont_traverse' in the type store
        # Getting the type of 'stypy_return_type' (line 15)
        stypy_return_type_285821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_285821)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dont_traverse'
        return stypy_return_type_285821


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 14, 0, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'only_base.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'only_base' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'only_base', only_base)
# Declaration of the 'html_only' class
# Getting the type of 'only_base' (line 18)
only_base_285822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 16), 'only_base')

class html_only(only_base_285822, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 18, 0, False)
        # Assigning a type to the variable 'self' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'html_only.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'html_only' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'html_only', html_only)
# Declaration of the 'latex_only' class
# Getting the type of 'only_base' (line 21)
only_base_285823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 17), 'only_base')

class latex_only(only_base_285823, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 21, 0, False)
        # Assigning a type to the variable 'self' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'latex_only.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'latex_only' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'latex_only', latex_only)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 24, 0, False)
    
    # Passed parameters checking function
    run.stypy_localization = localization
    run.stypy_type_of_self = None
    run.stypy_type_store = module_type_store
    run.stypy_function_name = 'run'
    run.stypy_param_names_list = ['content', 'node_class', 'state', 'content_offset']
    run.stypy_varargs_param_name = None
    run.stypy_kwargs_param_name = None
    run.stypy_call_defaults = defaults
    run.stypy_call_varargs = varargs
    run.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'run', ['content', 'node_class', 'state', 'content_offset'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'run', localization, ['content', 'node_class', 'state', 'content_offset'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'run(...)' code ##################

    
    # Assigning a Call to a Name (line 25):
    
    # Call to join(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'content' (line 25)
    content_285826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 21), 'content', False)
    # Processing the call keyword arguments (line 25)
    kwargs_285827 = {}
    unicode_285824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 11), 'unicode', u'\n')
    # Obtaining the member 'join' of a type (line 25)
    join_285825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 11), unicode_285824, 'join')
    # Calling join(args, kwargs) (line 25)
    join_call_result_285828 = invoke(stypy.reporting.localization.Localization(__file__, 25, 11), join_285825, *[content_285826], **kwargs_285827)
    
    # Assigning a type to the variable 'text' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'text', join_call_result_285828)
    
    # Assigning a Call to a Name (line 26):
    
    # Call to node_class(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'text' (line 26)
    text_285830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 22), 'text', False)
    # Processing the call keyword arguments (line 26)
    kwargs_285831 = {}
    # Getting the type of 'node_class' (line 26)
    node_class_285829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'node_class', False)
    # Calling node_class(args, kwargs) (line 26)
    node_class_call_result_285832 = invoke(stypy.reporting.localization.Localization(__file__, 26, 11), node_class_285829, *[text_285830], **kwargs_285831)
    
    # Assigning a type to the variable 'node' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'node', node_class_call_result_285832)
    
    # Call to nested_parse(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'content' (line 27)
    content_285835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 23), 'content', False)
    # Getting the type of 'content_offset' (line 27)
    content_offset_285836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 32), 'content_offset', False)
    # Getting the type of 'node' (line 27)
    node_285837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 48), 'node', False)
    # Processing the call keyword arguments (line 27)
    kwargs_285838 = {}
    # Getting the type of 'state' (line 27)
    state_285833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'state', False)
    # Obtaining the member 'nested_parse' of a type (line 27)
    nested_parse_285834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 4), state_285833, 'nested_parse')
    # Calling nested_parse(args, kwargs) (line 27)
    nested_parse_call_result_285839 = invoke(stypy.reporting.localization.Localization(__file__, 27, 4), nested_parse_285834, *[content_285835, content_offset_285836, node_285837], **kwargs_285838)
    
    
    # Obtaining an instance of the builtin type 'list' (line 28)
    list_285840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 28)
    # Adding element type (line 28)
    # Getting the type of 'node' (line 28)
    node_285841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'node')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 11), list_285840, node_285841)
    
    # Assigning a type to the variable 'stypy_return_type' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type', list_285840)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_285842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285842)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_285842

# Assigning a type to the variable 'run' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'run', run)

@norecursion
def html_only_directive(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'html_only_directive'
    module_type_store = module_type_store.open_function_context('html_only_directive', 30, 0, False)
    
    # Passed parameters checking function
    html_only_directive.stypy_localization = localization
    html_only_directive.stypy_type_of_self = None
    html_only_directive.stypy_type_store = module_type_store
    html_only_directive.stypy_function_name = 'html_only_directive'
    html_only_directive.stypy_param_names_list = ['name', 'arguments', 'options', 'content', 'lineno', 'content_offset', 'block_text', 'state', 'state_machine']
    html_only_directive.stypy_varargs_param_name = None
    html_only_directive.stypy_kwargs_param_name = None
    html_only_directive.stypy_call_defaults = defaults
    html_only_directive.stypy_call_varargs = varargs
    html_only_directive.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'html_only_directive', ['name', 'arguments', 'options', 'content', 'lineno', 'content_offset', 'block_text', 'state', 'state_machine'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'html_only_directive', localization, ['name', 'arguments', 'options', 'content', 'lineno', 'content_offset', 'block_text', 'state', 'state_machine'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'html_only_directive(...)' code ##################

    
    # Call to run(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'content' (line 32)
    content_285844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'content', False)
    # Getting the type of 'html_only' (line 32)
    html_only_285845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 24), 'html_only', False)
    # Getting the type of 'state' (line 32)
    state_285846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 35), 'state', False)
    # Getting the type of 'content_offset' (line 32)
    content_offset_285847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 42), 'content_offset', False)
    # Processing the call keyword arguments (line 32)
    kwargs_285848 = {}
    # Getting the type of 'run' (line 32)
    run_285843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 11), 'run', False)
    # Calling run(args, kwargs) (line 32)
    run_call_result_285849 = invoke(stypy.reporting.localization.Localization(__file__, 32, 11), run_285843, *[content_285844, html_only_285845, state_285846, content_offset_285847], **kwargs_285848)
    
    # Assigning a type to the variable 'stypy_return_type' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type', run_call_result_285849)
    
    # ################# End of 'html_only_directive(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'html_only_directive' in the type store
    # Getting the type of 'stypy_return_type' (line 30)
    stypy_return_type_285850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285850)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'html_only_directive'
    return stypy_return_type_285850

# Assigning a type to the variable 'html_only_directive' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'html_only_directive', html_only_directive)

@norecursion
def latex_only_directive(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'latex_only_directive'
    module_type_store = module_type_store.open_function_context('latex_only_directive', 34, 0, False)
    
    # Passed parameters checking function
    latex_only_directive.stypy_localization = localization
    latex_only_directive.stypy_type_of_self = None
    latex_only_directive.stypy_type_store = module_type_store
    latex_only_directive.stypy_function_name = 'latex_only_directive'
    latex_only_directive.stypy_param_names_list = ['name', 'arguments', 'options', 'content', 'lineno', 'content_offset', 'block_text', 'state', 'state_machine']
    latex_only_directive.stypy_varargs_param_name = None
    latex_only_directive.stypy_kwargs_param_name = None
    latex_only_directive.stypy_call_defaults = defaults
    latex_only_directive.stypy_call_varargs = varargs
    latex_only_directive.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'latex_only_directive', ['name', 'arguments', 'options', 'content', 'lineno', 'content_offset', 'block_text', 'state', 'state_machine'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'latex_only_directive', localization, ['name', 'arguments', 'options', 'content', 'lineno', 'content_offset', 'block_text', 'state', 'state_machine'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'latex_only_directive(...)' code ##################

    
    # Call to run(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'content' (line 36)
    content_285852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 15), 'content', False)
    # Getting the type of 'latex_only' (line 36)
    latex_only_285853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 24), 'latex_only', False)
    # Getting the type of 'state' (line 36)
    state_285854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 36), 'state', False)
    # Getting the type of 'content_offset' (line 36)
    content_offset_285855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 43), 'content_offset', False)
    # Processing the call keyword arguments (line 36)
    kwargs_285856 = {}
    # Getting the type of 'run' (line 36)
    run_285851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 11), 'run', False)
    # Calling run(args, kwargs) (line 36)
    run_call_result_285857 = invoke(stypy.reporting.localization.Localization(__file__, 36, 11), run_285851, *[content_285852, latex_only_285853, state_285854, content_offset_285855], **kwargs_285856)
    
    # Assigning a type to the variable 'stypy_return_type' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type', run_call_result_285857)
    
    # ################# End of 'latex_only_directive(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'latex_only_directive' in the type store
    # Getting the type of 'stypy_return_type' (line 34)
    stypy_return_type_285858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285858)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'latex_only_directive'
    return stypy_return_type_285858

# Assigning a type to the variable 'latex_only_directive' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'latex_only_directive', latex_only_directive)

@norecursion
def builder_inited(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'builder_inited'
    module_type_store = module_type_store.open_function_context('builder_inited', 38, 0, False)
    
    # Passed parameters checking function
    builder_inited.stypy_localization = localization
    builder_inited.stypy_type_of_self = None
    builder_inited.stypy_type_store = module_type_store
    builder_inited.stypy_function_name = 'builder_inited'
    builder_inited.stypy_param_names_list = ['app']
    builder_inited.stypy_varargs_param_name = None
    builder_inited.stypy_kwargs_param_name = None
    builder_inited.stypy_call_defaults = defaults
    builder_inited.stypy_call_varargs = varargs
    builder_inited.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'builder_inited', ['app'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'builder_inited', localization, ['app'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'builder_inited(...)' code ##################

    
    
    # Getting the type of 'app' (line 39)
    app_285859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 7), 'app')
    # Obtaining the member 'builder' of a type (line 39)
    builder_285860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 7), app_285859, 'builder')
    # Obtaining the member 'name' of a type (line 39)
    name_285861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 7), builder_285860, 'name')
    unicode_285862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 27), 'unicode', u'html')
    # Applying the binary operator '==' (line 39)
    result_eq_285863 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 7), '==', name_285861, unicode_285862)
    
    # Testing the type of an if condition (line 39)
    if_condition_285864 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 4), result_eq_285863)
    # Assigning a type to the variable 'if_condition_285864' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'if_condition_285864', if_condition_285864)
    # SSA begins for if statement (line 39)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Attribute (line 40):
    # Getting the type of 'only_base' (line 40)
    only_base_285865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 30), 'only_base')
    # Obtaining the member 'dont_traverse' of a type (line 40)
    dont_traverse_285866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 30), only_base_285865, 'dont_traverse')
    # Getting the type of 'latex_only' (line 40)
    latex_only_285867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'latex_only')
    # Setting the type of the member 'traverse' of a type (line 40)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), latex_only_285867, 'traverse', dont_traverse_285866)
    # SSA branch for the else part of an if statement (line 39)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Attribute to a Attribute (line 42):
    # Getting the type of 'only_base' (line 42)
    only_base_285868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 29), 'only_base')
    # Obtaining the member 'dont_traverse' of a type (line 42)
    dont_traverse_285869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 29), only_base_285868, 'dont_traverse')
    # Getting the type of 'html_only' (line 42)
    html_only_285870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'html_only')
    # Setting the type of the member 'traverse' of a type (line 42)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), html_only_285870, 'traverse', dont_traverse_285869)
    # SSA join for if statement (line 39)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'builder_inited(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'builder_inited' in the type store
    # Getting the type of 'stypy_return_type' (line 38)
    stypy_return_type_285871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285871)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'builder_inited'
    return stypy_return_type_285871

# Assigning a type to the variable 'builder_inited' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'builder_inited', builder_inited)

@norecursion
def setup(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'setup'
    module_type_store = module_type_store.open_function_context('setup', 45, 0, False)
    
    # Passed parameters checking function
    setup.stypy_localization = localization
    setup.stypy_type_of_self = None
    setup.stypy_type_store = module_type_store
    setup.stypy_function_name = 'setup'
    setup.stypy_param_names_list = ['app']
    setup.stypy_varargs_param_name = None
    setup.stypy_kwargs_param_name = None
    setup.stypy_call_defaults = defaults
    setup.stypy_call_varargs = varargs
    setup.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'setup', ['app'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'setup', localization, ['app'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'setup(...)' code ##################

    
    # Call to add_directive(...): (line 46)
    # Processing the call arguments (line 46)
    unicode_285874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 22), 'unicode', u'htmlonly')
    # Getting the type of 'html_only_directive' (line 46)
    html_only_directive_285875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 34), 'html_only_directive', False)
    # Getting the type of 'True' (line 46)
    True_285876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 55), 'True', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 46)
    tuple_285877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 62), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 46)
    # Adding element type (line 46)
    int_285878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 62), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 62), tuple_285877, int_285878)
    # Adding element type (line 46)
    int_285879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 65), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 62), tuple_285877, int_285879)
    # Adding element type (line 46)
    int_285880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 68), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 62), tuple_285877, int_285880)
    
    # Processing the call keyword arguments (line 46)
    kwargs_285881 = {}
    # Getting the type of 'app' (line 46)
    app_285872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'app', False)
    # Obtaining the member 'add_directive' of a type (line 46)
    add_directive_285873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 4), app_285872, 'add_directive')
    # Calling add_directive(args, kwargs) (line 46)
    add_directive_call_result_285882 = invoke(stypy.reporting.localization.Localization(__file__, 46, 4), add_directive_285873, *[unicode_285874, html_only_directive_285875, True_285876, tuple_285877], **kwargs_285881)
    
    
    # Call to add_directive(...): (line 47)
    # Processing the call arguments (line 47)
    unicode_285885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 22), 'unicode', u'latexonly')
    # Getting the type of 'latex_only_directive' (line 47)
    latex_only_directive_285886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 35), 'latex_only_directive', False)
    # Getting the type of 'True' (line 47)
    True_285887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 57), 'True', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 47)
    tuple_285888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 64), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 47)
    # Adding element type (line 47)
    int_285889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 64), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 64), tuple_285888, int_285889)
    # Adding element type (line 47)
    int_285890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 67), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 64), tuple_285888, int_285890)
    # Adding element type (line 47)
    int_285891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 70), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 64), tuple_285888, int_285891)
    
    # Processing the call keyword arguments (line 47)
    kwargs_285892 = {}
    # Getting the type of 'app' (line 47)
    app_285883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'app', False)
    # Obtaining the member 'add_directive' of a type (line 47)
    add_directive_285884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 4), app_285883, 'add_directive')
    # Calling add_directive(args, kwargs) (line 47)
    add_directive_call_result_285893 = invoke(stypy.reporting.localization.Localization(__file__, 47, 4), add_directive_285884, *[unicode_285885, latex_only_directive_285886, True_285887, tuple_285888], **kwargs_285892)
    

    @norecursion
    def visit_perform(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_perform'
        module_type_store = module_type_store.open_function_context('visit_perform', 55, 4, False)
        
        # Passed parameters checking function
        visit_perform.stypy_localization = localization
        visit_perform.stypy_type_of_self = None
        visit_perform.stypy_type_store = module_type_store
        visit_perform.stypy_function_name = 'visit_perform'
        visit_perform.stypy_param_names_list = ['self', 'node']
        visit_perform.stypy_varargs_param_name = None
        visit_perform.stypy_kwargs_param_name = None
        visit_perform.stypy_call_defaults = defaults
        visit_perform.stypy_call_varargs = varargs
        visit_perform.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'visit_perform', ['self', 'node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_perform', localization, ['self', 'node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_perform(...)' code ##################

        pass
        
        # ################# End of 'visit_perform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_perform' in the type store
        # Getting the type of 'stypy_return_type' (line 55)
        stypy_return_type_285894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_285894)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_perform'
        return stypy_return_type_285894

    # Assigning a type to the variable 'visit_perform' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'visit_perform', visit_perform)

    @norecursion
    def depart_perform(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'depart_perform'
        module_type_store = module_type_store.open_function_context('depart_perform', 58, 4, False)
        
        # Passed parameters checking function
        depart_perform.stypy_localization = localization
        depart_perform.stypy_type_of_self = None
        depart_perform.stypy_type_store = module_type_store
        depart_perform.stypy_function_name = 'depart_perform'
        depart_perform.stypy_param_names_list = ['self', 'node']
        depart_perform.stypy_varargs_param_name = None
        depart_perform.stypy_kwargs_param_name = None
        depart_perform.stypy_call_defaults = defaults
        depart_perform.stypy_call_varargs = varargs
        depart_perform.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'depart_perform', ['self', 'node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'depart_perform', localization, ['self', 'node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'depart_perform(...)' code ##################

        pass
        
        # ################# End of 'depart_perform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'depart_perform' in the type store
        # Getting the type of 'stypy_return_type' (line 58)
        stypy_return_type_285895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_285895)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'depart_perform'
        return stypy_return_type_285895

    # Assigning a type to the variable 'depart_perform' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'depart_perform', depart_perform)

    @norecursion
    def visit_ignore(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_ignore'
        module_type_store = module_type_store.open_function_context('visit_ignore', 61, 4, False)
        
        # Passed parameters checking function
        visit_ignore.stypy_localization = localization
        visit_ignore.stypy_type_of_self = None
        visit_ignore.stypy_type_store = module_type_store
        visit_ignore.stypy_function_name = 'visit_ignore'
        visit_ignore.stypy_param_names_list = ['self', 'node']
        visit_ignore.stypy_varargs_param_name = None
        visit_ignore.stypy_kwargs_param_name = None
        visit_ignore.stypy_call_defaults = defaults
        visit_ignore.stypy_call_varargs = varargs
        visit_ignore.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'visit_ignore', ['self', 'node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_ignore', localization, ['self', 'node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_ignore(...)' code ##################

        
        # Assigning a List to a Attribute (line 62):
        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_285896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        
        # Getting the type of 'node' (line 62)
        node_285897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'node')
        # Setting the type of the member 'children' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), node_285897, 'children', list_285896)
        
        # ################# End of 'visit_ignore(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_ignore' in the type store
        # Getting the type of 'stypy_return_type' (line 61)
        stypy_return_type_285898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_285898)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_ignore'
        return stypy_return_type_285898

    # Assigning a type to the variable 'visit_ignore' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'visit_ignore', visit_ignore)

    @norecursion
    def depart_ignore(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'depart_ignore'
        module_type_store = module_type_store.open_function_context('depart_ignore', 64, 4, False)
        
        # Passed parameters checking function
        depart_ignore.stypy_localization = localization
        depart_ignore.stypy_type_of_self = None
        depart_ignore.stypy_type_store = module_type_store
        depart_ignore.stypy_function_name = 'depart_ignore'
        depart_ignore.stypy_param_names_list = ['self', 'node']
        depart_ignore.stypy_varargs_param_name = None
        depart_ignore.stypy_kwargs_param_name = None
        depart_ignore.stypy_call_defaults = defaults
        depart_ignore.stypy_call_varargs = varargs
        depart_ignore.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'depart_ignore', ['self', 'node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'depart_ignore', localization, ['self', 'node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'depart_ignore(...)' code ##################

        
        # Assigning a List to a Attribute (line 65):
        
        # Obtaining an instance of the builtin type 'list' (line 65)
        list_285899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 65)
        
        # Getting the type of 'node' (line 65)
        node_285900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'node')
        # Setting the type of the member 'children' of a type (line 65)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), node_285900, 'children', list_285899)
        
        # ################# End of 'depart_ignore(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'depart_ignore' in the type store
        # Getting the type of 'stypy_return_type' (line 64)
        stypy_return_type_285901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_285901)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'depart_ignore'
        return stypy_return_type_285901

    # Assigning a type to the variable 'depart_ignore' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'depart_ignore', depart_ignore)
    
    # Call to add_node(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'html_only' (line 67)
    html_only_285904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 17), 'html_only', False)
    # Processing the call keyword arguments (line 67)
    
    # Obtaining an instance of the builtin type 'tuple' (line 68)
    tuple_285905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 68)
    # Adding element type (line 68)
    # Getting the type of 'visit_perform' (line 68)
    visit_perform_285906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 23), 'visit_perform', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 23), tuple_285905, visit_perform_285906)
    # Adding element type (line 68)
    # Getting the type of 'depart_perform' (line 68)
    depart_perform_285907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 38), 'depart_perform', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 23), tuple_285905, depart_perform_285907)
    
    keyword_285908 = tuple_285905
    
    # Obtaining an instance of the builtin type 'tuple' (line 69)
    tuple_285909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 69)
    # Adding element type (line 69)
    # Getting the type of 'visit_ignore' (line 69)
    visit_ignore_285910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 24), 'visit_ignore', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 24), tuple_285909, visit_ignore_285910)
    # Adding element type (line 69)
    # Getting the type of 'depart_ignore' (line 69)
    depart_ignore_285911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 38), 'depart_ignore', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 24), tuple_285909, depart_ignore_285911)
    
    keyword_285912 = tuple_285909
    kwargs_285913 = {'latex': keyword_285912, 'html': keyword_285908}
    # Getting the type of 'app' (line 67)
    app_285902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'app', False)
    # Obtaining the member 'add_node' of a type (line 67)
    add_node_285903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 4), app_285902, 'add_node')
    # Calling add_node(args, kwargs) (line 67)
    add_node_call_result_285914 = invoke(stypy.reporting.localization.Localization(__file__, 67, 4), add_node_285903, *[html_only_285904], **kwargs_285913)
    
    
    # Call to add_node(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'latex_only' (line 70)
    latex_only_285917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 17), 'latex_only', False)
    # Processing the call keyword arguments (line 70)
    
    # Obtaining an instance of the builtin type 'tuple' (line 71)
    tuple_285918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 71)
    # Adding element type (line 71)
    # Getting the type of 'visit_perform' (line 71)
    visit_perform_285919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 24), 'visit_perform', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 24), tuple_285918, visit_perform_285919)
    # Adding element type (line 71)
    # Getting the type of 'depart_perform' (line 71)
    depart_perform_285920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 39), 'depart_perform', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 24), tuple_285918, depart_perform_285920)
    
    keyword_285921 = tuple_285918
    
    # Obtaining an instance of the builtin type 'tuple' (line 72)
    tuple_285922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 72)
    # Adding element type (line 72)
    # Getting the type of 'visit_ignore' (line 72)
    visit_ignore_285923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 23), 'visit_ignore', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 23), tuple_285922, visit_ignore_285923)
    # Adding element type (line 72)
    # Getting the type of 'depart_ignore' (line 72)
    depart_ignore_285924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 37), 'depart_ignore', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 23), tuple_285922, depart_ignore_285924)
    
    keyword_285925 = tuple_285922
    kwargs_285926 = {'latex': keyword_285921, 'html': keyword_285925}
    # Getting the type of 'app' (line 70)
    app_285915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'app', False)
    # Obtaining the member 'add_node' of a type (line 70)
    add_node_285916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 4), app_285915, 'add_node')
    # Calling add_node(args, kwargs) (line 70)
    add_node_call_result_285927 = invoke(stypy.reporting.localization.Localization(__file__, 70, 4), add_node_285916, *[latex_only_285917], **kwargs_285926)
    
    
    # Assigning a Dict to a Name (line 74):
    
    # Obtaining an instance of the builtin type 'dict' (line 74)
    dict_285928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 74)
    # Adding element type (key, value) (line 74)
    unicode_285929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 16), 'unicode', u'parallel_read_safe')
    # Getting the type of 'True' (line 74)
    True_285930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 38), 'True')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 15), dict_285928, (unicode_285929, True_285930))
    # Adding element type (key, value) (line 74)
    unicode_285931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 44), 'unicode', u'parallel_write_safe')
    # Getting the type of 'True' (line 74)
    True_285932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 67), 'True')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 15), dict_285928, (unicode_285931, True_285932))
    
    # Assigning a type to the variable 'metadata' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'metadata', dict_285928)
    # Getting the type of 'metadata' (line 75)
    metadata_285933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 11), 'metadata')
    # Assigning a type to the variable 'stypy_return_type' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type', metadata_285933)
    
    # ################# End of 'setup(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'setup' in the type store
    # Getting the type of 'stypy_return_type' (line 45)
    stypy_return_type_285934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285934)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'setup'
    return stypy_return_type_285934

# Assigning a type to the variable 'setup' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'setup', setup)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

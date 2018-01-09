
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import six
5: 
6: import os
7: import sys
8: from hashlib import md5
9: 
10: from docutils import nodes
11: from docutils.parsers.rst import directives
12: import warnings
13: 
14: from matplotlib import rcParams
15: from matplotlib.mathtext import MathTextParser
16: rcParams['mathtext.fontset'] = 'cm'
17: mathtext_parser = MathTextParser("Bitmap")
18: 
19: # Define LaTeX math node:
20: class latex_math(nodes.General, nodes.Element):
21:     pass
22: 
23: def fontset_choice(arg):
24:     return directives.choice(arg, ['cm', 'stix', 'stixsans'])
25: 
26: options_spec = {'fontset': fontset_choice}
27: 
28: def math_role(role, rawtext, text, lineno, inliner,
29:               options={}, content=[]):
30:     i = rawtext.find('`')
31:     latex = rawtext[i+1:-1]
32:     node = latex_math(rawtext)
33:     node['latex'] = latex
34:     node['fontset'] = options.get('fontset', 'cm')
35:     return [node], []
36: math_role.options = options_spec
37: 
38: def math_directive(name, arguments, options, content, lineno,
39:                    content_offset, block_text, state, state_machine):
40:     latex = ''.join(content)
41:     node = latex_math(block_text)
42:     node['latex'] = latex
43:     node['fontset'] = options.get('fontset', 'cm')
44:     return [node]
45: 
46: # This uses mathtext to render the expression
47: def latex2png(latex, filename, fontset='cm'):
48:     latex = "$%s$" % latex
49:     orig_fontset = rcParams['mathtext.fontset']
50:     rcParams['mathtext.fontset'] = fontset
51:     if os.path.exists(filename):
52:         depth = mathtext_parser.get_depth(latex, dpi=100)
53:     else:
54:         try:
55:             depth = mathtext_parser.to_png(filename, latex, dpi=100)
56:         except:
57:             warnings.warn("Could not render math expression %s" % latex,
58:                           Warning)
59:             depth = 0
60:     rcParams['mathtext.fontset'] = orig_fontset
61:     sys.stdout.write("#")
62:     sys.stdout.flush()
63:     return depth
64: 
65: # LaTeX to HTML translation stuff:
66: def latex2html(node, source):
67:     inline = isinstance(node.parent, nodes.TextElement)
68:     latex = node['latex']
69:     name = 'math-%s' % md5(latex.encode()).hexdigest()[-10:]
70: 
71:     destdir = os.path.join(setup.app.builder.outdir, '_images', 'mathmpl')
72:     if not os.path.exists(destdir):
73:         os.makedirs(destdir)
74:     dest = os.path.join(destdir, '%s.png' % name)
75:     path = '/'.join((setup.app.builder.imgpath, 'mathmpl'))
76: 
77:     depth = latex2png(latex, dest, node['fontset'])
78: 
79:     if inline:
80:         cls = ''
81:     else:
82:         cls = 'class="center" '
83:     if inline and depth != 0:
84:         style = 'style="position: relative; bottom: -%dpx"' % (depth + 1)
85:     else:
86:         style = ''
87: 
88:     return '<img src="%s/%s.png" %s%s/>' % (path, name, cls, style)
89: 
90: 
91: def setup(app):
92:     setup.app = app
93: 
94:     # Add visit/depart methods to HTML-Translator:
95:     def visit_latex_math_html(self, node):
96:         source = self.document.attributes['source']
97:         self.body.append(latex2html(node, source))
98: 
99:     def depart_latex_math_html(self, node):
100:         pass
101: 
102:     # Add visit/depart methods to LaTeX-Translator:
103:     def visit_latex_math_latex(self, node):
104:         inline = isinstance(node.parent, nodes.TextElement)
105:         if inline:
106:             self.body.append('$%s$' % node['latex'])
107:         else:
108:             self.body.extend(['\\begin{equation}',
109:                               node['latex'],
110:                               '\\end{equation}'])
111: 
112:     def depart_latex_math_latex(self, node):
113:         pass
114: 
115:     app.add_node(latex_math,
116:                  html=(visit_latex_math_html, depart_latex_math_html),
117:                  latex=(visit_latex_math_latex, depart_latex_math_latex))
118:     app.add_role('math', math_role)
119:     app.add_directive('math', math_directive,
120:                       True, (0, 0, 0), **options_spec)
121: 
122:     metadata = {'parallel_read_safe': True, 'parallel_write_safe': True}
123:     return metadata
124: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import six' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/sphinxext/')
import_285465 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six')

if (type(import_285465) is not StypyTypeError):

    if (import_285465 != 'pyd_module'):
        __import__(import_285465)
        sys_modules_285466 = sys.modules[import_285465]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', sys_modules_285466.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', import_285465)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/sphinxext/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import os' statement (line 6)
import os

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import sys' statement (line 7)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from hashlib import md5' statement (line 8)
try:
    from hashlib import md5

except:
    md5 = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'hashlib', None, module_type_store, ['md5'], [md5])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from docutils import nodes' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/sphinxext/')
import_285467 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'docutils')

if (type(import_285467) is not StypyTypeError):

    if (import_285467 != 'pyd_module'):
        __import__(import_285467)
        sys_modules_285468 = sys.modules[import_285467]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'docutils', sys_modules_285468.module_type_store, module_type_store, ['nodes'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_285468, sys_modules_285468.module_type_store, module_type_store)
    else:
        from docutils import nodes

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'docutils', None, module_type_store, ['nodes'], [nodes])

else:
    # Assigning a type to the variable 'docutils' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'docutils', import_285467)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/sphinxext/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from docutils.parsers.rst import directives' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/sphinxext/')
import_285469 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'docutils.parsers.rst')

if (type(import_285469) is not StypyTypeError):

    if (import_285469 != 'pyd_module'):
        __import__(import_285469)
        sys_modules_285470 = sys.modules[import_285469]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'docutils.parsers.rst', sys_modules_285470.module_type_store, module_type_store, ['directives'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_285470, sys_modules_285470.module_type_store, module_type_store)
    else:
        from docutils.parsers.rst import directives

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'docutils.parsers.rst', None, module_type_store, ['directives'], [directives])

else:
    # Assigning a type to the variable 'docutils.parsers.rst' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'docutils.parsers.rst', import_285469)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/sphinxext/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import warnings' statement (line 12)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from matplotlib import rcParams' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/sphinxext/')
import_285471 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib')

if (type(import_285471) is not StypyTypeError):

    if (import_285471 != 'pyd_module'):
        __import__(import_285471)
        sys_modules_285472 = sys.modules[import_285471]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib', sys_modules_285472.module_type_store, module_type_store, ['rcParams'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_285472, sys_modules_285472.module_type_store, module_type_store)
    else:
        from matplotlib import rcParams

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib', None, module_type_store, ['rcParams'], [rcParams])

else:
    # Assigning a type to the variable 'matplotlib' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib', import_285471)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/sphinxext/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from matplotlib.mathtext import MathTextParser' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/sphinxext/')
import_285473 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.mathtext')

if (type(import_285473) is not StypyTypeError):

    if (import_285473 != 'pyd_module'):
        __import__(import_285473)
        sys_modules_285474 = sys.modules[import_285473]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.mathtext', sys_modules_285474.module_type_store, module_type_store, ['MathTextParser'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_285474, sys_modules_285474.module_type_store, module_type_store)
    else:
        from matplotlib.mathtext import MathTextParser

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.mathtext', None, module_type_store, ['MathTextParser'], [MathTextParser])

else:
    # Assigning a type to the variable 'matplotlib.mathtext' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.mathtext', import_285473)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/sphinxext/')


# Assigning a Str to a Subscript (line 16):
unicode_285475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 31), 'unicode', u'cm')
# Getting the type of 'rcParams' (line 16)
rcParams_285476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'rcParams')
unicode_285477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 9), 'unicode', u'mathtext.fontset')
# Storing an element on a container (line 16)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 0), rcParams_285476, (unicode_285477, unicode_285475))

# Assigning a Call to a Name (line 17):

# Call to MathTextParser(...): (line 17)
# Processing the call arguments (line 17)
unicode_285479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 33), 'unicode', u'Bitmap')
# Processing the call keyword arguments (line 17)
kwargs_285480 = {}
# Getting the type of 'MathTextParser' (line 17)
MathTextParser_285478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 18), 'MathTextParser', False)
# Calling MathTextParser(args, kwargs) (line 17)
MathTextParser_call_result_285481 = invoke(stypy.reporting.localization.Localization(__file__, 17, 18), MathTextParser_285478, *[unicode_285479], **kwargs_285480)

# Assigning a type to the variable 'mathtext_parser' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'mathtext_parser', MathTextParser_call_result_285481)
# Declaration of the 'latex_math' class
# Getting the type of 'nodes' (line 20)
nodes_285482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 17), 'nodes')
# Obtaining the member 'General' of a type (line 20)
General_285483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 17), nodes_285482, 'General')
# Getting the type of 'nodes' (line 20)
nodes_285484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 32), 'nodes')
# Obtaining the member 'Element' of a type (line 20)
Element_285485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 32), nodes_285484, 'Element')

class latex_math(General_285483, Element_285485, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 20, 0, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'latex_math.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'latex_math' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'latex_math', latex_math)

@norecursion
def fontset_choice(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'fontset_choice'
    module_type_store = module_type_store.open_function_context('fontset_choice', 23, 0, False)
    
    # Passed parameters checking function
    fontset_choice.stypy_localization = localization
    fontset_choice.stypy_type_of_self = None
    fontset_choice.stypy_type_store = module_type_store
    fontset_choice.stypy_function_name = 'fontset_choice'
    fontset_choice.stypy_param_names_list = ['arg']
    fontset_choice.stypy_varargs_param_name = None
    fontset_choice.stypy_kwargs_param_name = None
    fontset_choice.stypy_call_defaults = defaults
    fontset_choice.stypy_call_varargs = varargs
    fontset_choice.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fontset_choice', ['arg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fontset_choice', localization, ['arg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fontset_choice(...)' code ##################

    
    # Call to choice(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 'arg' (line 24)
    arg_285488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 29), 'arg', False)
    
    # Obtaining an instance of the builtin type 'list' (line 24)
    list_285489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 24)
    # Adding element type (line 24)
    unicode_285490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 35), 'unicode', u'cm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 34), list_285489, unicode_285490)
    # Adding element type (line 24)
    unicode_285491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 41), 'unicode', u'stix')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 34), list_285489, unicode_285491)
    # Adding element type (line 24)
    unicode_285492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 49), 'unicode', u'stixsans')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 34), list_285489, unicode_285492)
    
    # Processing the call keyword arguments (line 24)
    kwargs_285493 = {}
    # Getting the type of 'directives' (line 24)
    directives_285486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 11), 'directives', False)
    # Obtaining the member 'choice' of a type (line 24)
    choice_285487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 11), directives_285486, 'choice')
    # Calling choice(args, kwargs) (line 24)
    choice_call_result_285494 = invoke(stypy.reporting.localization.Localization(__file__, 24, 11), choice_285487, *[arg_285488, list_285489], **kwargs_285493)
    
    # Assigning a type to the variable 'stypy_return_type' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type', choice_call_result_285494)
    
    # ################# End of 'fontset_choice(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fontset_choice' in the type store
    # Getting the type of 'stypy_return_type' (line 23)
    stypy_return_type_285495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285495)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fontset_choice'
    return stypy_return_type_285495

# Assigning a type to the variable 'fontset_choice' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'fontset_choice', fontset_choice)

# Assigning a Dict to a Name (line 26):

# Obtaining an instance of the builtin type 'dict' (line 26)
dict_285496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 15), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 26)
# Adding element type (key, value) (line 26)
unicode_285497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 16), 'unicode', u'fontset')
# Getting the type of 'fontset_choice' (line 26)
fontset_choice_285498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 27), 'fontset_choice')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 15), dict_285496, (unicode_285497, fontset_choice_285498))

# Assigning a type to the variable 'options_spec' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'options_spec', dict_285496)

@norecursion
def math_role(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'dict' (line 29)
    dict_285499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 22), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 29)
    
    
    # Obtaining an instance of the builtin type 'list' (line 29)
    list_285500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 29)
    
    defaults = [dict_285499, list_285500]
    # Create a new context for function 'math_role'
    module_type_store = module_type_store.open_function_context('math_role', 28, 0, False)
    
    # Passed parameters checking function
    math_role.stypy_localization = localization
    math_role.stypy_type_of_self = None
    math_role.stypy_type_store = module_type_store
    math_role.stypy_function_name = 'math_role'
    math_role.stypy_param_names_list = ['role', 'rawtext', 'text', 'lineno', 'inliner', 'options', 'content']
    math_role.stypy_varargs_param_name = None
    math_role.stypy_kwargs_param_name = None
    math_role.stypy_call_defaults = defaults
    math_role.stypy_call_varargs = varargs
    math_role.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'math_role', ['role', 'rawtext', 'text', 'lineno', 'inliner', 'options', 'content'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'math_role', localization, ['role', 'rawtext', 'text', 'lineno', 'inliner', 'options', 'content'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'math_role(...)' code ##################

    
    # Assigning a Call to a Name (line 30):
    
    # Call to find(...): (line 30)
    # Processing the call arguments (line 30)
    unicode_285503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 21), 'unicode', u'`')
    # Processing the call keyword arguments (line 30)
    kwargs_285504 = {}
    # Getting the type of 'rawtext' (line 30)
    rawtext_285501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'rawtext', False)
    # Obtaining the member 'find' of a type (line 30)
    find_285502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), rawtext_285501, 'find')
    # Calling find(args, kwargs) (line 30)
    find_call_result_285505 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), find_285502, *[unicode_285503], **kwargs_285504)
    
    # Assigning a type to the variable 'i' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'i', find_call_result_285505)
    
    # Assigning a Subscript to a Name (line 31):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 31)
    i_285506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'i')
    int_285507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 22), 'int')
    # Applying the binary operator '+' (line 31)
    result_add_285508 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 20), '+', i_285506, int_285507)
    
    int_285509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 24), 'int')
    slice_285510 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 31, 12), result_add_285508, int_285509, None)
    # Getting the type of 'rawtext' (line 31)
    rawtext_285511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'rawtext')
    # Obtaining the member '__getitem__' of a type (line 31)
    getitem___285512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 12), rawtext_285511, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 31)
    subscript_call_result_285513 = invoke(stypy.reporting.localization.Localization(__file__, 31, 12), getitem___285512, slice_285510)
    
    # Assigning a type to the variable 'latex' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'latex', subscript_call_result_285513)
    
    # Assigning a Call to a Name (line 32):
    
    # Call to latex_math(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'rawtext' (line 32)
    rawtext_285515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 22), 'rawtext', False)
    # Processing the call keyword arguments (line 32)
    kwargs_285516 = {}
    # Getting the type of 'latex_math' (line 32)
    latex_math_285514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 11), 'latex_math', False)
    # Calling latex_math(args, kwargs) (line 32)
    latex_math_call_result_285517 = invoke(stypy.reporting.localization.Localization(__file__, 32, 11), latex_math_285514, *[rawtext_285515], **kwargs_285516)
    
    # Assigning a type to the variable 'node' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'node', latex_math_call_result_285517)
    
    # Assigning a Name to a Subscript (line 33):
    # Getting the type of 'latex' (line 33)
    latex_285518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 20), 'latex')
    # Getting the type of 'node' (line 33)
    node_285519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'node')
    unicode_285520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 9), 'unicode', u'latex')
    # Storing an element on a container (line 33)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 4), node_285519, (unicode_285520, latex_285518))
    
    # Assigning a Call to a Subscript (line 34):
    
    # Call to get(...): (line 34)
    # Processing the call arguments (line 34)
    unicode_285523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 34), 'unicode', u'fontset')
    unicode_285524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 45), 'unicode', u'cm')
    # Processing the call keyword arguments (line 34)
    kwargs_285525 = {}
    # Getting the type of 'options' (line 34)
    options_285521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 22), 'options', False)
    # Obtaining the member 'get' of a type (line 34)
    get_285522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 22), options_285521, 'get')
    # Calling get(args, kwargs) (line 34)
    get_call_result_285526 = invoke(stypy.reporting.localization.Localization(__file__, 34, 22), get_285522, *[unicode_285523, unicode_285524], **kwargs_285525)
    
    # Getting the type of 'node' (line 34)
    node_285527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'node')
    unicode_285528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 9), 'unicode', u'fontset')
    # Storing an element on a container (line 34)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 4), node_285527, (unicode_285528, get_call_result_285526))
    
    # Obtaining an instance of the builtin type 'tuple' (line 35)
    tuple_285529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 35)
    # Adding element type (line 35)
    
    # Obtaining an instance of the builtin type 'list' (line 35)
    list_285530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 35)
    # Adding element type (line 35)
    # Getting the type of 'node' (line 35)
    node_285531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'node')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 11), list_285530, node_285531)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 11), tuple_285529, list_285530)
    # Adding element type (line 35)
    
    # Obtaining an instance of the builtin type 'list' (line 35)
    list_285532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 35)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 11), tuple_285529, list_285532)
    
    # Assigning a type to the variable 'stypy_return_type' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type', tuple_285529)
    
    # ################# End of 'math_role(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'math_role' in the type store
    # Getting the type of 'stypy_return_type' (line 28)
    stypy_return_type_285533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285533)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'math_role'
    return stypy_return_type_285533

# Assigning a type to the variable 'math_role' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'math_role', math_role)

# Assigning a Name to a Attribute (line 36):
# Getting the type of 'options_spec' (line 36)
options_spec_285534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 20), 'options_spec')
# Getting the type of 'math_role' (line 36)
math_role_285535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'math_role')
# Setting the type of the member 'options' of a type (line 36)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 0), math_role_285535, 'options', options_spec_285534)

@norecursion
def math_directive(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'math_directive'
    module_type_store = module_type_store.open_function_context('math_directive', 38, 0, False)
    
    # Passed parameters checking function
    math_directive.stypy_localization = localization
    math_directive.stypy_type_of_self = None
    math_directive.stypy_type_store = module_type_store
    math_directive.stypy_function_name = 'math_directive'
    math_directive.stypy_param_names_list = ['name', 'arguments', 'options', 'content', 'lineno', 'content_offset', 'block_text', 'state', 'state_machine']
    math_directive.stypy_varargs_param_name = None
    math_directive.stypy_kwargs_param_name = None
    math_directive.stypy_call_defaults = defaults
    math_directive.stypy_call_varargs = varargs
    math_directive.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'math_directive', ['name', 'arguments', 'options', 'content', 'lineno', 'content_offset', 'block_text', 'state', 'state_machine'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'math_directive', localization, ['name', 'arguments', 'options', 'content', 'lineno', 'content_offset', 'block_text', 'state', 'state_machine'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'math_directive(...)' code ##################

    
    # Assigning a Call to a Name (line 40):
    
    # Call to join(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'content' (line 40)
    content_285538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 20), 'content', False)
    # Processing the call keyword arguments (line 40)
    kwargs_285539 = {}
    unicode_285536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 12), 'unicode', u'')
    # Obtaining the member 'join' of a type (line 40)
    join_285537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 12), unicode_285536, 'join')
    # Calling join(args, kwargs) (line 40)
    join_call_result_285540 = invoke(stypy.reporting.localization.Localization(__file__, 40, 12), join_285537, *[content_285538], **kwargs_285539)
    
    # Assigning a type to the variable 'latex' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'latex', join_call_result_285540)
    
    # Assigning a Call to a Name (line 41):
    
    # Call to latex_math(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'block_text' (line 41)
    block_text_285542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 22), 'block_text', False)
    # Processing the call keyword arguments (line 41)
    kwargs_285543 = {}
    # Getting the type of 'latex_math' (line 41)
    latex_math_285541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 11), 'latex_math', False)
    # Calling latex_math(args, kwargs) (line 41)
    latex_math_call_result_285544 = invoke(stypy.reporting.localization.Localization(__file__, 41, 11), latex_math_285541, *[block_text_285542], **kwargs_285543)
    
    # Assigning a type to the variable 'node' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'node', latex_math_call_result_285544)
    
    # Assigning a Name to a Subscript (line 42):
    # Getting the type of 'latex' (line 42)
    latex_285545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 20), 'latex')
    # Getting the type of 'node' (line 42)
    node_285546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'node')
    unicode_285547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 9), 'unicode', u'latex')
    # Storing an element on a container (line 42)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 4), node_285546, (unicode_285547, latex_285545))
    
    # Assigning a Call to a Subscript (line 43):
    
    # Call to get(...): (line 43)
    # Processing the call arguments (line 43)
    unicode_285550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 34), 'unicode', u'fontset')
    unicode_285551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 45), 'unicode', u'cm')
    # Processing the call keyword arguments (line 43)
    kwargs_285552 = {}
    # Getting the type of 'options' (line 43)
    options_285548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 22), 'options', False)
    # Obtaining the member 'get' of a type (line 43)
    get_285549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 22), options_285548, 'get')
    # Calling get(args, kwargs) (line 43)
    get_call_result_285553 = invoke(stypy.reporting.localization.Localization(__file__, 43, 22), get_285549, *[unicode_285550, unicode_285551], **kwargs_285552)
    
    # Getting the type of 'node' (line 43)
    node_285554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'node')
    unicode_285555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 9), 'unicode', u'fontset')
    # Storing an element on a container (line 43)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 4), node_285554, (unicode_285555, get_call_result_285553))
    
    # Obtaining an instance of the builtin type 'list' (line 44)
    list_285556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 44)
    # Adding element type (line 44)
    # Getting the type of 'node' (line 44)
    node_285557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'node')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 11), list_285556, node_285557)
    
    # Assigning a type to the variable 'stypy_return_type' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type', list_285556)
    
    # ################# End of 'math_directive(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'math_directive' in the type store
    # Getting the type of 'stypy_return_type' (line 38)
    stypy_return_type_285558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285558)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'math_directive'
    return stypy_return_type_285558

# Assigning a type to the variable 'math_directive' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'math_directive', math_directive)

@norecursion
def latex2png(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    unicode_285559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 39), 'unicode', u'cm')
    defaults = [unicode_285559]
    # Create a new context for function 'latex2png'
    module_type_store = module_type_store.open_function_context('latex2png', 47, 0, False)
    
    # Passed parameters checking function
    latex2png.stypy_localization = localization
    latex2png.stypy_type_of_self = None
    latex2png.stypy_type_store = module_type_store
    latex2png.stypy_function_name = 'latex2png'
    latex2png.stypy_param_names_list = ['latex', 'filename', 'fontset']
    latex2png.stypy_varargs_param_name = None
    latex2png.stypy_kwargs_param_name = None
    latex2png.stypy_call_defaults = defaults
    latex2png.stypy_call_varargs = varargs
    latex2png.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'latex2png', ['latex', 'filename', 'fontset'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'latex2png', localization, ['latex', 'filename', 'fontset'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'latex2png(...)' code ##################

    
    # Assigning a BinOp to a Name (line 48):
    unicode_285560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 12), 'unicode', u'$%s$')
    # Getting the type of 'latex' (line 48)
    latex_285561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 21), 'latex')
    # Applying the binary operator '%' (line 48)
    result_mod_285562 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 12), '%', unicode_285560, latex_285561)
    
    # Assigning a type to the variable 'latex' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'latex', result_mod_285562)
    
    # Assigning a Subscript to a Name (line 49):
    
    # Obtaining the type of the subscript
    unicode_285563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 28), 'unicode', u'mathtext.fontset')
    # Getting the type of 'rcParams' (line 49)
    rcParams_285564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 'rcParams')
    # Obtaining the member '__getitem__' of a type (line 49)
    getitem___285565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 19), rcParams_285564, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 49)
    subscript_call_result_285566 = invoke(stypy.reporting.localization.Localization(__file__, 49, 19), getitem___285565, unicode_285563)
    
    # Assigning a type to the variable 'orig_fontset' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'orig_fontset', subscript_call_result_285566)
    
    # Assigning a Name to a Subscript (line 50):
    # Getting the type of 'fontset' (line 50)
    fontset_285567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 35), 'fontset')
    # Getting the type of 'rcParams' (line 50)
    rcParams_285568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'rcParams')
    unicode_285569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 13), 'unicode', u'mathtext.fontset')
    # Storing an element on a container (line 50)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 4), rcParams_285568, (unicode_285569, fontset_285567))
    
    
    # Call to exists(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'filename' (line 51)
    filename_285573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 22), 'filename', False)
    # Processing the call keyword arguments (line 51)
    kwargs_285574 = {}
    # Getting the type of 'os' (line 51)
    os_285570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 7), 'os', False)
    # Obtaining the member 'path' of a type (line 51)
    path_285571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 7), os_285570, 'path')
    # Obtaining the member 'exists' of a type (line 51)
    exists_285572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 7), path_285571, 'exists')
    # Calling exists(args, kwargs) (line 51)
    exists_call_result_285575 = invoke(stypy.reporting.localization.Localization(__file__, 51, 7), exists_285572, *[filename_285573], **kwargs_285574)
    
    # Testing the type of an if condition (line 51)
    if_condition_285576 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 51, 4), exists_call_result_285575)
    # Assigning a type to the variable 'if_condition_285576' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'if_condition_285576', if_condition_285576)
    # SSA begins for if statement (line 51)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 52):
    
    # Call to get_depth(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'latex' (line 52)
    latex_285579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 42), 'latex', False)
    # Processing the call keyword arguments (line 52)
    int_285580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 53), 'int')
    keyword_285581 = int_285580
    kwargs_285582 = {'dpi': keyword_285581}
    # Getting the type of 'mathtext_parser' (line 52)
    mathtext_parser_285577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'mathtext_parser', False)
    # Obtaining the member 'get_depth' of a type (line 52)
    get_depth_285578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 16), mathtext_parser_285577, 'get_depth')
    # Calling get_depth(args, kwargs) (line 52)
    get_depth_call_result_285583 = invoke(stypy.reporting.localization.Localization(__file__, 52, 16), get_depth_285578, *[latex_285579], **kwargs_285582)
    
    # Assigning a type to the variable 'depth' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'depth', get_depth_call_result_285583)
    # SSA branch for the else part of an if statement (line 51)
    module_type_store.open_ssa_branch('else')
    
    
    # SSA begins for try-except statement (line 54)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 55):
    
    # Call to to_png(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'filename' (line 55)
    filename_285586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 43), 'filename', False)
    # Getting the type of 'latex' (line 55)
    latex_285587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 53), 'latex', False)
    # Processing the call keyword arguments (line 55)
    int_285588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 64), 'int')
    keyword_285589 = int_285588
    kwargs_285590 = {'dpi': keyword_285589}
    # Getting the type of 'mathtext_parser' (line 55)
    mathtext_parser_285584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 20), 'mathtext_parser', False)
    # Obtaining the member 'to_png' of a type (line 55)
    to_png_285585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 20), mathtext_parser_285584, 'to_png')
    # Calling to_png(args, kwargs) (line 55)
    to_png_call_result_285591 = invoke(stypy.reporting.localization.Localization(__file__, 55, 20), to_png_285585, *[filename_285586, latex_285587], **kwargs_285590)
    
    # Assigning a type to the variable 'depth' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'depth', to_png_call_result_285591)
    # SSA branch for the except part of a try statement (line 54)
    # SSA branch for the except '<any exception>' branch of a try statement (line 54)
    module_type_store.open_ssa_branch('except')
    
    # Call to warn(...): (line 57)
    # Processing the call arguments (line 57)
    unicode_285594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 26), 'unicode', u'Could not render math expression %s')
    # Getting the type of 'latex' (line 57)
    latex_285595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 66), 'latex', False)
    # Applying the binary operator '%' (line 57)
    result_mod_285596 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 26), '%', unicode_285594, latex_285595)
    
    # Getting the type of 'Warning' (line 58)
    Warning_285597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 26), 'Warning', False)
    # Processing the call keyword arguments (line 57)
    kwargs_285598 = {}
    # Getting the type of 'warnings' (line 57)
    warnings_285592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 57)
    warn_285593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), warnings_285592, 'warn')
    # Calling warn(args, kwargs) (line 57)
    warn_call_result_285599 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), warn_285593, *[result_mod_285596, Warning_285597], **kwargs_285598)
    
    
    # Assigning a Num to a Name (line 59):
    int_285600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 20), 'int')
    # Assigning a type to the variable 'depth' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'depth', int_285600)
    # SSA join for try-except statement (line 54)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 51)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 60):
    # Getting the type of 'orig_fontset' (line 60)
    orig_fontset_285601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 35), 'orig_fontset')
    # Getting the type of 'rcParams' (line 60)
    rcParams_285602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'rcParams')
    unicode_285603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 13), 'unicode', u'mathtext.fontset')
    # Storing an element on a container (line 60)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 4), rcParams_285602, (unicode_285603, orig_fontset_285601))
    
    # Call to write(...): (line 61)
    # Processing the call arguments (line 61)
    unicode_285607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 21), 'unicode', u'#')
    # Processing the call keyword arguments (line 61)
    kwargs_285608 = {}
    # Getting the type of 'sys' (line 61)
    sys_285604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'sys', False)
    # Obtaining the member 'stdout' of a type (line 61)
    stdout_285605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 4), sys_285604, 'stdout')
    # Obtaining the member 'write' of a type (line 61)
    write_285606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 4), stdout_285605, 'write')
    # Calling write(args, kwargs) (line 61)
    write_call_result_285609 = invoke(stypy.reporting.localization.Localization(__file__, 61, 4), write_285606, *[unicode_285607], **kwargs_285608)
    
    
    # Call to flush(...): (line 62)
    # Processing the call keyword arguments (line 62)
    kwargs_285613 = {}
    # Getting the type of 'sys' (line 62)
    sys_285610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'sys', False)
    # Obtaining the member 'stdout' of a type (line 62)
    stdout_285611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 4), sys_285610, 'stdout')
    # Obtaining the member 'flush' of a type (line 62)
    flush_285612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 4), stdout_285611, 'flush')
    # Calling flush(args, kwargs) (line 62)
    flush_call_result_285614 = invoke(stypy.reporting.localization.Localization(__file__, 62, 4), flush_285612, *[], **kwargs_285613)
    
    # Getting the type of 'depth' (line 63)
    depth_285615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 11), 'depth')
    # Assigning a type to the variable 'stypy_return_type' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'stypy_return_type', depth_285615)
    
    # ################# End of 'latex2png(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'latex2png' in the type store
    # Getting the type of 'stypy_return_type' (line 47)
    stypy_return_type_285616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285616)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'latex2png'
    return stypy_return_type_285616

# Assigning a type to the variable 'latex2png' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'latex2png', latex2png)

@norecursion
def latex2html(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'latex2html'
    module_type_store = module_type_store.open_function_context('latex2html', 66, 0, False)
    
    # Passed parameters checking function
    latex2html.stypy_localization = localization
    latex2html.stypy_type_of_self = None
    latex2html.stypy_type_store = module_type_store
    latex2html.stypy_function_name = 'latex2html'
    latex2html.stypy_param_names_list = ['node', 'source']
    latex2html.stypy_varargs_param_name = None
    latex2html.stypy_kwargs_param_name = None
    latex2html.stypy_call_defaults = defaults
    latex2html.stypy_call_varargs = varargs
    latex2html.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'latex2html', ['node', 'source'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'latex2html', localization, ['node', 'source'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'latex2html(...)' code ##################

    
    # Assigning a Call to a Name (line 67):
    
    # Call to isinstance(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'node' (line 67)
    node_285618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 24), 'node', False)
    # Obtaining the member 'parent' of a type (line 67)
    parent_285619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 24), node_285618, 'parent')
    # Getting the type of 'nodes' (line 67)
    nodes_285620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 37), 'nodes', False)
    # Obtaining the member 'TextElement' of a type (line 67)
    TextElement_285621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 37), nodes_285620, 'TextElement')
    # Processing the call keyword arguments (line 67)
    kwargs_285622 = {}
    # Getting the type of 'isinstance' (line 67)
    isinstance_285617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 13), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 67)
    isinstance_call_result_285623 = invoke(stypy.reporting.localization.Localization(__file__, 67, 13), isinstance_285617, *[parent_285619, TextElement_285621], **kwargs_285622)
    
    # Assigning a type to the variable 'inline' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'inline', isinstance_call_result_285623)
    
    # Assigning a Subscript to a Name (line 68):
    
    # Obtaining the type of the subscript
    unicode_285624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 17), 'unicode', u'latex')
    # Getting the type of 'node' (line 68)
    node_285625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'node')
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___285626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), node_285625, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 68)
    subscript_call_result_285627 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), getitem___285626, unicode_285624)
    
    # Assigning a type to the variable 'latex' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'latex', subscript_call_result_285627)
    
    # Assigning a BinOp to a Name (line 69):
    unicode_285628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 11), 'unicode', u'math-%s')
    
    # Obtaining the type of the subscript
    int_285629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 55), 'int')
    slice_285630 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 69, 23), int_285629, None, None)
    
    # Call to hexdigest(...): (line 69)
    # Processing the call keyword arguments (line 69)
    kwargs_285639 = {}
    
    # Call to md5(...): (line 69)
    # Processing the call arguments (line 69)
    
    # Call to encode(...): (line 69)
    # Processing the call keyword arguments (line 69)
    kwargs_285634 = {}
    # Getting the type of 'latex' (line 69)
    latex_285632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 27), 'latex', False)
    # Obtaining the member 'encode' of a type (line 69)
    encode_285633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 27), latex_285632, 'encode')
    # Calling encode(args, kwargs) (line 69)
    encode_call_result_285635 = invoke(stypy.reporting.localization.Localization(__file__, 69, 27), encode_285633, *[], **kwargs_285634)
    
    # Processing the call keyword arguments (line 69)
    kwargs_285636 = {}
    # Getting the type of 'md5' (line 69)
    md5_285631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 23), 'md5', False)
    # Calling md5(args, kwargs) (line 69)
    md5_call_result_285637 = invoke(stypy.reporting.localization.Localization(__file__, 69, 23), md5_285631, *[encode_call_result_285635], **kwargs_285636)
    
    # Obtaining the member 'hexdigest' of a type (line 69)
    hexdigest_285638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 23), md5_call_result_285637, 'hexdigest')
    # Calling hexdigest(args, kwargs) (line 69)
    hexdigest_call_result_285640 = invoke(stypy.reporting.localization.Localization(__file__, 69, 23), hexdigest_285638, *[], **kwargs_285639)
    
    # Obtaining the member '__getitem__' of a type (line 69)
    getitem___285641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 23), hexdigest_call_result_285640, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
    subscript_call_result_285642 = invoke(stypy.reporting.localization.Localization(__file__, 69, 23), getitem___285641, slice_285630)
    
    # Applying the binary operator '%' (line 69)
    result_mod_285643 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 11), '%', unicode_285628, subscript_call_result_285642)
    
    # Assigning a type to the variable 'name' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'name', result_mod_285643)
    
    # Assigning a Call to a Name (line 71):
    
    # Call to join(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'setup' (line 71)
    setup_285647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 27), 'setup', False)
    # Obtaining the member 'app' of a type (line 71)
    app_285648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 27), setup_285647, 'app')
    # Obtaining the member 'builder' of a type (line 71)
    builder_285649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 27), app_285648, 'builder')
    # Obtaining the member 'outdir' of a type (line 71)
    outdir_285650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 27), builder_285649, 'outdir')
    unicode_285651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 53), 'unicode', u'_images')
    unicode_285652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 64), 'unicode', u'mathmpl')
    # Processing the call keyword arguments (line 71)
    kwargs_285653 = {}
    # Getting the type of 'os' (line 71)
    os_285644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 14), 'os', False)
    # Obtaining the member 'path' of a type (line 71)
    path_285645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 14), os_285644, 'path')
    # Obtaining the member 'join' of a type (line 71)
    join_285646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 14), path_285645, 'join')
    # Calling join(args, kwargs) (line 71)
    join_call_result_285654 = invoke(stypy.reporting.localization.Localization(__file__, 71, 14), join_285646, *[outdir_285650, unicode_285651, unicode_285652], **kwargs_285653)
    
    # Assigning a type to the variable 'destdir' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'destdir', join_call_result_285654)
    
    
    
    # Call to exists(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'destdir' (line 72)
    destdir_285658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 26), 'destdir', False)
    # Processing the call keyword arguments (line 72)
    kwargs_285659 = {}
    # Getting the type of 'os' (line 72)
    os_285655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 72)
    path_285656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 11), os_285655, 'path')
    # Obtaining the member 'exists' of a type (line 72)
    exists_285657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 11), path_285656, 'exists')
    # Calling exists(args, kwargs) (line 72)
    exists_call_result_285660 = invoke(stypy.reporting.localization.Localization(__file__, 72, 11), exists_285657, *[destdir_285658], **kwargs_285659)
    
    # Applying the 'not' unary operator (line 72)
    result_not__285661 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 7), 'not', exists_call_result_285660)
    
    # Testing the type of an if condition (line 72)
    if_condition_285662 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 4), result_not__285661)
    # Assigning a type to the variable 'if_condition_285662' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'if_condition_285662', if_condition_285662)
    # SSA begins for if statement (line 72)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to makedirs(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'destdir' (line 73)
    destdir_285665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 20), 'destdir', False)
    # Processing the call keyword arguments (line 73)
    kwargs_285666 = {}
    # Getting the type of 'os' (line 73)
    os_285663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'os', False)
    # Obtaining the member 'makedirs' of a type (line 73)
    makedirs_285664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), os_285663, 'makedirs')
    # Calling makedirs(args, kwargs) (line 73)
    makedirs_call_result_285667 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), makedirs_285664, *[destdir_285665], **kwargs_285666)
    
    # SSA join for if statement (line 72)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 74):
    
    # Call to join(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'destdir' (line 74)
    destdir_285671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 24), 'destdir', False)
    unicode_285672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 33), 'unicode', u'%s.png')
    # Getting the type of 'name' (line 74)
    name_285673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 44), 'name', False)
    # Applying the binary operator '%' (line 74)
    result_mod_285674 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 33), '%', unicode_285672, name_285673)
    
    # Processing the call keyword arguments (line 74)
    kwargs_285675 = {}
    # Getting the type of 'os' (line 74)
    os_285668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 74)
    path_285669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 11), os_285668, 'path')
    # Obtaining the member 'join' of a type (line 74)
    join_285670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 11), path_285669, 'join')
    # Calling join(args, kwargs) (line 74)
    join_call_result_285676 = invoke(stypy.reporting.localization.Localization(__file__, 74, 11), join_285670, *[destdir_285671, result_mod_285674], **kwargs_285675)
    
    # Assigning a type to the variable 'dest' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'dest', join_call_result_285676)
    
    # Assigning a Call to a Name (line 75):
    
    # Call to join(...): (line 75)
    # Processing the call arguments (line 75)
    
    # Obtaining an instance of the builtin type 'tuple' (line 75)
    tuple_285679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 75)
    # Adding element type (line 75)
    # Getting the type of 'setup' (line 75)
    setup_285680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 21), 'setup', False)
    # Obtaining the member 'app' of a type (line 75)
    app_285681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 21), setup_285680, 'app')
    # Obtaining the member 'builder' of a type (line 75)
    builder_285682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 21), app_285681, 'builder')
    # Obtaining the member 'imgpath' of a type (line 75)
    imgpath_285683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 21), builder_285682, 'imgpath')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 21), tuple_285679, imgpath_285683)
    # Adding element type (line 75)
    unicode_285684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 48), 'unicode', u'mathmpl')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 21), tuple_285679, unicode_285684)
    
    # Processing the call keyword arguments (line 75)
    kwargs_285685 = {}
    unicode_285677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 11), 'unicode', u'/')
    # Obtaining the member 'join' of a type (line 75)
    join_285678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 11), unicode_285677, 'join')
    # Calling join(args, kwargs) (line 75)
    join_call_result_285686 = invoke(stypy.reporting.localization.Localization(__file__, 75, 11), join_285678, *[tuple_285679], **kwargs_285685)
    
    # Assigning a type to the variable 'path' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'path', join_call_result_285686)
    
    # Assigning a Call to a Name (line 77):
    
    # Call to latex2png(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'latex' (line 77)
    latex_285688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 22), 'latex', False)
    # Getting the type of 'dest' (line 77)
    dest_285689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 29), 'dest', False)
    
    # Obtaining the type of the subscript
    unicode_285690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 40), 'unicode', u'fontset')
    # Getting the type of 'node' (line 77)
    node_285691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 35), 'node', False)
    # Obtaining the member '__getitem__' of a type (line 77)
    getitem___285692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 35), node_285691, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 77)
    subscript_call_result_285693 = invoke(stypy.reporting.localization.Localization(__file__, 77, 35), getitem___285692, unicode_285690)
    
    # Processing the call keyword arguments (line 77)
    kwargs_285694 = {}
    # Getting the type of 'latex2png' (line 77)
    latex2png_285687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'latex2png', False)
    # Calling latex2png(args, kwargs) (line 77)
    latex2png_call_result_285695 = invoke(stypy.reporting.localization.Localization(__file__, 77, 12), latex2png_285687, *[latex_285688, dest_285689, subscript_call_result_285693], **kwargs_285694)
    
    # Assigning a type to the variable 'depth' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'depth', latex2png_call_result_285695)
    
    # Getting the type of 'inline' (line 79)
    inline_285696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 7), 'inline')
    # Testing the type of an if condition (line 79)
    if_condition_285697 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 4), inline_285696)
    # Assigning a type to the variable 'if_condition_285697' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'if_condition_285697', if_condition_285697)
    # SSA begins for if statement (line 79)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 80):
    unicode_285698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 14), 'unicode', u'')
    # Assigning a type to the variable 'cls' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'cls', unicode_285698)
    # SSA branch for the else part of an if statement (line 79)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 82):
    unicode_285699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 14), 'unicode', u'class="center" ')
    # Assigning a type to the variable 'cls' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'cls', unicode_285699)
    # SSA join for if statement (line 79)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'inline' (line 83)
    inline_285700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 7), 'inline')
    
    # Getting the type of 'depth' (line 83)
    depth_285701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 18), 'depth')
    int_285702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 27), 'int')
    # Applying the binary operator '!=' (line 83)
    result_ne_285703 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 18), '!=', depth_285701, int_285702)
    
    # Applying the binary operator 'and' (line 83)
    result_and_keyword_285704 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 7), 'and', inline_285700, result_ne_285703)
    
    # Testing the type of an if condition (line 83)
    if_condition_285705 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 4), result_and_keyword_285704)
    # Assigning a type to the variable 'if_condition_285705' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'if_condition_285705', if_condition_285705)
    # SSA begins for if statement (line 83)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 84):
    unicode_285706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 16), 'unicode', u'style="position: relative; bottom: -%dpx"')
    # Getting the type of 'depth' (line 84)
    depth_285707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 63), 'depth')
    int_285708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 71), 'int')
    # Applying the binary operator '+' (line 84)
    result_add_285709 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 63), '+', depth_285707, int_285708)
    
    # Applying the binary operator '%' (line 84)
    result_mod_285710 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 16), '%', unicode_285706, result_add_285709)
    
    # Assigning a type to the variable 'style' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'style', result_mod_285710)
    # SSA branch for the else part of an if statement (line 83)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 86):
    unicode_285711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 16), 'unicode', u'')
    # Assigning a type to the variable 'style' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'style', unicode_285711)
    # SSA join for if statement (line 83)
    module_type_store = module_type_store.join_ssa_context()
    
    unicode_285712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 11), 'unicode', u'<img src="%s/%s.png" %s%s/>')
    
    # Obtaining an instance of the builtin type 'tuple' (line 88)
    tuple_285713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 88)
    # Adding element type (line 88)
    # Getting the type of 'path' (line 88)
    path_285714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 44), 'path')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 44), tuple_285713, path_285714)
    # Adding element type (line 88)
    # Getting the type of 'name' (line 88)
    name_285715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 50), 'name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 44), tuple_285713, name_285715)
    # Adding element type (line 88)
    # Getting the type of 'cls' (line 88)
    cls_285716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 56), 'cls')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 44), tuple_285713, cls_285716)
    # Adding element type (line 88)
    # Getting the type of 'style' (line 88)
    style_285717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 61), 'style')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 44), tuple_285713, style_285717)
    
    # Applying the binary operator '%' (line 88)
    result_mod_285718 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 11), '%', unicode_285712, tuple_285713)
    
    # Assigning a type to the variable 'stypy_return_type' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type', result_mod_285718)
    
    # ################# End of 'latex2html(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'latex2html' in the type store
    # Getting the type of 'stypy_return_type' (line 66)
    stypy_return_type_285719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285719)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'latex2html'
    return stypy_return_type_285719

# Assigning a type to the variable 'latex2html' (line 66)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'latex2html', latex2html)

@norecursion
def setup(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'setup'
    module_type_store = module_type_store.open_function_context('setup', 91, 0, False)
    
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

    
    # Assigning a Name to a Attribute (line 92):
    # Getting the type of 'app' (line 92)
    app_285720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'app')
    # Getting the type of 'setup' (line 92)
    setup_285721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'setup')
    # Setting the type of the member 'app' of a type (line 92)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 4), setup_285721, 'app', app_285720)

    @norecursion
    def visit_latex_math_html(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_latex_math_html'
        module_type_store = module_type_store.open_function_context('visit_latex_math_html', 95, 4, False)
        
        # Passed parameters checking function
        visit_latex_math_html.stypy_localization = localization
        visit_latex_math_html.stypy_type_of_self = None
        visit_latex_math_html.stypy_type_store = module_type_store
        visit_latex_math_html.stypy_function_name = 'visit_latex_math_html'
        visit_latex_math_html.stypy_param_names_list = ['self', 'node']
        visit_latex_math_html.stypy_varargs_param_name = None
        visit_latex_math_html.stypy_kwargs_param_name = None
        visit_latex_math_html.stypy_call_defaults = defaults
        visit_latex_math_html.stypy_call_varargs = varargs
        visit_latex_math_html.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'visit_latex_math_html', ['self', 'node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_latex_math_html', localization, ['self', 'node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_latex_math_html(...)' code ##################

        
        # Assigning a Subscript to a Name (line 96):
        
        # Obtaining the type of the subscript
        unicode_285722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 42), 'unicode', u'source')
        # Getting the type of 'self' (line 96)
        self_285723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 17), 'self')
        # Obtaining the member 'document' of a type (line 96)
        document_285724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 17), self_285723, 'document')
        # Obtaining the member 'attributes' of a type (line 96)
        attributes_285725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 17), document_285724, 'attributes')
        # Obtaining the member '__getitem__' of a type (line 96)
        getitem___285726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 17), attributes_285725, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 96)
        subscript_call_result_285727 = invoke(stypy.reporting.localization.Localization(__file__, 96, 17), getitem___285726, unicode_285722)
        
        # Assigning a type to the variable 'source' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'source', subscript_call_result_285727)
        
        # Call to append(...): (line 97)
        # Processing the call arguments (line 97)
        
        # Call to latex2html(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'node' (line 97)
        node_285732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 36), 'node', False)
        # Getting the type of 'source' (line 97)
        source_285733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 42), 'source', False)
        # Processing the call keyword arguments (line 97)
        kwargs_285734 = {}
        # Getting the type of 'latex2html' (line 97)
        latex2html_285731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 25), 'latex2html', False)
        # Calling latex2html(args, kwargs) (line 97)
        latex2html_call_result_285735 = invoke(stypy.reporting.localization.Localization(__file__, 97, 25), latex2html_285731, *[node_285732, source_285733], **kwargs_285734)
        
        # Processing the call keyword arguments (line 97)
        kwargs_285736 = {}
        # Getting the type of 'self' (line 97)
        self_285728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'self', False)
        # Obtaining the member 'body' of a type (line 97)
        body_285729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), self_285728, 'body')
        # Obtaining the member 'append' of a type (line 97)
        append_285730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), body_285729, 'append')
        # Calling append(args, kwargs) (line 97)
        append_call_result_285737 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), append_285730, *[latex2html_call_result_285735], **kwargs_285736)
        
        
        # ################# End of 'visit_latex_math_html(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_latex_math_html' in the type store
        # Getting the type of 'stypy_return_type' (line 95)
        stypy_return_type_285738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_285738)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_latex_math_html'
        return stypy_return_type_285738

    # Assigning a type to the variable 'visit_latex_math_html' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'visit_latex_math_html', visit_latex_math_html)

    @norecursion
    def depart_latex_math_html(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'depart_latex_math_html'
        module_type_store = module_type_store.open_function_context('depart_latex_math_html', 99, 4, False)
        
        # Passed parameters checking function
        depart_latex_math_html.stypy_localization = localization
        depart_latex_math_html.stypy_type_of_self = None
        depart_latex_math_html.stypy_type_store = module_type_store
        depart_latex_math_html.stypy_function_name = 'depart_latex_math_html'
        depart_latex_math_html.stypy_param_names_list = ['self', 'node']
        depart_latex_math_html.stypy_varargs_param_name = None
        depart_latex_math_html.stypy_kwargs_param_name = None
        depart_latex_math_html.stypy_call_defaults = defaults
        depart_latex_math_html.stypy_call_varargs = varargs
        depart_latex_math_html.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'depart_latex_math_html', ['self', 'node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'depart_latex_math_html', localization, ['self', 'node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'depart_latex_math_html(...)' code ##################

        pass
        
        # ################# End of 'depart_latex_math_html(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'depart_latex_math_html' in the type store
        # Getting the type of 'stypy_return_type' (line 99)
        stypy_return_type_285739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_285739)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'depart_latex_math_html'
        return stypy_return_type_285739

    # Assigning a type to the variable 'depart_latex_math_html' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'depart_latex_math_html', depart_latex_math_html)

    @norecursion
    def visit_latex_math_latex(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_latex_math_latex'
        module_type_store = module_type_store.open_function_context('visit_latex_math_latex', 103, 4, False)
        
        # Passed parameters checking function
        visit_latex_math_latex.stypy_localization = localization
        visit_latex_math_latex.stypy_type_of_self = None
        visit_latex_math_latex.stypy_type_store = module_type_store
        visit_latex_math_latex.stypy_function_name = 'visit_latex_math_latex'
        visit_latex_math_latex.stypy_param_names_list = ['self', 'node']
        visit_latex_math_latex.stypy_varargs_param_name = None
        visit_latex_math_latex.stypy_kwargs_param_name = None
        visit_latex_math_latex.stypy_call_defaults = defaults
        visit_latex_math_latex.stypy_call_varargs = varargs
        visit_latex_math_latex.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'visit_latex_math_latex', ['self', 'node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_latex_math_latex', localization, ['self', 'node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_latex_math_latex(...)' code ##################

        
        # Assigning a Call to a Name (line 104):
        
        # Call to isinstance(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'node' (line 104)
        node_285741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 28), 'node', False)
        # Obtaining the member 'parent' of a type (line 104)
        parent_285742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 28), node_285741, 'parent')
        # Getting the type of 'nodes' (line 104)
        nodes_285743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 41), 'nodes', False)
        # Obtaining the member 'TextElement' of a type (line 104)
        TextElement_285744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 41), nodes_285743, 'TextElement')
        # Processing the call keyword arguments (line 104)
        kwargs_285745 = {}
        # Getting the type of 'isinstance' (line 104)
        isinstance_285740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 17), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 104)
        isinstance_call_result_285746 = invoke(stypy.reporting.localization.Localization(__file__, 104, 17), isinstance_285740, *[parent_285742, TextElement_285744], **kwargs_285745)
        
        # Assigning a type to the variable 'inline' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'inline', isinstance_call_result_285746)
        
        # Getting the type of 'inline' (line 105)
        inline_285747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 11), 'inline')
        # Testing the type of an if condition (line 105)
        if_condition_285748 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 8), inline_285747)
        # Assigning a type to the variable 'if_condition_285748' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'if_condition_285748', if_condition_285748)
        # SSA begins for if statement (line 105)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 106)
        # Processing the call arguments (line 106)
        unicode_285752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 29), 'unicode', u'$%s$')
        
        # Obtaining the type of the subscript
        unicode_285753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 43), 'unicode', u'latex')
        # Getting the type of 'node' (line 106)
        node_285754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 38), 'node', False)
        # Obtaining the member '__getitem__' of a type (line 106)
        getitem___285755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 38), node_285754, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 106)
        subscript_call_result_285756 = invoke(stypy.reporting.localization.Localization(__file__, 106, 38), getitem___285755, unicode_285753)
        
        # Applying the binary operator '%' (line 106)
        result_mod_285757 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 29), '%', unicode_285752, subscript_call_result_285756)
        
        # Processing the call keyword arguments (line 106)
        kwargs_285758 = {}
        # Getting the type of 'self' (line 106)
        self_285749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'self', False)
        # Obtaining the member 'body' of a type (line 106)
        body_285750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 12), self_285749, 'body')
        # Obtaining the member 'append' of a type (line 106)
        append_285751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 12), body_285750, 'append')
        # Calling append(args, kwargs) (line 106)
        append_call_result_285759 = invoke(stypy.reporting.localization.Localization(__file__, 106, 12), append_285751, *[result_mod_285757], **kwargs_285758)
        
        # SSA branch for the else part of an if statement (line 105)
        module_type_store.open_ssa_branch('else')
        
        # Call to extend(...): (line 108)
        # Processing the call arguments (line 108)
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_285763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        # Adding element type (line 108)
        unicode_285764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 30), 'unicode', u'\\begin{equation}')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 29), list_285763, unicode_285764)
        # Adding element type (line 108)
        
        # Obtaining the type of the subscript
        unicode_285765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 35), 'unicode', u'latex')
        # Getting the type of 'node' (line 109)
        node_285766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 30), 'node', False)
        # Obtaining the member '__getitem__' of a type (line 109)
        getitem___285767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 30), node_285766, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 109)
        subscript_call_result_285768 = invoke(stypy.reporting.localization.Localization(__file__, 109, 30), getitem___285767, unicode_285765)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 29), list_285763, subscript_call_result_285768)
        # Adding element type (line 108)
        unicode_285769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 30), 'unicode', u'\\end{equation}')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 29), list_285763, unicode_285769)
        
        # Processing the call keyword arguments (line 108)
        kwargs_285770 = {}
        # Getting the type of 'self' (line 108)
        self_285760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'self', False)
        # Obtaining the member 'body' of a type (line 108)
        body_285761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 12), self_285760, 'body')
        # Obtaining the member 'extend' of a type (line 108)
        extend_285762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 12), body_285761, 'extend')
        # Calling extend(args, kwargs) (line 108)
        extend_call_result_285771 = invoke(stypy.reporting.localization.Localization(__file__, 108, 12), extend_285762, *[list_285763], **kwargs_285770)
        
        # SSA join for if statement (line 105)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'visit_latex_math_latex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_latex_math_latex' in the type store
        # Getting the type of 'stypy_return_type' (line 103)
        stypy_return_type_285772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_285772)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_latex_math_latex'
        return stypy_return_type_285772

    # Assigning a type to the variable 'visit_latex_math_latex' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'visit_latex_math_latex', visit_latex_math_latex)

    @norecursion
    def depart_latex_math_latex(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'depart_latex_math_latex'
        module_type_store = module_type_store.open_function_context('depart_latex_math_latex', 112, 4, False)
        
        # Passed parameters checking function
        depart_latex_math_latex.stypy_localization = localization
        depart_latex_math_latex.stypy_type_of_self = None
        depart_latex_math_latex.stypy_type_store = module_type_store
        depart_latex_math_latex.stypy_function_name = 'depart_latex_math_latex'
        depart_latex_math_latex.stypy_param_names_list = ['self', 'node']
        depart_latex_math_latex.stypy_varargs_param_name = None
        depart_latex_math_latex.stypy_kwargs_param_name = None
        depart_latex_math_latex.stypy_call_defaults = defaults
        depart_latex_math_latex.stypy_call_varargs = varargs
        depart_latex_math_latex.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'depart_latex_math_latex', ['self', 'node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'depart_latex_math_latex', localization, ['self', 'node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'depart_latex_math_latex(...)' code ##################

        pass
        
        # ################# End of 'depart_latex_math_latex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'depart_latex_math_latex' in the type store
        # Getting the type of 'stypy_return_type' (line 112)
        stypy_return_type_285773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_285773)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'depart_latex_math_latex'
        return stypy_return_type_285773

    # Assigning a type to the variable 'depart_latex_math_latex' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'depart_latex_math_latex', depart_latex_math_latex)
    
    # Call to add_node(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'latex_math' (line 115)
    latex_math_285776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 17), 'latex_math', False)
    # Processing the call keyword arguments (line 115)
    
    # Obtaining an instance of the builtin type 'tuple' (line 116)
    tuple_285777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 116)
    # Adding element type (line 116)
    # Getting the type of 'visit_latex_math_html' (line 116)
    visit_latex_math_html_285778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 23), 'visit_latex_math_html', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 23), tuple_285777, visit_latex_math_html_285778)
    # Adding element type (line 116)
    # Getting the type of 'depart_latex_math_html' (line 116)
    depart_latex_math_html_285779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 46), 'depart_latex_math_html', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 23), tuple_285777, depart_latex_math_html_285779)
    
    keyword_285780 = tuple_285777
    
    # Obtaining an instance of the builtin type 'tuple' (line 117)
    tuple_285781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 117)
    # Adding element type (line 117)
    # Getting the type of 'visit_latex_math_latex' (line 117)
    visit_latex_math_latex_285782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 24), 'visit_latex_math_latex', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 24), tuple_285781, visit_latex_math_latex_285782)
    # Adding element type (line 117)
    # Getting the type of 'depart_latex_math_latex' (line 117)
    depart_latex_math_latex_285783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 48), 'depart_latex_math_latex', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 24), tuple_285781, depart_latex_math_latex_285783)
    
    keyword_285784 = tuple_285781
    kwargs_285785 = {'latex': keyword_285784, 'html': keyword_285780}
    # Getting the type of 'app' (line 115)
    app_285774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'app', False)
    # Obtaining the member 'add_node' of a type (line 115)
    add_node_285775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 4), app_285774, 'add_node')
    # Calling add_node(args, kwargs) (line 115)
    add_node_call_result_285786 = invoke(stypy.reporting.localization.Localization(__file__, 115, 4), add_node_285775, *[latex_math_285776], **kwargs_285785)
    
    
    # Call to add_role(...): (line 118)
    # Processing the call arguments (line 118)
    unicode_285789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 17), 'unicode', u'math')
    # Getting the type of 'math_role' (line 118)
    math_role_285790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 25), 'math_role', False)
    # Processing the call keyword arguments (line 118)
    kwargs_285791 = {}
    # Getting the type of 'app' (line 118)
    app_285787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'app', False)
    # Obtaining the member 'add_role' of a type (line 118)
    add_role_285788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 4), app_285787, 'add_role')
    # Calling add_role(args, kwargs) (line 118)
    add_role_call_result_285792 = invoke(stypy.reporting.localization.Localization(__file__, 118, 4), add_role_285788, *[unicode_285789, math_role_285790], **kwargs_285791)
    
    
    # Call to add_directive(...): (line 119)
    # Processing the call arguments (line 119)
    unicode_285795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 22), 'unicode', u'math')
    # Getting the type of 'math_directive' (line 119)
    math_directive_285796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 30), 'math_directive', False)
    # Getting the type of 'True' (line 120)
    True_285797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 22), 'True', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 120)
    tuple_285798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 120)
    # Adding element type (line 120)
    int_285799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 29), tuple_285798, int_285799)
    # Adding element type (line 120)
    int_285800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 29), tuple_285798, int_285800)
    # Adding element type (line 120)
    int_285801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 29), tuple_285798, int_285801)
    
    # Processing the call keyword arguments (line 119)
    # Getting the type of 'options_spec' (line 120)
    options_spec_285802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 41), 'options_spec', False)
    kwargs_285803 = {'options_spec_285802': options_spec_285802}
    # Getting the type of 'app' (line 119)
    app_285793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'app', False)
    # Obtaining the member 'add_directive' of a type (line 119)
    add_directive_285794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 4), app_285793, 'add_directive')
    # Calling add_directive(args, kwargs) (line 119)
    add_directive_call_result_285804 = invoke(stypy.reporting.localization.Localization(__file__, 119, 4), add_directive_285794, *[unicode_285795, math_directive_285796, True_285797, tuple_285798], **kwargs_285803)
    
    
    # Assigning a Dict to a Name (line 122):
    
    # Obtaining an instance of the builtin type 'dict' (line 122)
    dict_285805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 122)
    # Adding element type (key, value) (line 122)
    unicode_285806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 16), 'unicode', u'parallel_read_safe')
    # Getting the type of 'True' (line 122)
    True_285807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 38), 'True')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 15), dict_285805, (unicode_285806, True_285807))
    # Adding element type (key, value) (line 122)
    unicode_285808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 44), 'unicode', u'parallel_write_safe')
    # Getting the type of 'True' (line 122)
    True_285809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 67), 'True')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 15), dict_285805, (unicode_285808, True_285809))
    
    # Assigning a type to the variable 'metadata' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'metadata', dict_285805)
    # Getting the type of 'metadata' (line 123)
    metadata_285810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 11), 'metadata')
    # Assigning a type to the variable 'stypy_return_type' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type', metadata_285810)
    
    # ################# End of 'setup(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'setup' in the type store
    # Getting the type of 'stypy_return_type' (line 91)
    stypy_return_type_285811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285811)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'setup'
    return stypy_return_type_285811

# Assigning a type to the variable 'setup' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'setup', setup)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Tests which scan for certain occurrences in the code, they may not find
3: all of these occurrences but should catch almost all. This file was adapted
4: from numpy.
5: '''
6: 
7: 
8: from __future__ import division, absolute_import, print_function
9: 
10: import sys
11: import scipy
12: 
13: import pytest
14: 
15: 
16: if sys.version_info >= (3, 4):
17:     from pathlib import Path
18:     import ast
19:     import tokenize
20: 
21:     class ParseCall(ast.NodeVisitor):
22:         def __init__(self):
23:             self.ls = []
24: 
25:         def visit_Attribute(self, node):
26:             ast.NodeVisitor.generic_visit(self, node)
27:             self.ls.append(node.attr)
28: 
29:         def visit_Name(self, node):
30:             self.ls.append(node.id)
31: 
32:     class FindFuncs(ast.NodeVisitor):
33:         def __init__(self, filename):
34:             super().__init__()
35:             self.__filename = filename
36:             self.bad_filters = []
37:             self.bad_stacklevels = []
38: 
39:         def visit_Call(self, node):
40:             p = ParseCall()
41:             p.visit(node.func)
42:             ast.NodeVisitor.generic_visit(self, node)
43: 
44:             if p.ls[-1] == 'simplefilter' or p.ls[-1] == 'filterwarnings':
45:                 if node.args[0].s == "ignore":
46:                     self.bad_filters.append(
47:                         "{}:{}".format(self.__filename, node.lineno))
48: 
49:             if p.ls[-1] == 'warn' and (
50:                     len(p.ls) == 1 or p.ls[-2] == 'warnings'):
51: 
52:                 if self.__filename == "_lib/tests/test_warnings.py":
53:                     # This file
54:                     return
55: 
56:                 # See if stacklevel exists:
57:                 if len(node.args) == 3:
58:                     return
59:                 args = {kw.arg for kw in node.keywords}
60:                 if "stacklevel" not in args:
61:                     self.bad_stacklevels.append(
62:                         "{}:{}".format(self.__filename, node.lineno))
63: 
64: 
65: @pytest.fixture(scope="session")
66: def warning_calls():
67:     # combined "ignore" and stacklevel error
68:     base = Path(scipy.__file__).parent
69: 
70:     bad_filters = []
71:     bad_stacklevels = []
72:     
73:     for path in base.rglob("*.py"):
74:         # use tokenize to auto-detect encoding on systems where no
75:         # default encoding is defined (e.g. LANG='C')
76:         with tokenize.open(str(path)) as file:
77:             tree = ast.parse(file.read(), filename=str(path))
78:             finder = FindFuncs(path.relative_to(base))
79:             finder.visit(tree)
80:             bad_filters.extend(finder.bad_filters)
81:             bad_stacklevels.extend(finder.bad_stacklevels)
82: 
83:     return bad_filters, bad_stacklevels
84: 
85: 
86: @pytest.mark.slow
87: @pytest.mark.skipif(sys.version_info < (3, 4), reason="needs Python >= 3.4")
88: def test_warning_calls_filters(warning_calls):
89:     bad_filters, bad_stacklevels = warning_calls
90: 
91:     # There is still one missing occurance in optimize.py,
92:     # this is one that should be fixed and this removed then.
93:     bad_filters = [item for item in bad_filters
94:                    if 'optimize.py' not in item]
95: 
96:     if bad_filters:
97:         raise AssertionError(
98:             "warning ignore filter should not be used, instead, use\n"
99:             "scipy._lib._numpy_compat.suppress_warnings (in tests only);\n"
100:             "found in:\n    {}".format(
101:                 "\n    ".join(bad_filters)))
102: 
103: 
104: @pytest.mark.slow
105: @pytest.mark.skipif(sys.version_info < (3, 4), reason="needs Python >= 3.4")
106: @pytest.mark.xfail(reason="stacklevels currently missing")
107: def test_warning_calls_stacklevels(warning_calls):
108:     bad_filters, bad_stacklevels = warning_calls
109: 
110:     msg = ""
111: 
112:     if bad_filters:
113:         msg += ("warning ignore filter should not be used, instead, use\n"
114:                 "scipy._lib._numpy_compat.suppress_warnings (in tests only);\n"
115:                 "found in:\n    {}".format("\n    ".join(bad_filters)))
116:         msg += "\n\n"
117: 
118:     if bad_stacklevels:
119:         msg += "warnings should have an appropriate stacklevel:\n    {}".format(
120:                 "\n    ".join(bad_stacklevels))
121: 
122:     if msg:
123:         raise AssertionError(msg)
124: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_711852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'str', '\nTests which scan for certain occurrences in the code, they may not find\nall of these occurrences but should catch almost all. This file was adapted\nfrom numpy.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import sys' statement (line 10)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import scipy' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_711853 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy')

if (type(import_711853) is not StypyTypeError):

    if (import_711853 != 'pyd_module'):
        __import__(import_711853)
        sys_modules_711854 = sys.modules[import_711853]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy', sys_modules_711854.module_type_store, module_type_store)
    else:
        import scipy

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy', scipy, module_type_store)

else:
    # Assigning a type to the variable 'scipy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy', import_711853)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import pytest' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_711855 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'pytest')

if (type(import_711855) is not StypyTypeError):

    if (import_711855 != 'pyd_module'):
        __import__(import_711855)
        sys_modules_711856 = sys.modules[import_711855]
        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'pytest', sys_modules_711856.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'pytest', import_711855)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')



# Getting the type of 'sys' (line 16)
sys_711857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 3), 'sys')
# Obtaining the member 'version_info' of a type (line 16)
version_info_711858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 3), sys_711857, 'version_info')

# Obtaining an instance of the builtin type 'tuple' (line 16)
tuple_711859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 16)
# Adding element type (line 16)
int_711860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 24), tuple_711859, int_711860)
# Adding element type (line 16)
int_711861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 24), tuple_711859, int_711861)

# Applying the binary operator '>=' (line 16)
result_ge_711862 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 3), '>=', version_info_711858, tuple_711859)

# Testing the type of an if condition (line 16)
if_condition_711863 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 16, 0), result_ge_711862)
# Assigning a type to the variable 'if_condition_711863' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'if_condition_711863', if_condition_711863)
# SSA begins for if statement (line 16)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 4))

# 'from pathlib import Path' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_711864 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 4), 'pathlib')

if (type(import_711864) is not StypyTypeError):

    if (import_711864 != 'pyd_module'):
        __import__(import_711864)
        sys_modules_711865 = sys.modules[import_711864]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 4), 'pathlib', sys_modules_711865.module_type_store, module_type_store, ['Path'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 4), __file__, sys_modules_711865, sys_modules_711865.module_type_store, module_type_store)
    else:
        from pathlib import Path

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 4), 'pathlib', None, module_type_store, ['Path'], [Path])

else:
    # Assigning a type to the variable 'pathlib' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'pathlib', import_711864)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 4))

# 'import ast' statement (line 18)
import ast

import_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'ast', ast, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 4))

# 'import tokenize' statement (line 19)
import tokenize

import_module(stypy.reporting.localization.Localization(__file__, 19, 4), 'tokenize', tokenize, module_type_store)

# Declaration of the 'ParseCall' class
# Getting the type of 'ast' (line 21)
ast_711866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 20), 'ast')
# Obtaining the member 'NodeVisitor' of a type (line 21)
NodeVisitor_711867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 20), ast_711866, 'NodeVisitor')

class ParseCall(NodeVisitor_711867, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 22, 8, False)
        # Assigning a type to the variable 'self' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ParseCall.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a List to a Attribute (line 23):
        
        # Assigning a List to a Attribute (line 23):
        
        # Obtaining an instance of the builtin type 'list' (line 23)
        list_711868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 23)
        
        # Getting the type of 'self' (line 23)
        self_711869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'self')
        # Setting the type of the member 'ls' of a type (line 23)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 12), self_711869, 'ls', list_711868)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def visit_Attribute(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Attribute'
        module_type_store = module_type_store.open_function_context('visit_Attribute', 25, 8, False)
        # Assigning a type to the variable 'self' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        ParseCall.visit_Attribute.__dict__.__setitem__('stypy_localization', localization)
        ParseCall.visit_Attribute.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ParseCall.visit_Attribute.__dict__.__setitem__('stypy_type_store', module_type_store)
        ParseCall.visit_Attribute.__dict__.__setitem__('stypy_function_name', 'ParseCall.visit_Attribute')
        ParseCall.visit_Attribute.__dict__.__setitem__('stypy_param_names_list', ['node'])
        ParseCall.visit_Attribute.__dict__.__setitem__('stypy_varargs_param_name', None)
        ParseCall.visit_Attribute.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ParseCall.visit_Attribute.__dict__.__setitem__('stypy_call_defaults', defaults)
        ParseCall.visit_Attribute.__dict__.__setitem__('stypy_call_varargs', varargs)
        ParseCall.visit_Attribute.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ParseCall.visit_Attribute.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ParseCall.visit_Attribute', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Attribute', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Attribute(...)' code ##################

        
        # Call to generic_visit(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'self' (line 26)
        self_711873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 42), 'self', False)
        # Getting the type of 'node' (line 26)
        node_711874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 48), 'node', False)
        # Processing the call keyword arguments (line 26)
        kwargs_711875 = {}
        # Getting the type of 'ast' (line 26)
        ast_711870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'ast', False)
        # Obtaining the member 'NodeVisitor' of a type (line 26)
        NodeVisitor_711871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 12), ast_711870, 'NodeVisitor')
        # Obtaining the member 'generic_visit' of a type (line 26)
        generic_visit_711872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 12), NodeVisitor_711871, 'generic_visit')
        # Calling generic_visit(args, kwargs) (line 26)
        generic_visit_call_result_711876 = invoke(stypy.reporting.localization.Localization(__file__, 26, 12), generic_visit_711872, *[self_711873, node_711874], **kwargs_711875)
        
        
        # Call to append(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'node' (line 27)
        node_711880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 27), 'node', False)
        # Obtaining the member 'attr' of a type (line 27)
        attr_711881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 27), node_711880, 'attr')
        # Processing the call keyword arguments (line 27)
        kwargs_711882 = {}
        # Getting the type of 'self' (line 27)
        self_711877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'self', False)
        # Obtaining the member 'ls' of a type (line 27)
        ls_711878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 12), self_711877, 'ls')
        # Obtaining the member 'append' of a type (line 27)
        append_711879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 12), ls_711878, 'append')
        # Calling append(args, kwargs) (line 27)
        append_call_result_711883 = invoke(stypy.reporting.localization.Localization(__file__, 27, 12), append_711879, *[attr_711881], **kwargs_711882)
        
        
        # ################# End of 'visit_Attribute(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Attribute' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_711884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_711884)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Attribute'
        return stypy_return_type_711884


    @norecursion
    def visit_Name(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Name'
        module_type_store = module_type_store.open_function_context('visit_Name', 29, 8, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        ParseCall.visit_Name.__dict__.__setitem__('stypy_localization', localization)
        ParseCall.visit_Name.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ParseCall.visit_Name.__dict__.__setitem__('stypy_type_store', module_type_store)
        ParseCall.visit_Name.__dict__.__setitem__('stypy_function_name', 'ParseCall.visit_Name')
        ParseCall.visit_Name.__dict__.__setitem__('stypy_param_names_list', ['node'])
        ParseCall.visit_Name.__dict__.__setitem__('stypy_varargs_param_name', None)
        ParseCall.visit_Name.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ParseCall.visit_Name.__dict__.__setitem__('stypy_call_defaults', defaults)
        ParseCall.visit_Name.__dict__.__setitem__('stypy_call_varargs', varargs)
        ParseCall.visit_Name.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ParseCall.visit_Name.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ParseCall.visit_Name', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Name', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Name(...)' code ##################

        
        # Call to append(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'node' (line 30)
        node_711888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 27), 'node', False)
        # Obtaining the member 'id' of a type (line 30)
        id_711889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 27), node_711888, 'id')
        # Processing the call keyword arguments (line 30)
        kwargs_711890 = {}
        # Getting the type of 'self' (line 30)
        self_711885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'self', False)
        # Obtaining the member 'ls' of a type (line 30)
        ls_711886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 12), self_711885, 'ls')
        # Obtaining the member 'append' of a type (line 30)
        append_711887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 12), ls_711886, 'append')
        # Calling append(args, kwargs) (line 30)
        append_call_result_711891 = invoke(stypy.reporting.localization.Localization(__file__, 30, 12), append_711887, *[id_711889], **kwargs_711890)
        
        
        # ################# End of 'visit_Name(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Name' in the type store
        # Getting the type of 'stypy_return_type' (line 29)
        stypy_return_type_711892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_711892)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Name'
        return stypy_return_type_711892


# Assigning a type to the variable 'ParseCall' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'ParseCall', ParseCall)
# Declaration of the 'FindFuncs' class
# Getting the type of 'ast' (line 32)
ast_711893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 20), 'ast')
# Obtaining the member 'NodeVisitor' of a type (line 32)
NodeVisitor_711894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 20), ast_711893, 'NodeVisitor')

class FindFuncs(NodeVisitor_711894, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 33, 8, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FindFuncs.__init__', ['filename'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['filename'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 34)
        # Processing the call keyword arguments (line 34)
        kwargs_711899 = {}
        
        # Call to super(...): (line 34)
        # Processing the call keyword arguments (line 34)
        kwargs_711896 = {}
        # Getting the type of 'super' (line 34)
        super_711895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'super', False)
        # Calling super(args, kwargs) (line 34)
        super_call_result_711897 = invoke(stypy.reporting.localization.Localization(__file__, 34, 12), super_711895, *[], **kwargs_711896)
        
        # Obtaining the member '__init__' of a type (line 34)
        init___711898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), super_call_result_711897, '__init__')
        # Calling __init__(args, kwargs) (line 34)
        init___call_result_711900 = invoke(stypy.reporting.localization.Localization(__file__, 34, 12), init___711898, *[], **kwargs_711899)
        
        
        # Assigning a Name to a Attribute (line 35):
        
        # Assigning a Name to a Attribute (line 35):
        # Getting the type of 'filename' (line 35)
        filename_711901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 30), 'filename')
        # Getting the type of 'self' (line 35)
        self_711902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'self')
        # Setting the type of the member '__filename' of a type (line 35)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 12), self_711902, '__filename', filename_711901)
        
        # Assigning a List to a Attribute (line 36):
        
        # Assigning a List to a Attribute (line 36):
        
        # Obtaining an instance of the builtin type 'list' (line 36)
        list_711903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 36)
        
        # Getting the type of 'self' (line 36)
        self_711904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'self')
        # Setting the type of the member 'bad_filters' of a type (line 36)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), self_711904, 'bad_filters', list_711903)
        
        # Assigning a List to a Attribute (line 37):
        
        # Assigning a List to a Attribute (line 37):
        
        # Obtaining an instance of the builtin type 'list' (line 37)
        list_711905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 37)
        
        # Getting the type of 'self' (line 37)
        self_711906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'self')
        # Setting the type of the member 'bad_stacklevels' of a type (line 37)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 12), self_711906, 'bad_stacklevels', list_711905)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def visit_Call(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Call'
        module_type_store = module_type_store.open_function_context('visit_Call', 39, 8, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        FindFuncs.visit_Call.__dict__.__setitem__('stypy_localization', localization)
        FindFuncs.visit_Call.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FindFuncs.visit_Call.__dict__.__setitem__('stypy_type_store', module_type_store)
        FindFuncs.visit_Call.__dict__.__setitem__('stypy_function_name', 'FindFuncs.visit_Call')
        FindFuncs.visit_Call.__dict__.__setitem__('stypy_param_names_list', ['node'])
        FindFuncs.visit_Call.__dict__.__setitem__('stypy_varargs_param_name', None)
        FindFuncs.visit_Call.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FindFuncs.visit_Call.__dict__.__setitem__('stypy_call_defaults', defaults)
        FindFuncs.visit_Call.__dict__.__setitem__('stypy_call_varargs', varargs)
        FindFuncs.visit_Call.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FindFuncs.visit_Call.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FindFuncs.visit_Call', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Call', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Call(...)' code ##################

        
        # Assigning a Call to a Name (line 40):
        
        # Assigning a Call to a Name (line 40):
        
        # Call to ParseCall(...): (line 40)
        # Processing the call keyword arguments (line 40)
        kwargs_711908 = {}
        # Getting the type of 'ParseCall' (line 40)
        ParseCall_711907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'ParseCall', False)
        # Calling ParseCall(args, kwargs) (line 40)
        ParseCall_call_result_711909 = invoke(stypy.reporting.localization.Localization(__file__, 40, 16), ParseCall_711907, *[], **kwargs_711908)
        
        # Assigning a type to the variable 'p' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'p', ParseCall_call_result_711909)
        
        # Call to visit(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'node' (line 41)
        node_711912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 20), 'node', False)
        # Obtaining the member 'func' of a type (line 41)
        func_711913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 20), node_711912, 'func')
        # Processing the call keyword arguments (line 41)
        kwargs_711914 = {}
        # Getting the type of 'p' (line 41)
        p_711910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'p', False)
        # Obtaining the member 'visit' of a type (line 41)
        visit_711911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 12), p_711910, 'visit')
        # Calling visit(args, kwargs) (line 41)
        visit_call_result_711915 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), visit_711911, *[func_711913], **kwargs_711914)
        
        
        # Call to generic_visit(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'self' (line 42)
        self_711919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 42), 'self', False)
        # Getting the type of 'node' (line 42)
        node_711920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 48), 'node', False)
        # Processing the call keyword arguments (line 42)
        kwargs_711921 = {}
        # Getting the type of 'ast' (line 42)
        ast_711916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'ast', False)
        # Obtaining the member 'NodeVisitor' of a type (line 42)
        NodeVisitor_711917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 12), ast_711916, 'NodeVisitor')
        # Obtaining the member 'generic_visit' of a type (line 42)
        generic_visit_711918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 12), NodeVisitor_711917, 'generic_visit')
        # Calling generic_visit(args, kwargs) (line 42)
        generic_visit_call_result_711922 = invoke(stypy.reporting.localization.Localization(__file__, 42, 12), generic_visit_711918, *[self_711919, node_711920], **kwargs_711921)
        
        
        
        # Evaluating a boolean operation
        
        
        # Obtaining the type of the subscript
        int_711923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 20), 'int')
        # Getting the type of 'p' (line 44)
        p_711924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'p')
        # Obtaining the member 'ls' of a type (line 44)
        ls_711925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 15), p_711924, 'ls')
        # Obtaining the member '__getitem__' of a type (line 44)
        getitem___711926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 15), ls_711925, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 44)
        subscript_call_result_711927 = invoke(stypy.reporting.localization.Localization(__file__, 44, 15), getitem___711926, int_711923)
        
        str_711928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 27), 'str', 'simplefilter')
        # Applying the binary operator '==' (line 44)
        result_eq_711929 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 15), '==', subscript_call_result_711927, str_711928)
        
        
        
        # Obtaining the type of the subscript
        int_711930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 50), 'int')
        # Getting the type of 'p' (line 44)
        p_711931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 45), 'p')
        # Obtaining the member 'ls' of a type (line 44)
        ls_711932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 45), p_711931, 'ls')
        # Obtaining the member '__getitem__' of a type (line 44)
        getitem___711933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 45), ls_711932, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 44)
        subscript_call_result_711934 = invoke(stypy.reporting.localization.Localization(__file__, 44, 45), getitem___711933, int_711930)
        
        str_711935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 57), 'str', 'filterwarnings')
        # Applying the binary operator '==' (line 44)
        result_eq_711936 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 45), '==', subscript_call_result_711934, str_711935)
        
        # Applying the binary operator 'or' (line 44)
        result_or_keyword_711937 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 15), 'or', result_eq_711929, result_eq_711936)
        
        # Testing the type of an if condition (line 44)
        if_condition_711938 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 12), result_or_keyword_711937)
        # Assigning a type to the variable 'if_condition_711938' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'if_condition_711938', if_condition_711938)
        # SSA begins for if statement (line 44)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Obtaining the type of the subscript
        int_711939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 29), 'int')
        # Getting the type of 'node' (line 45)
        node_711940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 19), 'node')
        # Obtaining the member 'args' of a type (line 45)
        args_711941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 19), node_711940, 'args')
        # Obtaining the member '__getitem__' of a type (line 45)
        getitem___711942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 19), args_711941, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 45)
        subscript_call_result_711943 = invoke(stypy.reporting.localization.Localization(__file__, 45, 19), getitem___711942, int_711939)
        
        # Obtaining the member 's' of a type (line 45)
        s_711944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 19), subscript_call_result_711943, 's')
        str_711945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 37), 'str', 'ignore')
        # Applying the binary operator '==' (line 45)
        result_eq_711946 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 19), '==', s_711944, str_711945)
        
        # Testing the type of an if condition (line 45)
        if_condition_711947 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 16), result_eq_711946)
        # Assigning a type to the variable 'if_condition_711947' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 16), 'if_condition_711947', if_condition_711947)
        # SSA begins for if statement (line 45)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 46)
        # Processing the call arguments (line 46)
        
        # Call to format(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'self' (line 47)
        self_711953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 39), 'self', False)
        # Obtaining the member '__filename' of a type (line 47)
        filename_711954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 39), self_711953, '__filename')
        # Getting the type of 'node' (line 47)
        node_711955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 56), 'node', False)
        # Obtaining the member 'lineno' of a type (line 47)
        lineno_711956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 56), node_711955, 'lineno')
        # Processing the call keyword arguments (line 47)
        kwargs_711957 = {}
        str_711951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 24), 'str', '{}:{}')
        # Obtaining the member 'format' of a type (line 47)
        format_711952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), str_711951, 'format')
        # Calling format(args, kwargs) (line 47)
        format_call_result_711958 = invoke(stypy.reporting.localization.Localization(__file__, 47, 24), format_711952, *[filename_711954, lineno_711956], **kwargs_711957)
        
        # Processing the call keyword arguments (line 46)
        kwargs_711959 = {}
        # Getting the type of 'self' (line 46)
        self_711948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'self', False)
        # Obtaining the member 'bad_filters' of a type (line 46)
        bad_filters_711949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 20), self_711948, 'bad_filters')
        # Obtaining the member 'append' of a type (line 46)
        append_711950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 20), bad_filters_711949, 'append')
        # Calling append(args, kwargs) (line 46)
        append_call_result_711960 = invoke(stypy.reporting.localization.Localization(__file__, 46, 20), append_711950, *[format_call_result_711958], **kwargs_711959)
        
        # SSA join for if statement (line 45)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 44)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        
        # Obtaining the type of the subscript
        int_711961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 20), 'int')
        # Getting the type of 'p' (line 49)
        p_711962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'p')
        # Obtaining the member 'ls' of a type (line 49)
        ls_711963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 15), p_711962, 'ls')
        # Obtaining the member '__getitem__' of a type (line 49)
        getitem___711964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 15), ls_711963, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
        subscript_call_result_711965 = invoke(stypy.reporting.localization.Localization(__file__, 49, 15), getitem___711964, int_711961)
        
        str_711966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 27), 'str', 'warn')
        # Applying the binary operator '==' (line 49)
        result_eq_711967 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 15), '==', subscript_call_result_711965, str_711966)
        
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'p' (line 50)
        p_711969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 24), 'p', False)
        # Obtaining the member 'ls' of a type (line 50)
        ls_711970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 24), p_711969, 'ls')
        # Processing the call keyword arguments (line 50)
        kwargs_711971 = {}
        # Getting the type of 'len' (line 50)
        len_711968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 20), 'len', False)
        # Calling len(args, kwargs) (line 50)
        len_call_result_711972 = invoke(stypy.reporting.localization.Localization(__file__, 50, 20), len_711968, *[ls_711970], **kwargs_711971)
        
        int_711973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 33), 'int')
        # Applying the binary operator '==' (line 50)
        result_eq_711974 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 20), '==', len_call_result_711972, int_711973)
        
        
        
        # Obtaining the type of the subscript
        int_711975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 43), 'int')
        # Getting the type of 'p' (line 50)
        p_711976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 38), 'p')
        # Obtaining the member 'ls' of a type (line 50)
        ls_711977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 38), p_711976, 'ls')
        # Obtaining the member '__getitem__' of a type (line 50)
        getitem___711978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 38), ls_711977, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 50)
        subscript_call_result_711979 = invoke(stypy.reporting.localization.Localization(__file__, 50, 38), getitem___711978, int_711975)
        
        str_711980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 50), 'str', 'warnings')
        # Applying the binary operator '==' (line 50)
        result_eq_711981 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 38), '==', subscript_call_result_711979, str_711980)
        
        # Applying the binary operator 'or' (line 50)
        result_or_keyword_711982 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 20), 'or', result_eq_711974, result_eq_711981)
        
        # Applying the binary operator 'and' (line 49)
        result_and_keyword_711983 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 15), 'and', result_eq_711967, result_or_keyword_711982)
        
        # Testing the type of an if condition (line 49)
        if_condition_711984 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 12), result_and_keyword_711983)
        # Assigning a type to the variable 'if_condition_711984' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'if_condition_711984', if_condition_711984)
        # SSA begins for if statement (line 49)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'self' (line 52)
        self_711985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 19), 'self')
        # Obtaining the member '__filename' of a type (line 52)
        filename_711986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 19), self_711985, '__filename')
        str_711987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 38), 'str', '_lib/tests/test_warnings.py')
        # Applying the binary operator '==' (line 52)
        result_eq_711988 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 19), '==', filename_711986, str_711987)
        
        # Testing the type of an if condition (line 52)
        if_condition_711989 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 52, 16), result_eq_711988)
        # Assigning a type to the variable 'if_condition_711989' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'if_condition_711989', if_condition_711989)
        # SSA begins for if statement (line 52)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 52)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to len(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'node' (line 57)
        node_711991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 23), 'node', False)
        # Obtaining the member 'args' of a type (line 57)
        args_711992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 23), node_711991, 'args')
        # Processing the call keyword arguments (line 57)
        kwargs_711993 = {}
        # Getting the type of 'len' (line 57)
        len_711990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 19), 'len', False)
        # Calling len(args, kwargs) (line 57)
        len_call_result_711994 = invoke(stypy.reporting.localization.Localization(__file__, 57, 19), len_711990, *[args_711992], **kwargs_711993)
        
        int_711995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 37), 'int')
        # Applying the binary operator '==' (line 57)
        result_eq_711996 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 19), '==', len_call_result_711994, int_711995)
        
        # Testing the type of an if condition (line 57)
        if_condition_711997 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 16), result_eq_711996)
        # Assigning a type to the variable 'if_condition_711997' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'if_condition_711997', if_condition_711997)
        # SSA begins for if statement (line 57)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 20), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 57)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a SetComp to a Name (line 59):
        
        # Assigning a SetComp to a Name (line 59):
        # Calculating set comprehension
        module_type_store = module_type_store.open_function_context('set comprehension expression', 59, 24, True)
        # Calculating comprehension expression
        # Getting the type of 'node' (line 59)
        node_712000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 41), 'node')
        # Obtaining the member 'keywords' of a type (line 59)
        keywords_712001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 41), node_712000, 'keywords')
        comprehension_712002 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 24), keywords_712001)
        # Assigning a type to the variable 'kw' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 24), 'kw', comprehension_712002)
        # Getting the type of 'kw' (line 59)
        kw_711998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 24), 'kw')
        # Obtaining the member 'arg' of a type (line 59)
        arg_711999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 24), kw_711998, 'arg')
        set_712003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 24), 'set')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 24), set_712003, arg_711999)
        # Assigning a type to the variable 'args' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'args', set_712003)
        
        
        str_712004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 19), 'str', 'stacklevel')
        # Getting the type of 'args' (line 60)
        args_712005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 39), 'args')
        # Applying the binary operator 'notin' (line 60)
        result_contains_712006 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 19), 'notin', str_712004, args_712005)
        
        # Testing the type of an if condition (line 60)
        if_condition_712007 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 16), result_contains_712006)
        # Assigning a type to the variable 'if_condition_712007' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'if_condition_712007', if_condition_712007)
        # SSA begins for if statement (line 60)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 61)
        # Processing the call arguments (line 61)
        
        # Call to format(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'self' (line 62)
        self_712013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 39), 'self', False)
        # Obtaining the member '__filename' of a type (line 62)
        filename_712014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 39), self_712013, '__filename')
        # Getting the type of 'node' (line 62)
        node_712015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 56), 'node', False)
        # Obtaining the member 'lineno' of a type (line 62)
        lineno_712016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 56), node_712015, 'lineno')
        # Processing the call keyword arguments (line 62)
        kwargs_712017 = {}
        str_712011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 24), 'str', '{}:{}')
        # Obtaining the member 'format' of a type (line 62)
        format_712012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 24), str_712011, 'format')
        # Calling format(args, kwargs) (line 62)
        format_call_result_712018 = invoke(stypy.reporting.localization.Localization(__file__, 62, 24), format_712012, *[filename_712014, lineno_712016], **kwargs_712017)
        
        # Processing the call keyword arguments (line 61)
        kwargs_712019 = {}
        # Getting the type of 'self' (line 61)
        self_712008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 20), 'self', False)
        # Obtaining the member 'bad_stacklevels' of a type (line 61)
        bad_stacklevels_712009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 20), self_712008, 'bad_stacklevels')
        # Obtaining the member 'append' of a type (line 61)
        append_712010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 20), bad_stacklevels_712009, 'append')
        # Calling append(args, kwargs) (line 61)
        append_call_result_712020 = invoke(stypy.reporting.localization.Localization(__file__, 61, 20), append_712010, *[format_call_result_712018], **kwargs_712019)
        
        # SSA join for if statement (line 60)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 49)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'visit_Call(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Call' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_712021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_712021)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Call'
        return stypy_return_type_712021


# Assigning a type to the variable 'FindFuncs' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'FindFuncs', FindFuncs)
# SSA join for if statement (line 16)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def warning_calls(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'warning_calls'
    module_type_store = module_type_store.open_function_context('warning_calls', 65, 0, False)
    
    # Passed parameters checking function
    warning_calls.stypy_localization = localization
    warning_calls.stypy_type_of_self = None
    warning_calls.stypy_type_store = module_type_store
    warning_calls.stypy_function_name = 'warning_calls'
    warning_calls.stypy_param_names_list = []
    warning_calls.stypy_varargs_param_name = None
    warning_calls.stypy_kwargs_param_name = None
    warning_calls.stypy_call_defaults = defaults
    warning_calls.stypy_call_varargs = varargs
    warning_calls.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'warning_calls', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'warning_calls', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'warning_calls(...)' code ##################

    
    # Assigning a Attribute to a Name (line 68):
    
    # Assigning a Attribute to a Name (line 68):
    
    # Call to Path(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'scipy' (line 68)
    scipy_712023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'scipy', False)
    # Obtaining the member '__file__' of a type (line 68)
    file___712024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 16), scipy_712023, '__file__')
    # Processing the call keyword arguments (line 68)
    kwargs_712025 = {}
    # Getting the type of 'Path' (line 68)
    Path_712022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'Path', False)
    # Calling Path(args, kwargs) (line 68)
    Path_call_result_712026 = invoke(stypy.reporting.localization.Localization(__file__, 68, 11), Path_712022, *[file___712024], **kwargs_712025)
    
    # Obtaining the member 'parent' of a type (line 68)
    parent_712027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 11), Path_call_result_712026, 'parent')
    # Assigning a type to the variable 'base' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'base', parent_712027)
    
    # Assigning a List to a Name (line 70):
    
    # Assigning a List to a Name (line 70):
    
    # Obtaining an instance of the builtin type 'list' (line 70)
    list_712028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 70)
    
    # Assigning a type to the variable 'bad_filters' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'bad_filters', list_712028)
    
    # Assigning a List to a Name (line 71):
    
    # Assigning a List to a Name (line 71):
    
    # Obtaining an instance of the builtin type 'list' (line 71)
    list_712029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 71)
    
    # Assigning a type to the variable 'bad_stacklevels' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'bad_stacklevels', list_712029)
    
    
    # Call to rglob(...): (line 73)
    # Processing the call arguments (line 73)
    str_712032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 27), 'str', '*.py')
    # Processing the call keyword arguments (line 73)
    kwargs_712033 = {}
    # Getting the type of 'base' (line 73)
    base_712030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'base', False)
    # Obtaining the member 'rglob' of a type (line 73)
    rglob_712031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 16), base_712030, 'rglob')
    # Calling rglob(args, kwargs) (line 73)
    rglob_call_result_712034 = invoke(stypy.reporting.localization.Localization(__file__, 73, 16), rglob_712031, *[str_712032], **kwargs_712033)
    
    # Testing the type of a for loop iterable (line 73)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 73, 4), rglob_call_result_712034)
    # Getting the type of the for loop variable (line 73)
    for_loop_var_712035 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 73, 4), rglob_call_result_712034)
    # Assigning a type to the variable 'path' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'path', for_loop_var_712035)
    # SSA begins for a for statement (line 73)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to open(...): (line 76)
    # Processing the call arguments (line 76)
    
    # Call to str(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'path' (line 76)
    path_712039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 31), 'path', False)
    # Processing the call keyword arguments (line 76)
    kwargs_712040 = {}
    # Getting the type of 'str' (line 76)
    str_712038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 27), 'str', False)
    # Calling str(args, kwargs) (line 76)
    str_call_result_712041 = invoke(stypy.reporting.localization.Localization(__file__, 76, 27), str_712038, *[path_712039], **kwargs_712040)
    
    # Processing the call keyword arguments (line 76)
    kwargs_712042 = {}
    # Getting the type of 'tokenize' (line 76)
    tokenize_712036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 13), 'tokenize', False)
    # Obtaining the member 'open' of a type (line 76)
    open_712037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 13), tokenize_712036, 'open')
    # Calling open(args, kwargs) (line 76)
    open_call_result_712043 = invoke(stypy.reporting.localization.Localization(__file__, 76, 13), open_712037, *[str_call_result_712041], **kwargs_712042)
    
    with_712044 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 76, 13), open_call_result_712043, 'with parameter', '__enter__', '__exit__')

    if with_712044:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 76)
        enter___712045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 13), open_call_result_712043, '__enter__')
        with_enter_712046 = invoke(stypy.reporting.localization.Localization(__file__, 76, 13), enter___712045)
        # Assigning a type to the variable 'file' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 13), 'file', with_enter_712046)
        
        # Assigning a Call to a Name (line 77):
        
        # Assigning a Call to a Name (line 77):
        
        # Call to parse(...): (line 77)
        # Processing the call arguments (line 77)
        
        # Call to read(...): (line 77)
        # Processing the call keyword arguments (line 77)
        kwargs_712051 = {}
        # Getting the type of 'file' (line 77)
        file_712049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 29), 'file', False)
        # Obtaining the member 'read' of a type (line 77)
        read_712050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 29), file_712049, 'read')
        # Calling read(args, kwargs) (line 77)
        read_call_result_712052 = invoke(stypy.reporting.localization.Localization(__file__, 77, 29), read_712050, *[], **kwargs_712051)
        
        # Processing the call keyword arguments (line 77)
        
        # Call to str(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'path' (line 77)
        path_712054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 55), 'path', False)
        # Processing the call keyword arguments (line 77)
        kwargs_712055 = {}
        # Getting the type of 'str' (line 77)
        str_712053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 51), 'str', False)
        # Calling str(args, kwargs) (line 77)
        str_call_result_712056 = invoke(stypy.reporting.localization.Localization(__file__, 77, 51), str_712053, *[path_712054], **kwargs_712055)
        
        keyword_712057 = str_call_result_712056
        kwargs_712058 = {'filename': keyword_712057}
        # Getting the type of 'ast' (line 77)
        ast_712047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 19), 'ast', False)
        # Obtaining the member 'parse' of a type (line 77)
        parse_712048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 19), ast_712047, 'parse')
        # Calling parse(args, kwargs) (line 77)
        parse_call_result_712059 = invoke(stypy.reporting.localization.Localization(__file__, 77, 19), parse_712048, *[read_call_result_712052], **kwargs_712058)
        
        # Assigning a type to the variable 'tree' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'tree', parse_call_result_712059)
        
        # Assigning a Call to a Name (line 78):
        
        # Assigning a Call to a Name (line 78):
        
        # Call to FindFuncs(...): (line 78)
        # Processing the call arguments (line 78)
        
        # Call to relative_to(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'base' (line 78)
        base_712063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 48), 'base', False)
        # Processing the call keyword arguments (line 78)
        kwargs_712064 = {}
        # Getting the type of 'path' (line 78)
        path_712061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 31), 'path', False)
        # Obtaining the member 'relative_to' of a type (line 78)
        relative_to_712062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 31), path_712061, 'relative_to')
        # Calling relative_to(args, kwargs) (line 78)
        relative_to_call_result_712065 = invoke(stypy.reporting.localization.Localization(__file__, 78, 31), relative_to_712062, *[base_712063], **kwargs_712064)
        
        # Processing the call keyword arguments (line 78)
        kwargs_712066 = {}
        # Getting the type of 'FindFuncs' (line 78)
        FindFuncs_712060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 21), 'FindFuncs', False)
        # Calling FindFuncs(args, kwargs) (line 78)
        FindFuncs_call_result_712067 = invoke(stypy.reporting.localization.Localization(__file__, 78, 21), FindFuncs_712060, *[relative_to_call_result_712065], **kwargs_712066)
        
        # Assigning a type to the variable 'finder' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'finder', FindFuncs_call_result_712067)
        
        # Call to visit(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'tree' (line 79)
        tree_712070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 25), 'tree', False)
        # Processing the call keyword arguments (line 79)
        kwargs_712071 = {}
        # Getting the type of 'finder' (line 79)
        finder_712068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'finder', False)
        # Obtaining the member 'visit' of a type (line 79)
        visit_712069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), finder_712068, 'visit')
        # Calling visit(args, kwargs) (line 79)
        visit_call_result_712072 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), visit_712069, *[tree_712070], **kwargs_712071)
        
        
        # Call to extend(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'finder' (line 80)
        finder_712075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 31), 'finder', False)
        # Obtaining the member 'bad_filters' of a type (line 80)
        bad_filters_712076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 31), finder_712075, 'bad_filters')
        # Processing the call keyword arguments (line 80)
        kwargs_712077 = {}
        # Getting the type of 'bad_filters' (line 80)
        bad_filters_712073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'bad_filters', False)
        # Obtaining the member 'extend' of a type (line 80)
        extend_712074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 12), bad_filters_712073, 'extend')
        # Calling extend(args, kwargs) (line 80)
        extend_call_result_712078 = invoke(stypy.reporting.localization.Localization(__file__, 80, 12), extend_712074, *[bad_filters_712076], **kwargs_712077)
        
        
        # Call to extend(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'finder' (line 81)
        finder_712081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 35), 'finder', False)
        # Obtaining the member 'bad_stacklevels' of a type (line 81)
        bad_stacklevels_712082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 35), finder_712081, 'bad_stacklevels')
        # Processing the call keyword arguments (line 81)
        kwargs_712083 = {}
        # Getting the type of 'bad_stacklevels' (line 81)
        bad_stacklevels_712079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'bad_stacklevels', False)
        # Obtaining the member 'extend' of a type (line 81)
        extend_712080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 12), bad_stacklevels_712079, 'extend')
        # Calling extend(args, kwargs) (line 81)
        extend_call_result_712084 = invoke(stypy.reporting.localization.Localization(__file__, 81, 12), extend_712080, *[bad_stacklevels_712082], **kwargs_712083)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 76)
        exit___712085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 13), open_call_result_712043, '__exit__')
        with_exit_712086 = invoke(stypy.reporting.localization.Localization(__file__, 76, 13), exit___712085, None, None, None)

    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 83)
    tuple_712087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 83)
    # Adding element type (line 83)
    # Getting the type of 'bad_filters' (line 83)
    bad_filters_712088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 11), 'bad_filters')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 11), tuple_712087, bad_filters_712088)
    # Adding element type (line 83)
    # Getting the type of 'bad_stacklevels' (line 83)
    bad_stacklevels_712089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 24), 'bad_stacklevels')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 11), tuple_712087, bad_stacklevels_712089)
    
    # Assigning a type to the variable 'stypy_return_type' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'stypy_return_type', tuple_712087)
    
    # ################# End of 'warning_calls(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'warning_calls' in the type store
    # Getting the type of 'stypy_return_type' (line 65)
    stypy_return_type_712090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_712090)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'warning_calls'
    return stypy_return_type_712090

# Assigning a type to the variable 'warning_calls' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'warning_calls', warning_calls)

@norecursion
def test_warning_calls_filters(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_warning_calls_filters'
    module_type_store = module_type_store.open_function_context('test_warning_calls_filters', 86, 0, False)
    
    # Passed parameters checking function
    test_warning_calls_filters.stypy_localization = localization
    test_warning_calls_filters.stypy_type_of_self = None
    test_warning_calls_filters.stypy_type_store = module_type_store
    test_warning_calls_filters.stypy_function_name = 'test_warning_calls_filters'
    test_warning_calls_filters.stypy_param_names_list = ['warning_calls']
    test_warning_calls_filters.stypy_varargs_param_name = None
    test_warning_calls_filters.stypy_kwargs_param_name = None
    test_warning_calls_filters.stypy_call_defaults = defaults
    test_warning_calls_filters.stypy_call_varargs = varargs
    test_warning_calls_filters.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_warning_calls_filters', ['warning_calls'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_warning_calls_filters', localization, ['warning_calls'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_warning_calls_filters(...)' code ##################

    
    # Assigning a Name to a Tuple (line 89):
    
    # Assigning a Subscript to a Name (line 89):
    
    # Obtaining the type of the subscript
    int_712091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 4), 'int')
    # Getting the type of 'warning_calls' (line 89)
    warning_calls_712092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 35), 'warning_calls')
    # Obtaining the member '__getitem__' of a type (line 89)
    getitem___712093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 4), warning_calls_712092, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
    subscript_call_result_712094 = invoke(stypy.reporting.localization.Localization(__file__, 89, 4), getitem___712093, int_712091)
    
    # Assigning a type to the variable 'tuple_var_assignment_711848' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'tuple_var_assignment_711848', subscript_call_result_712094)
    
    # Assigning a Subscript to a Name (line 89):
    
    # Obtaining the type of the subscript
    int_712095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 4), 'int')
    # Getting the type of 'warning_calls' (line 89)
    warning_calls_712096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 35), 'warning_calls')
    # Obtaining the member '__getitem__' of a type (line 89)
    getitem___712097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 4), warning_calls_712096, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
    subscript_call_result_712098 = invoke(stypy.reporting.localization.Localization(__file__, 89, 4), getitem___712097, int_712095)
    
    # Assigning a type to the variable 'tuple_var_assignment_711849' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'tuple_var_assignment_711849', subscript_call_result_712098)
    
    # Assigning a Name to a Name (line 89):
    # Getting the type of 'tuple_var_assignment_711848' (line 89)
    tuple_var_assignment_711848_712099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'tuple_var_assignment_711848')
    # Assigning a type to the variable 'bad_filters' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'bad_filters', tuple_var_assignment_711848_712099)
    
    # Assigning a Name to a Name (line 89):
    # Getting the type of 'tuple_var_assignment_711849' (line 89)
    tuple_var_assignment_711849_712100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'tuple_var_assignment_711849')
    # Assigning a type to the variable 'bad_stacklevels' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 17), 'bad_stacklevels', tuple_var_assignment_711849_712100)
    
    # Assigning a ListComp to a Name (line 93):
    
    # Assigning a ListComp to a Name (line 93):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'bad_filters' (line 93)
    bad_filters_712105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 36), 'bad_filters')
    comprehension_712106 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 19), bad_filters_712105)
    # Assigning a type to the variable 'item' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 19), 'item', comprehension_712106)
    
    str_712102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 22), 'str', 'optimize.py')
    # Getting the type of 'item' (line 94)
    item_712103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 43), 'item')
    # Applying the binary operator 'notin' (line 94)
    result_contains_712104 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 22), 'notin', str_712102, item_712103)
    
    # Getting the type of 'item' (line 93)
    item_712101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 19), 'item')
    list_712107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 19), list_712107, item_712101)
    # Assigning a type to the variable 'bad_filters' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'bad_filters', list_712107)
    
    # Getting the type of 'bad_filters' (line 96)
    bad_filters_712108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 7), 'bad_filters')
    # Testing the type of an if condition (line 96)
    if_condition_712109 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 4), bad_filters_712108)
    # Assigning a type to the variable 'if_condition_712109' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'if_condition_712109', if_condition_712109)
    # SSA begins for if statement (line 96)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to AssertionError(...): (line 97)
    # Processing the call arguments (line 97)
    
    # Call to format(...): (line 98)
    # Processing the call arguments (line 98)
    
    # Call to join(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'bad_filters' (line 101)
    bad_filters_712115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'bad_filters', False)
    # Processing the call keyword arguments (line 101)
    kwargs_712116 = {}
    str_712113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 16), 'str', '\n    ')
    # Obtaining the member 'join' of a type (line 101)
    join_712114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 16), str_712113, 'join')
    # Calling join(args, kwargs) (line 101)
    join_call_result_712117 = invoke(stypy.reporting.localization.Localization(__file__, 101, 16), join_712114, *[bad_filters_712115], **kwargs_712116)
    
    # Processing the call keyword arguments (line 98)
    kwargs_712118 = {}
    str_712111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 12), 'str', 'warning ignore filter should not be used, instead, use\nscipy._lib._numpy_compat.suppress_warnings (in tests only);\nfound in:\n    {}')
    # Obtaining the member 'format' of a type (line 98)
    format_712112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 12), str_712111, 'format')
    # Calling format(args, kwargs) (line 98)
    format_call_result_712119 = invoke(stypy.reporting.localization.Localization(__file__, 98, 12), format_712112, *[join_call_result_712117], **kwargs_712118)
    
    # Processing the call keyword arguments (line 97)
    kwargs_712120 = {}
    # Getting the type of 'AssertionError' (line 97)
    AssertionError_712110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 14), 'AssertionError', False)
    # Calling AssertionError(args, kwargs) (line 97)
    AssertionError_call_result_712121 = invoke(stypy.reporting.localization.Localization(__file__, 97, 14), AssertionError_712110, *[format_call_result_712119], **kwargs_712120)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 97, 8), AssertionError_call_result_712121, 'raise parameter', BaseException)
    # SSA join for if statement (line 96)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_warning_calls_filters(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_warning_calls_filters' in the type store
    # Getting the type of 'stypy_return_type' (line 86)
    stypy_return_type_712122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_712122)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_warning_calls_filters'
    return stypy_return_type_712122

# Assigning a type to the variable 'test_warning_calls_filters' (line 86)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'test_warning_calls_filters', test_warning_calls_filters)

@norecursion
def test_warning_calls_stacklevels(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_warning_calls_stacklevels'
    module_type_store = module_type_store.open_function_context('test_warning_calls_stacklevels', 104, 0, False)
    
    # Passed parameters checking function
    test_warning_calls_stacklevels.stypy_localization = localization
    test_warning_calls_stacklevels.stypy_type_of_self = None
    test_warning_calls_stacklevels.stypy_type_store = module_type_store
    test_warning_calls_stacklevels.stypy_function_name = 'test_warning_calls_stacklevels'
    test_warning_calls_stacklevels.stypy_param_names_list = ['warning_calls']
    test_warning_calls_stacklevels.stypy_varargs_param_name = None
    test_warning_calls_stacklevels.stypy_kwargs_param_name = None
    test_warning_calls_stacklevels.stypy_call_defaults = defaults
    test_warning_calls_stacklevels.stypy_call_varargs = varargs
    test_warning_calls_stacklevels.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_warning_calls_stacklevels', ['warning_calls'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_warning_calls_stacklevels', localization, ['warning_calls'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_warning_calls_stacklevels(...)' code ##################

    
    # Assigning a Name to a Tuple (line 108):
    
    # Assigning a Subscript to a Name (line 108):
    
    # Obtaining the type of the subscript
    int_712123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 4), 'int')
    # Getting the type of 'warning_calls' (line 108)
    warning_calls_712124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 35), 'warning_calls')
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___712125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 4), warning_calls_712124, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_712126 = invoke(stypy.reporting.localization.Localization(__file__, 108, 4), getitem___712125, int_712123)
    
    # Assigning a type to the variable 'tuple_var_assignment_711850' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'tuple_var_assignment_711850', subscript_call_result_712126)
    
    # Assigning a Subscript to a Name (line 108):
    
    # Obtaining the type of the subscript
    int_712127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 4), 'int')
    # Getting the type of 'warning_calls' (line 108)
    warning_calls_712128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 35), 'warning_calls')
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___712129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 4), warning_calls_712128, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_712130 = invoke(stypy.reporting.localization.Localization(__file__, 108, 4), getitem___712129, int_712127)
    
    # Assigning a type to the variable 'tuple_var_assignment_711851' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'tuple_var_assignment_711851', subscript_call_result_712130)
    
    # Assigning a Name to a Name (line 108):
    # Getting the type of 'tuple_var_assignment_711850' (line 108)
    tuple_var_assignment_711850_712131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'tuple_var_assignment_711850')
    # Assigning a type to the variable 'bad_filters' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'bad_filters', tuple_var_assignment_711850_712131)
    
    # Assigning a Name to a Name (line 108):
    # Getting the type of 'tuple_var_assignment_711851' (line 108)
    tuple_var_assignment_711851_712132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'tuple_var_assignment_711851')
    # Assigning a type to the variable 'bad_stacklevels' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 17), 'bad_stacklevels', tuple_var_assignment_711851_712132)
    
    # Assigning a Str to a Name (line 110):
    
    # Assigning a Str to a Name (line 110):
    str_712133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 10), 'str', '')
    # Assigning a type to the variable 'msg' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'msg', str_712133)
    
    # Getting the type of 'bad_filters' (line 112)
    bad_filters_712134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 7), 'bad_filters')
    # Testing the type of an if condition (line 112)
    if_condition_712135 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 4), bad_filters_712134)
    # Assigning a type to the variable 'if_condition_712135' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'if_condition_712135', if_condition_712135)
    # SSA begins for if statement (line 112)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'msg' (line 113)
    msg_712136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'msg')
    
    # Call to format(...): (line 113)
    # Processing the call arguments (line 113)
    
    # Call to join(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'bad_filters' (line 115)
    bad_filters_712141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 57), 'bad_filters', False)
    # Processing the call keyword arguments (line 115)
    kwargs_712142 = {}
    str_712139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 43), 'str', '\n    ')
    # Obtaining the member 'join' of a type (line 115)
    join_712140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 43), str_712139, 'join')
    # Calling join(args, kwargs) (line 115)
    join_call_result_712143 = invoke(stypy.reporting.localization.Localization(__file__, 115, 43), join_712140, *[bad_filters_712141], **kwargs_712142)
    
    # Processing the call keyword arguments (line 113)
    kwargs_712144 = {}
    str_712137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 16), 'str', 'warning ignore filter should not be used, instead, use\nscipy._lib._numpy_compat.suppress_warnings (in tests only);\nfound in:\n    {}')
    # Obtaining the member 'format' of a type (line 113)
    format_712138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 16), str_712137, 'format')
    # Calling format(args, kwargs) (line 113)
    format_call_result_712145 = invoke(stypy.reporting.localization.Localization(__file__, 113, 16), format_712138, *[join_call_result_712143], **kwargs_712144)
    
    # Applying the binary operator '+=' (line 113)
    result_iadd_712146 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 8), '+=', msg_712136, format_call_result_712145)
    # Assigning a type to the variable 'msg' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'msg', result_iadd_712146)
    
    
    # Getting the type of 'msg' (line 116)
    msg_712147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'msg')
    str_712148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 15), 'str', '\n\n')
    # Applying the binary operator '+=' (line 116)
    result_iadd_712149 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 8), '+=', msg_712147, str_712148)
    # Assigning a type to the variable 'msg' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'msg', result_iadd_712149)
    
    # SSA join for if statement (line 112)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'bad_stacklevels' (line 118)
    bad_stacklevels_712150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 7), 'bad_stacklevels')
    # Testing the type of an if condition (line 118)
    if_condition_712151 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 118, 4), bad_stacklevels_712150)
    # Assigning a type to the variable 'if_condition_712151' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'if_condition_712151', if_condition_712151)
    # SSA begins for if statement (line 118)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'msg' (line 119)
    msg_712152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'msg')
    
    # Call to format(...): (line 119)
    # Processing the call arguments (line 119)
    
    # Call to join(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'bad_stacklevels' (line 120)
    bad_stacklevels_712157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 30), 'bad_stacklevels', False)
    # Processing the call keyword arguments (line 120)
    kwargs_712158 = {}
    str_712155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 16), 'str', '\n    ')
    # Obtaining the member 'join' of a type (line 120)
    join_712156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 16), str_712155, 'join')
    # Calling join(args, kwargs) (line 120)
    join_call_result_712159 = invoke(stypy.reporting.localization.Localization(__file__, 120, 16), join_712156, *[bad_stacklevels_712157], **kwargs_712158)
    
    # Processing the call keyword arguments (line 119)
    kwargs_712160 = {}
    str_712153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 15), 'str', 'warnings should have an appropriate stacklevel:\n    {}')
    # Obtaining the member 'format' of a type (line 119)
    format_712154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 15), str_712153, 'format')
    # Calling format(args, kwargs) (line 119)
    format_call_result_712161 = invoke(stypy.reporting.localization.Localization(__file__, 119, 15), format_712154, *[join_call_result_712159], **kwargs_712160)
    
    # Applying the binary operator '+=' (line 119)
    result_iadd_712162 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 8), '+=', msg_712152, format_call_result_712161)
    # Assigning a type to the variable 'msg' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'msg', result_iadd_712162)
    
    # SSA join for if statement (line 118)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'msg' (line 122)
    msg_712163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 7), 'msg')
    # Testing the type of an if condition (line 122)
    if_condition_712164 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 4), msg_712163)
    # Assigning a type to the variable 'if_condition_712164' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'if_condition_712164', if_condition_712164)
    # SSA begins for if statement (line 122)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to AssertionError(...): (line 123)
    # Processing the call arguments (line 123)
    # Getting the type of 'msg' (line 123)
    msg_712166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 29), 'msg', False)
    # Processing the call keyword arguments (line 123)
    kwargs_712167 = {}
    # Getting the type of 'AssertionError' (line 123)
    AssertionError_712165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 14), 'AssertionError', False)
    # Calling AssertionError(args, kwargs) (line 123)
    AssertionError_call_result_712168 = invoke(stypy.reporting.localization.Localization(__file__, 123, 14), AssertionError_712165, *[msg_712166], **kwargs_712167)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 123, 8), AssertionError_call_result_712168, 'raise parameter', BaseException)
    # SSA join for if statement (line 122)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_warning_calls_stacklevels(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_warning_calls_stacklevels' in the type store
    # Getting the type of 'stypy_return_type' (line 104)
    stypy_return_type_712169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_712169)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_warning_calls_stacklevels'
    return stypy_return_type_712169

# Assigning a type to the variable 'test_warning_calls_stacklevels' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'test_warning_calls_stacklevels', test_warning_calls_stacklevels)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

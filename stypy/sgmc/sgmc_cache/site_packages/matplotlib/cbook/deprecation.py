
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import warnings
2: import functools
3: 
4: 
5: class MatplotlibDeprecationWarning(UserWarning):
6:     '''
7:     A class for issuing deprecation warnings for Matplotlib users.
8: 
9:     In light of the fact that Python builtin DeprecationWarnings are ignored
10:     by default as of Python 2.7 (see link below), this class was put in to
11:     allow for the signaling of deprecation, but via UserWarnings which are not
12:     ignored by default.
13: 
14:     https://docs.python.org/dev/whatsnew/2.7.html#the-future-for-python-2-x
15:     '''
16:     pass
17: 
18: 
19: mplDeprecation = MatplotlibDeprecationWarning
20: 
21: 
22: def _generate_deprecation_message(since, message='', name='',
23:                                   alternative='', pending=False,
24:                                   obj_type='attribute',
25:                                   addendum=''):
26: 
27:     if not message:
28: 
29:         if pending:
30:             message = (
31:                 'The %(name)s %(obj_type)s will be deprecated in a '
32:                 'future version.')
33:         else:
34:             message = (
35:                 'The %(name)s %(obj_type)s was deprecated in version '
36:                 '%(since)s.')
37: 
38:     altmessage = ''
39:     if alternative:
40:         altmessage = ' Use %s instead.' % alternative
41: 
42:     message = ((message % {
43:         'func': name,
44:         'name': name,
45:         'alternative': alternative,
46:         'obj_type': obj_type,
47:         'since': since}) +
48:         altmessage)
49: 
50:     if addendum:
51:         message += addendum
52: 
53:     return message
54: 
55: 
56: def warn_deprecated(
57:         since, message='', name='', alternative='', pending=False,
58:         obj_type='attribute', addendum=''):
59:     '''
60:     Used to display deprecation warning in a standard way.
61: 
62:     Parameters
63:     ----------
64:     since : str
65:         The release at which this API became deprecated.
66: 
67:     message : str, optional
68:         Override the default deprecation message.  The format
69:         specifier `%(name)s` may be used for the name of the function,
70:         and `%(alternative)s` may be used in the deprecation message
71:         to insert the name of an alternative to the deprecated
72:         function.  `%(obj_type)s` may be used to insert a friendly name
73:         for the type of object being deprecated.
74: 
75:     name : str, optional
76:         The name of the deprecated object.
77: 
78:     alternative : str, optional
79:         An alternative function that the user may use in place of the
80:         deprecated function.  The deprecation warning will tell the user
81:         about this alternative if provided.
82: 
83:     pending : bool, optional
84:         If True, uses a PendingDeprecationWarning instead of a
85:         DeprecationWarning.
86: 
87:     obj_type : str, optional
88:         The object type being deprecated.
89: 
90:     addendum : str, optional
91:         Additional text appended directly to the final message.
92: 
93:     Examples
94:     --------
95: 
96:         Basic example::
97: 
98:             # To warn of the deprecation of "matplotlib.name_of_module"
99:             warn_deprecated('1.4.0', name='matplotlib.name_of_module',
100:                             obj_type='module')
101: 
102:     '''
103:     message = _generate_deprecation_message(
104:                 since, message, name, alternative, pending, obj_type)
105: 
106:     warnings.warn(message, mplDeprecation, stacklevel=1)
107: 
108: 
109: def deprecated(since, message='', name='', alternative='', pending=False,
110:                obj_type=None, addendum=''):
111:     '''
112:     Decorator to mark a function or a class as deprecated.
113: 
114:     Parameters
115:     ----------
116:     since : str
117:         The release at which this API became deprecated.  This is
118:         required.
119: 
120:     message : str, optional
121:         Override the default deprecation message.  The format
122:         specifier `%(name)s` may be used for the name of the object,
123:         and `%(alternative)s` may be used in the deprecation message
124:         to insert the name of an alternative to the deprecated
125:         object.  `%(obj_type)s` may be used to insert a friendly name
126:         for the type of object being deprecated.
127: 
128:     name : str, optional
129:         The name of the deprecated object; if not provided the name
130:         is automatically determined from the passed in object,
131:         though this is useful in the case of renamed functions, where
132:         the new function is just assigned to the name of the
133:         deprecated function.  For example::
134: 
135:             def new_function():
136:                 ...
137:             oldFunction = new_function
138: 
139:     alternative : str, optional
140:         An alternative object that the user may use in place of the
141:         deprecated object.  The deprecation warning will tell the user
142:         about this alternative if provided.
143: 
144:     pending : bool, optional
145:         If True, uses a PendingDeprecationWarning instead of a
146:         DeprecationWarning.
147: 
148:     addendum : str, optional
149:         Additional text appended directly to the final message.
150: 
151:     Examples
152:     --------
153: 
154:         Basic example::
155: 
156:             @deprecated('1.4.0')
157:             def the_function_to_deprecate():
158:                 pass
159: 
160:     '''
161: 
162:     def deprecate(obj, message=message, name=name, alternative=alternative,
163:                   pending=pending, addendum=addendum):
164:         import textwrap
165: 
166:         if not name:
167:             name = obj.__name__
168: 
169:         if isinstance(obj, type):
170:             obj_type = "class"
171:             old_doc = obj.__doc__
172:             func = obj.__init__
173: 
174:             def finalize(wrapper, new_doc):
175:                 try:
176:                     obj.__doc__ = new_doc
177:                 except (AttributeError, TypeError):
178:                     # cls.__doc__ is not writeable on Py2.
179:                     # TypeError occurs on PyPy
180:                     pass
181:                 obj.__init__ = wrapper
182:                 return obj
183:         else:
184:             obj_type = "function"
185:             if isinstance(obj, classmethod):
186:                 func = obj.__func__
187:                 old_doc = func.__doc__
188: 
189:                 def finalize(wrapper, new_doc):
190:                     wrapper = functools.wraps(func)(wrapper)
191:                     wrapper.__doc__ = new_doc
192:                     return classmethod(wrapper)
193:             else:
194:                 func = obj
195:                 old_doc = func.__doc__
196: 
197:                 def finalize(wrapper, new_doc):
198:                     wrapper = functools.wraps(func)(wrapper)
199:                     wrapper.__doc__ = new_doc
200:                     return wrapper
201: 
202:         message = _generate_deprecation_message(
203:                     since, message, name, alternative, pending,
204:                     obj_type, addendum)
205: 
206:         def wrapper(*args, **kwargs):
207:             warnings.warn(message, mplDeprecation, stacklevel=2)
208:             return func(*args, **kwargs)
209: 
210:         old_doc = textwrap.dedent(old_doc or '').strip('\n')
211:         message = message.strip()
212:         new_doc = (('\n.. deprecated:: %(since)s'
213:                     '\n    %(message)s\n\n' %
214:                     {'since': since, 'message': message}) + old_doc)
215:         if not old_doc:
216:             # This is to prevent a spurious 'unexected unindent' warning from
217:             # docutils when the original docstring was blank.
218:             new_doc += r'\ '
219: 
220:         return finalize(wrapper, new_doc)
221: 
222:     return deprecate
223: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import warnings' statement (line 1)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import functools' statement (line 2)
import functools

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'functools', functools, module_type_store)

# Declaration of the 'MatplotlibDeprecationWarning' class
# Getting the type of 'UserWarning' (line 5)
UserWarning_273153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 35), 'UserWarning')

class MatplotlibDeprecationWarning(UserWarning_273153, ):
    str_273154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, (-1)), 'str', '\n    A class for issuing deprecation warnings for Matplotlib users.\n\n    In light of the fact that Python builtin DeprecationWarnings are ignored\n    by default as of Python 2.7 (see link below), this class was put in to\n    allow for the signaling of deprecation, but via UserWarnings which are not\n    ignored by default.\n\n    https://docs.python.org/dev/whatsnew/2.7.html#the-future-for-python-2-x\n    ')
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 5, 0, False)
        # Assigning a type to the variable 'self' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatplotlibDeprecationWarning.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'MatplotlibDeprecationWarning' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'MatplotlibDeprecationWarning', MatplotlibDeprecationWarning)

# Assigning a Name to a Name (line 19):
# Getting the type of 'MatplotlibDeprecationWarning' (line 19)
MatplotlibDeprecationWarning_273155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 17), 'MatplotlibDeprecationWarning')
# Assigning a type to the variable 'mplDeprecation' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'mplDeprecation', MatplotlibDeprecationWarning_273155)

@norecursion
def _generate_deprecation_message(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_273156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 49), 'str', '')
    str_273157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 58), 'str', '')
    str_273158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 46), 'str', '')
    # Getting the type of 'False' (line 23)
    False_273159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 58), 'False')
    str_273160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 43), 'str', 'attribute')
    str_273161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 43), 'str', '')
    defaults = [str_273156, str_273157, str_273158, False_273159, str_273160, str_273161]
    # Create a new context for function '_generate_deprecation_message'
    module_type_store = module_type_store.open_function_context('_generate_deprecation_message', 22, 0, False)
    
    # Passed parameters checking function
    _generate_deprecation_message.stypy_localization = localization
    _generate_deprecation_message.stypy_type_of_self = None
    _generate_deprecation_message.stypy_type_store = module_type_store
    _generate_deprecation_message.stypy_function_name = '_generate_deprecation_message'
    _generate_deprecation_message.stypy_param_names_list = ['since', 'message', 'name', 'alternative', 'pending', 'obj_type', 'addendum']
    _generate_deprecation_message.stypy_varargs_param_name = None
    _generate_deprecation_message.stypy_kwargs_param_name = None
    _generate_deprecation_message.stypy_call_defaults = defaults
    _generate_deprecation_message.stypy_call_varargs = varargs
    _generate_deprecation_message.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_generate_deprecation_message', ['since', 'message', 'name', 'alternative', 'pending', 'obj_type', 'addendum'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_generate_deprecation_message', localization, ['since', 'message', 'name', 'alternative', 'pending', 'obj_type', 'addendum'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_generate_deprecation_message(...)' code ##################

    
    
    # Getting the type of 'message' (line 27)
    message_273162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 11), 'message')
    # Applying the 'not' unary operator (line 27)
    result_not__273163 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 7), 'not', message_273162)
    
    # Testing the type of an if condition (line 27)
    if_condition_273164 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 4), result_not__273163)
    # Assigning a type to the variable 'if_condition_273164' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'if_condition_273164', if_condition_273164)
    # SSA begins for if statement (line 27)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'pending' (line 29)
    pending_273165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 11), 'pending')
    # Testing the type of an if condition (line 29)
    if_condition_273166 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 8), pending_273165)
    # Assigning a type to the variable 'if_condition_273166' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'if_condition_273166', if_condition_273166)
    # SSA begins for if statement (line 29)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 30):
    str_273167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 16), 'str', 'The %(name)s %(obj_type)s will be deprecated in a future version.')
    # Assigning a type to the variable 'message' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'message', str_273167)
    # SSA branch for the else part of an if statement (line 29)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 34):
    str_273168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 16), 'str', 'The %(name)s %(obj_type)s was deprecated in version %(since)s.')
    # Assigning a type to the variable 'message' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'message', str_273168)
    # SSA join for if statement (line 29)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 27)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Str to a Name (line 38):
    str_273169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 17), 'str', '')
    # Assigning a type to the variable 'altmessage' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'altmessage', str_273169)
    
    # Getting the type of 'alternative' (line 39)
    alternative_273170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 7), 'alternative')
    # Testing the type of an if condition (line 39)
    if_condition_273171 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 4), alternative_273170)
    # Assigning a type to the variable 'if_condition_273171' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'if_condition_273171', if_condition_273171)
    # SSA begins for if statement (line 39)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 40):
    str_273172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 21), 'str', ' Use %s instead.')
    # Getting the type of 'alternative' (line 40)
    alternative_273173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 42), 'alternative')
    # Applying the binary operator '%' (line 40)
    result_mod_273174 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 21), '%', str_273172, alternative_273173)
    
    # Assigning a type to the variable 'altmessage' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'altmessage', result_mod_273174)
    # SSA join for if statement (line 39)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 42):
    # Getting the type of 'message' (line 42)
    message_273175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 16), 'message')
    
    # Obtaining an instance of the builtin type 'dict' (line 42)
    dict_273176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 26), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 42)
    # Adding element type (key, value) (line 42)
    str_273177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 8), 'str', 'func')
    # Getting the type of 'name' (line 43)
    name_273178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 16), 'name')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 26), dict_273176, (str_273177, name_273178))
    # Adding element type (key, value) (line 42)
    str_273179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 8), 'str', 'name')
    # Getting the type of 'name' (line 44)
    name_273180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), 'name')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 26), dict_273176, (str_273179, name_273180))
    # Adding element type (key, value) (line 42)
    str_273181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 8), 'str', 'alternative')
    # Getting the type of 'alternative' (line 45)
    alternative_273182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 23), 'alternative')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 26), dict_273176, (str_273181, alternative_273182))
    # Adding element type (key, value) (line 42)
    str_273183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 8), 'str', 'obj_type')
    # Getting the type of 'obj_type' (line 46)
    obj_type_273184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'obj_type')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 26), dict_273176, (str_273183, obj_type_273184))
    # Adding element type (key, value) (line 42)
    str_273185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 8), 'str', 'since')
    # Getting the type of 'since' (line 47)
    since_273186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 17), 'since')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 26), dict_273176, (str_273185, since_273186))
    
    # Applying the binary operator '%' (line 42)
    result_mod_273187 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 16), '%', message_273175, dict_273176)
    
    # Getting the type of 'altmessage' (line 48)
    altmessage_273188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'altmessage')
    # Applying the binary operator '+' (line 42)
    result_add_273189 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 15), '+', result_mod_273187, altmessage_273188)
    
    # Assigning a type to the variable 'message' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'message', result_add_273189)
    
    # Getting the type of 'addendum' (line 50)
    addendum_273190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 7), 'addendum')
    # Testing the type of an if condition (line 50)
    if_condition_273191 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 4), addendum_273190)
    # Assigning a type to the variable 'if_condition_273191' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'if_condition_273191', if_condition_273191)
    # SSA begins for if statement (line 50)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'message' (line 51)
    message_273192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'message')
    # Getting the type of 'addendum' (line 51)
    addendum_273193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 19), 'addendum')
    # Applying the binary operator '+=' (line 51)
    result_iadd_273194 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 8), '+=', message_273192, addendum_273193)
    # Assigning a type to the variable 'message' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'message', result_iadd_273194)
    
    # SSA join for if statement (line 50)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'message' (line 53)
    message_273195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 11), 'message')
    # Assigning a type to the variable 'stypy_return_type' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type', message_273195)
    
    # ################# End of '_generate_deprecation_message(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_generate_deprecation_message' in the type store
    # Getting the type of 'stypy_return_type' (line 22)
    stypy_return_type_273196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_273196)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_generate_deprecation_message'
    return stypy_return_type_273196

# Assigning a type to the variable '_generate_deprecation_message' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), '_generate_deprecation_message', _generate_deprecation_message)

@norecursion
def warn_deprecated(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_273197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 23), 'str', '')
    str_273198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 32), 'str', '')
    str_273199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 48), 'str', '')
    # Getting the type of 'False' (line 57)
    False_273200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 60), 'False')
    str_273201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 17), 'str', 'attribute')
    str_273202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 39), 'str', '')
    defaults = [str_273197, str_273198, str_273199, False_273200, str_273201, str_273202]
    # Create a new context for function 'warn_deprecated'
    module_type_store = module_type_store.open_function_context('warn_deprecated', 56, 0, False)
    
    # Passed parameters checking function
    warn_deprecated.stypy_localization = localization
    warn_deprecated.stypy_type_of_self = None
    warn_deprecated.stypy_type_store = module_type_store
    warn_deprecated.stypy_function_name = 'warn_deprecated'
    warn_deprecated.stypy_param_names_list = ['since', 'message', 'name', 'alternative', 'pending', 'obj_type', 'addendum']
    warn_deprecated.stypy_varargs_param_name = None
    warn_deprecated.stypy_kwargs_param_name = None
    warn_deprecated.stypy_call_defaults = defaults
    warn_deprecated.stypy_call_varargs = varargs
    warn_deprecated.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'warn_deprecated', ['since', 'message', 'name', 'alternative', 'pending', 'obj_type', 'addendum'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'warn_deprecated', localization, ['since', 'message', 'name', 'alternative', 'pending', 'obj_type', 'addendum'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'warn_deprecated(...)' code ##################

    str_273203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, (-1)), 'str', '\n    Used to display deprecation warning in a standard way.\n\n    Parameters\n    ----------\n    since : str\n        The release at which this API became deprecated.\n\n    message : str, optional\n        Override the default deprecation message.  The format\n        specifier `%(name)s` may be used for the name of the function,\n        and `%(alternative)s` may be used in the deprecation message\n        to insert the name of an alternative to the deprecated\n        function.  `%(obj_type)s` may be used to insert a friendly name\n        for the type of object being deprecated.\n\n    name : str, optional\n        The name of the deprecated object.\n\n    alternative : str, optional\n        An alternative function that the user may use in place of the\n        deprecated function.  The deprecation warning will tell the user\n        about this alternative if provided.\n\n    pending : bool, optional\n        If True, uses a PendingDeprecationWarning instead of a\n        DeprecationWarning.\n\n    obj_type : str, optional\n        The object type being deprecated.\n\n    addendum : str, optional\n        Additional text appended directly to the final message.\n\n    Examples\n    --------\n\n        Basic example::\n\n            # To warn of the deprecation of "matplotlib.name_of_module"\n            warn_deprecated(\'1.4.0\', name=\'matplotlib.name_of_module\',\n                            obj_type=\'module\')\n\n    ')
    
    # Assigning a Call to a Name (line 103):
    
    # Call to _generate_deprecation_message(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'since' (line 104)
    since_273205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'since', False)
    # Getting the type of 'message' (line 104)
    message_273206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 23), 'message', False)
    # Getting the type of 'name' (line 104)
    name_273207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 32), 'name', False)
    # Getting the type of 'alternative' (line 104)
    alternative_273208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 38), 'alternative', False)
    # Getting the type of 'pending' (line 104)
    pending_273209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 51), 'pending', False)
    # Getting the type of 'obj_type' (line 104)
    obj_type_273210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 60), 'obj_type', False)
    # Processing the call keyword arguments (line 103)
    kwargs_273211 = {}
    # Getting the type of '_generate_deprecation_message' (line 103)
    _generate_deprecation_message_273204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 14), '_generate_deprecation_message', False)
    # Calling _generate_deprecation_message(args, kwargs) (line 103)
    _generate_deprecation_message_call_result_273212 = invoke(stypy.reporting.localization.Localization(__file__, 103, 14), _generate_deprecation_message_273204, *[since_273205, message_273206, name_273207, alternative_273208, pending_273209, obj_type_273210], **kwargs_273211)
    
    # Assigning a type to the variable 'message' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'message', _generate_deprecation_message_call_result_273212)
    
    # Call to warn(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'message' (line 106)
    message_273215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 18), 'message', False)
    # Getting the type of 'mplDeprecation' (line 106)
    mplDeprecation_273216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 27), 'mplDeprecation', False)
    # Processing the call keyword arguments (line 106)
    int_273217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 54), 'int')
    keyword_273218 = int_273217
    kwargs_273219 = {'stacklevel': keyword_273218}
    # Getting the type of 'warnings' (line 106)
    warnings_273213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 106)
    warn_273214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 4), warnings_273213, 'warn')
    # Calling warn(args, kwargs) (line 106)
    warn_call_result_273220 = invoke(stypy.reporting.localization.Localization(__file__, 106, 4), warn_273214, *[message_273215, mplDeprecation_273216], **kwargs_273219)
    
    
    # ################# End of 'warn_deprecated(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'warn_deprecated' in the type store
    # Getting the type of 'stypy_return_type' (line 56)
    stypy_return_type_273221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_273221)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'warn_deprecated'
    return stypy_return_type_273221

# Assigning a type to the variable 'warn_deprecated' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'warn_deprecated', warn_deprecated)

@norecursion
def deprecated(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_273222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 30), 'str', '')
    str_273223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 39), 'str', '')
    str_273224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 55), 'str', '')
    # Getting the type of 'False' (line 109)
    False_273225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 67), 'False')
    # Getting the type of 'None' (line 110)
    None_273226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 24), 'None')
    str_273227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 39), 'str', '')
    defaults = [str_273222, str_273223, str_273224, False_273225, None_273226, str_273227]
    # Create a new context for function 'deprecated'
    module_type_store = module_type_store.open_function_context('deprecated', 109, 0, False)
    
    # Passed parameters checking function
    deprecated.stypy_localization = localization
    deprecated.stypy_type_of_self = None
    deprecated.stypy_type_store = module_type_store
    deprecated.stypy_function_name = 'deprecated'
    deprecated.stypy_param_names_list = ['since', 'message', 'name', 'alternative', 'pending', 'obj_type', 'addendum']
    deprecated.stypy_varargs_param_name = None
    deprecated.stypy_kwargs_param_name = None
    deprecated.stypy_call_defaults = defaults
    deprecated.stypy_call_varargs = varargs
    deprecated.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'deprecated', ['since', 'message', 'name', 'alternative', 'pending', 'obj_type', 'addendum'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'deprecated', localization, ['since', 'message', 'name', 'alternative', 'pending', 'obj_type', 'addendum'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'deprecated(...)' code ##################

    str_273228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, (-1)), 'str', "\n    Decorator to mark a function or a class as deprecated.\n\n    Parameters\n    ----------\n    since : str\n        The release at which this API became deprecated.  This is\n        required.\n\n    message : str, optional\n        Override the default deprecation message.  The format\n        specifier `%(name)s` may be used for the name of the object,\n        and `%(alternative)s` may be used in the deprecation message\n        to insert the name of an alternative to the deprecated\n        object.  `%(obj_type)s` may be used to insert a friendly name\n        for the type of object being deprecated.\n\n    name : str, optional\n        The name of the deprecated object; if not provided the name\n        is automatically determined from the passed in object,\n        though this is useful in the case of renamed functions, where\n        the new function is just assigned to the name of the\n        deprecated function.  For example::\n\n            def new_function():\n                ...\n            oldFunction = new_function\n\n    alternative : str, optional\n        An alternative object that the user may use in place of the\n        deprecated object.  The deprecation warning will tell the user\n        about this alternative if provided.\n\n    pending : bool, optional\n        If True, uses a PendingDeprecationWarning instead of a\n        DeprecationWarning.\n\n    addendum : str, optional\n        Additional text appended directly to the final message.\n\n    Examples\n    --------\n\n        Basic example::\n\n            @deprecated('1.4.0')\n            def the_function_to_deprecate():\n                pass\n\n    ")

    @norecursion
    def deprecate(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'message' (line 162)
        message_273229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 31), 'message')
        # Getting the type of 'name' (line 162)
        name_273230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 45), 'name')
        # Getting the type of 'alternative' (line 162)
        alternative_273231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 63), 'alternative')
        # Getting the type of 'pending' (line 163)
        pending_273232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 26), 'pending')
        # Getting the type of 'addendum' (line 163)
        addendum_273233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 44), 'addendum')
        defaults = [message_273229, name_273230, alternative_273231, pending_273232, addendum_273233]
        # Create a new context for function 'deprecate'
        module_type_store = module_type_store.open_function_context('deprecate', 162, 4, False)
        
        # Passed parameters checking function
        deprecate.stypy_localization = localization
        deprecate.stypy_type_of_self = None
        deprecate.stypy_type_store = module_type_store
        deprecate.stypy_function_name = 'deprecate'
        deprecate.stypy_param_names_list = ['obj', 'message', 'name', 'alternative', 'pending', 'addendum']
        deprecate.stypy_varargs_param_name = None
        deprecate.stypy_kwargs_param_name = None
        deprecate.stypy_call_defaults = defaults
        deprecate.stypy_call_varargs = varargs
        deprecate.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'deprecate', ['obj', 'message', 'name', 'alternative', 'pending', 'addendum'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'deprecate', localization, ['obj', 'message', 'name', 'alternative', 'pending', 'addendum'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'deprecate(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 164, 8))
        
        # 'import textwrap' statement (line 164)
        import textwrap

        import_module(stypy.reporting.localization.Localization(__file__, 164, 8), 'textwrap', textwrap, module_type_store)
        
        
        
        # Getting the type of 'name' (line 166)
        name_273234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 15), 'name')
        # Applying the 'not' unary operator (line 166)
        result_not__273235 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 11), 'not', name_273234)
        
        # Testing the type of an if condition (line 166)
        if_condition_273236 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 166, 8), result_not__273235)
        # Assigning a type to the variable 'if_condition_273236' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'if_condition_273236', if_condition_273236)
        # SSA begins for if statement (line 166)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 167):
        # Getting the type of 'obj' (line 167)
        obj_273237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 19), 'obj')
        # Obtaining the member '__name__' of a type (line 167)
        name___273238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 19), obj_273237, '__name__')
        # Assigning a type to the variable 'name' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'name', name___273238)
        # SSA join for if statement (line 166)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 169)
        # Getting the type of 'type' (line 169)
        type_273239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 27), 'type')
        # Getting the type of 'obj' (line 169)
        obj_273240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 22), 'obj')
        
        (may_be_273241, more_types_in_union_273242) = may_be_subtype(type_273239, obj_273240)

        if may_be_273241:

            if more_types_in_union_273242:
                # Runtime conditional SSA (line 169)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'obj' (line 169)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'obj', remove_not_subtype_from_union(obj_273240, type))
            
            # Assigning a Str to a Name (line 170):
            str_273243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 23), 'str', 'class')
            # Assigning a type to the variable 'obj_type' (line 170)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'obj_type', str_273243)
            
            # Assigning a Attribute to a Name (line 171):
            # Getting the type of 'obj' (line 171)
            obj_273244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 22), 'obj')
            # Obtaining the member '__doc__' of a type (line 171)
            doc___273245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 22), obj_273244, '__doc__')
            # Assigning a type to the variable 'old_doc' (line 171)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'old_doc', doc___273245)
            
            # Assigning a Attribute to a Name (line 172):
            # Getting the type of 'obj' (line 172)
            obj_273246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 19), 'obj')
            # Obtaining the member '__init__' of a type (line 172)
            init___273247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 19), obj_273246, '__init__')
            # Assigning a type to the variable 'func' (line 172)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'func', init___273247)

            @norecursion
            def finalize(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'finalize'
                module_type_store = module_type_store.open_function_context('finalize', 174, 12, False)
                
                # Passed parameters checking function
                finalize.stypy_localization = localization
                finalize.stypy_type_of_self = None
                finalize.stypy_type_store = module_type_store
                finalize.stypy_function_name = 'finalize'
                finalize.stypy_param_names_list = ['wrapper', 'new_doc']
                finalize.stypy_varargs_param_name = None
                finalize.stypy_kwargs_param_name = None
                finalize.stypy_call_defaults = defaults
                finalize.stypy_call_varargs = varargs
                finalize.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, 'finalize', ['wrapper', 'new_doc'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'finalize', localization, ['wrapper', 'new_doc'], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'finalize(...)' code ##################

                
                
                # SSA begins for try-except statement (line 175)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
                
                # Assigning a Name to a Attribute (line 176):
                # Getting the type of 'new_doc' (line 176)
                new_doc_273248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 34), 'new_doc')
                # Getting the type of 'obj' (line 176)
                obj_273249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 20), 'obj')
                # Setting the type of the member '__doc__' of a type (line 176)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 20), obj_273249, '__doc__', new_doc_273248)
                # SSA branch for the except part of a try statement (line 175)
                # SSA branch for the except 'Tuple' branch of a try statement (line 175)
                module_type_store.open_ssa_branch('except')
                pass
                # SSA join for try-except statement (line 175)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # Assigning a Name to a Attribute (line 181):
                # Getting the type of 'wrapper' (line 181)
                wrapper_273250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 31), 'wrapper')
                # Getting the type of 'obj' (line 181)
                obj_273251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 'obj')
                # Setting the type of the member '__init__' of a type (line 181)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 16), obj_273251, '__init__', wrapper_273250)
                # Getting the type of 'obj' (line 182)
                obj_273252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 23), 'obj')
                # Assigning a type to the variable 'stypy_return_type' (line 182)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 16), 'stypy_return_type', obj_273252)
                
                # ################# End of 'finalize(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'finalize' in the type store
                # Getting the type of 'stypy_return_type' (line 174)
                stypy_return_type_273253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_273253)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'finalize'
                return stypy_return_type_273253

            # Assigning a type to the variable 'finalize' (line 174)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'finalize', finalize)

            if more_types_in_union_273242:
                # Runtime conditional SSA for else branch (line 169)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_273241) or more_types_in_union_273242):
            # Assigning a type to the variable 'obj' (line 169)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'obj', remove_subtype_from_union(obj_273240, type))
            
            # Assigning a Str to a Name (line 184):
            str_273254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 23), 'str', 'function')
            # Assigning a type to the variable 'obj_type' (line 184)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'obj_type', str_273254)
            
            # Type idiom detected: calculating its left and rigth part (line 185)
            # Getting the type of 'classmethod' (line 185)
            classmethod_273255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 31), 'classmethod')
            # Getting the type of 'obj' (line 185)
            obj_273256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 26), 'obj')
            
            (may_be_273257, more_types_in_union_273258) = may_be_subtype(classmethod_273255, obj_273256)

            if may_be_273257:

                if more_types_in_union_273258:
                    # Runtime conditional SSA (line 185)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'obj' (line 185)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'obj', remove_not_subtype_from_union(obj_273256, classmethod))
                
                # Assigning a Attribute to a Name (line 186):
                # Getting the type of 'obj' (line 186)
                obj_273259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 23), 'obj')
                # Obtaining the member '__func__' of a type (line 186)
                func___273260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 23), obj_273259, '__func__')
                # Assigning a type to the variable 'func' (line 186)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 16), 'func', func___273260)
                
                # Assigning a Attribute to a Name (line 187):
                # Getting the type of 'func' (line 187)
                func_273261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 26), 'func')
                # Obtaining the member '__doc__' of a type (line 187)
                doc___273262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 26), func_273261, '__doc__')
                # Assigning a type to the variable 'old_doc' (line 187)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 16), 'old_doc', doc___273262)

                @norecursion
                def finalize(localization, *varargs, **kwargs):
                    global module_type_store
                    # Assign values to the parameters with defaults
                    defaults = []
                    # Create a new context for function 'finalize'
                    module_type_store = module_type_store.open_function_context('finalize', 189, 16, False)
                    
                    # Passed parameters checking function
                    finalize.stypy_localization = localization
                    finalize.stypy_type_of_self = None
                    finalize.stypy_type_store = module_type_store
                    finalize.stypy_function_name = 'finalize'
                    finalize.stypy_param_names_list = ['wrapper', 'new_doc']
                    finalize.stypy_varargs_param_name = None
                    finalize.stypy_kwargs_param_name = None
                    finalize.stypy_call_defaults = defaults
                    finalize.stypy_call_varargs = varargs
                    finalize.stypy_call_kwargs = kwargs
                    arguments = process_argument_values(localization, None, module_type_store, 'finalize', ['wrapper', 'new_doc'], None, None, defaults, varargs, kwargs)

                    if is_error_type(arguments):
                        # Destroy the current context
                        module_type_store = module_type_store.close_function_context()
                        return arguments

                    # Initialize method data
                    init_call_information(module_type_store, 'finalize', localization, ['wrapper', 'new_doc'], arguments)
                    
                    # Default return type storage variable (SSA)
                    # Assigning a type to the variable 'stypy_return_type'
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                    
                    
                    # ################# Begin of 'finalize(...)' code ##################

                    
                    # Assigning a Call to a Name (line 190):
                    
                    # Call to (...): (line 190)
                    # Processing the call arguments (line 190)
                    # Getting the type of 'wrapper' (line 190)
                    wrapper_273268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 52), 'wrapper', False)
                    # Processing the call keyword arguments (line 190)
                    kwargs_273269 = {}
                    
                    # Call to wraps(...): (line 190)
                    # Processing the call arguments (line 190)
                    # Getting the type of 'func' (line 190)
                    func_273265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 46), 'func', False)
                    # Processing the call keyword arguments (line 190)
                    kwargs_273266 = {}
                    # Getting the type of 'functools' (line 190)
                    functools_273263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 30), 'functools', False)
                    # Obtaining the member 'wraps' of a type (line 190)
                    wraps_273264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 30), functools_273263, 'wraps')
                    # Calling wraps(args, kwargs) (line 190)
                    wraps_call_result_273267 = invoke(stypy.reporting.localization.Localization(__file__, 190, 30), wraps_273264, *[func_273265], **kwargs_273266)
                    
                    # Calling (args, kwargs) (line 190)
                    _call_result_273270 = invoke(stypy.reporting.localization.Localization(__file__, 190, 30), wraps_call_result_273267, *[wrapper_273268], **kwargs_273269)
                    
                    # Assigning a type to the variable 'wrapper' (line 190)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 20), 'wrapper', _call_result_273270)
                    
                    # Assigning a Name to a Attribute (line 191):
                    # Getting the type of 'new_doc' (line 191)
                    new_doc_273271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 38), 'new_doc')
                    # Getting the type of 'wrapper' (line 191)
                    wrapper_273272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 20), 'wrapper')
                    # Setting the type of the member '__doc__' of a type (line 191)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 20), wrapper_273272, '__doc__', new_doc_273271)
                    
                    # Call to classmethod(...): (line 192)
                    # Processing the call arguments (line 192)
                    # Getting the type of 'wrapper' (line 192)
                    wrapper_273274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 39), 'wrapper', False)
                    # Processing the call keyword arguments (line 192)
                    kwargs_273275 = {}
                    # Getting the type of 'classmethod' (line 192)
                    classmethod_273273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 27), 'classmethod', False)
                    # Calling classmethod(args, kwargs) (line 192)
                    classmethod_call_result_273276 = invoke(stypy.reporting.localization.Localization(__file__, 192, 27), classmethod_273273, *[wrapper_273274], **kwargs_273275)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 192)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 20), 'stypy_return_type', classmethod_call_result_273276)
                    
                    # ################# End of 'finalize(...)' code ##################

                    # Teardown call information
                    teardown_call_information(localization, arguments)
                    
                    # Storing the return type of function 'finalize' in the type store
                    # Getting the type of 'stypy_return_type' (line 189)
                    stypy_return_type_273277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 16), 'stypy_return_type')
                    module_type_store.store_return_type_of_current_context(stypy_return_type_273277)
                    
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    
                    # Return type of the function 'finalize'
                    return stypy_return_type_273277

                # Assigning a type to the variable 'finalize' (line 189)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 16), 'finalize', finalize)

                if more_types_in_union_273258:
                    # Runtime conditional SSA for else branch (line 185)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_273257) or more_types_in_union_273258):
                # Assigning a type to the variable 'obj' (line 185)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'obj', remove_subtype_from_union(obj_273256, classmethod))
                
                # Assigning a Name to a Name (line 194):
                # Getting the type of 'obj' (line 194)
                obj_273278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 23), 'obj')
                # Assigning a type to the variable 'func' (line 194)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 16), 'func', obj_273278)
                
                # Assigning a Attribute to a Name (line 195):
                # Getting the type of 'func' (line 195)
                func_273279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 26), 'func')
                # Obtaining the member '__doc__' of a type (line 195)
                doc___273280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 26), func_273279, '__doc__')
                # Assigning a type to the variable 'old_doc' (line 195)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 16), 'old_doc', doc___273280)

                @norecursion
                def finalize(localization, *varargs, **kwargs):
                    global module_type_store
                    # Assign values to the parameters with defaults
                    defaults = []
                    # Create a new context for function 'finalize'
                    module_type_store = module_type_store.open_function_context('finalize', 197, 16, False)
                    
                    # Passed parameters checking function
                    finalize.stypy_localization = localization
                    finalize.stypy_type_of_self = None
                    finalize.stypy_type_store = module_type_store
                    finalize.stypy_function_name = 'finalize'
                    finalize.stypy_param_names_list = ['wrapper', 'new_doc']
                    finalize.stypy_varargs_param_name = None
                    finalize.stypy_kwargs_param_name = None
                    finalize.stypy_call_defaults = defaults
                    finalize.stypy_call_varargs = varargs
                    finalize.stypy_call_kwargs = kwargs
                    arguments = process_argument_values(localization, None, module_type_store, 'finalize', ['wrapper', 'new_doc'], None, None, defaults, varargs, kwargs)

                    if is_error_type(arguments):
                        # Destroy the current context
                        module_type_store = module_type_store.close_function_context()
                        return arguments

                    # Initialize method data
                    init_call_information(module_type_store, 'finalize', localization, ['wrapper', 'new_doc'], arguments)
                    
                    # Default return type storage variable (SSA)
                    # Assigning a type to the variable 'stypy_return_type'
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                    
                    
                    # ################# Begin of 'finalize(...)' code ##################

                    
                    # Assigning a Call to a Name (line 198):
                    
                    # Call to (...): (line 198)
                    # Processing the call arguments (line 198)
                    # Getting the type of 'wrapper' (line 198)
                    wrapper_273286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 52), 'wrapper', False)
                    # Processing the call keyword arguments (line 198)
                    kwargs_273287 = {}
                    
                    # Call to wraps(...): (line 198)
                    # Processing the call arguments (line 198)
                    # Getting the type of 'func' (line 198)
                    func_273283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 46), 'func', False)
                    # Processing the call keyword arguments (line 198)
                    kwargs_273284 = {}
                    # Getting the type of 'functools' (line 198)
                    functools_273281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 30), 'functools', False)
                    # Obtaining the member 'wraps' of a type (line 198)
                    wraps_273282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 30), functools_273281, 'wraps')
                    # Calling wraps(args, kwargs) (line 198)
                    wraps_call_result_273285 = invoke(stypy.reporting.localization.Localization(__file__, 198, 30), wraps_273282, *[func_273283], **kwargs_273284)
                    
                    # Calling (args, kwargs) (line 198)
                    _call_result_273288 = invoke(stypy.reporting.localization.Localization(__file__, 198, 30), wraps_call_result_273285, *[wrapper_273286], **kwargs_273287)
                    
                    # Assigning a type to the variable 'wrapper' (line 198)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 20), 'wrapper', _call_result_273288)
                    
                    # Assigning a Name to a Attribute (line 199):
                    # Getting the type of 'new_doc' (line 199)
                    new_doc_273289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 38), 'new_doc')
                    # Getting the type of 'wrapper' (line 199)
                    wrapper_273290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'wrapper')
                    # Setting the type of the member '__doc__' of a type (line 199)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 20), wrapper_273290, '__doc__', new_doc_273289)
                    # Getting the type of 'wrapper' (line 200)
                    wrapper_273291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 27), 'wrapper')
                    # Assigning a type to the variable 'stypy_return_type' (line 200)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 20), 'stypy_return_type', wrapper_273291)
                    
                    # ################# End of 'finalize(...)' code ##################

                    # Teardown call information
                    teardown_call_information(localization, arguments)
                    
                    # Storing the return type of function 'finalize' in the type store
                    # Getting the type of 'stypy_return_type' (line 197)
                    stypy_return_type_273292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 16), 'stypy_return_type')
                    module_type_store.store_return_type_of_current_context(stypy_return_type_273292)
                    
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    
                    # Return type of the function 'finalize'
                    return stypy_return_type_273292

                # Assigning a type to the variable 'finalize' (line 197)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 16), 'finalize', finalize)

                if (may_be_273257 and more_types_in_union_273258):
                    # SSA join for if statement (line 185)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_273241 and more_types_in_union_273242):
                # SSA join for if statement (line 169)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 202):
        
        # Call to _generate_deprecation_message(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'since' (line 203)
        since_273294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 20), 'since', False)
        # Getting the type of 'message' (line 203)
        message_273295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 27), 'message', False)
        # Getting the type of 'name' (line 203)
        name_273296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 36), 'name', False)
        # Getting the type of 'alternative' (line 203)
        alternative_273297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 42), 'alternative', False)
        # Getting the type of 'pending' (line 203)
        pending_273298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 55), 'pending', False)
        # Getting the type of 'obj_type' (line 204)
        obj_type_273299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 20), 'obj_type', False)
        # Getting the type of 'addendum' (line 204)
        addendum_273300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 30), 'addendum', False)
        # Processing the call keyword arguments (line 202)
        kwargs_273301 = {}
        # Getting the type of '_generate_deprecation_message' (line 202)
        _generate_deprecation_message_273293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 18), '_generate_deprecation_message', False)
        # Calling _generate_deprecation_message(args, kwargs) (line 202)
        _generate_deprecation_message_call_result_273302 = invoke(stypy.reporting.localization.Localization(__file__, 202, 18), _generate_deprecation_message_273293, *[since_273294, message_273295, name_273296, alternative_273297, pending_273298, obj_type_273299, addendum_273300], **kwargs_273301)
        
        # Assigning a type to the variable 'message' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'message', _generate_deprecation_message_call_result_273302)

        @norecursion
        def wrapper(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'wrapper'
            module_type_store = module_type_store.open_function_context('wrapper', 206, 8, False)
            
            # Passed parameters checking function
            wrapper.stypy_localization = localization
            wrapper.stypy_type_of_self = None
            wrapper.stypy_type_store = module_type_store
            wrapper.stypy_function_name = 'wrapper'
            wrapper.stypy_param_names_list = []
            wrapper.stypy_varargs_param_name = 'args'
            wrapper.stypy_kwargs_param_name = 'kwargs'
            wrapper.stypy_call_defaults = defaults
            wrapper.stypy_call_varargs = varargs
            wrapper.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'wrapper', [], 'args', 'kwargs', defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'wrapper', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'wrapper(...)' code ##################

            
            # Call to warn(...): (line 207)
            # Processing the call arguments (line 207)
            # Getting the type of 'message' (line 207)
            message_273305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 26), 'message', False)
            # Getting the type of 'mplDeprecation' (line 207)
            mplDeprecation_273306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 35), 'mplDeprecation', False)
            # Processing the call keyword arguments (line 207)
            int_273307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 62), 'int')
            keyword_273308 = int_273307
            kwargs_273309 = {'stacklevel': keyword_273308}
            # Getting the type of 'warnings' (line 207)
            warnings_273303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'warnings', False)
            # Obtaining the member 'warn' of a type (line 207)
            warn_273304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 12), warnings_273303, 'warn')
            # Calling warn(args, kwargs) (line 207)
            warn_call_result_273310 = invoke(stypy.reporting.localization.Localization(__file__, 207, 12), warn_273304, *[message_273305, mplDeprecation_273306], **kwargs_273309)
            
            
            # Call to func(...): (line 208)
            # Getting the type of 'args' (line 208)
            args_273312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 25), 'args', False)
            # Processing the call keyword arguments (line 208)
            # Getting the type of 'kwargs' (line 208)
            kwargs_273313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 33), 'kwargs', False)
            kwargs_273314 = {'kwargs_273313': kwargs_273313}
            # Getting the type of 'func' (line 208)
            func_273311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 19), 'func', False)
            # Calling func(args, kwargs) (line 208)
            func_call_result_273315 = invoke(stypy.reporting.localization.Localization(__file__, 208, 19), func_273311, *[args_273312], **kwargs_273314)
            
            # Assigning a type to the variable 'stypy_return_type' (line 208)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'stypy_return_type', func_call_result_273315)
            
            # ################# End of 'wrapper(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'wrapper' in the type store
            # Getting the type of 'stypy_return_type' (line 206)
            stypy_return_type_273316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_273316)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'wrapper'
            return stypy_return_type_273316

        # Assigning a type to the variable 'wrapper' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'wrapper', wrapper)
        
        # Assigning a Call to a Name (line 210):
        
        # Call to strip(...): (line 210)
        # Processing the call arguments (line 210)
        str_273325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 55), 'str', '\n')
        # Processing the call keyword arguments (line 210)
        kwargs_273326 = {}
        
        # Call to dedent(...): (line 210)
        # Processing the call arguments (line 210)
        
        # Evaluating a boolean operation
        # Getting the type of 'old_doc' (line 210)
        old_doc_273319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 34), 'old_doc', False)
        str_273320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 45), 'str', '')
        # Applying the binary operator 'or' (line 210)
        result_or_keyword_273321 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 34), 'or', old_doc_273319, str_273320)
        
        # Processing the call keyword arguments (line 210)
        kwargs_273322 = {}
        # Getting the type of 'textwrap' (line 210)
        textwrap_273317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 18), 'textwrap', False)
        # Obtaining the member 'dedent' of a type (line 210)
        dedent_273318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 18), textwrap_273317, 'dedent')
        # Calling dedent(args, kwargs) (line 210)
        dedent_call_result_273323 = invoke(stypy.reporting.localization.Localization(__file__, 210, 18), dedent_273318, *[result_or_keyword_273321], **kwargs_273322)
        
        # Obtaining the member 'strip' of a type (line 210)
        strip_273324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 18), dedent_call_result_273323, 'strip')
        # Calling strip(args, kwargs) (line 210)
        strip_call_result_273327 = invoke(stypy.reporting.localization.Localization(__file__, 210, 18), strip_273324, *[str_273325], **kwargs_273326)
        
        # Assigning a type to the variable 'old_doc' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'old_doc', strip_call_result_273327)
        
        # Assigning a Call to a Name (line 211):
        
        # Call to strip(...): (line 211)
        # Processing the call keyword arguments (line 211)
        kwargs_273330 = {}
        # Getting the type of 'message' (line 211)
        message_273328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 18), 'message', False)
        # Obtaining the member 'strip' of a type (line 211)
        strip_273329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 18), message_273328, 'strip')
        # Calling strip(args, kwargs) (line 211)
        strip_call_result_273331 = invoke(stypy.reporting.localization.Localization(__file__, 211, 18), strip_273329, *[], **kwargs_273330)
        
        # Assigning a type to the variable 'message' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'message', strip_call_result_273331)
        
        # Assigning a BinOp to a Name (line 212):
        str_273332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 20), 'str', '\n.. deprecated:: %(since)s\n    %(message)s\n\n')
        
        # Obtaining an instance of the builtin type 'dict' (line 214)
        dict_273333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 20), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 214)
        # Adding element type (key, value) (line 214)
        str_273334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 21), 'str', 'since')
        # Getting the type of 'since' (line 214)
        since_273335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 30), 'since')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 20), dict_273333, (str_273334, since_273335))
        # Adding element type (key, value) (line 214)
        str_273336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 37), 'str', 'message')
        # Getting the type of 'message' (line 214)
        message_273337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 48), 'message')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 20), dict_273333, (str_273336, message_273337))
        
        # Applying the binary operator '%' (line 212)
        result_mod_273338 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 20), '%', str_273332, dict_273333)
        
        # Getting the type of 'old_doc' (line 214)
        old_doc_273339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 60), 'old_doc')
        # Applying the binary operator '+' (line 212)
        result_add_273340 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 19), '+', result_mod_273338, old_doc_273339)
        
        # Assigning a type to the variable 'new_doc' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'new_doc', result_add_273340)
        
        
        # Getting the type of 'old_doc' (line 215)
        old_doc_273341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 15), 'old_doc')
        # Applying the 'not' unary operator (line 215)
        result_not__273342 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 11), 'not', old_doc_273341)
        
        # Testing the type of an if condition (line 215)
        if_condition_273343 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 8), result_not__273342)
        # Assigning a type to the variable 'if_condition_273343' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'if_condition_273343', if_condition_273343)
        # SSA begins for if statement (line 215)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'new_doc' (line 218)
        new_doc_273344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'new_doc')
        str_273345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 23), 'str', '\\ ')
        # Applying the binary operator '+=' (line 218)
        result_iadd_273346 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 12), '+=', new_doc_273344, str_273345)
        # Assigning a type to the variable 'new_doc' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'new_doc', result_iadd_273346)
        
        # SSA join for if statement (line 215)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to finalize(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'wrapper' (line 220)
        wrapper_273348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 24), 'wrapper', False)
        # Getting the type of 'new_doc' (line 220)
        new_doc_273349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 33), 'new_doc', False)
        # Processing the call keyword arguments (line 220)
        kwargs_273350 = {}
        # Getting the type of 'finalize' (line 220)
        finalize_273347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 15), 'finalize', False)
        # Calling finalize(args, kwargs) (line 220)
        finalize_call_result_273351 = invoke(stypy.reporting.localization.Localization(__file__, 220, 15), finalize_273347, *[wrapper_273348, new_doc_273349], **kwargs_273350)
        
        # Assigning a type to the variable 'stypy_return_type' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'stypy_return_type', finalize_call_result_273351)
        
        # ################# End of 'deprecate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'deprecate' in the type store
        # Getting the type of 'stypy_return_type' (line 162)
        stypy_return_type_273352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_273352)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'deprecate'
        return stypy_return_type_273352

    # Assigning a type to the variable 'deprecate' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'deprecate', deprecate)
    # Getting the type of 'deprecate' (line 222)
    deprecate_273353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 11), 'deprecate')
    # Assigning a type to the variable 'stypy_return_type' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'stypy_return_type', deprecate_273353)
    
    # ################# End of 'deprecated(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'deprecated' in the type store
    # Getting the type of 'stypy_return_type' (line 109)
    stypy_return_type_273354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_273354)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'deprecated'
    return stypy_return_type_273354

# Assigning a type to the variable 'deprecated' (line 109)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'deprecated', deprecated)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # #########################     LICENSE     ############################ #
2: 
3: # Copyright (c) 2005-2015, Michele Simionato
4: # All rights reserved.
5: 
6: # Redistribution and use in source and binary forms, with or without
7: # modification, are permitted provided that the following conditions are
8: # met:
9: 
10: #   Redistributions of source code must retain the above copyright
11: #   notice, this list of conditions and the following disclaimer.
12: #   Redistributions in bytecode form must reproduce the above copyright
13: #   notice, this list of conditions and the following disclaimer in
14: #   the documentation and/or other materials provided with the
15: #   distribution.
16: 
17: # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
18: # "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
19: # LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
20: # A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
21: # HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
22: # INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
23: # BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
24: # OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
25: # ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
26: # TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
27: # USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
28: # DAMAGE.
29: 
30: '''
31: Decorator module, see http://pypi.python.org/pypi/decorator
32: for the documentation.
33: '''
34: from __future__ import print_function
35: 
36: import re
37: import sys
38: import inspect
39: import operator
40: import itertools
41: import collections
42: 
43: __version__ = '4.0.5'
44: 
45: if sys.version >= '3':
46:     from inspect import getfullargspec
47: 
48:     def get_init(cls):
49:         return cls.__init__
50: else:
51:     class getfullargspec(object):
52:         "A quick and dirty replacement for getfullargspec for Python 2.X"
53:         def __init__(self, f):
54:             self.args, self.varargs, self.varkw, self.defaults = \
55:                 inspect.getargspec(f)
56:             self.kwonlyargs = []
57:             self.kwonlydefaults = None
58: 
59:         def __iter__(self):
60:             yield self.args
61:             yield self.varargs
62:             yield self.varkw
63:             yield self.defaults
64: 
65:         getargspec = inspect.getargspec
66: 
67:     def get_init(cls):
68:         return cls.__init__.__func__
69: 
70: # getargspec has been deprecated in Python 3.5
71: ArgSpec = collections.namedtuple(
72:     'ArgSpec', 'args varargs varkw defaults')
73: 
74: 
75: def getargspec(f):
76:     '''A replacement for inspect.getargspec'''
77:     spec = getfullargspec(f)
78:     return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)
79: 
80: DEF = re.compile(r'\s*def\s*([_\w][_\w\d]*)\s*\(')
81: 
82: 
83: # basic functionality
84: class FunctionMaker(object):
85:     '''
86:     An object with the ability to create functions with a given signature.
87:     It has attributes name, doc, module, signature, defaults, dict and
88:     methods update and make.
89:     '''
90: 
91:     # Atomic get-and-increment provided by the GIL
92:     _compile_count = itertools.count()
93: 
94:     def __init__(self, func=None, name=None, signature=None,
95:                  defaults=None, doc=None, module=None, funcdict=None):
96:         self.shortsignature = signature
97:         if func:
98:             # func can be a class or a callable, but not an instance method
99:             self.name = func.__name__
100:             if self.name == '<lambda>':  # small hack for lambda functions
101:                 self.name = '_lambda_'
102:             self.doc = func.__doc__
103:             self.module = func.__module__
104:             if inspect.isfunction(func):
105:                 argspec = getfullargspec(func)
106:                 self.annotations = getattr(func, '__annotations__', {})
107:                 for a in ('args', 'varargs', 'varkw', 'defaults', 'kwonlyargs',
108:                           'kwonlydefaults'):
109:                     setattr(self, a, getattr(argspec, a))
110:                 for i, arg in enumerate(self.args):
111:                     setattr(self, 'arg%d' % i, arg)
112:                 if sys.version < '3':  # easy way
113:                     self.shortsignature = self.signature = (
114:                         inspect.formatargspec(
115:                             formatvalue=lambda val: "", *argspec)[1:-1])
116:                 else:  # Python 3 way
117:                     allargs = list(self.args)
118:                     allshortargs = list(self.args)
119:                     if self.varargs:
120:                         allargs.append('*' + self.varargs)
121:                         allshortargs.append('*' + self.varargs)
122:                     elif self.kwonlyargs:
123:                         allargs.append('*')  # single star syntax
124:                     for a in self.kwonlyargs:
125:                         allargs.append('%s=None' % a)
126:                         allshortargs.append('%s=%s' % (a, a))
127:                     if self.varkw:
128:                         allargs.append('**' + self.varkw)
129:                         allshortargs.append('**' + self.varkw)
130:                     self.signature = ', '.join(allargs)
131:                     self.shortsignature = ', '.join(allshortargs)
132:                 self.dict = func.__dict__.copy()
133:         # func=None happens when decorating a caller
134:         if name:
135:             self.name = name
136:         if signature is not None:
137:             self.signature = signature
138:         if defaults:
139:             self.defaults = defaults
140:         if doc:
141:             self.doc = doc
142:         if module:
143:             self.module = module
144:         if funcdict:
145:             self.dict = funcdict
146:         # check existence required attributes
147:         assert hasattr(self, 'name')
148:         if not hasattr(self, 'signature'):
149:             raise TypeError('You are decorating a non function: %s' % func)
150: 
151:     def update(self, func, **kw):
152:         "Update the signature of func with the data in self"
153:         func.__name__ = self.name
154:         func.__doc__ = getattr(self, 'doc', None)
155:         func.__dict__ = getattr(self, 'dict', {})
156:         func.__defaults__ = getattr(self, 'defaults', ())
157:         func.__kwdefaults__ = getattr(self, 'kwonlydefaults', None)
158:         func.__annotations__ = getattr(self, 'annotations', None)
159:         try:
160:             frame = sys._getframe(3)
161:         except AttributeError:  # for IronPython and similar implementations
162:             callermodule = '?'
163:         else:
164:             callermodule = frame.f_globals.get('__name__', '?')
165:         func.__module__ = getattr(self, 'module', callermodule)
166:         func.__dict__.update(kw)
167: 
168:     def make(self, src_templ, evaldict=None, addsource=False, **attrs):
169:         "Make a new function from a given template and update the signature"
170:         src = src_templ % vars(self)  # expand name and signature
171:         evaldict = evaldict or {}
172:         mo = DEF.match(src)
173:         if mo is None:
174:             raise SyntaxError('not a valid function template\n%s' % src)
175:         name = mo.group(1)  # extract the function name
176:         names = set([name] + [arg.strip(' *') for arg in
177:                               self.shortsignature.split(',')])
178:         for n in names:
179:             if n in ('_func_', '_call_'):
180:                 raise NameError('%s is overridden in\n%s' % (n, src))
181:         if not src.endswith('\n'):  # add a newline just for safety
182:             src += '\n'  # this is needed in old versions of Python
183: 
184:         # Ensure each generated function has a unique filename for profilers
185:         # (such as cProfile) that depend on the tuple of (<filename>,
186:         # <definition line>, <function name>) being unique.
187:         filename = '<decorator-gen-%d>' % (next(self._compile_count),)
188:         try:
189:             code = compile(src, filename, 'single')
190:             exec(code, evaldict)
191:         except:
192:             print('Error in generated code:', file=sys.stderr)
193:             print(src, file=sys.stderr)
194:             raise
195:         func = evaldict[name]
196:         if addsource:
197:             attrs['__source__'] = src
198:         self.update(func, **attrs)
199:         return func
200: 
201:     @classmethod
202:     def create(cls, obj, body, evaldict, defaults=None,
203:                doc=None, module=None, addsource=True, **attrs):
204:         '''
205:         Create a function from the strings name, signature and body.
206:         evaldict is the evaluation dictionary. If addsource is true an
207:         attribute __source__ is added to the result. The attributes attrs
208:         are added, if any.
209:         '''
210:         if isinstance(obj, str):  # "name(signature)"
211:             name, rest = obj.strip().split('(', 1)
212:             signature = rest[:-1]  # strip a right parens
213:             func = None
214:         else:  # a function
215:             name = None
216:             signature = None
217:             func = obj
218:         self = cls(func, name, signature, defaults, doc, module)
219:         ibody = '\n'.join('    ' + line for line in body.splitlines())
220:         return self.make('def %(name)s(%(signature)s):\n' + ibody,
221:                          evaldict, addsource, **attrs)
222: 
223: 
224: def decorate(func, caller):
225:     '''
226:     decorate(func, caller) decorates a function using a caller.
227:     '''
228:     evaldict = func.__globals__.copy()
229:     evaldict['_call_'] = caller
230:     evaldict['_func_'] = func
231:     fun = FunctionMaker.create(
232:         func, "return _call_(_func_, %(shortsignature)s)",
233:         evaldict, __wrapped__=func)
234:     if hasattr(func, '__qualname__'):
235:         fun.__qualname__ = func.__qualname__
236:     return fun
237: 
238: 
239: def decorator(caller, _func=None):
240:     '''decorator(caller) converts a caller function into a decorator'''
241:     if _func is not None:  # return a decorated function
242:         # this is obsolete behavior; you should use decorate instead
243:         return decorate(_func, caller)
244:     # else return a decorator function
245:     if inspect.isclass(caller):
246:         name = caller.__name__.lower()
247:         callerfunc = get_init(caller)
248:         doc = 'decorator(%s) converts functions/generators into ' \
249:             'factories of %s objects' % (caller.__name__, caller.__name__)
250:     elif inspect.isfunction(caller):
251:         if caller.__name__ == '<lambda>':
252:             name = '_lambda_'
253:         else:
254:             name = caller.__name__
255:         callerfunc = caller
256:         doc = caller.__doc__
257:     else:  # assume caller is an object with a __call__ method
258:         name = caller.__class__.__name__.lower()
259:         callerfunc = caller.__call__.__func__
260:         doc = caller.__call__.__doc__
261:     evaldict = callerfunc.__globals__.copy()
262:     evaldict['_call_'] = caller
263:     evaldict['_decorate_'] = decorate
264:     return FunctionMaker.create(
265:         '%s(func)' % name, 'return _decorate_(func, _call_)',
266:         evaldict, doc=doc, module=caller.__module__,
267:         __wrapped__=caller)
268: 
269: 
270: # ####################### contextmanager ####################### #
271: 
272: try:  # Python >= 3.2
273:     from contextlib import _GeneratorContextManager
274: except ImportError:  # Python >= 2.5
275:     from contextlib import GeneratorContextManager as _GeneratorContextManager
276: 
277: 
278: class ContextManager(_GeneratorContextManager):
279:     def __call__(self, func):
280:         '''Context manager decorator'''
281:         return FunctionMaker.create(
282:             func, "with _self_: return _func_(%(shortsignature)s)",
283:             dict(_self_=self, _func_=func), __wrapped__=func)
284: 
285: init = getfullargspec(_GeneratorContextManager.__init__)
286: n_args = len(init.args)
287: if n_args == 2 and not init.varargs:  # (self, genobj) Python 2.7
288:     def __init__(self, g, *a, **k):
289:         return _GeneratorContextManager.__init__(self, g(*a, **k))
290:     ContextManager.__init__ = __init__
291: elif n_args == 2 and init.varargs:  # (self, gen, *a, **k) Python 3.4
292:     pass
293: elif n_args == 4:  # (self, gen, args, kwds) Python 3.5
294:     def __init__(self, g, *a, **k):
295:         return _GeneratorContextManager.__init__(self, g, a, k)
296:     ContextManager.__init__ = __init__
297: 
298: contextmanager = decorator(ContextManager)
299: 
300: 
301: # ############################ dispatch_on ############################ #
302: 
303: def append(a, vancestors):
304:     '''
305:     Append ``a`` to the list of the virtual ancestors, unless it is already
306:     included.
307:     '''
308:     add = True
309:     for j, va in enumerate(vancestors):
310:         if issubclass(va, a):
311:             add = False
312:             break
313:         if issubclass(a, va):
314:             vancestors[j] = a
315:             add = False
316:     if add:
317:         vancestors.append(a)
318: 
319: 
320: # inspired from simplegeneric by P.J. Eby and functools.singledispatch
321: def dispatch_on(*dispatch_args):
322:     '''
323:     Factory of decorators turning a function into a generic function
324:     dispatching on the given arguments.
325:     '''
326:     assert dispatch_args, 'No dispatch args passed'
327:     dispatch_str = '(%s,)' % ', '.join(dispatch_args)
328: 
329:     def check(arguments, wrong=operator.ne, msg=''):
330:         '''Make sure one passes the expected number of arguments'''
331:         if wrong(len(arguments), len(dispatch_args)):
332:             raise TypeError('Expected %d arguments, got %d%s' %
333:                             (len(dispatch_args), len(arguments), msg))
334: 
335:     def gen_func_dec(func):
336:         '''Decorator turning a function into a generic function'''
337: 
338:         # first check the dispatch arguments
339:         argset = set(getfullargspec(func).args)
340:         if not set(dispatch_args) <= argset:
341:             raise NameError('Unknown dispatch arguments %s' % dispatch_str)
342: 
343:         typemap = {}
344: 
345:         def vancestors(*types):
346:             '''
347:             Get a list of sets of virtual ancestors for the given types
348:             '''
349:             check(types)
350:             ras = [[] for _ in range(len(dispatch_args))]
351:             for types_ in typemap:
352:                 for t, type_, ra in zip(types, types_, ras):
353:                     if issubclass(t, type_) and type_ not in t.__mro__:
354:                         append(type_, ra)
355:             return [set(ra) for ra in ras]
356: 
357:         def ancestors(*types):
358:             '''
359:             Get a list of virtual MROs, one for each type
360:             '''
361:             check(types)
362:             lists = []
363:             for t, vas in zip(types, vancestors(*types)):
364:                 n_vas = len(vas)
365:                 if n_vas > 1:
366:                     raise RuntimeError(
367:                         'Ambiguous dispatch for %s: %s' % (t, vas))
368:                 elif n_vas == 1:
369:                     va, = vas
370:                     mro = type('t', (t, va), {}).__mro__[1:]
371:                 else:
372:                     mro = t.__mro__
373:                 lists.append(mro[:-1])  # discard t and object
374:             return lists
375: 
376:         def register(*types):
377:             '''
378:             Decorator to register an implementation for the given types
379:             '''
380:             check(types)
381: 
382:             def dec(f):
383:                 check(getfullargspec(f).args, operator.lt, ' in ' + f.__name__)
384:                 typemap[types] = f
385:                 return f
386:             return dec
387: 
388:         def dispatch_info(*types):
389:             '''
390:             An utility to introspect the dispatch algorithm
391:             '''
392:             check(types)
393:             lst = []
394:             for anc in itertools.product(*ancestors(*types)):
395:                 lst.append(tuple(a.__name__ for a in anc))
396:             return lst
397: 
398:         def _dispatch(dispatch_args, *args, **kw):
399:             types = tuple(type(arg) for arg in dispatch_args)
400:             try:  # fast path
401:                 f = typemap[types]
402:             except KeyError:
403:                 pass
404:             else:
405:                 return f(*args, **kw)
406:             combinations = itertools.product(*ancestors(*types))
407:             next(combinations)  # the first one has been already tried
408:             for types_ in combinations:
409:                 f = typemap.get(types_)
410:                 if f is not None:
411:                     return f(*args, **kw)
412: 
413:             # else call the default implementation
414:             return func(*args, **kw)
415: 
416:         return FunctionMaker.create(
417:             func, 'return _f_(%s, %%(shortsignature)s)' % dispatch_str,
418:             dict(_f_=_dispatch), register=register, default=func,
419:             typemap=typemap, vancestors=vancestors, ancestors=ancestors,
420:             dispatch_info=dispatch_info, __wrapped__=func)
421: 
422:     gen_func_dec.__name__ = 'dispatch_on' + dispatch_str
423:     return gen_func_dec
424: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_706367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, (-1)), 'str', '\nDecorator module, see http://pypi.python.org/pypi/decorator\nfor the documentation.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 36, 0))

# 'import re' statement (line 36)
import re

import_module(stypy.reporting.localization.Localization(__file__, 36, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 37, 0))

# 'import sys' statement (line 37)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 38, 0))

# 'import inspect' statement (line 38)
import inspect

import_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'inspect', inspect, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 39, 0))

# 'import operator' statement (line 39)
import operator

import_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'operator', operator, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 40, 0))

# 'import itertools' statement (line 40)
import itertools

import_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'itertools', itertools, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 41, 0))

# 'import collections' statement (line 41)
import collections

import_module(stypy.reporting.localization.Localization(__file__, 41, 0), 'collections', collections, module_type_store)


# Assigning a Str to a Name (line 43):

# Assigning a Str to a Name (line 43):
str_706368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 14), 'str', '4.0.5')
# Assigning a type to the variable '__version__' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), '__version__', str_706368)


# Getting the type of 'sys' (line 45)
sys_706369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 3), 'sys')
# Obtaining the member 'version' of a type (line 45)
version_706370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 3), sys_706369, 'version')
str_706371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 18), 'str', '3')
# Applying the binary operator '>=' (line 45)
result_ge_706372 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 3), '>=', version_706370, str_706371)

# Testing the type of an if condition (line 45)
if_condition_706373 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 0), result_ge_706372)
# Assigning a type to the variable 'if_condition_706373' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'if_condition_706373', if_condition_706373)
# SSA begins for if statement (line 45)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 46, 4))

# 'from inspect import getfullargspec' statement (line 46)
try:
    from inspect import getfullargspec

except:
    getfullargspec = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 46, 4), 'inspect', None, module_type_store, ['getfullargspec'], [getfullargspec])


@norecursion
def get_init(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_init'
    module_type_store = module_type_store.open_function_context('get_init', 48, 4, False)
    
    # Passed parameters checking function
    get_init.stypy_localization = localization
    get_init.stypy_type_of_self = None
    get_init.stypy_type_store = module_type_store
    get_init.stypy_function_name = 'get_init'
    get_init.stypy_param_names_list = ['cls']
    get_init.stypy_varargs_param_name = None
    get_init.stypy_kwargs_param_name = None
    get_init.stypy_call_defaults = defaults
    get_init.stypy_call_varargs = varargs
    get_init.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_init', ['cls'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_init', localization, ['cls'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_init(...)' code ##################

    # Getting the type of 'cls' (line 49)
    cls_706374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'cls')
    # Obtaining the member '__init__' of a type (line 49)
    init___706375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 15), cls_706374, '__init__')
    # Assigning a type to the variable 'stypy_return_type' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'stypy_return_type', init___706375)
    
    # ################# End of 'get_init(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_init' in the type store
    # Getting the type of 'stypy_return_type' (line 48)
    stypy_return_type_706376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_706376)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_init'
    return stypy_return_type_706376

# Assigning a type to the variable 'get_init' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'get_init', get_init)
# SSA branch for the else part of an if statement (line 45)
module_type_store.open_ssa_branch('else')
# Declaration of the 'getfullargspec' class

class getfullargspec(object, ):
    str_706377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 8), 'str', 'A quick and dirty replacement for getfullargspec for Python 2.X')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 53, 8, False)
        # Assigning a type to the variable 'self' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'getfullargspec.__init__', ['f'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['f'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Tuple (line 54):
        
        # Assigning a Subscript to a Name (line 54):
        
        # Obtaining the type of the subscript
        int_706378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 12), 'int')
        
        # Call to getargspec(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'f' (line 55)
        f_706381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 35), 'f', False)
        # Processing the call keyword arguments (line 55)
        kwargs_706382 = {}
        # Getting the type of 'inspect' (line 55)
        inspect_706379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'inspect', False)
        # Obtaining the member 'getargspec' of a type (line 55)
        getargspec_706380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 16), inspect_706379, 'getargspec')
        # Calling getargspec(args, kwargs) (line 55)
        getargspec_call_result_706383 = invoke(stypy.reporting.localization.Localization(__file__, 55, 16), getargspec_706380, *[f_706381], **kwargs_706382)
        
        # Obtaining the member '__getitem__' of a type (line 54)
        getitem___706384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), getargspec_call_result_706383, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 54)
        subscript_call_result_706385 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), getitem___706384, int_706378)
        
        # Assigning a type to the variable 'tuple_var_assignment_706360' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'tuple_var_assignment_706360', subscript_call_result_706385)
        
        # Assigning a Subscript to a Name (line 54):
        
        # Obtaining the type of the subscript
        int_706386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 12), 'int')
        
        # Call to getargspec(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'f' (line 55)
        f_706389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 35), 'f', False)
        # Processing the call keyword arguments (line 55)
        kwargs_706390 = {}
        # Getting the type of 'inspect' (line 55)
        inspect_706387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'inspect', False)
        # Obtaining the member 'getargspec' of a type (line 55)
        getargspec_706388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 16), inspect_706387, 'getargspec')
        # Calling getargspec(args, kwargs) (line 55)
        getargspec_call_result_706391 = invoke(stypy.reporting.localization.Localization(__file__, 55, 16), getargspec_706388, *[f_706389], **kwargs_706390)
        
        # Obtaining the member '__getitem__' of a type (line 54)
        getitem___706392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), getargspec_call_result_706391, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 54)
        subscript_call_result_706393 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), getitem___706392, int_706386)
        
        # Assigning a type to the variable 'tuple_var_assignment_706361' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'tuple_var_assignment_706361', subscript_call_result_706393)
        
        # Assigning a Subscript to a Name (line 54):
        
        # Obtaining the type of the subscript
        int_706394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 12), 'int')
        
        # Call to getargspec(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'f' (line 55)
        f_706397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 35), 'f', False)
        # Processing the call keyword arguments (line 55)
        kwargs_706398 = {}
        # Getting the type of 'inspect' (line 55)
        inspect_706395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'inspect', False)
        # Obtaining the member 'getargspec' of a type (line 55)
        getargspec_706396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 16), inspect_706395, 'getargspec')
        # Calling getargspec(args, kwargs) (line 55)
        getargspec_call_result_706399 = invoke(stypy.reporting.localization.Localization(__file__, 55, 16), getargspec_706396, *[f_706397], **kwargs_706398)
        
        # Obtaining the member '__getitem__' of a type (line 54)
        getitem___706400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), getargspec_call_result_706399, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 54)
        subscript_call_result_706401 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), getitem___706400, int_706394)
        
        # Assigning a type to the variable 'tuple_var_assignment_706362' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'tuple_var_assignment_706362', subscript_call_result_706401)
        
        # Assigning a Subscript to a Name (line 54):
        
        # Obtaining the type of the subscript
        int_706402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 12), 'int')
        
        # Call to getargspec(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'f' (line 55)
        f_706405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 35), 'f', False)
        # Processing the call keyword arguments (line 55)
        kwargs_706406 = {}
        # Getting the type of 'inspect' (line 55)
        inspect_706403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'inspect', False)
        # Obtaining the member 'getargspec' of a type (line 55)
        getargspec_706404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 16), inspect_706403, 'getargspec')
        # Calling getargspec(args, kwargs) (line 55)
        getargspec_call_result_706407 = invoke(stypy.reporting.localization.Localization(__file__, 55, 16), getargspec_706404, *[f_706405], **kwargs_706406)
        
        # Obtaining the member '__getitem__' of a type (line 54)
        getitem___706408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), getargspec_call_result_706407, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 54)
        subscript_call_result_706409 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), getitem___706408, int_706402)
        
        # Assigning a type to the variable 'tuple_var_assignment_706363' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'tuple_var_assignment_706363', subscript_call_result_706409)
        
        # Assigning a Name to a Attribute (line 54):
        # Getting the type of 'tuple_var_assignment_706360' (line 54)
        tuple_var_assignment_706360_706410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'tuple_var_assignment_706360')
        # Getting the type of 'self' (line 54)
        self_706411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'self')
        # Setting the type of the member 'args' of a type (line 54)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), self_706411, 'args', tuple_var_assignment_706360_706410)
        
        # Assigning a Name to a Attribute (line 54):
        # Getting the type of 'tuple_var_assignment_706361' (line 54)
        tuple_var_assignment_706361_706412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'tuple_var_assignment_706361')
        # Getting the type of 'self' (line 54)
        self_706413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 23), 'self')
        # Setting the type of the member 'varargs' of a type (line 54)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 23), self_706413, 'varargs', tuple_var_assignment_706361_706412)
        
        # Assigning a Name to a Attribute (line 54):
        # Getting the type of 'tuple_var_assignment_706362' (line 54)
        tuple_var_assignment_706362_706414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'tuple_var_assignment_706362')
        # Getting the type of 'self' (line 54)
        self_706415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 37), 'self')
        # Setting the type of the member 'varkw' of a type (line 54)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 37), self_706415, 'varkw', tuple_var_assignment_706362_706414)
        
        # Assigning a Name to a Attribute (line 54):
        # Getting the type of 'tuple_var_assignment_706363' (line 54)
        tuple_var_assignment_706363_706416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'tuple_var_assignment_706363')
        # Getting the type of 'self' (line 54)
        self_706417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 49), 'self')
        # Setting the type of the member 'defaults' of a type (line 54)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 49), self_706417, 'defaults', tuple_var_assignment_706363_706416)
        
        # Assigning a List to a Attribute (line 56):
        
        # Assigning a List to a Attribute (line 56):
        
        # Obtaining an instance of the builtin type 'list' (line 56)
        list_706418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 56)
        
        # Getting the type of 'self' (line 56)
        self_706419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'self')
        # Setting the type of the member 'kwonlyargs' of a type (line 56)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 12), self_706419, 'kwonlyargs', list_706418)
        
        # Assigning a Name to a Attribute (line 57):
        
        # Assigning a Name to a Attribute (line 57):
        # Getting the type of 'None' (line 57)
        None_706420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 34), 'None')
        # Getting the type of 'self' (line 57)
        self_706421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'self')
        # Setting the type of the member 'kwonlydefaults' of a type (line 57)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), self_706421, 'kwonlydefaults', None_706420)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __iter__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__iter__'
        module_type_store = module_type_store.open_function_context('__iter__', 59, 8, False)
        # Assigning a type to the variable 'self' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        getfullargspec.__iter__.__dict__.__setitem__('stypy_localization', localization)
        getfullargspec.__iter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        getfullargspec.__iter__.__dict__.__setitem__('stypy_type_store', module_type_store)
        getfullargspec.__iter__.__dict__.__setitem__('stypy_function_name', 'getfullargspec.__iter__')
        getfullargspec.__iter__.__dict__.__setitem__('stypy_param_names_list', [])
        getfullargspec.__iter__.__dict__.__setitem__('stypy_varargs_param_name', None)
        getfullargspec.__iter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        getfullargspec.__iter__.__dict__.__setitem__('stypy_call_defaults', defaults)
        getfullargspec.__iter__.__dict__.__setitem__('stypy_call_varargs', varargs)
        getfullargspec.__iter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        getfullargspec.__iter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'getfullargspec.__iter__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__iter__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__iter__(...)' code ##################

        # Creating a generator
        # Getting the type of 'self' (line 60)
        self_706422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 18), 'self')
        # Obtaining the member 'args' of a type (line 60)
        args_706423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 18), self_706422, 'args')
        GeneratorType_706424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 12), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 12), GeneratorType_706424, args_706423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'stypy_return_type', GeneratorType_706424)
        # Creating a generator
        # Getting the type of 'self' (line 61)
        self_706425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 18), 'self')
        # Obtaining the member 'varargs' of a type (line 61)
        varargs_706426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 18), self_706425, 'varargs')
        GeneratorType_706427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 12), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 12), GeneratorType_706427, varargs_706426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'stypy_return_type', GeneratorType_706427)
        # Creating a generator
        # Getting the type of 'self' (line 62)
        self_706428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 18), 'self')
        # Obtaining the member 'varkw' of a type (line 62)
        varkw_706429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 18), self_706428, 'varkw')
        GeneratorType_706430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 12), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 12), GeneratorType_706430, varkw_706429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'stypy_return_type', GeneratorType_706430)
        # Creating a generator
        # Getting the type of 'self' (line 63)
        self_706431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 18), 'self')
        # Obtaining the member 'defaults' of a type (line 63)
        defaults_706432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 18), self_706431, 'defaults')
        GeneratorType_706433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 12), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 12), GeneratorType_706433, defaults_706432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'stypy_return_type', GeneratorType_706433)
        
        # ################# End of '__iter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__iter__' in the type store
        # Getting the type of 'stypy_return_type' (line 59)
        stypy_return_type_706434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_706434)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__iter__'
        return stypy_return_type_706434

    
    # Assigning a Attribute to a Name (line 65):

# Assigning a type to the variable 'getfullargspec' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'getfullargspec', getfullargspec)

# Assigning a Attribute to a Name (line 65):
# Getting the type of 'inspect' (line 65)
inspect_706435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 21), 'inspect')
# Obtaining the member 'getargspec' of a type (line 65)
getargspec_706436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 21), inspect_706435, 'getargspec')
# Getting the type of 'getfullargspec'
getfullargspec_706437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'getfullargspec')
# Setting the type of the member 'getargspec' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), getfullargspec_706437, 'getargspec', getargspec_706436)

@norecursion
def get_init(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_init'
    module_type_store = module_type_store.open_function_context('get_init', 67, 4, False)
    
    # Passed parameters checking function
    get_init.stypy_localization = localization
    get_init.stypy_type_of_self = None
    get_init.stypy_type_store = module_type_store
    get_init.stypy_function_name = 'get_init'
    get_init.stypy_param_names_list = ['cls']
    get_init.stypy_varargs_param_name = None
    get_init.stypy_kwargs_param_name = None
    get_init.stypy_call_defaults = defaults
    get_init.stypy_call_varargs = varargs
    get_init.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_init', ['cls'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_init', localization, ['cls'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_init(...)' code ##################

    # Getting the type of 'cls' (line 68)
    cls_706438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 15), 'cls')
    # Obtaining the member '__init__' of a type (line 68)
    init___706439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 15), cls_706438, '__init__')
    # Obtaining the member '__func__' of a type (line 68)
    func___706440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 15), init___706439, '__func__')
    # Assigning a type to the variable 'stypy_return_type' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'stypy_return_type', func___706440)
    
    # ################# End of 'get_init(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_init' in the type store
    # Getting the type of 'stypy_return_type' (line 67)
    stypy_return_type_706441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_706441)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_init'
    return stypy_return_type_706441

# Assigning a type to the variable 'get_init' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'get_init', get_init)
# SSA join for if statement (line 45)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Call to a Name (line 71):

# Assigning a Call to a Name (line 71):

# Call to namedtuple(...): (line 71)
# Processing the call arguments (line 71)
str_706444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 4), 'str', 'ArgSpec')
str_706445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 15), 'str', 'args varargs varkw defaults')
# Processing the call keyword arguments (line 71)
kwargs_706446 = {}
# Getting the type of 'collections' (line 71)
collections_706442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 10), 'collections', False)
# Obtaining the member 'namedtuple' of a type (line 71)
namedtuple_706443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 10), collections_706442, 'namedtuple')
# Calling namedtuple(args, kwargs) (line 71)
namedtuple_call_result_706447 = invoke(stypy.reporting.localization.Localization(__file__, 71, 10), namedtuple_706443, *[str_706444, str_706445], **kwargs_706446)

# Assigning a type to the variable 'ArgSpec' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'ArgSpec', namedtuple_call_result_706447)

@norecursion
def getargspec(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'getargspec'
    module_type_store = module_type_store.open_function_context('getargspec', 75, 0, False)
    
    # Passed parameters checking function
    getargspec.stypy_localization = localization
    getargspec.stypy_type_of_self = None
    getargspec.stypy_type_store = module_type_store
    getargspec.stypy_function_name = 'getargspec'
    getargspec.stypy_param_names_list = ['f']
    getargspec.stypy_varargs_param_name = None
    getargspec.stypy_kwargs_param_name = None
    getargspec.stypy_call_defaults = defaults
    getargspec.stypy_call_varargs = varargs
    getargspec.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getargspec', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getargspec', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getargspec(...)' code ##################

    str_706448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 4), 'str', 'A replacement for inspect.getargspec')
    
    # Assigning a Call to a Name (line 77):
    
    # Assigning a Call to a Name (line 77):
    
    # Call to getfullargspec(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'f' (line 77)
    f_706450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 26), 'f', False)
    # Processing the call keyword arguments (line 77)
    kwargs_706451 = {}
    # Getting the type of 'getfullargspec' (line 77)
    getfullargspec_706449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), 'getfullargspec', False)
    # Calling getfullargspec(args, kwargs) (line 77)
    getfullargspec_call_result_706452 = invoke(stypy.reporting.localization.Localization(__file__, 77, 11), getfullargspec_706449, *[f_706450], **kwargs_706451)
    
    # Assigning a type to the variable 'spec' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'spec', getfullargspec_call_result_706452)
    
    # Call to ArgSpec(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'spec' (line 78)
    spec_706454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 19), 'spec', False)
    # Obtaining the member 'args' of a type (line 78)
    args_706455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 19), spec_706454, 'args')
    # Getting the type of 'spec' (line 78)
    spec_706456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 30), 'spec', False)
    # Obtaining the member 'varargs' of a type (line 78)
    varargs_706457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 30), spec_706456, 'varargs')
    # Getting the type of 'spec' (line 78)
    spec_706458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 44), 'spec', False)
    # Obtaining the member 'varkw' of a type (line 78)
    varkw_706459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 44), spec_706458, 'varkw')
    # Getting the type of 'spec' (line 78)
    spec_706460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 56), 'spec', False)
    # Obtaining the member 'defaults' of a type (line 78)
    defaults_706461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 56), spec_706460, 'defaults')
    # Processing the call keyword arguments (line 78)
    kwargs_706462 = {}
    # Getting the type of 'ArgSpec' (line 78)
    ArgSpec_706453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 11), 'ArgSpec', False)
    # Calling ArgSpec(args, kwargs) (line 78)
    ArgSpec_call_result_706463 = invoke(stypy.reporting.localization.Localization(__file__, 78, 11), ArgSpec_706453, *[args_706455, varargs_706457, varkw_706459, defaults_706461], **kwargs_706462)
    
    # Assigning a type to the variable 'stypy_return_type' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type', ArgSpec_call_result_706463)
    
    # ################# End of 'getargspec(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getargspec' in the type store
    # Getting the type of 'stypy_return_type' (line 75)
    stypy_return_type_706464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_706464)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getargspec'
    return stypy_return_type_706464

# Assigning a type to the variable 'getargspec' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'getargspec', getargspec)

# Assigning a Call to a Name (line 80):

# Assigning a Call to a Name (line 80):

# Call to compile(...): (line 80)
# Processing the call arguments (line 80)
str_706467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 17), 'str', '\\s*def\\s*([_\\w][_\\w\\d]*)\\s*\\(')
# Processing the call keyword arguments (line 80)
kwargs_706468 = {}
# Getting the type of 're' (line 80)
re_706465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 6), 're', False)
# Obtaining the member 'compile' of a type (line 80)
compile_706466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 6), re_706465, 'compile')
# Calling compile(args, kwargs) (line 80)
compile_call_result_706469 = invoke(stypy.reporting.localization.Localization(__file__, 80, 6), compile_706466, *[str_706467], **kwargs_706468)

# Assigning a type to the variable 'DEF' (line 80)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'DEF', compile_call_result_706469)
# Declaration of the 'FunctionMaker' class

class FunctionMaker(object, ):
    str_706470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, (-1)), 'str', '\n    An object with the ability to create functions with a given signature.\n    It has attributes name, doc, module, signature, defaults, dict and\n    methods update and make.\n    ')
    
    # Assigning a Call to a Name (line 92):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 94)
        None_706471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 28), 'None')
        # Getting the type of 'None' (line 94)
        None_706472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 39), 'None')
        # Getting the type of 'None' (line 94)
        None_706473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 55), 'None')
        # Getting the type of 'None' (line 95)
        None_706474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 26), 'None')
        # Getting the type of 'None' (line 95)
        None_706475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 36), 'None')
        # Getting the type of 'None' (line 95)
        None_706476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 49), 'None')
        # Getting the type of 'None' (line 95)
        None_706477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 64), 'None')
        defaults = [None_706471, None_706472, None_706473, None_706474, None_706475, None_706476, None_706477]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 94, 4, False)
        # Assigning a type to the variable 'self' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionMaker.__init__', ['func', 'name', 'signature', 'defaults', 'doc', 'module', 'funcdict'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['func', 'name', 'signature', 'defaults', 'doc', 'module', 'funcdict'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 96):
        
        # Assigning a Name to a Attribute (line 96):
        # Getting the type of 'signature' (line 96)
        signature_706478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 30), 'signature')
        # Getting the type of 'self' (line 96)
        self_706479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'self')
        # Setting the type of the member 'shortsignature' of a type (line 96)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), self_706479, 'shortsignature', signature_706478)
        
        # Getting the type of 'func' (line 97)
        func_706480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 'func')
        # Testing the type of an if condition (line 97)
        if_condition_706481 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 8), func_706480)
        # Assigning a type to the variable 'if_condition_706481' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'if_condition_706481', if_condition_706481)
        # SSA begins for if statement (line 97)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 99):
        
        # Assigning a Attribute to a Attribute (line 99):
        # Getting the type of 'func' (line 99)
        func_706482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 24), 'func')
        # Obtaining the member '__name__' of a type (line 99)
        name___706483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 24), func_706482, '__name__')
        # Getting the type of 'self' (line 99)
        self_706484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'self')
        # Setting the type of the member 'name' of a type (line 99)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 12), self_706484, 'name', name___706483)
        
        
        # Getting the type of 'self' (line 100)
        self_706485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 15), 'self')
        # Obtaining the member 'name' of a type (line 100)
        name_706486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 15), self_706485, 'name')
        str_706487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 28), 'str', '<lambda>')
        # Applying the binary operator '==' (line 100)
        result_eq_706488 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 15), '==', name_706486, str_706487)
        
        # Testing the type of an if condition (line 100)
        if_condition_706489 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 12), result_eq_706488)
        # Assigning a type to the variable 'if_condition_706489' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'if_condition_706489', if_condition_706489)
        # SSA begins for if statement (line 100)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Attribute (line 101):
        
        # Assigning a Str to a Attribute (line 101):
        str_706490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 28), 'str', '_lambda_')
        # Getting the type of 'self' (line 101)
        self_706491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'self')
        # Setting the type of the member 'name' of a type (line 101)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 16), self_706491, 'name', str_706490)
        # SSA join for if statement (line 100)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Attribute (line 102):
        
        # Assigning a Attribute to a Attribute (line 102):
        # Getting the type of 'func' (line 102)
        func_706492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 23), 'func')
        # Obtaining the member '__doc__' of a type (line 102)
        doc___706493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 23), func_706492, '__doc__')
        # Getting the type of 'self' (line 102)
        self_706494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'self')
        # Setting the type of the member 'doc' of a type (line 102)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), self_706494, 'doc', doc___706493)
        
        # Assigning a Attribute to a Attribute (line 103):
        
        # Assigning a Attribute to a Attribute (line 103):
        # Getting the type of 'func' (line 103)
        func_706495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 26), 'func')
        # Obtaining the member '__module__' of a type (line 103)
        module___706496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 26), func_706495, '__module__')
        # Getting the type of 'self' (line 103)
        self_706497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'self')
        # Setting the type of the member 'module' of a type (line 103)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), self_706497, 'module', module___706496)
        
        
        # Call to isfunction(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'func' (line 104)
        func_706500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 34), 'func', False)
        # Processing the call keyword arguments (line 104)
        kwargs_706501 = {}
        # Getting the type of 'inspect' (line 104)
        inspect_706498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 'inspect', False)
        # Obtaining the member 'isfunction' of a type (line 104)
        isfunction_706499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 15), inspect_706498, 'isfunction')
        # Calling isfunction(args, kwargs) (line 104)
        isfunction_call_result_706502 = invoke(stypy.reporting.localization.Localization(__file__, 104, 15), isfunction_706499, *[func_706500], **kwargs_706501)
        
        # Testing the type of an if condition (line 104)
        if_condition_706503 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 12), isfunction_call_result_706502)
        # Assigning a type to the variable 'if_condition_706503' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'if_condition_706503', if_condition_706503)
        # SSA begins for if statement (line 104)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 105):
        
        # Assigning a Call to a Name (line 105):
        
        # Call to getfullargspec(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'func' (line 105)
        func_706505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 41), 'func', False)
        # Processing the call keyword arguments (line 105)
        kwargs_706506 = {}
        # Getting the type of 'getfullargspec' (line 105)
        getfullargspec_706504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 26), 'getfullargspec', False)
        # Calling getfullargspec(args, kwargs) (line 105)
        getfullargspec_call_result_706507 = invoke(stypy.reporting.localization.Localization(__file__, 105, 26), getfullargspec_706504, *[func_706505], **kwargs_706506)
        
        # Assigning a type to the variable 'argspec' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'argspec', getfullargspec_call_result_706507)
        
        # Assigning a Call to a Attribute (line 106):
        
        # Assigning a Call to a Attribute (line 106):
        
        # Call to getattr(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'func' (line 106)
        func_706509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 43), 'func', False)
        str_706510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 49), 'str', '__annotations__')
        
        # Obtaining an instance of the builtin type 'dict' (line 106)
        dict_706511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 68), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 106)
        
        # Processing the call keyword arguments (line 106)
        kwargs_706512 = {}
        # Getting the type of 'getattr' (line 106)
        getattr_706508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 35), 'getattr', False)
        # Calling getattr(args, kwargs) (line 106)
        getattr_call_result_706513 = invoke(stypy.reporting.localization.Localization(__file__, 106, 35), getattr_706508, *[func_706509, str_706510, dict_706511], **kwargs_706512)
        
        # Getting the type of 'self' (line 106)
        self_706514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'self')
        # Setting the type of the member 'annotations' of a type (line 106)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 16), self_706514, 'annotations', getattr_call_result_706513)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 107)
        tuple_706515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 107)
        # Adding element type (line 107)
        str_706516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 26), 'str', 'args')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 26), tuple_706515, str_706516)
        # Adding element type (line 107)
        str_706517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 34), 'str', 'varargs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 26), tuple_706515, str_706517)
        # Adding element type (line 107)
        str_706518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 45), 'str', 'varkw')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 26), tuple_706515, str_706518)
        # Adding element type (line 107)
        str_706519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 54), 'str', 'defaults')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 26), tuple_706515, str_706519)
        # Adding element type (line 107)
        str_706520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 66), 'str', 'kwonlyargs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 26), tuple_706515, str_706520)
        # Adding element type (line 107)
        str_706521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 26), 'str', 'kwonlydefaults')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 26), tuple_706515, str_706521)
        
        # Testing the type of a for loop iterable (line 107)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 107, 16), tuple_706515)
        # Getting the type of the for loop variable (line 107)
        for_loop_var_706522 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 107, 16), tuple_706515)
        # Assigning a type to the variable 'a' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'a', for_loop_var_706522)
        # SSA begins for a for statement (line 107)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to setattr(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'self' (line 109)
        self_706524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 28), 'self', False)
        # Getting the type of 'a' (line 109)
        a_706525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 34), 'a', False)
        
        # Call to getattr(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'argspec' (line 109)
        argspec_706527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 45), 'argspec', False)
        # Getting the type of 'a' (line 109)
        a_706528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 54), 'a', False)
        # Processing the call keyword arguments (line 109)
        kwargs_706529 = {}
        # Getting the type of 'getattr' (line 109)
        getattr_706526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 37), 'getattr', False)
        # Calling getattr(args, kwargs) (line 109)
        getattr_call_result_706530 = invoke(stypy.reporting.localization.Localization(__file__, 109, 37), getattr_706526, *[argspec_706527, a_706528], **kwargs_706529)
        
        # Processing the call keyword arguments (line 109)
        kwargs_706531 = {}
        # Getting the type of 'setattr' (line 109)
        setattr_706523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 20), 'setattr', False)
        # Calling setattr(args, kwargs) (line 109)
        setattr_call_result_706532 = invoke(stypy.reporting.localization.Localization(__file__, 109, 20), setattr_706523, *[self_706524, a_706525, getattr_call_result_706530], **kwargs_706531)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to enumerate(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'self' (line 110)
        self_706534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 40), 'self', False)
        # Obtaining the member 'args' of a type (line 110)
        args_706535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 40), self_706534, 'args')
        # Processing the call keyword arguments (line 110)
        kwargs_706536 = {}
        # Getting the type of 'enumerate' (line 110)
        enumerate_706533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 30), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 110)
        enumerate_call_result_706537 = invoke(stypy.reporting.localization.Localization(__file__, 110, 30), enumerate_706533, *[args_706535], **kwargs_706536)
        
        # Testing the type of a for loop iterable (line 110)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 110, 16), enumerate_call_result_706537)
        # Getting the type of the for loop variable (line 110)
        for_loop_var_706538 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 110, 16), enumerate_call_result_706537)
        # Assigning a type to the variable 'i' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 16), for_loop_var_706538))
        # Assigning a type to the variable 'arg' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'arg', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 16), for_loop_var_706538))
        # SSA begins for a for statement (line 110)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to setattr(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'self' (line 111)
        self_706540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 28), 'self', False)
        str_706541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 34), 'str', 'arg%d')
        # Getting the type of 'i' (line 111)
        i_706542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 44), 'i', False)
        # Applying the binary operator '%' (line 111)
        result_mod_706543 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 34), '%', str_706541, i_706542)
        
        # Getting the type of 'arg' (line 111)
        arg_706544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 47), 'arg', False)
        # Processing the call keyword arguments (line 111)
        kwargs_706545 = {}
        # Getting the type of 'setattr' (line 111)
        setattr_706539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 20), 'setattr', False)
        # Calling setattr(args, kwargs) (line 111)
        setattr_call_result_706546 = invoke(stypy.reporting.localization.Localization(__file__, 111, 20), setattr_706539, *[self_706540, result_mod_706543, arg_706544], **kwargs_706545)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'sys' (line 112)
        sys_706547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 19), 'sys')
        # Obtaining the member 'version' of a type (line 112)
        version_706548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 19), sys_706547, 'version')
        str_706549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 33), 'str', '3')
        # Applying the binary operator '<' (line 112)
        result_lt_706550 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 19), '<', version_706548, str_706549)
        
        # Testing the type of an if condition (line 112)
        if_condition_706551 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 16), result_lt_706550)
        # Assigning a type to the variable 'if_condition_706551' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'if_condition_706551', if_condition_706551)
        # SSA begins for if statement (line 112)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Subscript to a Attribute (line 113):
        
        # Obtaining the type of the subscript
        int_706552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 66), 'int')
        int_706553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 68), 'int')
        slice_706554 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 114, 24), int_706552, int_706553, None)
        
        # Call to formatargspec(...): (line 114)
        # Getting the type of 'argspec' (line 115)
        argspec_706557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 57), 'argspec', False)
        # Processing the call keyword arguments (line 114)

        @norecursion
        def _stypy_temp_lambda_576(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_576'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_576', 115, 40, True)
            # Passed parameters checking function
            _stypy_temp_lambda_576.stypy_localization = localization
            _stypy_temp_lambda_576.stypy_type_of_self = None
            _stypy_temp_lambda_576.stypy_type_store = module_type_store
            _stypy_temp_lambda_576.stypy_function_name = '_stypy_temp_lambda_576'
            _stypy_temp_lambda_576.stypy_param_names_list = ['val']
            _stypy_temp_lambda_576.stypy_varargs_param_name = None
            _stypy_temp_lambda_576.stypy_kwargs_param_name = None
            _stypy_temp_lambda_576.stypy_call_defaults = defaults
            _stypy_temp_lambda_576.stypy_call_varargs = varargs
            _stypy_temp_lambda_576.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_576', ['val'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_576', ['val'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            str_706558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 52), 'str', '')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 40), 'stypy_return_type', str_706558)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_576' in the type store
            # Getting the type of 'stypy_return_type' (line 115)
            stypy_return_type_706559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 40), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_706559)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_576'
            return stypy_return_type_706559

        # Assigning a type to the variable '_stypy_temp_lambda_576' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 40), '_stypy_temp_lambda_576', _stypy_temp_lambda_576)
        # Getting the type of '_stypy_temp_lambda_576' (line 115)
        _stypy_temp_lambda_576_706560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 40), '_stypy_temp_lambda_576')
        keyword_706561 = _stypy_temp_lambda_576_706560
        kwargs_706562 = {'formatvalue': keyword_706561}
        # Getting the type of 'inspect' (line 114)
        inspect_706555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 24), 'inspect', False)
        # Obtaining the member 'formatargspec' of a type (line 114)
        formatargspec_706556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 24), inspect_706555, 'formatargspec')
        # Calling formatargspec(args, kwargs) (line 114)
        formatargspec_call_result_706563 = invoke(stypy.reporting.localization.Localization(__file__, 114, 24), formatargspec_706556, *[argspec_706557], **kwargs_706562)
        
        # Obtaining the member '__getitem__' of a type (line 114)
        getitem___706564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 24), formatargspec_call_result_706563, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 114)
        subscript_call_result_706565 = invoke(stypy.reporting.localization.Localization(__file__, 114, 24), getitem___706564, slice_706554)
        
        # Getting the type of 'self' (line 113)
        self_706566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 42), 'self')
        # Setting the type of the member 'signature' of a type (line 113)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 42), self_706566, 'signature', subscript_call_result_706565)
        
        # Assigning a Attribute to a Attribute (line 113):
        # Getting the type of 'self' (line 113)
        self_706567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 42), 'self')
        # Obtaining the member 'signature' of a type (line 113)
        signature_706568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 42), self_706567, 'signature')
        # Getting the type of 'self' (line 113)
        self_706569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'self')
        # Setting the type of the member 'shortsignature' of a type (line 113)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 20), self_706569, 'shortsignature', signature_706568)
        # SSA branch for the else part of an if statement (line 112)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 117):
        
        # Assigning a Call to a Name (line 117):
        
        # Call to list(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'self' (line 117)
        self_706571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 35), 'self', False)
        # Obtaining the member 'args' of a type (line 117)
        args_706572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 35), self_706571, 'args')
        # Processing the call keyword arguments (line 117)
        kwargs_706573 = {}
        # Getting the type of 'list' (line 117)
        list_706570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 30), 'list', False)
        # Calling list(args, kwargs) (line 117)
        list_call_result_706574 = invoke(stypy.reporting.localization.Localization(__file__, 117, 30), list_706570, *[args_706572], **kwargs_706573)
        
        # Assigning a type to the variable 'allargs' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 20), 'allargs', list_call_result_706574)
        
        # Assigning a Call to a Name (line 118):
        
        # Assigning a Call to a Name (line 118):
        
        # Call to list(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'self' (line 118)
        self_706576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 40), 'self', False)
        # Obtaining the member 'args' of a type (line 118)
        args_706577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 40), self_706576, 'args')
        # Processing the call keyword arguments (line 118)
        kwargs_706578 = {}
        # Getting the type of 'list' (line 118)
        list_706575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 35), 'list', False)
        # Calling list(args, kwargs) (line 118)
        list_call_result_706579 = invoke(stypy.reporting.localization.Localization(__file__, 118, 35), list_706575, *[args_706577], **kwargs_706578)
        
        # Assigning a type to the variable 'allshortargs' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 20), 'allshortargs', list_call_result_706579)
        
        # Getting the type of 'self' (line 119)
        self_706580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 23), 'self')
        # Obtaining the member 'varargs' of a type (line 119)
        varargs_706581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 23), self_706580, 'varargs')
        # Testing the type of an if condition (line 119)
        if_condition_706582 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 20), varargs_706581)
        # Assigning a type to the variable 'if_condition_706582' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 20), 'if_condition_706582', if_condition_706582)
        # SSA begins for if statement (line 119)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 120)
        # Processing the call arguments (line 120)
        str_706585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 39), 'str', '*')
        # Getting the type of 'self' (line 120)
        self_706586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 45), 'self', False)
        # Obtaining the member 'varargs' of a type (line 120)
        varargs_706587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 45), self_706586, 'varargs')
        # Applying the binary operator '+' (line 120)
        result_add_706588 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 39), '+', str_706585, varargs_706587)
        
        # Processing the call keyword arguments (line 120)
        kwargs_706589 = {}
        # Getting the type of 'allargs' (line 120)
        allargs_706583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 24), 'allargs', False)
        # Obtaining the member 'append' of a type (line 120)
        append_706584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 24), allargs_706583, 'append')
        # Calling append(args, kwargs) (line 120)
        append_call_result_706590 = invoke(stypy.reporting.localization.Localization(__file__, 120, 24), append_706584, *[result_add_706588], **kwargs_706589)
        
        
        # Call to append(...): (line 121)
        # Processing the call arguments (line 121)
        str_706593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 44), 'str', '*')
        # Getting the type of 'self' (line 121)
        self_706594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 50), 'self', False)
        # Obtaining the member 'varargs' of a type (line 121)
        varargs_706595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 50), self_706594, 'varargs')
        # Applying the binary operator '+' (line 121)
        result_add_706596 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 44), '+', str_706593, varargs_706595)
        
        # Processing the call keyword arguments (line 121)
        kwargs_706597 = {}
        # Getting the type of 'allshortargs' (line 121)
        allshortargs_706591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 24), 'allshortargs', False)
        # Obtaining the member 'append' of a type (line 121)
        append_706592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 24), allshortargs_706591, 'append')
        # Calling append(args, kwargs) (line 121)
        append_call_result_706598 = invoke(stypy.reporting.localization.Localization(__file__, 121, 24), append_706592, *[result_add_706596], **kwargs_706597)
        
        # SSA branch for the else part of an if statement (line 119)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'self' (line 122)
        self_706599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 25), 'self')
        # Obtaining the member 'kwonlyargs' of a type (line 122)
        kwonlyargs_706600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 25), self_706599, 'kwonlyargs')
        # Testing the type of an if condition (line 122)
        if_condition_706601 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 25), kwonlyargs_706600)
        # Assigning a type to the variable 'if_condition_706601' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 25), 'if_condition_706601', if_condition_706601)
        # SSA begins for if statement (line 122)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 123)
        # Processing the call arguments (line 123)
        str_706604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 39), 'str', '*')
        # Processing the call keyword arguments (line 123)
        kwargs_706605 = {}
        # Getting the type of 'allargs' (line 123)
        allargs_706602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 24), 'allargs', False)
        # Obtaining the member 'append' of a type (line 123)
        append_706603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 24), allargs_706602, 'append')
        # Calling append(args, kwargs) (line 123)
        append_call_result_706606 = invoke(stypy.reporting.localization.Localization(__file__, 123, 24), append_706603, *[str_706604], **kwargs_706605)
        
        # SSA join for if statement (line 122)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 119)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 124)
        self_706607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 29), 'self')
        # Obtaining the member 'kwonlyargs' of a type (line 124)
        kwonlyargs_706608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 29), self_706607, 'kwonlyargs')
        # Testing the type of a for loop iterable (line 124)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 124, 20), kwonlyargs_706608)
        # Getting the type of the for loop variable (line 124)
        for_loop_var_706609 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 124, 20), kwonlyargs_706608)
        # Assigning a type to the variable 'a' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'a', for_loop_var_706609)
        # SSA begins for a for statement (line 124)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 125)
        # Processing the call arguments (line 125)
        str_706612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 39), 'str', '%s=None')
        # Getting the type of 'a' (line 125)
        a_706613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 51), 'a', False)
        # Applying the binary operator '%' (line 125)
        result_mod_706614 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 39), '%', str_706612, a_706613)
        
        # Processing the call keyword arguments (line 125)
        kwargs_706615 = {}
        # Getting the type of 'allargs' (line 125)
        allargs_706610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 24), 'allargs', False)
        # Obtaining the member 'append' of a type (line 125)
        append_706611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 24), allargs_706610, 'append')
        # Calling append(args, kwargs) (line 125)
        append_call_result_706616 = invoke(stypy.reporting.localization.Localization(__file__, 125, 24), append_706611, *[result_mod_706614], **kwargs_706615)
        
        
        # Call to append(...): (line 126)
        # Processing the call arguments (line 126)
        str_706619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 44), 'str', '%s=%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 126)
        tuple_706620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 126)
        # Adding element type (line 126)
        # Getting the type of 'a' (line 126)
        a_706621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 55), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 55), tuple_706620, a_706621)
        # Adding element type (line 126)
        # Getting the type of 'a' (line 126)
        a_706622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 58), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 55), tuple_706620, a_706622)
        
        # Applying the binary operator '%' (line 126)
        result_mod_706623 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 44), '%', str_706619, tuple_706620)
        
        # Processing the call keyword arguments (line 126)
        kwargs_706624 = {}
        # Getting the type of 'allshortargs' (line 126)
        allshortargs_706617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 24), 'allshortargs', False)
        # Obtaining the member 'append' of a type (line 126)
        append_706618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 24), allshortargs_706617, 'append')
        # Calling append(args, kwargs) (line 126)
        append_call_result_706625 = invoke(stypy.reporting.localization.Localization(__file__, 126, 24), append_706618, *[result_mod_706623], **kwargs_706624)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 127)
        self_706626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'self')
        # Obtaining the member 'varkw' of a type (line 127)
        varkw_706627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 23), self_706626, 'varkw')
        # Testing the type of an if condition (line 127)
        if_condition_706628 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 127, 20), varkw_706627)
        # Assigning a type to the variable 'if_condition_706628' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 20), 'if_condition_706628', if_condition_706628)
        # SSA begins for if statement (line 127)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 128)
        # Processing the call arguments (line 128)
        str_706631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 39), 'str', '**')
        # Getting the type of 'self' (line 128)
        self_706632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 46), 'self', False)
        # Obtaining the member 'varkw' of a type (line 128)
        varkw_706633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 46), self_706632, 'varkw')
        # Applying the binary operator '+' (line 128)
        result_add_706634 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 39), '+', str_706631, varkw_706633)
        
        # Processing the call keyword arguments (line 128)
        kwargs_706635 = {}
        # Getting the type of 'allargs' (line 128)
        allargs_706629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 24), 'allargs', False)
        # Obtaining the member 'append' of a type (line 128)
        append_706630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 24), allargs_706629, 'append')
        # Calling append(args, kwargs) (line 128)
        append_call_result_706636 = invoke(stypy.reporting.localization.Localization(__file__, 128, 24), append_706630, *[result_add_706634], **kwargs_706635)
        
        
        # Call to append(...): (line 129)
        # Processing the call arguments (line 129)
        str_706639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 44), 'str', '**')
        # Getting the type of 'self' (line 129)
        self_706640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 51), 'self', False)
        # Obtaining the member 'varkw' of a type (line 129)
        varkw_706641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 51), self_706640, 'varkw')
        # Applying the binary operator '+' (line 129)
        result_add_706642 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 44), '+', str_706639, varkw_706641)
        
        # Processing the call keyword arguments (line 129)
        kwargs_706643 = {}
        # Getting the type of 'allshortargs' (line 129)
        allshortargs_706637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 24), 'allshortargs', False)
        # Obtaining the member 'append' of a type (line 129)
        append_706638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 24), allshortargs_706637, 'append')
        # Calling append(args, kwargs) (line 129)
        append_call_result_706644 = invoke(stypy.reporting.localization.Localization(__file__, 129, 24), append_706638, *[result_add_706642], **kwargs_706643)
        
        # SSA join for if statement (line 127)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 130):
        
        # Assigning a Call to a Attribute (line 130):
        
        # Call to join(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'allargs' (line 130)
        allargs_706647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 47), 'allargs', False)
        # Processing the call keyword arguments (line 130)
        kwargs_706648 = {}
        str_706645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 37), 'str', ', ')
        # Obtaining the member 'join' of a type (line 130)
        join_706646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 37), str_706645, 'join')
        # Calling join(args, kwargs) (line 130)
        join_call_result_706649 = invoke(stypy.reporting.localization.Localization(__file__, 130, 37), join_706646, *[allargs_706647], **kwargs_706648)
        
        # Getting the type of 'self' (line 130)
        self_706650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 20), 'self')
        # Setting the type of the member 'signature' of a type (line 130)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 20), self_706650, 'signature', join_call_result_706649)
        
        # Assigning a Call to a Attribute (line 131):
        
        # Assigning a Call to a Attribute (line 131):
        
        # Call to join(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'allshortargs' (line 131)
        allshortargs_706653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 52), 'allshortargs', False)
        # Processing the call keyword arguments (line 131)
        kwargs_706654 = {}
        str_706651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 42), 'str', ', ')
        # Obtaining the member 'join' of a type (line 131)
        join_706652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 42), str_706651, 'join')
        # Calling join(args, kwargs) (line 131)
        join_call_result_706655 = invoke(stypy.reporting.localization.Localization(__file__, 131, 42), join_706652, *[allshortargs_706653], **kwargs_706654)
        
        # Getting the type of 'self' (line 131)
        self_706656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 20), 'self')
        # Setting the type of the member 'shortsignature' of a type (line 131)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 20), self_706656, 'shortsignature', join_call_result_706655)
        # SSA join for if statement (line 112)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 132):
        
        # Assigning a Call to a Attribute (line 132):
        
        # Call to copy(...): (line 132)
        # Processing the call keyword arguments (line 132)
        kwargs_706660 = {}
        # Getting the type of 'func' (line 132)
        func_706657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 28), 'func', False)
        # Obtaining the member '__dict__' of a type (line 132)
        dict___706658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 28), func_706657, '__dict__')
        # Obtaining the member 'copy' of a type (line 132)
        copy_706659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 28), dict___706658, 'copy')
        # Calling copy(args, kwargs) (line 132)
        copy_call_result_706661 = invoke(stypy.reporting.localization.Localization(__file__, 132, 28), copy_706659, *[], **kwargs_706660)
        
        # Getting the type of 'self' (line 132)
        self_706662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'self')
        # Setting the type of the member 'dict' of a type (line 132)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 16), self_706662, 'dict', copy_call_result_706661)
        # SSA join for if statement (line 104)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 97)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'name' (line 134)
        name_706663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 11), 'name')
        # Testing the type of an if condition (line 134)
        if_condition_706664 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 8), name_706663)
        # Assigning a type to the variable 'if_condition_706664' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'if_condition_706664', if_condition_706664)
        # SSA begins for if statement (line 134)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 135):
        
        # Assigning a Name to a Attribute (line 135):
        # Getting the type of 'name' (line 135)
        name_706665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 24), 'name')
        # Getting the type of 'self' (line 135)
        self_706666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'self')
        # Setting the type of the member 'name' of a type (line 135)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 12), self_706666, 'name', name_706665)
        # SSA join for if statement (line 134)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 136)
        # Getting the type of 'signature' (line 136)
        signature_706667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'signature')
        # Getting the type of 'None' (line 136)
        None_706668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 28), 'None')
        
        (may_be_706669, more_types_in_union_706670) = may_not_be_none(signature_706667, None_706668)

        if may_be_706669:

            if more_types_in_union_706670:
                # Runtime conditional SSA (line 136)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 137):
            
            # Assigning a Name to a Attribute (line 137):
            # Getting the type of 'signature' (line 137)
            signature_706671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 29), 'signature')
            # Getting the type of 'self' (line 137)
            self_706672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'self')
            # Setting the type of the member 'signature' of a type (line 137)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 12), self_706672, 'signature', signature_706671)

            if more_types_in_union_706670:
                # SSA join for if statement (line 136)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'defaults' (line 138)
        defaults_706673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 11), 'defaults')
        # Testing the type of an if condition (line 138)
        if_condition_706674 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 138, 8), defaults_706673)
        # Assigning a type to the variable 'if_condition_706674' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'if_condition_706674', if_condition_706674)
        # SSA begins for if statement (line 138)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 139):
        
        # Assigning a Name to a Attribute (line 139):
        # Getting the type of 'defaults' (line 139)
        defaults_706675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 28), 'defaults')
        # Getting the type of 'self' (line 139)
        self_706676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'self')
        # Setting the type of the member 'defaults' of a type (line 139)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 12), self_706676, 'defaults', defaults_706675)
        # SSA join for if statement (line 138)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'doc' (line 140)
        doc_706677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 11), 'doc')
        # Testing the type of an if condition (line 140)
        if_condition_706678 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 140, 8), doc_706677)
        # Assigning a type to the variable 'if_condition_706678' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'if_condition_706678', if_condition_706678)
        # SSA begins for if statement (line 140)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 141):
        
        # Assigning a Name to a Attribute (line 141):
        # Getting the type of 'doc' (line 141)
        doc_706679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 23), 'doc')
        # Getting the type of 'self' (line 141)
        self_706680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'self')
        # Setting the type of the member 'doc' of a type (line 141)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 12), self_706680, 'doc', doc_706679)
        # SSA join for if statement (line 140)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'module' (line 142)
        module_706681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 11), 'module')
        # Testing the type of an if condition (line 142)
        if_condition_706682 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 142, 8), module_706681)
        # Assigning a type to the variable 'if_condition_706682' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'if_condition_706682', if_condition_706682)
        # SSA begins for if statement (line 142)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 143):
        
        # Assigning a Name to a Attribute (line 143):
        # Getting the type of 'module' (line 143)
        module_706683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 26), 'module')
        # Getting the type of 'self' (line 143)
        self_706684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'self')
        # Setting the type of the member 'module' of a type (line 143)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 12), self_706684, 'module', module_706683)
        # SSA join for if statement (line 142)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'funcdict' (line 144)
        funcdict_706685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 11), 'funcdict')
        # Testing the type of an if condition (line 144)
        if_condition_706686 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 8), funcdict_706685)
        # Assigning a type to the variable 'if_condition_706686' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'if_condition_706686', if_condition_706686)
        # SSA begins for if statement (line 144)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 145):
        
        # Assigning a Name to a Attribute (line 145):
        # Getting the type of 'funcdict' (line 145)
        funcdict_706687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 24), 'funcdict')
        # Getting the type of 'self' (line 145)
        self_706688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'self')
        # Setting the type of the member 'dict' of a type (line 145)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 12), self_706688, 'dict', funcdict_706687)
        # SSA join for if statement (line 144)
        module_type_store = module_type_store.join_ssa_context()
        
        # Evaluating assert statement condition
        
        # Call to hasattr(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'self' (line 147)
        self_706690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 23), 'self', False)
        str_706691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 29), 'str', 'name')
        # Processing the call keyword arguments (line 147)
        kwargs_706692 = {}
        # Getting the type of 'hasattr' (line 147)
        hasattr_706689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 15), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 147)
        hasattr_call_result_706693 = invoke(stypy.reporting.localization.Localization(__file__, 147, 15), hasattr_706689, *[self_706690, str_706691], **kwargs_706692)
        
        
        # Type idiom detected: calculating its left and rigth part (line 148)
        str_706694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 29), 'str', 'signature')
        # Getting the type of 'self' (line 148)
        self_706695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 23), 'self')
        
        (may_be_706696, more_types_in_union_706697) = may_not_provide_member(str_706694, self_706695)

        if may_be_706696:

            if more_types_in_union_706697:
                # Runtime conditional SSA (line 148)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'self' (line 148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'self', remove_member_provider_from_union(self_706695, 'signature'))
            
            # Call to TypeError(...): (line 149)
            # Processing the call arguments (line 149)
            str_706699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 28), 'str', 'You are decorating a non function: %s')
            # Getting the type of 'func' (line 149)
            func_706700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 70), 'func', False)
            # Applying the binary operator '%' (line 149)
            result_mod_706701 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 28), '%', str_706699, func_706700)
            
            # Processing the call keyword arguments (line 149)
            kwargs_706702 = {}
            # Getting the type of 'TypeError' (line 149)
            TypeError_706698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 18), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 149)
            TypeError_call_result_706703 = invoke(stypy.reporting.localization.Localization(__file__, 149, 18), TypeError_706698, *[result_mod_706701], **kwargs_706702)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 149, 12), TypeError_call_result_706703, 'raise parameter', BaseException)

            if more_types_in_union_706697:
                # SSA join for if statement (line 148)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def update(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update'
        module_type_store = module_type_store.open_function_context('update', 151, 4, False)
        # Assigning a type to the variable 'self' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FunctionMaker.update.__dict__.__setitem__('stypy_localization', localization)
        FunctionMaker.update.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FunctionMaker.update.__dict__.__setitem__('stypy_type_store', module_type_store)
        FunctionMaker.update.__dict__.__setitem__('stypy_function_name', 'FunctionMaker.update')
        FunctionMaker.update.__dict__.__setitem__('stypy_param_names_list', ['func'])
        FunctionMaker.update.__dict__.__setitem__('stypy_varargs_param_name', None)
        FunctionMaker.update.__dict__.__setitem__('stypy_kwargs_param_name', 'kw')
        FunctionMaker.update.__dict__.__setitem__('stypy_call_defaults', defaults)
        FunctionMaker.update.__dict__.__setitem__('stypy_call_varargs', varargs)
        FunctionMaker.update.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FunctionMaker.update.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionMaker.update', ['func'], None, 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update', localization, ['func'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update(...)' code ##################

        str_706704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 8), 'str', 'Update the signature of func with the data in self')
        
        # Assigning a Attribute to a Attribute (line 153):
        
        # Assigning a Attribute to a Attribute (line 153):
        # Getting the type of 'self' (line 153)
        self_706705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 24), 'self')
        # Obtaining the member 'name' of a type (line 153)
        name_706706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 24), self_706705, 'name')
        # Getting the type of 'func' (line 153)
        func_706707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'func')
        # Setting the type of the member '__name__' of a type (line 153)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), func_706707, '__name__', name_706706)
        
        # Assigning a Call to a Attribute (line 154):
        
        # Assigning a Call to a Attribute (line 154):
        
        # Call to getattr(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'self' (line 154)
        self_706709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 31), 'self', False)
        str_706710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 37), 'str', 'doc')
        # Getting the type of 'None' (line 154)
        None_706711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 44), 'None', False)
        # Processing the call keyword arguments (line 154)
        kwargs_706712 = {}
        # Getting the type of 'getattr' (line 154)
        getattr_706708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 23), 'getattr', False)
        # Calling getattr(args, kwargs) (line 154)
        getattr_call_result_706713 = invoke(stypy.reporting.localization.Localization(__file__, 154, 23), getattr_706708, *[self_706709, str_706710, None_706711], **kwargs_706712)
        
        # Getting the type of 'func' (line 154)
        func_706714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'func')
        # Setting the type of the member '__doc__' of a type (line 154)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), func_706714, '__doc__', getattr_call_result_706713)
        
        # Assigning a Call to a Attribute (line 155):
        
        # Assigning a Call to a Attribute (line 155):
        
        # Call to getattr(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'self' (line 155)
        self_706716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 32), 'self', False)
        str_706717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 38), 'str', 'dict')
        
        # Obtaining an instance of the builtin type 'dict' (line 155)
        dict_706718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 46), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 155)
        
        # Processing the call keyword arguments (line 155)
        kwargs_706719 = {}
        # Getting the type of 'getattr' (line 155)
        getattr_706715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 24), 'getattr', False)
        # Calling getattr(args, kwargs) (line 155)
        getattr_call_result_706720 = invoke(stypy.reporting.localization.Localization(__file__, 155, 24), getattr_706715, *[self_706716, str_706717, dict_706718], **kwargs_706719)
        
        # Getting the type of 'func' (line 155)
        func_706721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'func')
        # Setting the type of the member '__dict__' of a type (line 155)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), func_706721, '__dict__', getattr_call_result_706720)
        
        # Assigning a Call to a Attribute (line 156):
        
        # Assigning a Call to a Attribute (line 156):
        
        # Call to getattr(...): (line 156)
        # Processing the call arguments (line 156)
        # Getting the type of 'self' (line 156)
        self_706723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 36), 'self', False)
        str_706724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 42), 'str', 'defaults')
        
        # Obtaining an instance of the builtin type 'tuple' (line 156)
        tuple_706725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 54), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 156)
        
        # Processing the call keyword arguments (line 156)
        kwargs_706726 = {}
        # Getting the type of 'getattr' (line 156)
        getattr_706722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 28), 'getattr', False)
        # Calling getattr(args, kwargs) (line 156)
        getattr_call_result_706727 = invoke(stypy.reporting.localization.Localization(__file__, 156, 28), getattr_706722, *[self_706723, str_706724, tuple_706725], **kwargs_706726)
        
        # Getting the type of 'func' (line 156)
        func_706728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'func')
        # Setting the type of the member '__defaults__' of a type (line 156)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 8), func_706728, '__defaults__', getattr_call_result_706727)
        
        # Assigning a Call to a Attribute (line 157):
        
        # Assigning a Call to a Attribute (line 157):
        
        # Call to getattr(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'self' (line 157)
        self_706730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 38), 'self', False)
        str_706731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 44), 'str', 'kwonlydefaults')
        # Getting the type of 'None' (line 157)
        None_706732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 62), 'None', False)
        # Processing the call keyword arguments (line 157)
        kwargs_706733 = {}
        # Getting the type of 'getattr' (line 157)
        getattr_706729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 30), 'getattr', False)
        # Calling getattr(args, kwargs) (line 157)
        getattr_call_result_706734 = invoke(stypy.reporting.localization.Localization(__file__, 157, 30), getattr_706729, *[self_706730, str_706731, None_706732], **kwargs_706733)
        
        # Getting the type of 'func' (line 157)
        func_706735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'func')
        # Setting the type of the member '__kwdefaults__' of a type (line 157)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), func_706735, '__kwdefaults__', getattr_call_result_706734)
        
        # Assigning a Call to a Attribute (line 158):
        
        # Assigning a Call to a Attribute (line 158):
        
        # Call to getattr(...): (line 158)
        # Processing the call arguments (line 158)
        # Getting the type of 'self' (line 158)
        self_706737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 39), 'self', False)
        str_706738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 45), 'str', 'annotations')
        # Getting the type of 'None' (line 158)
        None_706739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 60), 'None', False)
        # Processing the call keyword arguments (line 158)
        kwargs_706740 = {}
        # Getting the type of 'getattr' (line 158)
        getattr_706736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 31), 'getattr', False)
        # Calling getattr(args, kwargs) (line 158)
        getattr_call_result_706741 = invoke(stypy.reporting.localization.Localization(__file__, 158, 31), getattr_706736, *[self_706737, str_706738, None_706739], **kwargs_706740)
        
        # Getting the type of 'func' (line 158)
        func_706742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'func')
        # Setting the type of the member '__annotations__' of a type (line 158)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), func_706742, '__annotations__', getattr_call_result_706741)
        
        
        # SSA begins for try-except statement (line 159)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 160):
        
        # Assigning a Call to a Name (line 160):
        
        # Call to _getframe(...): (line 160)
        # Processing the call arguments (line 160)
        int_706745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 34), 'int')
        # Processing the call keyword arguments (line 160)
        kwargs_706746 = {}
        # Getting the type of 'sys' (line 160)
        sys_706743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), 'sys', False)
        # Obtaining the member '_getframe' of a type (line 160)
        _getframe_706744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 20), sys_706743, '_getframe')
        # Calling _getframe(args, kwargs) (line 160)
        _getframe_call_result_706747 = invoke(stypy.reporting.localization.Localization(__file__, 160, 20), _getframe_706744, *[int_706745], **kwargs_706746)
        
        # Assigning a type to the variable 'frame' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'frame', _getframe_call_result_706747)
        # SSA branch for the except part of a try statement (line 159)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 159)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Str to a Name (line 162):
        
        # Assigning a Str to a Name (line 162):
        str_706748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 27), 'str', '?')
        # Assigning a type to the variable 'callermodule' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'callermodule', str_706748)
        # SSA branch for the else branch of a try statement (line 159)
        module_type_store.open_ssa_branch('except else')
        
        # Assigning a Call to a Name (line 164):
        
        # Assigning a Call to a Name (line 164):
        
        # Call to get(...): (line 164)
        # Processing the call arguments (line 164)
        str_706752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 47), 'str', '__name__')
        str_706753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 59), 'str', '?')
        # Processing the call keyword arguments (line 164)
        kwargs_706754 = {}
        # Getting the type of 'frame' (line 164)
        frame_706749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 27), 'frame', False)
        # Obtaining the member 'f_globals' of a type (line 164)
        f_globals_706750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 27), frame_706749, 'f_globals')
        # Obtaining the member 'get' of a type (line 164)
        get_706751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 27), f_globals_706750, 'get')
        # Calling get(args, kwargs) (line 164)
        get_call_result_706755 = invoke(stypy.reporting.localization.Localization(__file__, 164, 27), get_706751, *[str_706752, str_706753], **kwargs_706754)
        
        # Assigning a type to the variable 'callermodule' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'callermodule', get_call_result_706755)
        # SSA join for try-except statement (line 159)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 165):
        
        # Assigning a Call to a Attribute (line 165):
        
        # Call to getattr(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'self' (line 165)
        self_706757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 34), 'self', False)
        str_706758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 40), 'str', 'module')
        # Getting the type of 'callermodule' (line 165)
        callermodule_706759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 50), 'callermodule', False)
        # Processing the call keyword arguments (line 165)
        kwargs_706760 = {}
        # Getting the type of 'getattr' (line 165)
        getattr_706756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 26), 'getattr', False)
        # Calling getattr(args, kwargs) (line 165)
        getattr_call_result_706761 = invoke(stypy.reporting.localization.Localization(__file__, 165, 26), getattr_706756, *[self_706757, str_706758, callermodule_706759], **kwargs_706760)
        
        # Getting the type of 'func' (line 165)
        func_706762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'func')
        # Setting the type of the member '__module__' of a type (line 165)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 8), func_706762, '__module__', getattr_call_result_706761)
        
        # Call to update(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'kw' (line 166)
        kw_706766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 29), 'kw', False)
        # Processing the call keyword arguments (line 166)
        kwargs_706767 = {}
        # Getting the type of 'func' (line 166)
        func_706763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'func', False)
        # Obtaining the member '__dict__' of a type (line 166)
        dict___706764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), func_706763, '__dict__')
        # Obtaining the member 'update' of a type (line 166)
        update_706765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), dict___706764, 'update')
        # Calling update(args, kwargs) (line 166)
        update_call_result_706768 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), update_706765, *[kw_706766], **kwargs_706767)
        
        
        # ################# End of 'update(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update' in the type store
        # Getting the type of 'stypy_return_type' (line 151)
        stypy_return_type_706769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_706769)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update'
        return stypy_return_type_706769


    @norecursion
    def make(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 168)
        None_706770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 39), 'None')
        # Getting the type of 'False' (line 168)
        False_706771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 55), 'False')
        defaults = [None_706770, False_706771]
        # Create a new context for function 'make'
        module_type_store = module_type_store.open_function_context('make', 168, 4, False)
        # Assigning a type to the variable 'self' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FunctionMaker.make.__dict__.__setitem__('stypy_localization', localization)
        FunctionMaker.make.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FunctionMaker.make.__dict__.__setitem__('stypy_type_store', module_type_store)
        FunctionMaker.make.__dict__.__setitem__('stypy_function_name', 'FunctionMaker.make')
        FunctionMaker.make.__dict__.__setitem__('stypy_param_names_list', ['src_templ', 'evaldict', 'addsource'])
        FunctionMaker.make.__dict__.__setitem__('stypy_varargs_param_name', None)
        FunctionMaker.make.__dict__.__setitem__('stypy_kwargs_param_name', 'attrs')
        FunctionMaker.make.__dict__.__setitem__('stypy_call_defaults', defaults)
        FunctionMaker.make.__dict__.__setitem__('stypy_call_varargs', varargs)
        FunctionMaker.make.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FunctionMaker.make.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionMaker.make', ['src_templ', 'evaldict', 'addsource'], None, 'attrs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'make', localization, ['src_templ', 'evaldict', 'addsource'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'make(...)' code ##################

        str_706772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 8), 'str', 'Make a new function from a given template and update the signature')
        
        # Assigning a BinOp to a Name (line 170):
        
        # Assigning a BinOp to a Name (line 170):
        # Getting the type of 'src_templ' (line 170)
        src_templ_706773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 14), 'src_templ')
        
        # Call to vars(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'self' (line 170)
        self_706775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 31), 'self', False)
        # Processing the call keyword arguments (line 170)
        kwargs_706776 = {}
        # Getting the type of 'vars' (line 170)
        vars_706774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 26), 'vars', False)
        # Calling vars(args, kwargs) (line 170)
        vars_call_result_706777 = invoke(stypy.reporting.localization.Localization(__file__, 170, 26), vars_706774, *[self_706775], **kwargs_706776)
        
        # Applying the binary operator '%' (line 170)
        result_mod_706778 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 14), '%', src_templ_706773, vars_call_result_706777)
        
        # Assigning a type to the variable 'src' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'src', result_mod_706778)
        
        # Assigning a BoolOp to a Name (line 171):
        
        # Assigning a BoolOp to a Name (line 171):
        
        # Evaluating a boolean operation
        # Getting the type of 'evaldict' (line 171)
        evaldict_706779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 19), 'evaldict')
        
        # Obtaining an instance of the builtin type 'dict' (line 171)
        dict_706780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 31), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 171)
        
        # Applying the binary operator 'or' (line 171)
        result_or_keyword_706781 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 19), 'or', evaldict_706779, dict_706780)
        
        # Assigning a type to the variable 'evaldict' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'evaldict', result_or_keyword_706781)
        
        # Assigning a Call to a Name (line 172):
        
        # Assigning a Call to a Name (line 172):
        
        # Call to match(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'src' (line 172)
        src_706784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 23), 'src', False)
        # Processing the call keyword arguments (line 172)
        kwargs_706785 = {}
        # Getting the type of 'DEF' (line 172)
        DEF_706782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 13), 'DEF', False)
        # Obtaining the member 'match' of a type (line 172)
        match_706783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 13), DEF_706782, 'match')
        # Calling match(args, kwargs) (line 172)
        match_call_result_706786 = invoke(stypy.reporting.localization.Localization(__file__, 172, 13), match_706783, *[src_706784], **kwargs_706785)
        
        # Assigning a type to the variable 'mo' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'mo', match_call_result_706786)
        
        # Type idiom detected: calculating its left and rigth part (line 173)
        # Getting the type of 'mo' (line 173)
        mo_706787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 11), 'mo')
        # Getting the type of 'None' (line 173)
        None_706788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 17), 'None')
        
        (may_be_706789, more_types_in_union_706790) = may_be_none(mo_706787, None_706788)

        if may_be_706789:

            if more_types_in_union_706790:
                # Runtime conditional SSA (line 173)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to SyntaxError(...): (line 174)
            # Processing the call arguments (line 174)
            str_706792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 30), 'str', 'not a valid function template\n%s')
            # Getting the type of 'src' (line 174)
            src_706793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 68), 'src', False)
            # Applying the binary operator '%' (line 174)
            result_mod_706794 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 30), '%', str_706792, src_706793)
            
            # Processing the call keyword arguments (line 174)
            kwargs_706795 = {}
            # Getting the type of 'SyntaxError' (line 174)
            SyntaxError_706791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 18), 'SyntaxError', False)
            # Calling SyntaxError(args, kwargs) (line 174)
            SyntaxError_call_result_706796 = invoke(stypy.reporting.localization.Localization(__file__, 174, 18), SyntaxError_706791, *[result_mod_706794], **kwargs_706795)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 174, 12), SyntaxError_call_result_706796, 'raise parameter', BaseException)

            if more_types_in_union_706790:
                # SSA join for if statement (line 173)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 175):
        
        # Assigning a Call to a Name (line 175):
        
        # Call to group(...): (line 175)
        # Processing the call arguments (line 175)
        int_706799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 24), 'int')
        # Processing the call keyword arguments (line 175)
        kwargs_706800 = {}
        # Getting the type of 'mo' (line 175)
        mo_706797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 15), 'mo', False)
        # Obtaining the member 'group' of a type (line 175)
        group_706798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 15), mo_706797, 'group')
        # Calling group(args, kwargs) (line 175)
        group_call_result_706801 = invoke(stypy.reporting.localization.Localization(__file__, 175, 15), group_706798, *[int_706799], **kwargs_706800)
        
        # Assigning a type to the variable 'name' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'name', group_call_result_706801)
        
        # Assigning a Call to a Name (line 176):
        
        # Assigning a Call to a Name (line 176):
        
        # Call to set(...): (line 176)
        # Processing the call arguments (line 176)
        
        # Obtaining an instance of the builtin type 'list' (line 176)
        list_706803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 176)
        # Adding element type (line 176)
        # Getting the type of 'name' (line 176)
        name_706804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 21), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 20), list_706803, name_706804)
        
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to split(...): (line 177)
        # Processing the call arguments (line 177)
        str_706813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 56), 'str', ',')
        # Processing the call keyword arguments (line 177)
        kwargs_706814 = {}
        # Getting the type of 'self' (line 177)
        self_706810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 30), 'self', False)
        # Obtaining the member 'shortsignature' of a type (line 177)
        shortsignature_706811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 30), self_706810, 'shortsignature')
        # Obtaining the member 'split' of a type (line 177)
        split_706812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 30), shortsignature_706811, 'split')
        # Calling split(args, kwargs) (line 177)
        split_call_result_706815 = invoke(stypy.reporting.localization.Localization(__file__, 177, 30), split_706812, *[str_706813], **kwargs_706814)
        
        comprehension_706816 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 30), split_call_result_706815)
        # Assigning a type to the variable 'arg' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 30), 'arg', comprehension_706816)
        
        # Call to strip(...): (line 176)
        # Processing the call arguments (line 176)
        str_706807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 40), 'str', ' *')
        # Processing the call keyword arguments (line 176)
        kwargs_706808 = {}
        # Getting the type of 'arg' (line 176)
        arg_706805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 30), 'arg', False)
        # Obtaining the member 'strip' of a type (line 176)
        strip_706806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 30), arg_706805, 'strip')
        # Calling strip(args, kwargs) (line 176)
        strip_call_result_706809 = invoke(stypy.reporting.localization.Localization(__file__, 176, 30), strip_706806, *[str_706807], **kwargs_706808)
        
        list_706817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 30), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 30), list_706817, strip_call_result_706809)
        # Applying the binary operator '+' (line 176)
        result_add_706818 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 20), '+', list_706803, list_706817)
        
        # Processing the call keyword arguments (line 176)
        kwargs_706819 = {}
        # Getting the type of 'set' (line 176)
        set_706802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 16), 'set', False)
        # Calling set(args, kwargs) (line 176)
        set_call_result_706820 = invoke(stypy.reporting.localization.Localization(__file__, 176, 16), set_706802, *[result_add_706818], **kwargs_706819)
        
        # Assigning a type to the variable 'names' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'names', set_call_result_706820)
        
        # Getting the type of 'names' (line 178)
        names_706821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 17), 'names')
        # Testing the type of a for loop iterable (line 178)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 178, 8), names_706821)
        # Getting the type of the for loop variable (line 178)
        for_loop_var_706822 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 178, 8), names_706821)
        # Assigning a type to the variable 'n' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'n', for_loop_var_706822)
        # SSA begins for a for statement (line 178)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'n' (line 179)
        n_706823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 15), 'n')
        
        # Obtaining an instance of the builtin type 'tuple' (line 179)
        tuple_706824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 179)
        # Adding element type (line 179)
        str_706825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 21), 'str', '_func_')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 21), tuple_706824, str_706825)
        # Adding element type (line 179)
        str_706826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 31), 'str', '_call_')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 21), tuple_706824, str_706826)
        
        # Applying the binary operator 'in' (line 179)
        result_contains_706827 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 15), 'in', n_706823, tuple_706824)
        
        # Testing the type of an if condition (line 179)
        if_condition_706828 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 179, 12), result_contains_706827)
        # Assigning a type to the variable 'if_condition_706828' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'if_condition_706828', if_condition_706828)
        # SSA begins for if statement (line 179)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to NameError(...): (line 180)
        # Processing the call arguments (line 180)
        str_706830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 32), 'str', '%s is overridden in\n%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 180)
        tuple_706831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 61), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 180)
        # Adding element type (line 180)
        # Getting the type of 'n' (line 180)
        n_706832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 61), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 61), tuple_706831, n_706832)
        # Adding element type (line 180)
        # Getting the type of 'src' (line 180)
        src_706833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 64), 'src', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 61), tuple_706831, src_706833)
        
        # Applying the binary operator '%' (line 180)
        result_mod_706834 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 32), '%', str_706830, tuple_706831)
        
        # Processing the call keyword arguments (line 180)
        kwargs_706835 = {}
        # Getting the type of 'NameError' (line 180)
        NameError_706829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 22), 'NameError', False)
        # Calling NameError(args, kwargs) (line 180)
        NameError_call_result_706836 = invoke(stypy.reporting.localization.Localization(__file__, 180, 22), NameError_706829, *[result_mod_706834], **kwargs_706835)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 180, 16), NameError_call_result_706836, 'raise parameter', BaseException)
        # SSA join for if statement (line 179)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to endswith(...): (line 181)
        # Processing the call arguments (line 181)
        str_706839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 28), 'str', '\n')
        # Processing the call keyword arguments (line 181)
        kwargs_706840 = {}
        # Getting the type of 'src' (line 181)
        src_706837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 15), 'src', False)
        # Obtaining the member 'endswith' of a type (line 181)
        endswith_706838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 15), src_706837, 'endswith')
        # Calling endswith(args, kwargs) (line 181)
        endswith_call_result_706841 = invoke(stypy.reporting.localization.Localization(__file__, 181, 15), endswith_706838, *[str_706839], **kwargs_706840)
        
        # Applying the 'not' unary operator (line 181)
        result_not__706842 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 11), 'not', endswith_call_result_706841)
        
        # Testing the type of an if condition (line 181)
        if_condition_706843 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 181, 8), result_not__706842)
        # Assigning a type to the variable 'if_condition_706843' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'if_condition_706843', if_condition_706843)
        # SSA begins for if statement (line 181)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'src' (line 182)
        src_706844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'src')
        str_706845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 19), 'str', '\n')
        # Applying the binary operator '+=' (line 182)
        result_iadd_706846 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 12), '+=', src_706844, str_706845)
        # Assigning a type to the variable 'src' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'src', result_iadd_706846)
        
        # SSA join for if statement (line 181)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 187):
        
        # Assigning a BinOp to a Name (line 187):
        str_706847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 19), 'str', '<decorator-gen-%d>')
        
        # Obtaining an instance of the builtin type 'tuple' (line 187)
        tuple_706848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 187)
        # Adding element type (line 187)
        
        # Call to next(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'self' (line 187)
        self_706850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 48), 'self', False)
        # Obtaining the member '_compile_count' of a type (line 187)
        _compile_count_706851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 48), self_706850, '_compile_count')
        # Processing the call keyword arguments (line 187)
        kwargs_706852 = {}
        # Getting the type of 'next' (line 187)
        next_706849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 43), 'next', False)
        # Calling next(args, kwargs) (line 187)
        next_call_result_706853 = invoke(stypy.reporting.localization.Localization(__file__, 187, 43), next_706849, *[_compile_count_706851], **kwargs_706852)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 43), tuple_706848, next_call_result_706853)
        
        # Applying the binary operator '%' (line 187)
        result_mod_706854 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 19), '%', str_706847, tuple_706848)
        
        # Assigning a type to the variable 'filename' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'filename', result_mod_706854)
        
        
        # SSA begins for try-except statement (line 188)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 189):
        
        # Assigning a Call to a Name (line 189):
        
        # Call to compile(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'src' (line 189)
        src_706856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 27), 'src', False)
        # Getting the type of 'filename' (line 189)
        filename_706857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 32), 'filename', False)
        str_706858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 42), 'str', 'single')
        # Processing the call keyword arguments (line 189)
        kwargs_706859 = {}
        # Getting the type of 'compile' (line 189)
        compile_706855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 19), 'compile', False)
        # Calling compile(args, kwargs) (line 189)
        compile_call_result_706860 = invoke(stypy.reporting.localization.Localization(__file__, 189, 19), compile_706855, *[src_706856, filename_706857, str_706858], **kwargs_706859)
        
        # Assigning a type to the variable 'code' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'code', compile_call_result_706860)
        # Dynamic code evaluation using an exec statement
        # Getting the type of 'code' (line 190)
        code_706861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 17), 'code')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 190, 12), code_706861, 'exec parameter', 'StringType', 'FileType', 'CodeType')
        enable_usage_of_dynamic_types_warning(stypy.reporting.localization.Localization(__file__, 190, 12))
        # SSA branch for the except part of a try statement (line 188)
        # SSA branch for the except '<any exception>' branch of a try statement (line 188)
        module_type_store.open_ssa_branch('except')
        
        # Call to print(...): (line 192)
        # Processing the call arguments (line 192)
        str_706863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 18), 'str', 'Error in generated code:')
        # Processing the call keyword arguments (line 192)
        # Getting the type of 'sys' (line 192)
        sys_706864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 51), 'sys', False)
        # Obtaining the member 'stderr' of a type (line 192)
        stderr_706865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 51), sys_706864, 'stderr')
        keyword_706866 = stderr_706865
        kwargs_706867 = {'file': keyword_706866}
        # Getting the type of 'print' (line 192)
        print_706862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'print', False)
        # Calling print(args, kwargs) (line 192)
        print_call_result_706868 = invoke(stypy.reporting.localization.Localization(__file__, 192, 12), print_706862, *[str_706863], **kwargs_706867)
        
        
        # Call to print(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'src' (line 193)
        src_706870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 18), 'src', False)
        # Processing the call keyword arguments (line 193)
        # Getting the type of 'sys' (line 193)
        sys_706871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 28), 'sys', False)
        # Obtaining the member 'stderr' of a type (line 193)
        stderr_706872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 28), sys_706871, 'stderr')
        keyword_706873 = stderr_706872
        kwargs_706874 = {'file': keyword_706873}
        # Getting the type of 'print' (line 193)
        print_706869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'print', False)
        # Calling print(args, kwargs) (line 193)
        print_call_result_706875 = invoke(stypy.reporting.localization.Localization(__file__, 193, 12), print_706869, *[src_706870], **kwargs_706874)
        
        # SSA join for try-except statement (line 188)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 195):
        
        # Assigning a Subscript to a Name (line 195):
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 195)
        name_706876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 24), 'name')
        # Getting the type of 'evaldict' (line 195)
        evaldict_706877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 15), 'evaldict')
        # Obtaining the member '__getitem__' of a type (line 195)
        getitem___706878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 15), evaldict_706877, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 195)
        subscript_call_result_706879 = invoke(stypy.reporting.localization.Localization(__file__, 195, 15), getitem___706878, name_706876)
        
        # Assigning a type to the variable 'func' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'func', subscript_call_result_706879)
        
        # Getting the type of 'addsource' (line 196)
        addsource_706880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 11), 'addsource')
        # Testing the type of an if condition (line 196)
        if_condition_706881 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 196, 8), addsource_706880)
        # Assigning a type to the variable 'if_condition_706881' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'if_condition_706881', if_condition_706881)
        # SSA begins for if statement (line 196)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 197):
        
        # Assigning a Name to a Subscript (line 197):
        # Getting the type of 'src' (line 197)
        src_706882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 34), 'src')
        # Getting the type of 'attrs' (line 197)
        attrs_706883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'attrs')
        str_706884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 18), 'str', '__source__')
        # Storing an element on a container (line 197)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 12), attrs_706883, (str_706884, src_706882))
        # SSA join for if statement (line 196)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to update(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'func' (line 198)
        func_706887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 20), 'func', False)
        # Processing the call keyword arguments (line 198)
        # Getting the type of 'attrs' (line 198)
        attrs_706888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 28), 'attrs', False)
        kwargs_706889 = {'attrs_706888': attrs_706888}
        # Getting the type of 'self' (line 198)
        self_706885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'self', False)
        # Obtaining the member 'update' of a type (line 198)
        update_706886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), self_706885, 'update')
        # Calling update(args, kwargs) (line 198)
        update_call_result_706890 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), update_706886, *[func_706887], **kwargs_706889)
        
        # Getting the type of 'func' (line 199)
        func_706891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 15), 'func')
        # Assigning a type to the variable 'stypy_return_type' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'stypy_return_type', func_706891)
        
        # ################# End of 'make(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'make' in the type store
        # Getting the type of 'stypy_return_type' (line 168)
        stypy_return_type_706892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_706892)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'make'
        return stypy_return_type_706892


    @norecursion
    def create(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 202)
        None_706893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 50), 'None')
        # Getting the type of 'None' (line 203)
        None_706894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 19), 'None')
        # Getting the type of 'None' (line 203)
        None_706895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 32), 'None')
        # Getting the type of 'True' (line 203)
        True_706896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 48), 'True')
        defaults = [None_706893, None_706894, None_706895, True_706896]
        # Create a new context for function 'create'
        module_type_store = module_type_store.open_function_context('create', 201, 4, False)
        # Assigning a type to the variable 'self' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FunctionMaker.create.__dict__.__setitem__('stypy_localization', localization)
        FunctionMaker.create.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FunctionMaker.create.__dict__.__setitem__('stypy_type_store', module_type_store)
        FunctionMaker.create.__dict__.__setitem__('stypy_function_name', 'FunctionMaker.create')
        FunctionMaker.create.__dict__.__setitem__('stypy_param_names_list', ['obj', 'body', 'evaldict', 'defaults', 'doc', 'module', 'addsource'])
        FunctionMaker.create.__dict__.__setitem__('stypy_varargs_param_name', None)
        FunctionMaker.create.__dict__.__setitem__('stypy_kwargs_param_name', 'attrs')
        FunctionMaker.create.__dict__.__setitem__('stypy_call_defaults', defaults)
        FunctionMaker.create.__dict__.__setitem__('stypy_call_varargs', varargs)
        FunctionMaker.create.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FunctionMaker.create.__dict__.__setitem__('stypy_declared_arg_number', 8)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionMaker.create', ['obj', 'body', 'evaldict', 'defaults', 'doc', 'module', 'addsource'], None, 'attrs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create', localization, ['obj', 'body', 'evaldict', 'defaults', 'doc', 'module', 'addsource'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create(...)' code ##################

        str_706897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, (-1)), 'str', '\n        Create a function from the strings name, signature and body.\n        evaldict is the evaluation dictionary. If addsource is true an\n        attribute __source__ is added to the result. The attributes attrs\n        are added, if any.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 210)
        # Getting the type of 'str' (line 210)
        str_706898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 27), 'str')
        # Getting the type of 'obj' (line 210)
        obj_706899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 22), 'obj')
        
        (may_be_706900, more_types_in_union_706901) = may_be_subtype(str_706898, obj_706899)

        if may_be_706900:

            if more_types_in_union_706901:
                # Runtime conditional SSA (line 210)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'obj' (line 210)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'obj', remove_not_subtype_from_union(obj_706899, str))
            
            # Assigning a Call to a Tuple (line 211):
            
            # Assigning a Subscript to a Name (line 211):
            
            # Obtaining the type of the subscript
            int_706902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 12), 'int')
            
            # Call to split(...): (line 211)
            # Processing the call arguments (line 211)
            str_706908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 43), 'str', '(')
            int_706909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 48), 'int')
            # Processing the call keyword arguments (line 211)
            kwargs_706910 = {}
            
            # Call to strip(...): (line 211)
            # Processing the call keyword arguments (line 211)
            kwargs_706905 = {}
            # Getting the type of 'obj' (line 211)
            obj_706903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 25), 'obj', False)
            # Obtaining the member 'strip' of a type (line 211)
            strip_706904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 25), obj_706903, 'strip')
            # Calling strip(args, kwargs) (line 211)
            strip_call_result_706906 = invoke(stypy.reporting.localization.Localization(__file__, 211, 25), strip_706904, *[], **kwargs_706905)
            
            # Obtaining the member 'split' of a type (line 211)
            split_706907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 25), strip_call_result_706906, 'split')
            # Calling split(args, kwargs) (line 211)
            split_call_result_706911 = invoke(stypy.reporting.localization.Localization(__file__, 211, 25), split_706907, *[str_706908, int_706909], **kwargs_706910)
            
            # Obtaining the member '__getitem__' of a type (line 211)
            getitem___706912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 12), split_call_result_706911, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 211)
            subscript_call_result_706913 = invoke(stypy.reporting.localization.Localization(__file__, 211, 12), getitem___706912, int_706902)
            
            # Assigning a type to the variable 'tuple_var_assignment_706364' (line 211)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'tuple_var_assignment_706364', subscript_call_result_706913)
            
            # Assigning a Subscript to a Name (line 211):
            
            # Obtaining the type of the subscript
            int_706914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 12), 'int')
            
            # Call to split(...): (line 211)
            # Processing the call arguments (line 211)
            str_706920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 43), 'str', '(')
            int_706921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 48), 'int')
            # Processing the call keyword arguments (line 211)
            kwargs_706922 = {}
            
            # Call to strip(...): (line 211)
            # Processing the call keyword arguments (line 211)
            kwargs_706917 = {}
            # Getting the type of 'obj' (line 211)
            obj_706915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 25), 'obj', False)
            # Obtaining the member 'strip' of a type (line 211)
            strip_706916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 25), obj_706915, 'strip')
            # Calling strip(args, kwargs) (line 211)
            strip_call_result_706918 = invoke(stypy.reporting.localization.Localization(__file__, 211, 25), strip_706916, *[], **kwargs_706917)
            
            # Obtaining the member 'split' of a type (line 211)
            split_706919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 25), strip_call_result_706918, 'split')
            # Calling split(args, kwargs) (line 211)
            split_call_result_706923 = invoke(stypy.reporting.localization.Localization(__file__, 211, 25), split_706919, *[str_706920, int_706921], **kwargs_706922)
            
            # Obtaining the member '__getitem__' of a type (line 211)
            getitem___706924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 12), split_call_result_706923, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 211)
            subscript_call_result_706925 = invoke(stypy.reporting.localization.Localization(__file__, 211, 12), getitem___706924, int_706914)
            
            # Assigning a type to the variable 'tuple_var_assignment_706365' (line 211)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'tuple_var_assignment_706365', subscript_call_result_706925)
            
            # Assigning a Name to a Name (line 211):
            # Getting the type of 'tuple_var_assignment_706364' (line 211)
            tuple_var_assignment_706364_706926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'tuple_var_assignment_706364')
            # Assigning a type to the variable 'name' (line 211)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'name', tuple_var_assignment_706364_706926)
            
            # Assigning a Name to a Name (line 211):
            # Getting the type of 'tuple_var_assignment_706365' (line 211)
            tuple_var_assignment_706365_706927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'tuple_var_assignment_706365')
            # Assigning a type to the variable 'rest' (line 211)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 18), 'rest', tuple_var_assignment_706365_706927)
            
            # Assigning a Subscript to a Name (line 212):
            
            # Assigning a Subscript to a Name (line 212):
            
            # Obtaining the type of the subscript
            int_706928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 30), 'int')
            slice_706929 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 212, 24), None, int_706928, None)
            # Getting the type of 'rest' (line 212)
            rest_706930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 24), 'rest')
            # Obtaining the member '__getitem__' of a type (line 212)
            getitem___706931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 24), rest_706930, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 212)
            subscript_call_result_706932 = invoke(stypy.reporting.localization.Localization(__file__, 212, 24), getitem___706931, slice_706929)
            
            # Assigning a type to the variable 'signature' (line 212)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'signature', subscript_call_result_706932)
            
            # Assigning a Name to a Name (line 213):
            
            # Assigning a Name to a Name (line 213):
            # Getting the type of 'None' (line 213)
            None_706933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 19), 'None')
            # Assigning a type to the variable 'func' (line 213)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'func', None_706933)

            if more_types_in_union_706901:
                # Runtime conditional SSA for else branch (line 210)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_706900) or more_types_in_union_706901):
            # Assigning a type to the variable 'obj' (line 210)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'obj', remove_subtype_from_union(obj_706899, str))
            
            # Assigning a Name to a Name (line 215):
            
            # Assigning a Name to a Name (line 215):
            # Getting the type of 'None' (line 215)
            None_706934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 19), 'None')
            # Assigning a type to the variable 'name' (line 215)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'name', None_706934)
            
            # Assigning a Name to a Name (line 216):
            
            # Assigning a Name to a Name (line 216):
            # Getting the type of 'None' (line 216)
            None_706935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 24), 'None')
            # Assigning a type to the variable 'signature' (line 216)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'signature', None_706935)
            
            # Assigning a Name to a Name (line 217):
            
            # Assigning a Name to a Name (line 217):
            # Getting the type of 'obj' (line 217)
            obj_706936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 19), 'obj')
            # Assigning a type to the variable 'func' (line 217)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'func', obj_706936)

            if (may_be_706900 and more_types_in_union_706901):
                # SSA join for if statement (line 210)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 218):
        
        # Assigning a Call to a Name (line 218):
        
        # Call to cls(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'func' (line 218)
        func_706938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 19), 'func', False)
        # Getting the type of 'name' (line 218)
        name_706939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 25), 'name', False)
        # Getting the type of 'signature' (line 218)
        signature_706940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 31), 'signature', False)
        # Getting the type of 'defaults' (line 218)
        defaults_706941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 42), 'defaults', False)
        # Getting the type of 'doc' (line 218)
        doc_706942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 52), 'doc', False)
        # Getting the type of 'module' (line 218)
        module_706943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 57), 'module', False)
        # Processing the call keyword arguments (line 218)
        kwargs_706944 = {}
        # Getting the type of 'cls' (line 218)
        cls_706937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 15), 'cls', False)
        # Calling cls(args, kwargs) (line 218)
        cls_call_result_706945 = invoke(stypy.reporting.localization.Localization(__file__, 218, 15), cls_706937, *[func_706938, name_706939, signature_706940, defaults_706941, doc_706942, module_706943], **kwargs_706944)
        
        # Assigning a type to the variable 'self' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'self', cls_call_result_706945)
        
        # Assigning a Call to a Name (line 219):
        
        # Assigning a Call to a Name (line 219):
        
        # Call to join(...): (line 219)
        # Processing the call arguments (line 219)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 219, 26, True)
        # Calculating comprehension expression
        
        # Call to splitlines(...): (line 219)
        # Processing the call keyword arguments (line 219)
        kwargs_706953 = {}
        # Getting the type of 'body' (line 219)
        body_706951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 52), 'body', False)
        # Obtaining the member 'splitlines' of a type (line 219)
        splitlines_706952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 52), body_706951, 'splitlines')
        # Calling splitlines(args, kwargs) (line 219)
        splitlines_call_result_706954 = invoke(stypy.reporting.localization.Localization(__file__, 219, 52), splitlines_706952, *[], **kwargs_706953)
        
        comprehension_706955 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 26), splitlines_call_result_706954)
        # Assigning a type to the variable 'line' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 26), 'line', comprehension_706955)
        str_706948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 26), 'str', '    ')
        # Getting the type of 'line' (line 219)
        line_706949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 35), 'line', False)
        # Applying the binary operator '+' (line 219)
        result_add_706950 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 26), '+', str_706948, line_706949)
        
        list_706956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 26), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 26), list_706956, result_add_706950)
        # Processing the call keyword arguments (line 219)
        kwargs_706957 = {}
        str_706946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 16), 'str', '\n')
        # Obtaining the member 'join' of a type (line 219)
        join_706947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 16), str_706946, 'join')
        # Calling join(args, kwargs) (line 219)
        join_call_result_706958 = invoke(stypy.reporting.localization.Localization(__file__, 219, 16), join_706947, *[list_706956], **kwargs_706957)
        
        # Assigning a type to the variable 'ibody' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'ibody', join_call_result_706958)
        
        # Call to make(...): (line 220)
        # Processing the call arguments (line 220)
        str_706961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 25), 'str', 'def %(name)s(%(signature)s):\n')
        # Getting the type of 'ibody' (line 220)
        ibody_706962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 60), 'ibody', False)
        # Applying the binary operator '+' (line 220)
        result_add_706963 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 25), '+', str_706961, ibody_706962)
        
        # Getting the type of 'evaldict' (line 221)
        evaldict_706964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 25), 'evaldict', False)
        # Getting the type of 'addsource' (line 221)
        addsource_706965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 35), 'addsource', False)
        # Processing the call keyword arguments (line 220)
        # Getting the type of 'attrs' (line 221)
        attrs_706966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 48), 'attrs', False)
        kwargs_706967 = {'attrs_706966': attrs_706966}
        # Getting the type of 'self' (line 220)
        self_706959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 15), 'self', False)
        # Obtaining the member 'make' of a type (line 220)
        make_706960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 15), self_706959, 'make')
        # Calling make(args, kwargs) (line 220)
        make_call_result_706968 = invoke(stypy.reporting.localization.Localization(__file__, 220, 15), make_706960, *[result_add_706963, evaldict_706964, addsource_706965], **kwargs_706967)
        
        # Assigning a type to the variable 'stypy_return_type' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'stypy_return_type', make_call_result_706968)
        
        # ################# End of 'create(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create' in the type store
        # Getting the type of 'stypy_return_type' (line 201)
        stypy_return_type_706969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_706969)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create'
        return stypy_return_type_706969


# Assigning a type to the variable 'FunctionMaker' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'FunctionMaker', FunctionMaker)

# Assigning a Call to a Name (line 92):

# Call to count(...): (line 92)
# Processing the call keyword arguments (line 92)
kwargs_706972 = {}
# Getting the type of 'itertools' (line 92)
itertools_706970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 21), 'itertools', False)
# Obtaining the member 'count' of a type (line 92)
count_706971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 21), itertools_706970, 'count')
# Calling count(args, kwargs) (line 92)
count_call_result_706973 = invoke(stypy.reporting.localization.Localization(__file__, 92, 21), count_706971, *[], **kwargs_706972)

# Getting the type of 'FunctionMaker'
FunctionMaker_706974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FunctionMaker')
# Setting the type of the member '_compile_count' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FunctionMaker_706974, '_compile_count', count_call_result_706973)

@norecursion
def decorate(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'decorate'
    module_type_store = module_type_store.open_function_context('decorate', 224, 0, False)
    
    # Passed parameters checking function
    decorate.stypy_localization = localization
    decorate.stypy_type_of_self = None
    decorate.stypy_type_store = module_type_store
    decorate.stypy_function_name = 'decorate'
    decorate.stypy_param_names_list = ['func', 'caller']
    decorate.stypy_varargs_param_name = None
    decorate.stypy_kwargs_param_name = None
    decorate.stypy_call_defaults = defaults
    decorate.stypy_call_varargs = varargs
    decorate.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'decorate', ['func', 'caller'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'decorate', localization, ['func', 'caller'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'decorate(...)' code ##################

    str_706975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, (-1)), 'str', '\n    decorate(func, caller) decorates a function using a caller.\n    ')
    
    # Assigning a Call to a Name (line 228):
    
    # Assigning a Call to a Name (line 228):
    
    # Call to copy(...): (line 228)
    # Processing the call keyword arguments (line 228)
    kwargs_706979 = {}
    # Getting the type of 'func' (line 228)
    func_706976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 15), 'func', False)
    # Obtaining the member '__globals__' of a type (line 228)
    globals___706977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 15), func_706976, '__globals__')
    # Obtaining the member 'copy' of a type (line 228)
    copy_706978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 15), globals___706977, 'copy')
    # Calling copy(args, kwargs) (line 228)
    copy_call_result_706980 = invoke(stypy.reporting.localization.Localization(__file__, 228, 15), copy_706978, *[], **kwargs_706979)
    
    # Assigning a type to the variable 'evaldict' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'evaldict', copy_call_result_706980)
    
    # Assigning a Name to a Subscript (line 229):
    
    # Assigning a Name to a Subscript (line 229):
    # Getting the type of 'caller' (line 229)
    caller_706981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 25), 'caller')
    # Getting the type of 'evaldict' (line 229)
    evaldict_706982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'evaldict')
    str_706983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 13), 'str', '_call_')
    # Storing an element on a container (line 229)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 4), evaldict_706982, (str_706983, caller_706981))
    
    # Assigning a Name to a Subscript (line 230):
    
    # Assigning a Name to a Subscript (line 230):
    # Getting the type of 'func' (line 230)
    func_706984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 25), 'func')
    # Getting the type of 'evaldict' (line 230)
    evaldict_706985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'evaldict')
    str_706986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 13), 'str', '_func_')
    # Storing an element on a container (line 230)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 4), evaldict_706985, (str_706986, func_706984))
    
    # Assigning a Call to a Name (line 231):
    
    # Assigning a Call to a Name (line 231):
    
    # Call to create(...): (line 231)
    # Processing the call arguments (line 231)
    # Getting the type of 'func' (line 232)
    func_706989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'func', False)
    str_706990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 14), 'str', 'return _call_(_func_, %(shortsignature)s)')
    # Getting the type of 'evaldict' (line 233)
    evaldict_706991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'evaldict', False)
    # Processing the call keyword arguments (line 231)
    # Getting the type of 'func' (line 233)
    func_706992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 30), 'func', False)
    keyword_706993 = func_706992
    kwargs_706994 = {'__wrapped__': keyword_706993}
    # Getting the type of 'FunctionMaker' (line 231)
    FunctionMaker_706987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 10), 'FunctionMaker', False)
    # Obtaining the member 'create' of a type (line 231)
    create_706988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 10), FunctionMaker_706987, 'create')
    # Calling create(args, kwargs) (line 231)
    create_call_result_706995 = invoke(stypy.reporting.localization.Localization(__file__, 231, 10), create_706988, *[func_706989, str_706990, evaldict_706991], **kwargs_706994)
    
    # Assigning a type to the variable 'fun' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'fun', create_call_result_706995)
    
    # Type idiom detected: calculating its left and rigth part (line 234)
    str_706996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 21), 'str', '__qualname__')
    # Getting the type of 'func' (line 234)
    func_706997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 15), 'func')
    
    (may_be_706998, more_types_in_union_706999) = may_provide_member(str_706996, func_706997)

    if may_be_706998:

        if more_types_in_union_706999:
            # Runtime conditional SSA (line 234)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'func' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'func', remove_not_member_provider_from_union(func_706997, '__qualname__'))
        
        # Assigning a Attribute to a Attribute (line 235):
        
        # Assigning a Attribute to a Attribute (line 235):
        # Getting the type of 'func' (line 235)
        func_707000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 27), 'func')
        # Obtaining the member '__qualname__' of a type (line 235)
        qualname___707001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 27), func_707000, '__qualname__')
        # Getting the type of 'fun' (line 235)
        fun_707002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'fun')
        # Setting the type of the member '__qualname__' of a type (line 235)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), fun_707002, '__qualname__', qualname___707001)

        if more_types_in_union_706999:
            # SSA join for if statement (line 234)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'fun' (line 236)
    fun_707003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 11), 'fun')
    # Assigning a type to the variable 'stypy_return_type' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'stypy_return_type', fun_707003)
    
    # ################# End of 'decorate(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'decorate' in the type store
    # Getting the type of 'stypy_return_type' (line 224)
    stypy_return_type_707004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_707004)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'decorate'
    return stypy_return_type_707004

# Assigning a type to the variable 'decorate' (line 224)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 0), 'decorate', decorate)

@norecursion
def decorator(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 239)
    None_707005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 28), 'None')
    defaults = [None_707005]
    # Create a new context for function 'decorator'
    module_type_store = module_type_store.open_function_context('decorator', 239, 0, False)
    
    # Passed parameters checking function
    decorator.stypy_localization = localization
    decorator.stypy_type_of_self = None
    decorator.stypy_type_store = module_type_store
    decorator.stypy_function_name = 'decorator'
    decorator.stypy_param_names_list = ['caller', '_func']
    decorator.stypy_varargs_param_name = None
    decorator.stypy_kwargs_param_name = None
    decorator.stypy_call_defaults = defaults
    decorator.stypy_call_varargs = varargs
    decorator.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'decorator', ['caller', '_func'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'decorator', localization, ['caller', '_func'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'decorator(...)' code ##################

    str_707006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 4), 'str', 'decorator(caller) converts a caller function into a decorator')
    
    # Type idiom detected: calculating its left and rigth part (line 241)
    # Getting the type of '_func' (line 241)
    _func_707007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), '_func')
    # Getting the type of 'None' (line 241)
    None_707008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 20), 'None')
    
    (may_be_707009, more_types_in_union_707010) = may_not_be_none(_func_707007, None_707008)

    if may_be_707009:

        if more_types_in_union_707010:
            # Runtime conditional SSA (line 241)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to decorate(...): (line 243)
        # Processing the call arguments (line 243)
        # Getting the type of '_func' (line 243)
        _func_707012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 24), '_func', False)
        # Getting the type of 'caller' (line 243)
        caller_707013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 31), 'caller', False)
        # Processing the call keyword arguments (line 243)
        kwargs_707014 = {}
        # Getting the type of 'decorate' (line 243)
        decorate_707011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 15), 'decorate', False)
        # Calling decorate(args, kwargs) (line 243)
        decorate_call_result_707015 = invoke(stypy.reporting.localization.Localization(__file__, 243, 15), decorate_707011, *[_func_707012, caller_707013], **kwargs_707014)
        
        # Assigning a type to the variable 'stypy_return_type' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'stypy_return_type', decorate_call_result_707015)

        if more_types_in_union_707010:
            # SSA join for if statement (line 241)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to isclass(...): (line 245)
    # Processing the call arguments (line 245)
    # Getting the type of 'caller' (line 245)
    caller_707018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 23), 'caller', False)
    # Processing the call keyword arguments (line 245)
    kwargs_707019 = {}
    # Getting the type of 'inspect' (line 245)
    inspect_707016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 7), 'inspect', False)
    # Obtaining the member 'isclass' of a type (line 245)
    isclass_707017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 7), inspect_707016, 'isclass')
    # Calling isclass(args, kwargs) (line 245)
    isclass_call_result_707020 = invoke(stypy.reporting.localization.Localization(__file__, 245, 7), isclass_707017, *[caller_707018], **kwargs_707019)
    
    # Testing the type of an if condition (line 245)
    if_condition_707021 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 245, 4), isclass_call_result_707020)
    # Assigning a type to the variable 'if_condition_707021' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'if_condition_707021', if_condition_707021)
    # SSA begins for if statement (line 245)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 246):
    
    # Assigning a Call to a Name (line 246):
    
    # Call to lower(...): (line 246)
    # Processing the call keyword arguments (line 246)
    kwargs_707025 = {}
    # Getting the type of 'caller' (line 246)
    caller_707022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 15), 'caller', False)
    # Obtaining the member '__name__' of a type (line 246)
    name___707023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 15), caller_707022, '__name__')
    # Obtaining the member 'lower' of a type (line 246)
    lower_707024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 15), name___707023, 'lower')
    # Calling lower(args, kwargs) (line 246)
    lower_call_result_707026 = invoke(stypy.reporting.localization.Localization(__file__, 246, 15), lower_707024, *[], **kwargs_707025)
    
    # Assigning a type to the variable 'name' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'name', lower_call_result_707026)
    
    # Assigning a Call to a Name (line 247):
    
    # Assigning a Call to a Name (line 247):
    
    # Call to get_init(...): (line 247)
    # Processing the call arguments (line 247)
    # Getting the type of 'caller' (line 247)
    caller_707028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 30), 'caller', False)
    # Processing the call keyword arguments (line 247)
    kwargs_707029 = {}
    # Getting the type of 'get_init' (line 247)
    get_init_707027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 21), 'get_init', False)
    # Calling get_init(args, kwargs) (line 247)
    get_init_call_result_707030 = invoke(stypy.reporting.localization.Localization(__file__, 247, 21), get_init_707027, *[caller_707028], **kwargs_707029)
    
    # Assigning a type to the variable 'callerfunc' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'callerfunc', get_init_call_result_707030)
    
    # Assigning a BinOp to a Name (line 248):
    
    # Assigning a BinOp to a Name (line 248):
    str_707031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 14), 'str', 'decorator(%s) converts functions/generators into factories of %s objects')
    
    # Obtaining an instance of the builtin type 'tuple' (line 249)
    tuple_707032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 249)
    # Adding element type (line 249)
    # Getting the type of 'caller' (line 249)
    caller_707033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 41), 'caller')
    # Obtaining the member '__name__' of a type (line 249)
    name___707034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 41), caller_707033, '__name__')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 41), tuple_707032, name___707034)
    # Adding element type (line 249)
    # Getting the type of 'caller' (line 249)
    caller_707035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 58), 'caller')
    # Obtaining the member '__name__' of a type (line 249)
    name___707036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 58), caller_707035, '__name__')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 41), tuple_707032, name___707036)
    
    # Applying the binary operator '%' (line 248)
    result_mod_707037 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 14), '%', str_707031, tuple_707032)
    
    # Assigning a type to the variable 'doc' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'doc', result_mod_707037)
    # SSA branch for the else part of an if statement (line 245)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isfunction(...): (line 250)
    # Processing the call arguments (line 250)
    # Getting the type of 'caller' (line 250)
    caller_707040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 28), 'caller', False)
    # Processing the call keyword arguments (line 250)
    kwargs_707041 = {}
    # Getting the type of 'inspect' (line 250)
    inspect_707038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 9), 'inspect', False)
    # Obtaining the member 'isfunction' of a type (line 250)
    isfunction_707039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 9), inspect_707038, 'isfunction')
    # Calling isfunction(args, kwargs) (line 250)
    isfunction_call_result_707042 = invoke(stypy.reporting.localization.Localization(__file__, 250, 9), isfunction_707039, *[caller_707040], **kwargs_707041)
    
    # Testing the type of an if condition (line 250)
    if_condition_707043 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 250, 9), isfunction_call_result_707042)
    # Assigning a type to the variable 'if_condition_707043' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 9), 'if_condition_707043', if_condition_707043)
    # SSA begins for if statement (line 250)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'caller' (line 251)
    caller_707044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 11), 'caller')
    # Obtaining the member '__name__' of a type (line 251)
    name___707045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 11), caller_707044, '__name__')
    str_707046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 30), 'str', '<lambda>')
    # Applying the binary operator '==' (line 251)
    result_eq_707047 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 11), '==', name___707045, str_707046)
    
    # Testing the type of an if condition (line 251)
    if_condition_707048 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 251, 8), result_eq_707047)
    # Assigning a type to the variable 'if_condition_707048' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'if_condition_707048', if_condition_707048)
    # SSA begins for if statement (line 251)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 252):
    
    # Assigning a Str to a Name (line 252):
    str_707049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 19), 'str', '_lambda_')
    # Assigning a type to the variable 'name' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'name', str_707049)
    # SSA branch for the else part of an if statement (line 251)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Attribute to a Name (line 254):
    
    # Assigning a Attribute to a Name (line 254):
    # Getting the type of 'caller' (line 254)
    caller_707050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 19), 'caller')
    # Obtaining the member '__name__' of a type (line 254)
    name___707051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 19), caller_707050, '__name__')
    # Assigning a type to the variable 'name' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'name', name___707051)
    # SSA join for if statement (line 251)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 255):
    
    # Assigning a Name to a Name (line 255):
    # Getting the type of 'caller' (line 255)
    caller_707052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 21), 'caller')
    # Assigning a type to the variable 'callerfunc' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'callerfunc', caller_707052)
    
    # Assigning a Attribute to a Name (line 256):
    
    # Assigning a Attribute to a Name (line 256):
    # Getting the type of 'caller' (line 256)
    caller_707053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 14), 'caller')
    # Obtaining the member '__doc__' of a type (line 256)
    doc___707054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 14), caller_707053, '__doc__')
    # Assigning a type to the variable 'doc' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'doc', doc___707054)
    # SSA branch for the else part of an if statement (line 250)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 258):
    
    # Assigning a Call to a Name (line 258):
    
    # Call to lower(...): (line 258)
    # Processing the call keyword arguments (line 258)
    kwargs_707059 = {}
    # Getting the type of 'caller' (line 258)
    caller_707055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 15), 'caller', False)
    # Obtaining the member '__class__' of a type (line 258)
    class___707056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 15), caller_707055, '__class__')
    # Obtaining the member '__name__' of a type (line 258)
    name___707057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 15), class___707056, '__name__')
    # Obtaining the member 'lower' of a type (line 258)
    lower_707058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 15), name___707057, 'lower')
    # Calling lower(args, kwargs) (line 258)
    lower_call_result_707060 = invoke(stypy.reporting.localization.Localization(__file__, 258, 15), lower_707058, *[], **kwargs_707059)
    
    # Assigning a type to the variable 'name' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'name', lower_call_result_707060)
    
    # Assigning a Attribute to a Name (line 259):
    
    # Assigning a Attribute to a Name (line 259):
    # Getting the type of 'caller' (line 259)
    caller_707061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 21), 'caller')
    # Obtaining the member '__call__' of a type (line 259)
    call___707062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 21), caller_707061, '__call__')
    # Obtaining the member '__func__' of a type (line 259)
    func___707063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 21), call___707062, '__func__')
    # Assigning a type to the variable 'callerfunc' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'callerfunc', func___707063)
    
    # Assigning a Attribute to a Name (line 260):
    
    # Assigning a Attribute to a Name (line 260):
    # Getting the type of 'caller' (line 260)
    caller_707064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 14), 'caller')
    # Obtaining the member '__call__' of a type (line 260)
    call___707065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 14), caller_707064, '__call__')
    # Obtaining the member '__doc__' of a type (line 260)
    doc___707066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 14), call___707065, '__doc__')
    # Assigning a type to the variable 'doc' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'doc', doc___707066)
    # SSA join for if statement (line 250)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 245)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 261):
    
    # Assigning a Call to a Name (line 261):
    
    # Call to copy(...): (line 261)
    # Processing the call keyword arguments (line 261)
    kwargs_707070 = {}
    # Getting the type of 'callerfunc' (line 261)
    callerfunc_707067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 15), 'callerfunc', False)
    # Obtaining the member '__globals__' of a type (line 261)
    globals___707068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 15), callerfunc_707067, '__globals__')
    # Obtaining the member 'copy' of a type (line 261)
    copy_707069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 15), globals___707068, 'copy')
    # Calling copy(args, kwargs) (line 261)
    copy_call_result_707071 = invoke(stypy.reporting.localization.Localization(__file__, 261, 15), copy_707069, *[], **kwargs_707070)
    
    # Assigning a type to the variable 'evaldict' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'evaldict', copy_call_result_707071)
    
    # Assigning a Name to a Subscript (line 262):
    
    # Assigning a Name to a Subscript (line 262):
    # Getting the type of 'caller' (line 262)
    caller_707072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 25), 'caller')
    # Getting the type of 'evaldict' (line 262)
    evaldict_707073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'evaldict')
    str_707074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 13), 'str', '_call_')
    # Storing an element on a container (line 262)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 4), evaldict_707073, (str_707074, caller_707072))
    
    # Assigning a Name to a Subscript (line 263):
    
    # Assigning a Name to a Subscript (line 263):
    # Getting the type of 'decorate' (line 263)
    decorate_707075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 29), 'decorate')
    # Getting the type of 'evaldict' (line 263)
    evaldict_707076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'evaldict')
    str_707077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 13), 'str', '_decorate_')
    # Storing an element on a container (line 263)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 4), evaldict_707076, (str_707077, decorate_707075))
    
    # Call to create(...): (line 264)
    # Processing the call arguments (line 264)
    str_707080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 8), 'str', '%s(func)')
    # Getting the type of 'name' (line 265)
    name_707081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 21), 'name', False)
    # Applying the binary operator '%' (line 265)
    result_mod_707082 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 8), '%', str_707080, name_707081)
    
    str_707083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 27), 'str', 'return _decorate_(func, _call_)')
    # Getting the type of 'evaldict' (line 266)
    evaldict_707084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'evaldict', False)
    # Processing the call keyword arguments (line 264)
    # Getting the type of 'doc' (line 266)
    doc_707085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 22), 'doc', False)
    keyword_707086 = doc_707085
    # Getting the type of 'caller' (line 266)
    caller_707087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 34), 'caller', False)
    # Obtaining the member '__module__' of a type (line 266)
    module___707088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 34), caller_707087, '__module__')
    keyword_707089 = module___707088
    # Getting the type of 'caller' (line 267)
    caller_707090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 20), 'caller', False)
    keyword_707091 = caller_707090
    kwargs_707092 = {'doc': keyword_707086, '__wrapped__': keyword_707091, 'module': keyword_707089}
    # Getting the type of 'FunctionMaker' (line 264)
    FunctionMaker_707078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 11), 'FunctionMaker', False)
    # Obtaining the member 'create' of a type (line 264)
    create_707079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 11), FunctionMaker_707078, 'create')
    # Calling create(args, kwargs) (line 264)
    create_call_result_707093 = invoke(stypy.reporting.localization.Localization(__file__, 264, 11), create_707079, *[result_mod_707082, str_707083, evaldict_707084], **kwargs_707092)
    
    # Assigning a type to the variable 'stypy_return_type' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'stypy_return_type', create_call_result_707093)
    
    # ################# End of 'decorator(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'decorator' in the type store
    # Getting the type of 'stypy_return_type' (line 239)
    stypy_return_type_707094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_707094)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'decorator'
    return stypy_return_type_707094

# Assigning a type to the variable 'decorator' (line 239)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 0), 'decorator', decorator)


# SSA begins for try-except statement (line 272)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 273, 4))

# 'from contextlib import _GeneratorContextManager' statement (line 273)
try:
    from contextlib import _GeneratorContextManager

except:
    _GeneratorContextManager = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 273, 4), 'contextlib', None, module_type_store, ['_GeneratorContextManager'], [_GeneratorContextManager])

# SSA branch for the except part of a try statement (line 272)
# SSA branch for the except 'ImportError' branch of a try statement (line 272)
module_type_store.open_ssa_branch('except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 275, 4))

# 'from contextlib import _GeneratorContextManager' statement (line 275)
try:
    from contextlib import GeneratorContextManager as _GeneratorContextManager

except:
    _GeneratorContextManager = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 275, 4), 'contextlib', None, module_type_store, ['GeneratorContextManager'], [_GeneratorContextManager])
# Adding an alias
module_type_store.add_alias('_GeneratorContextManager', 'GeneratorContextManager')

# SSA join for try-except statement (line 272)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'ContextManager' class
# Getting the type of '_GeneratorContextManager' (line 278)
_GeneratorContextManager_707095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 21), '_GeneratorContextManager')

class ContextManager(_GeneratorContextManager_707095, ):

    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 279, 4, False)
        # Assigning a type to the variable 'self' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ContextManager.__call__.__dict__.__setitem__('stypy_localization', localization)
        ContextManager.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ContextManager.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ContextManager.__call__.__dict__.__setitem__('stypy_function_name', 'ContextManager.__call__')
        ContextManager.__call__.__dict__.__setitem__('stypy_param_names_list', ['func'])
        ContextManager.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ContextManager.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ContextManager.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ContextManager.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ContextManager.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ContextManager.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ContextManager.__call__', ['func'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['func'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        str_707096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 8), 'str', 'Context manager decorator')
        
        # Call to create(...): (line 281)
        # Processing the call arguments (line 281)
        # Getting the type of 'func' (line 282)
        func_707099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'func', False)
        str_707100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 18), 'str', 'with _self_: return _func_(%(shortsignature)s)')
        
        # Call to dict(...): (line 283)
        # Processing the call keyword arguments (line 283)
        # Getting the type of 'self' (line 283)
        self_707102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 24), 'self', False)
        keyword_707103 = self_707102
        # Getting the type of 'func' (line 283)
        func_707104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 37), 'func', False)
        keyword_707105 = func_707104
        kwargs_707106 = {'_func_': keyword_707105, '_self_': keyword_707103}
        # Getting the type of 'dict' (line 283)
        dict_707101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'dict', False)
        # Calling dict(args, kwargs) (line 283)
        dict_call_result_707107 = invoke(stypy.reporting.localization.Localization(__file__, 283, 12), dict_707101, *[], **kwargs_707106)
        
        # Processing the call keyword arguments (line 281)
        # Getting the type of 'func' (line 283)
        func_707108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 56), 'func', False)
        keyword_707109 = func_707108
        kwargs_707110 = {'__wrapped__': keyword_707109}
        # Getting the type of 'FunctionMaker' (line 281)
        FunctionMaker_707097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 15), 'FunctionMaker', False)
        # Obtaining the member 'create' of a type (line 281)
        create_707098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 15), FunctionMaker_707097, 'create')
        # Calling create(args, kwargs) (line 281)
        create_call_result_707111 = invoke(stypy.reporting.localization.Localization(__file__, 281, 15), create_707098, *[func_707099, str_707100, dict_call_result_707107], **kwargs_707110)
        
        # Assigning a type to the variable 'stypy_return_type' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'stypy_return_type', create_call_result_707111)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 279)
        stypy_return_type_707112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_707112)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_707112


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 278, 0, False)
        # Assigning a type to the variable 'self' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ContextManager.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ContextManager' (line 278)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 0), 'ContextManager', ContextManager)

# Assigning a Call to a Name (line 285):

# Assigning a Call to a Name (line 285):

# Call to getfullargspec(...): (line 285)
# Processing the call arguments (line 285)
# Getting the type of '_GeneratorContextManager' (line 285)
_GeneratorContextManager_707114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 22), '_GeneratorContextManager', False)
# Obtaining the member '__init__' of a type (line 285)
init___707115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 22), _GeneratorContextManager_707114, '__init__')
# Processing the call keyword arguments (line 285)
kwargs_707116 = {}
# Getting the type of 'getfullargspec' (line 285)
getfullargspec_707113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 7), 'getfullargspec', False)
# Calling getfullargspec(args, kwargs) (line 285)
getfullargspec_call_result_707117 = invoke(stypy.reporting.localization.Localization(__file__, 285, 7), getfullargspec_707113, *[init___707115], **kwargs_707116)

# Assigning a type to the variable 'init' (line 285)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 0), 'init', getfullargspec_call_result_707117)

# Assigning a Call to a Name (line 286):

# Assigning a Call to a Name (line 286):

# Call to len(...): (line 286)
# Processing the call arguments (line 286)
# Getting the type of 'init' (line 286)
init_707119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 13), 'init', False)
# Obtaining the member 'args' of a type (line 286)
args_707120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 13), init_707119, 'args')
# Processing the call keyword arguments (line 286)
kwargs_707121 = {}
# Getting the type of 'len' (line 286)
len_707118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 9), 'len', False)
# Calling len(args, kwargs) (line 286)
len_call_result_707122 = invoke(stypy.reporting.localization.Localization(__file__, 286, 9), len_707118, *[args_707120], **kwargs_707121)

# Assigning a type to the variable 'n_args' (line 286)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 0), 'n_args', len_call_result_707122)


# Evaluating a boolean operation

# Getting the type of 'n_args' (line 287)
n_args_707123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 3), 'n_args')
int_707124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 13), 'int')
# Applying the binary operator '==' (line 287)
result_eq_707125 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 3), '==', n_args_707123, int_707124)


# Getting the type of 'init' (line 287)
init_707126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 23), 'init')
# Obtaining the member 'varargs' of a type (line 287)
varargs_707127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 23), init_707126, 'varargs')
# Applying the 'not' unary operator (line 287)
result_not__707128 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 19), 'not', varargs_707127)

# Applying the binary operator 'and' (line 287)
result_and_keyword_707129 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 3), 'and', result_eq_707125, result_not__707128)

# Testing the type of an if condition (line 287)
if_condition_707130 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 287, 0), result_and_keyword_707129)
# Assigning a type to the variable 'if_condition_707130' (line 287)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 0), 'if_condition_707130', if_condition_707130)
# SSA begins for if statement (line 287)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

@norecursion
def __init__(type_of_self, localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__init__'
    module_type_store = module_type_store.open_function_context('__init__', 288, 4, False)
    
    # Passed parameters checking function
    arguments = process_argument_values(localization, None, module_type_store, '__init__', ['self', 'g'], 'a', 'k', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return

    # Initialize method data
    init_call_information(module_type_store, '__init__', localization, ['self', 'g'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__init__(...)' code ##################

    
    # Call to __init__(...): (line 289)
    # Processing the call arguments (line 289)
    # Getting the type of 'self' (line 289)
    self_707133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 49), 'self', False)
    
    # Call to g(...): (line 289)
    # Getting the type of 'a' (line 289)
    a_707135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 58), 'a', False)
    # Processing the call keyword arguments (line 289)
    # Getting the type of 'k' (line 289)
    k_707136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 63), 'k', False)
    kwargs_707137 = {'k_707136': k_707136}
    # Getting the type of 'g' (line 289)
    g_707134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 55), 'g', False)
    # Calling g(args, kwargs) (line 289)
    g_call_result_707138 = invoke(stypy.reporting.localization.Localization(__file__, 289, 55), g_707134, *[a_707135], **kwargs_707137)
    
    # Processing the call keyword arguments (line 289)
    kwargs_707139 = {}
    # Getting the type of '_GeneratorContextManager' (line 289)
    _GeneratorContextManager_707131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 15), '_GeneratorContextManager', False)
    # Obtaining the member '__init__' of a type (line 289)
    init___707132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 15), _GeneratorContextManager_707131, '__init__')
    # Calling __init__(args, kwargs) (line 289)
    init___call_result_707140 = invoke(stypy.reporting.localization.Localization(__file__, 289, 15), init___707132, *[self_707133, g_call_result_707138], **kwargs_707139)
    
    # Assigning a type to the variable 'stypy_return_type' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'stypy_return_type', init___call_result_707140)
    
    # ################# End of '__init__(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()

# Assigning a type to the variable '__init__' (line 288)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), '__init__', __init__)

# Assigning a Name to a Attribute (line 290):

# Assigning a Name to a Attribute (line 290):
# Getting the type of '__init__' (line 290)
init___707141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 30), '__init__')
# Getting the type of 'ContextManager' (line 290)
ContextManager_707142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'ContextManager')
# Setting the type of the member '__init__' of a type (line 290)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 4), ContextManager_707142, '__init__', init___707141)
# SSA branch for the else part of an if statement (line 287)
module_type_store.open_ssa_branch('else')


# Evaluating a boolean operation

# Getting the type of 'n_args' (line 291)
n_args_707143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 5), 'n_args')
int_707144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 15), 'int')
# Applying the binary operator '==' (line 291)
result_eq_707145 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 5), '==', n_args_707143, int_707144)

# Getting the type of 'init' (line 291)
init_707146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 21), 'init')
# Obtaining the member 'varargs' of a type (line 291)
varargs_707147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 21), init_707146, 'varargs')
# Applying the binary operator 'and' (line 291)
result_and_keyword_707148 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 5), 'and', result_eq_707145, varargs_707147)

# Testing the type of an if condition (line 291)
if_condition_707149 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 291, 5), result_and_keyword_707148)
# Assigning a type to the variable 'if_condition_707149' (line 291)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 5), 'if_condition_707149', if_condition_707149)
# SSA begins for if statement (line 291)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
pass
# SSA branch for the else part of an if statement (line 291)
module_type_store.open_ssa_branch('else')


# Getting the type of 'n_args' (line 293)
n_args_707150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 5), 'n_args')
int_707151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 15), 'int')
# Applying the binary operator '==' (line 293)
result_eq_707152 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 5), '==', n_args_707150, int_707151)

# Testing the type of an if condition (line 293)
if_condition_707153 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 293, 5), result_eq_707152)
# Assigning a type to the variable 'if_condition_707153' (line 293)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 5), 'if_condition_707153', if_condition_707153)
# SSA begins for if statement (line 293)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

@norecursion
def __init__(type_of_self, localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__init__'
    module_type_store = module_type_store.open_function_context('__init__', 294, 4, False)
    
    # Passed parameters checking function
    arguments = process_argument_values(localization, None, module_type_store, '__init__', ['self', 'g'], 'a', 'k', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return

    # Initialize method data
    init_call_information(module_type_store, '__init__', localization, ['self', 'g'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__init__(...)' code ##################

    
    # Call to __init__(...): (line 295)
    # Processing the call arguments (line 295)
    # Getting the type of 'self' (line 295)
    self_707156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 49), 'self', False)
    # Getting the type of 'g' (line 295)
    g_707157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 55), 'g', False)
    # Getting the type of 'a' (line 295)
    a_707158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 58), 'a', False)
    # Getting the type of 'k' (line 295)
    k_707159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 61), 'k', False)
    # Processing the call keyword arguments (line 295)
    kwargs_707160 = {}
    # Getting the type of '_GeneratorContextManager' (line 295)
    _GeneratorContextManager_707154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 15), '_GeneratorContextManager', False)
    # Obtaining the member '__init__' of a type (line 295)
    init___707155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 15), _GeneratorContextManager_707154, '__init__')
    # Calling __init__(args, kwargs) (line 295)
    init___call_result_707161 = invoke(stypy.reporting.localization.Localization(__file__, 295, 15), init___707155, *[self_707156, g_707157, a_707158, k_707159], **kwargs_707160)
    
    # Assigning a type to the variable 'stypy_return_type' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'stypy_return_type', init___call_result_707161)
    
    # ################# End of '__init__(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()

# Assigning a type to the variable '__init__' (line 294)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), '__init__', __init__)

# Assigning a Name to a Attribute (line 296):

# Assigning a Name to a Attribute (line 296):
# Getting the type of '__init__' (line 296)
init___707162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 30), '__init__')
# Getting the type of 'ContextManager' (line 296)
ContextManager_707163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 'ContextManager')
# Setting the type of the member '__init__' of a type (line 296)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 4), ContextManager_707163, '__init__', init___707162)
# SSA join for if statement (line 293)
module_type_store = module_type_store.join_ssa_context()

# SSA join for if statement (line 291)
module_type_store = module_type_store.join_ssa_context()

# SSA join for if statement (line 287)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Call to a Name (line 298):

# Assigning a Call to a Name (line 298):

# Call to decorator(...): (line 298)
# Processing the call arguments (line 298)
# Getting the type of 'ContextManager' (line 298)
ContextManager_707165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 27), 'ContextManager', False)
# Processing the call keyword arguments (line 298)
kwargs_707166 = {}
# Getting the type of 'decorator' (line 298)
decorator_707164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 17), 'decorator', False)
# Calling decorator(args, kwargs) (line 298)
decorator_call_result_707167 = invoke(stypy.reporting.localization.Localization(__file__, 298, 17), decorator_707164, *[ContextManager_707165], **kwargs_707166)

# Assigning a type to the variable 'contextmanager' (line 298)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 0), 'contextmanager', decorator_call_result_707167)

@norecursion
def append(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'append'
    module_type_store = module_type_store.open_function_context('append', 303, 0, False)
    
    # Passed parameters checking function
    append.stypy_localization = localization
    append.stypy_type_of_self = None
    append.stypy_type_store = module_type_store
    append.stypy_function_name = 'append'
    append.stypy_param_names_list = ['a', 'vancestors']
    append.stypy_varargs_param_name = None
    append.stypy_kwargs_param_name = None
    append.stypy_call_defaults = defaults
    append.stypy_call_varargs = varargs
    append.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'append', ['a', 'vancestors'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'append', localization, ['a', 'vancestors'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'append(...)' code ##################

    str_707168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, (-1)), 'str', '\n    Append ``a`` to the list of the virtual ancestors, unless it is already\n    included.\n    ')
    
    # Assigning a Name to a Name (line 308):
    
    # Assigning a Name to a Name (line 308):
    # Getting the type of 'True' (line 308)
    True_707169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 10), 'True')
    # Assigning a type to the variable 'add' (line 308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'add', True_707169)
    
    
    # Call to enumerate(...): (line 309)
    # Processing the call arguments (line 309)
    # Getting the type of 'vancestors' (line 309)
    vancestors_707171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 27), 'vancestors', False)
    # Processing the call keyword arguments (line 309)
    kwargs_707172 = {}
    # Getting the type of 'enumerate' (line 309)
    enumerate_707170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 17), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 309)
    enumerate_call_result_707173 = invoke(stypy.reporting.localization.Localization(__file__, 309, 17), enumerate_707170, *[vancestors_707171], **kwargs_707172)
    
    # Testing the type of a for loop iterable (line 309)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 309, 4), enumerate_call_result_707173)
    # Getting the type of the for loop variable (line 309)
    for_loop_var_707174 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 309, 4), enumerate_call_result_707173)
    # Assigning a type to the variable 'j' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 4), for_loop_var_707174))
    # Assigning a type to the variable 'va' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'va', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 4), for_loop_var_707174))
    # SSA begins for a for statement (line 309)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to issubclass(...): (line 310)
    # Processing the call arguments (line 310)
    # Getting the type of 'va' (line 310)
    va_707176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 22), 'va', False)
    # Getting the type of 'a' (line 310)
    a_707177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 26), 'a', False)
    # Processing the call keyword arguments (line 310)
    kwargs_707178 = {}
    # Getting the type of 'issubclass' (line 310)
    issubclass_707175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 11), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 310)
    issubclass_call_result_707179 = invoke(stypy.reporting.localization.Localization(__file__, 310, 11), issubclass_707175, *[va_707176, a_707177], **kwargs_707178)
    
    # Testing the type of an if condition (line 310)
    if_condition_707180 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 310, 8), issubclass_call_result_707179)
    # Assigning a type to the variable 'if_condition_707180' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'if_condition_707180', if_condition_707180)
    # SSA begins for if statement (line 310)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 311):
    
    # Assigning a Name to a Name (line 311):
    # Getting the type of 'False' (line 311)
    False_707181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 18), 'False')
    # Assigning a type to the variable 'add' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'add', False_707181)
    # SSA join for if statement (line 310)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to issubclass(...): (line 313)
    # Processing the call arguments (line 313)
    # Getting the type of 'a' (line 313)
    a_707183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 22), 'a', False)
    # Getting the type of 'va' (line 313)
    va_707184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 25), 'va', False)
    # Processing the call keyword arguments (line 313)
    kwargs_707185 = {}
    # Getting the type of 'issubclass' (line 313)
    issubclass_707182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 11), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 313)
    issubclass_call_result_707186 = invoke(stypy.reporting.localization.Localization(__file__, 313, 11), issubclass_707182, *[a_707183, va_707184], **kwargs_707185)
    
    # Testing the type of an if condition (line 313)
    if_condition_707187 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 313, 8), issubclass_call_result_707186)
    # Assigning a type to the variable 'if_condition_707187' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'if_condition_707187', if_condition_707187)
    # SSA begins for if statement (line 313)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 314):
    
    # Assigning a Name to a Subscript (line 314):
    # Getting the type of 'a' (line 314)
    a_707188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 28), 'a')
    # Getting the type of 'vancestors' (line 314)
    vancestors_707189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'vancestors')
    # Getting the type of 'j' (line 314)
    j_707190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 23), 'j')
    # Storing an element on a container (line 314)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 12), vancestors_707189, (j_707190, a_707188))
    
    # Assigning a Name to a Name (line 315):
    
    # Assigning a Name to a Name (line 315):
    # Getting the type of 'False' (line 315)
    False_707191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 18), 'False')
    # Assigning a type to the variable 'add' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'add', False_707191)
    # SSA join for if statement (line 313)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'add' (line 316)
    add_707192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 7), 'add')
    # Testing the type of an if condition (line 316)
    if_condition_707193 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 316, 4), add_707192)
    # Assigning a type to the variable 'if_condition_707193' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'if_condition_707193', if_condition_707193)
    # SSA begins for if statement (line 316)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 317)
    # Processing the call arguments (line 317)
    # Getting the type of 'a' (line 317)
    a_707196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 26), 'a', False)
    # Processing the call keyword arguments (line 317)
    kwargs_707197 = {}
    # Getting the type of 'vancestors' (line 317)
    vancestors_707194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'vancestors', False)
    # Obtaining the member 'append' of a type (line 317)
    append_707195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 8), vancestors_707194, 'append')
    # Calling append(args, kwargs) (line 317)
    append_call_result_707198 = invoke(stypy.reporting.localization.Localization(__file__, 317, 8), append_707195, *[a_707196], **kwargs_707197)
    
    # SSA join for if statement (line 316)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'append(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'append' in the type store
    # Getting the type of 'stypy_return_type' (line 303)
    stypy_return_type_707199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_707199)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'append'
    return stypy_return_type_707199

# Assigning a type to the variable 'append' (line 303)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 0), 'append', append)

@norecursion
def dispatch_on(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'dispatch_on'
    module_type_store = module_type_store.open_function_context('dispatch_on', 321, 0, False)
    
    # Passed parameters checking function
    dispatch_on.stypy_localization = localization
    dispatch_on.stypy_type_of_self = None
    dispatch_on.stypy_type_store = module_type_store
    dispatch_on.stypy_function_name = 'dispatch_on'
    dispatch_on.stypy_param_names_list = []
    dispatch_on.stypy_varargs_param_name = 'dispatch_args'
    dispatch_on.stypy_kwargs_param_name = None
    dispatch_on.stypy_call_defaults = defaults
    dispatch_on.stypy_call_varargs = varargs
    dispatch_on.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dispatch_on', [], 'dispatch_args', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dispatch_on', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dispatch_on(...)' code ##################

    str_707200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, (-1)), 'str', '\n    Factory of decorators turning a function into a generic function\n    dispatching on the given arguments.\n    ')
    # Evaluating assert statement condition
    # Getting the type of 'dispatch_args' (line 326)
    dispatch_args_707201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 11), 'dispatch_args')
    
    # Assigning a BinOp to a Name (line 327):
    
    # Assigning a BinOp to a Name (line 327):
    str_707202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 19), 'str', '(%s,)')
    
    # Call to join(...): (line 327)
    # Processing the call arguments (line 327)
    # Getting the type of 'dispatch_args' (line 327)
    dispatch_args_707205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 39), 'dispatch_args', False)
    # Processing the call keyword arguments (line 327)
    kwargs_707206 = {}
    str_707203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 29), 'str', ', ')
    # Obtaining the member 'join' of a type (line 327)
    join_707204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 29), str_707203, 'join')
    # Calling join(args, kwargs) (line 327)
    join_call_result_707207 = invoke(stypy.reporting.localization.Localization(__file__, 327, 29), join_707204, *[dispatch_args_707205], **kwargs_707206)
    
    # Applying the binary operator '%' (line 327)
    result_mod_707208 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 19), '%', str_707202, join_call_result_707207)
    
    # Assigning a type to the variable 'dispatch_str' (line 327)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'dispatch_str', result_mod_707208)

    @norecursion
    def check(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'operator' (line 329)
        operator_707209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 31), 'operator')
        # Obtaining the member 'ne' of a type (line 329)
        ne_707210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 31), operator_707209, 'ne')
        str_707211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 48), 'str', '')
        defaults = [ne_707210, str_707211]
        # Create a new context for function 'check'
        module_type_store = module_type_store.open_function_context('check', 329, 4, False)
        
        # Passed parameters checking function
        check.stypy_localization = localization
        check.stypy_type_of_self = None
        check.stypy_type_store = module_type_store
        check.stypy_function_name = 'check'
        check.stypy_param_names_list = ['arguments', 'wrong', 'msg']
        check.stypy_varargs_param_name = None
        check.stypy_kwargs_param_name = None
        check.stypy_call_defaults = defaults
        check.stypy_call_varargs = varargs
        check.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'check', ['arguments', 'wrong', 'msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check', localization, ['arguments', 'wrong', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check(...)' code ##################

        str_707212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 8), 'str', 'Make sure one passes the expected number of arguments')
        
        
        # Call to wrong(...): (line 331)
        # Processing the call arguments (line 331)
        
        # Call to len(...): (line 331)
        # Processing the call arguments (line 331)
        # Getting the type of 'arguments' (line 331)
        arguments_707215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 21), 'arguments', False)
        # Processing the call keyword arguments (line 331)
        kwargs_707216 = {}
        # Getting the type of 'len' (line 331)
        len_707214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 17), 'len', False)
        # Calling len(args, kwargs) (line 331)
        len_call_result_707217 = invoke(stypy.reporting.localization.Localization(__file__, 331, 17), len_707214, *[arguments_707215], **kwargs_707216)
        
        
        # Call to len(...): (line 331)
        # Processing the call arguments (line 331)
        # Getting the type of 'dispatch_args' (line 331)
        dispatch_args_707219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 37), 'dispatch_args', False)
        # Processing the call keyword arguments (line 331)
        kwargs_707220 = {}
        # Getting the type of 'len' (line 331)
        len_707218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 33), 'len', False)
        # Calling len(args, kwargs) (line 331)
        len_call_result_707221 = invoke(stypy.reporting.localization.Localization(__file__, 331, 33), len_707218, *[dispatch_args_707219], **kwargs_707220)
        
        # Processing the call keyword arguments (line 331)
        kwargs_707222 = {}
        # Getting the type of 'wrong' (line 331)
        wrong_707213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 11), 'wrong', False)
        # Calling wrong(args, kwargs) (line 331)
        wrong_call_result_707223 = invoke(stypy.reporting.localization.Localization(__file__, 331, 11), wrong_707213, *[len_call_result_707217, len_call_result_707221], **kwargs_707222)
        
        # Testing the type of an if condition (line 331)
        if_condition_707224 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 331, 8), wrong_call_result_707223)
        # Assigning a type to the variable 'if_condition_707224' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'if_condition_707224', if_condition_707224)
        # SSA begins for if statement (line 331)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 332)
        # Processing the call arguments (line 332)
        str_707226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 28), 'str', 'Expected %d arguments, got %d%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 333)
        tuple_707227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 333)
        # Adding element type (line 333)
        
        # Call to len(...): (line 333)
        # Processing the call arguments (line 333)
        # Getting the type of 'dispatch_args' (line 333)
        dispatch_args_707229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 33), 'dispatch_args', False)
        # Processing the call keyword arguments (line 333)
        kwargs_707230 = {}
        # Getting the type of 'len' (line 333)
        len_707228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 29), 'len', False)
        # Calling len(args, kwargs) (line 333)
        len_call_result_707231 = invoke(stypy.reporting.localization.Localization(__file__, 333, 29), len_707228, *[dispatch_args_707229], **kwargs_707230)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 29), tuple_707227, len_call_result_707231)
        # Adding element type (line 333)
        
        # Call to len(...): (line 333)
        # Processing the call arguments (line 333)
        # Getting the type of 'arguments' (line 333)
        arguments_707233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 53), 'arguments', False)
        # Processing the call keyword arguments (line 333)
        kwargs_707234 = {}
        # Getting the type of 'len' (line 333)
        len_707232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 49), 'len', False)
        # Calling len(args, kwargs) (line 333)
        len_call_result_707235 = invoke(stypy.reporting.localization.Localization(__file__, 333, 49), len_707232, *[arguments_707233], **kwargs_707234)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 29), tuple_707227, len_call_result_707235)
        # Adding element type (line 333)
        # Getting the type of 'msg' (line 333)
        msg_707236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 65), 'msg', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 29), tuple_707227, msg_707236)
        
        # Applying the binary operator '%' (line 332)
        result_mod_707237 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 28), '%', str_707226, tuple_707227)
        
        # Processing the call keyword arguments (line 332)
        kwargs_707238 = {}
        # Getting the type of 'TypeError' (line 332)
        TypeError_707225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 332)
        TypeError_call_result_707239 = invoke(stypy.reporting.localization.Localization(__file__, 332, 18), TypeError_707225, *[result_mod_707237], **kwargs_707238)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 332, 12), TypeError_call_result_707239, 'raise parameter', BaseException)
        # SSA join for if statement (line 331)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check' in the type store
        # Getting the type of 'stypy_return_type' (line 329)
        stypy_return_type_707240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_707240)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check'
        return stypy_return_type_707240

    # Assigning a type to the variable 'check' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'check', check)

    @norecursion
    def gen_func_dec(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'gen_func_dec'
        module_type_store = module_type_store.open_function_context('gen_func_dec', 335, 4, False)
        
        # Passed parameters checking function
        gen_func_dec.stypy_localization = localization
        gen_func_dec.stypy_type_of_self = None
        gen_func_dec.stypy_type_store = module_type_store
        gen_func_dec.stypy_function_name = 'gen_func_dec'
        gen_func_dec.stypy_param_names_list = ['func']
        gen_func_dec.stypy_varargs_param_name = None
        gen_func_dec.stypy_kwargs_param_name = None
        gen_func_dec.stypy_call_defaults = defaults
        gen_func_dec.stypy_call_varargs = varargs
        gen_func_dec.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'gen_func_dec', ['func'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'gen_func_dec', localization, ['func'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'gen_func_dec(...)' code ##################

        str_707241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 8), 'str', 'Decorator turning a function into a generic function')
        
        # Assigning a Call to a Name (line 339):
        
        # Assigning a Call to a Name (line 339):
        
        # Call to set(...): (line 339)
        # Processing the call arguments (line 339)
        
        # Call to getfullargspec(...): (line 339)
        # Processing the call arguments (line 339)
        # Getting the type of 'func' (line 339)
        func_707244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 36), 'func', False)
        # Processing the call keyword arguments (line 339)
        kwargs_707245 = {}
        # Getting the type of 'getfullargspec' (line 339)
        getfullargspec_707243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 21), 'getfullargspec', False)
        # Calling getfullargspec(args, kwargs) (line 339)
        getfullargspec_call_result_707246 = invoke(stypy.reporting.localization.Localization(__file__, 339, 21), getfullargspec_707243, *[func_707244], **kwargs_707245)
        
        # Obtaining the member 'args' of a type (line 339)
        args_707247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 21), getfullargspec_call_result_707246, 'args')
        # Processing the call keyword arguments (line 339)
        kwargs_707248 = {}
        # Getting the type of 'set' (line 339)
        set_707242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 17), 'set', False)
        # Calling set(args, kwargs) (line 339)
        set_call_result_707249 = invoke(stypy.reporting.localization.Localization(__file__, 339, 17), set_707242, *[args_707247], **kwargs_707248)
        
        # Assigning a type to the variable 'argset' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'argset', set_call_result_707249)
        
        
        
        
        # Call to set(...): (line 340)
        # Processing the call arguments (line 340)
        # Getting the type of 'dispatch_args' (line 340)
        dispatch_args_707251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 19), 'dispatch_args', False)
        # Processing the call keyword arguments (line 340)
        kwargs_707252 = {}
        # Getting the type of 'set' (line 340)
        set_707250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 15), 'set', False)
        # Calling set(args, kwargs) (line 340)
        set_call_result_707253 = invoke(stypy.reporting.localization.Localization(__file__, 340, 15), set_707250, *[dispatch_args_707251], **kwargs_707252)
        
        # Getting the type of 'argset' (line 340)
        argset_707254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 37), 'argset')
        # Applying the binary operator '<=' (line 340)
        result_le_707255 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 15), '<=', set_call_result_707253, argset_707254)
        
        # Applying the 'not' unary operator (line 340)
        result_not__707256 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 11), 'not', result_le_707255)
        
        # Testing the type of an if condition (line 340)
        if_condition_707257 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 340, 8), result_not__707256)
        # Assigning a type to the variable 'if_condition_707257' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'if_condition_707257', if_condition_707257)
        # SSA begins for if statement (line 340)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to NameError(...): (line 341)
        # Processing the call arguments (line 341)
        str_707259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 28), 'str', 'Unknown dispatch arguments %s')
        # Getting the type of 'dispatch_str' (line 341)
        dispatch_str_707260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 62), 'dispatch_str', False)
        # Applying the binary operator '%' (line 341)
        result_mod_707261 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 28), '%', str_707259, dispatch_str_707260)
        
        # Processing the call keyword arguments (line 341)
        kwargs_707262 = {}
        # Getting the type of 'NameError' (line 341)
        NameError_707258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 18), 'NameError', False)
        # Calling NameError(args, kwargs) (line 341)
        NameError_call_result_707263 = invoke(stypy.reporting.localization.Localization(__file__, 341, 18), NameError_707258, *[result_mod_707261], **kwargs_707262)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 341, 12), NameError_call_result_707263, 'raise parameter', BaseException)
        # SSA join for if statement (line 340)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Dict to a Name (line 343):
        
        # Assigning a Dict to a Name (line 343):
        
        # Obtaining an instance of the builtin type 'dict' (line 343)
        dict_707264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 18), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 343)
        
        # Assigning a type to the variable 'typemap' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'typemap', dict_707264)

        @norecursion
        def vancestors(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'vancestors'
            module_type_store = module_type_store.open_function_context('vancestors', 345, 8, False)
            
            # Passed parameters checking function
            vancestors.stypy_localization = localization
            vancestors.stypy_type_of_self = None
            vancestors.stypy_type_store = module_type_store
            vancestors.stypy_function_name = 'vancestors'
            vancestors.stypy_param_names_list = []
            vancestors.stypy_varargs_param_name = 'types'
            vancestors.stypy_kwargs_param_name = None
            vancestors.stypy_call_defaults = defaults
            vancestors.stypy_call_varargs = varargs
            vancestors.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'vancestors', [], 'types', None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'vancestors', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'vancestors(...)' code ##################

            str_707265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, (-1)), 'str', '\n            Get a list of sets of virtual ancestors for the given types\n            ')
            
            # Call to check(...): (line 349)
            # Processing the call arguments (line 349)
            # Getting the type of 'types' (line 349)
            types_707267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 18), 'types', False)
            # Processing the call keyword arguments (line 349)
            kwargs_707268 = {}
            # Getting the type of 'check' (line 349)
            check_707266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 12), 'check', False)
            # Calling check(args, kwargs) (line 349)
            check_call_result_707269 = invoke(stypy.reporting.localization.Localization(__file__, 349, 12), check_707266, *[types_707267], **kwargs_707268)
            
            
            # Assigning a ListComp to a Name (line 350):
            
            # Assigning a ListComp to a Name (line 350):
            # Calculating list comprehension
            # Calculating comprehension expression
            
            # Call to range(...): (line 350)
            # Processing the call arguments (line 350)
            
            # Call to len(...): (line 350)
            # Processing the call arguments (line 350)
            # Getting the type of 'dispatch_args' (line 350)
            dispatch_args_707273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 41), 'dispatch_args', False)
            # Processing the call keyword arguments (line 350)
            kwargs_707274 = {}
            # Getting the type of 'len' (line 350)
            len_707272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 37), 'len', False)
            # Calling len(args, kwargs) (line 350)
            len_call_result_707275 = invoke(stypy.reporting.localization.Localization(__file__, 350, 37), len_707272, *[dispatch_args_707273], **kwargs_707274)
            
            # Processing the call keyword arguments (line 350)
            kwargs_707276 = {}
            # Getting the type of 'range' (line 350)
            range_707271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 31), 'range', False)
            # Calling range(args, kwargs) (line 350)
            range_call_result_707277 = invoke(stypy.reporting.localization.Localization(__file__, 350, 31), range_707271, *[len_call_result_707275], **kwargs_707276)
            
            comprehension_707278 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 19), range_call_result_707277)
            # Assigning a type to the variable '_' (line 350)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 19), '_', comprehension_707278)
            
            # Obtaining an instance of the builtin type 'list' (line 350)
            list_707270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 19), 'list')
            # Adding type elements to the builtin type 'list' instance (line 350)
            
            list_707279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 19), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 19), list_707279, list_707270)
            # Assigning a type to the variable 'ras' (line 350)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 12), 'ras', list_707279)
            
            # Getting the type of 'typemap' (line 351)
            typemap_707280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 26), 'typemap')
            # Testing the type of a for loop iterable (line 351)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 351, 12), typemap_707280)
            # Getting the type of the for loop variable (line 351)
            for_loop_var_707281 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 351, 12), typemap_707280)
            # Assigning a type to the variable 'types_' (line 351)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'types_', for_loop_var_707281)
            # SSA begins for a for statement (line 351)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to zip(...): (line 352)
            # Processing the call arguments (line 352)
            # Getting the type of 'types' (line 352)
            types_707283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 40), 'types', False)
            # Getting the type of 'types_' (line 352)
            types__707284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 47), 'types_', False)
            # Getting the type of 'ras' (line 352)
            ras_707285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 55), 'ras', False)
            # Processing the call keyword arguments (line 352)
            kwargs_707286 = {}
            # Getting the type of 'zip' (line 352)
            zip_707282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 36), 'zip', False)
            # Calling zip(args, kwargs) (line 352)
            zip_call_result_707287 = invoke(stypy.reporting.localization.Localization(__file__, 352, 36), zip_707282, *[types_707283, types__707284, ras_707285], **kwargs_707286)
            
            # Testing the type of a for loop iterable (line 352)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 352, 16), zip_call_result_707287)
            # Getting the type of the for loop variable (line 352)
            for_loop_var_707288 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 352, 16), zip_call_result_707287)
            # Assigning a type to the variable 't' (line 352)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 16), 't', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 16), for_loop_var_707288))
            # Assigning a type to the variable 'type_' (line 352)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 16), 'type_', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 16), for_loop_var_707288))
            # Assigning a type to the variable 'ra' (line 352)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 16), 'ra', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 16), for_loop_var_707288))
            # SSA begins for a for statement (line 352)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Evaluating a boolean operation
            
            # Call to issubclass(...): (line 353)
            # Processing the call arguments (line 353)
            # Getting the type of 't' (line 353)
            t_707290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 34), 't', False)
            # Getting the type of 'type_' (line 353)
            type__707291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 37), 'type_', False)
            # Processing the call keyword arguments (line 353)
            kwargs_707292 = {}
            # Getting the type of 'issubclass' (line 353)
            issubclass_707289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 23), 'issubclass', False)
            # Calling issubclass(args, kwargs) (line 353)
            issubclass_call_result_707293 = invoke(stypy.reporting.localization.Localization(__file__, 353, 23), issubclass_707289, *[t_707290, type__707291], **kwargs_707292)
            
            
            # Getting the type of 'type_' (line 353)
            type__707294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 48), 'type_')
            # Getting the type of 't' (line 353)
            t_707295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 61), 't')
            # Obtaining the member '__mro__' of a type (line 353)
            mro___707296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 61), t_707295, '__mro__')
            # Applying the binary operator 'notin' (line 353)
            result_contains_707297 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 48), 'notin', type__707294, mro___707296)
            
            # Applying the binary operator 'and' (line 353)
            result_and_keyword_707298 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 23), 'and', issubclass_call_result_707293, result_contains_707297)
            
            # Testing the type of an if condition (line 353)
            if_condition_707299 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 353, 20), result_and_keyword_707298)
            # Assigning a type to the variable 'if_condition_707299' (line 353)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 20), 'if_condition_707299', if_condition_707299)
            # SSA begins for if statement (line 353)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 354)
            # Processing the call arguments (line 354)
            # Getting the type of 'type_' (line 354)
            type__707301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 31), 'type_', False)
            # Getting the type of 'ra' (line 354)
            ra_707302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 38), 'ra', False)
            # Processing the call keyword arguments (line 354)
            kwargs_707303 = {}
            # Getting the type of 'append' (line 354)
            append_707300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 24), 'append', False)
            # Calling append(args, kwargs) (line 354)
            append_call_result_707304 = invoke(stypy.reporting.localization.Localization(__file__, 354, 24), append_707300, *[type__707301, ra_707302], **kwargs_707303)
            
            # SSA join for if statement (line 353)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'ras' (line 355)
            ras_707309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 38), 'ras')
            comprehension_707310 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 20), ras_707309)
            # Assigning a type to the variable 'ra' (line 355)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 20), 'ra', comprehension_707310)
            
            # Call to set(...): (line 355)
            # Processing the call arguments (line 355)
            # Getting the type of 'ra' (line 355)
            ra_707306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 24), 'ra', False)
            # Processing the call keyword arguments (line 355)
            kwargs_707307 = {}
            # Getting the type of 'set' (line 355)
            set_707305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 20), 'set', False)
            # Calling set(args, kwargs) (line 355)
            set_call_result_707308 = invoke(stypy.reporting.localization.Localization(__file__, 355, 20), set_707305, *[ra_707306], **kwargs_707307)
            
            list_707311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 20), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 20), list_707311, set_call_result_707308)
            # Assigning a type to the variable 'stypy_return_type' (line 355)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 12), 'stypy_return_type', list_707311)
            
            # ################# End of 'vancestors(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'vancestors' in the type store
            # Getting the type of 'stypy_return_type' (line 345)
            stypy_return_type_707312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_707312)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'vancestors'
            return stypy_return_type_707312

        # Assigning a type to the variable 'vancestors' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'vancestors', vancestors)

        @norecursion
        def ancestors(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'ancestors'
            module_type_store = module_type_store.open_function_context('ancestors', 357, 8, False)
            
            # Passed parameters checking function
            ancestors.stypy_localization = localization
            ancestors.stypy_type_of_self = None
            ancestors.stypy_type_store = module_type_store
            ancestors.stypy_function_name = 'ancestors'
            ancestors.stypy_param_names_list = []
            ancestors.stypy_varargs_param_name = 'types'
            ancestors.stypy_kwargs_param_name = None
            ancestors.stypy_call_defaults = defaults
            ancestors.stypy_call_varargs = varargs
            ancestors.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'ancestors', [], 'types', None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'ancestors', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'ancestors(...)' code ##################

            str_707313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, (-1)), 'str', '\n            Get a list of virtual MROs, one for each type\n            ')
            
            # Call to check(...): (line 361)
            # Processing the call arguments (line 361)
            # Getting the type of 'types' (line 361)
            types_707315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 18), 'types', False)
            # Processing the call keyword arguments (line 361)
            kwargs_707316 = {}
            # Getting the type of 'check' (line 361)
            check_707314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'check', False)
            # Calling check(args, kwargs) (line 361)
            check_call_result_707317 = invoke(stypy.reporting.localization.Localization(__file__, 361, 12), check_707314, *[types_707315], **kwargs_707316)
            
            
            # Assigning a List to a Name (line 362):
            
            # Assigning a List to a Name (line 362):
            
            # Obtaining an instance of the builtin type 'list' (line 362)
            list_707318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 20), 'list')
            # Adding type elements to the builtin type 'list' instance (line 362)
            
            # Assigning a type to the variable 'lists' (line 362)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'lists', list_707318)
            
            
            # Call to zip(...): (line 363)
            # Processing the call arguments (line 363)
            # Getting the type of 'types' (line 363)
            types_707320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 30), 'types', False)
            
            # Call to vancestors(...): (line 363)
            # Getting the type of 'types' (line 363)
            types_707322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 49), 'types', False)
            # Processing the call keyword arguments (line 363)
            kwargs_707323 = {}
            # Getting the type of 'vancestors' (line 363)
            vancestors_707321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 37), 'vancestors', False)
            # Calling vancestors(args, kwargs) (line 363)
            vancestors_call_result_707324 = invoke(stypy.reporting.localization.Localization(__file__, 363, 37), vancestors_707321, *[types_707322], **kwargs_707323)
            
            # Processing the call keyword arguments (line 363)
            kwargs_707325 = {}
            # Getting the type of 'zip' (line 363)
            zip_707319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 26), 'zip', False)
            # Calling zip(args, kwargs) (line 363)
            zip_call_result_707326 = invoke(stypy.reporting.localization.Localization(__file__, 363, 26), zip_707319, *[types_707320, vancestors_call_result_707324], **kwargs_707325)
            
            # Testing the type of a for loop iterable (line 363)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 363, 12), zip_call_result_707326)
            # Getting the type of the for loop variable (line 363)
            for_loop_var_707327 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 363, 12), zip_call_result_707326)
            # Assigning a type to the variable 't' (line 363)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 't', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 12), for_loop_var_707327))
            # Assigning a type to the variable 'vas' (line 363)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'vas', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 12), for_loop_var_707327))
            # SSA begins for a for statement (line 363)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 364):
            
            # Assigning a Call to a Name (line 364):
            
            # Call to len(...): (line 364)
            # Processing the call arguments (line 364)
            # Getting the type of 'vas' (line 364)
            vas_707329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 28), 'vas', False)
            # Processing the call keyword arguments (line 364)
            kwargs_707330 = {}
            # Getting the type of 'len' (line 364)
            len_707328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 24), 'len', False)
            # Calling len(args, kwargs) (line 364)
            len_call_result_707331 = invoke(stypy.reporting.localization.Localization(__file__, 364, 24), len_707328, *[vas_707329], **kwargs_707330)
            
            # Assigning a type to the variable 'n_vas' (line 364)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 16), 'n_vas', len_call_result_707331)
            
            
            # Getting the type of 'n_vas' (line 365)
            n_vas_707332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 19), 'n_vas')
            int_707333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 27), 'int')
            # Applying the binary operator '>' (line 365)
            result_gt_707334 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 19), '>', n_vas_707332, int_707333)
            
            # Testing the type of an if condition (line 365)
            if_condition_707335 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 365, 16), result_gt_707334)
            # Assigning a type to the variable 'if_condition_707335' (line 365)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 16), 'if_condition_707335', if_condition_707335)
            # SSA begins for if statement (line 365)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to RuntimeError(...): (line 366)
            # Processing the call arguments (line 366)
            str_707337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 24), 'str', 'Ambiguous dispatch for %s: %s')
            
            # Obtaining an instance of the builtin type 'tuple' (line 367)
            tuple_707338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 59), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 367)
            # Adding element type (line 367)
            # Getting the type of 't' (line 367)
            t_707339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 59), 't', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 59), tuple_707338, t_707339)
            # Adding element type (line 367)
            # Getting the type of 'vas' (line 367)
            vas_707340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 62), 'vas', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 59), tuple_707338, vas_707340)
            
            # Applying the binary operator '%' (line 367)
            result_mod_707341 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 24), '%', str_707337, tuple_707338)
            
            # Processing the call keyword arguments (line 366)
            kwargs_707342 = {}
            # Getting the type of 'RuntimeError' (line 366)
            RuntimeError_707336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 26), 'RuntimeError', False)
            # Calling RuntimeError(args, kwargs) (line 366)
            RuntimeError_call_result_707343 = invoke(stypy.reporting.localization.Localization(__file__, 366, 26), RuntimeError_707336, *[result_mod_707341], **kwargs_707342)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 366, 20), RuntimeError_call_result_707343, 'raise parameter', BaseException)
            # SSA branch for the else part of an if statement (line 365)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'n_vas' (line 368)
            n_vas_707344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 21), 'n_vas')
            int_707345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 30), 'int')
            # Applying the binary operator '==' (line 368)
            result_eq_707346 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 21), '==', n_vas_707344, int_707345)
            
            # Testing the type of an if condition (line 368)
            if_condition_707347 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 368, 21), result_eq_707346)
            # Assigning a type to the variable 'if_condition_707347' (line 368)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 21), 'if_condition_707347', if_condition_707347)
            # SSA begins for if statement (line 368)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Tuple (line 369):
            
            # Assigning a Subscript to a Name (line 369):
            
            # Obtaining the type of the subscript
            int_707348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 20), 'int')
            # Getting the type of 'vas' (line 369)
            vas_707349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 26), 'vas')
            # Obtaining the member '__getitem__' of a type (line 369)
            getitem___707350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 20), vas_707349, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 369)
            subscript_call_result_707351 = invoke(stypy.reporting.localization.Localization(__file__, 369, 20), getitem___707350, int_707348)
            
            # Assigning a type to the variable 'tuple_var_assignment_706366' (line 369)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 20), 'tuple_var_assignment_706366', subscript_call_result_707351)
            
            # Assigning a Name to a Name (line 369):
            # Getting the type of 'tuple_var_assignment_706366' (line 369)
            tuple_var_assignment_706366_707352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 20), 'tuple_var_assignment_706366')
            # Assigning a type to the variable 'va' (line 369)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 20), 'va', tuple_var_assignment_706366_707352)
            
            # Assigning a Subscript to a Name (line 370):
            
            # Assigning a Subscript to a Name (line 370):
            
            # Obtaining the type of the subscript
            int_707353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 57), 'int')
            slice_707354 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 370, 26), int_707353, None, None)
            
            # Call to type(...): (line 370)
            # Processing the call arguments (line 370)
            str_707356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 31), 'str', 't')
            
            # Obtaining an instance of the builtin type 'tuple' (line 370)
            tuple_707357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 37), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 370)
            # Adding element type (line 370)
            # Getting the type of 't' (line 370)
            t_707358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 37), 't', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 37), tuple_707357, t_707358)
            # Adding element type (line 370)
            # Getting the type of 'va' (line 370)
            va_707359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 40), 'va', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 37), tuple_707357, va_707359)
            
            
            # Obtaining an instance of the builtin type 'dict' (line 370)
            dict_707360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 45), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 370)
            
            # Processing the call keyword arguments (line 370)
            kwargs_707361 = {}
            # Getting the type of 'type' (line 370)
            type_707355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 26), 'type', False)
            # Calling type(args, kwargs) (line 370)
            type_call_result_707362 = invoke(stypy.reporting.localization.Localization(__file__, 370, 26), type_707355, *[str_707356, tuple_707357, dict_707360], **kwargs_707361)
            
            # Obtaining the member '__mro__' of a type (line 370)
            mro___707363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 26), type_call_result_707362, '__mro__')
            # Obtaining the member '__getitem__' of a type (line 370)
            getitem___707364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 26), mro___707363, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 370)
            subscript_call_result_707365 = invoke(stypy.reporting.localization.Localization(__file__, 370, 26), getitem___707364, slice_707354)
            
            # Assigning a type to the variable 'mro' (line 370)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 20), 'mro', subscript_call_result_707365)
            # SSA branch for the else part of an if statement (line 368)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Attribute to a Name (line 372):
            
            # Assigning a Attribute to a Name (line 372):
            # Getting the type of 't' (line 372)
            t_707366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 26), 't')
            # Obtaining the member '__mro__' of a type (line 372)
            mro___707367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 26), t_707366, '__mro__')
            # Assigning a type to the variable 'mro' (line 372)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 20), 'mro', mro___707367)
            # SSA join for if statement (line 368)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 365)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to append(...): (line 373)
            # Processing the call arguments (line 373)
            
            # Obtaining the type of the subscript
            int_707370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 34), 'int')
            slice_707371 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 373, 29), None, int_707370, None)
            # Getting the type of 'mro' (line 373)
            mro_707372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 29), 'mro', False)
            # Obtaining the member '__getitem__' of a type (line 373)
            getitem___707373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 29), mro_707372, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 373)
            subscript_call_result_707374 = invoke(stypy.reporting.localization.Localization(__file__, 373, 29), getitem___707373, slice_707371)
            
            # Processing the call keyword arguments (line 373)
            kwargs_707375 = {}
            # Getting the type of 'lists' (line 373)
            lists_707368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 16), 'lists', False)
            # Obtaining the member 'append' of a type (line 373)
            append_707369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 16), lists_707368, 'append')
            # Calling append(args, kwargs) (line 373)
            append_call_result_707376 = invoke(stypy.reporting.localization.Localization(__file__, 373, 16), append_707369, *[subscript_call_result_707374], **kwargs_707375)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # Getting the type of 'lists' (line 374)
            lists_707377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 19), 'lists')
            # Assigning a type to the variable 'stypy_return_type' (line 374)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'stypy_return_type', lists_707377)
            
            # ################# End of 'ancestors(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'ancestors' in the type store
            # Getting the type of 'stypy_return_type' (line 357)
            stypy_return_type_707378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_707378)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'ancestors'
            return stypy_return_type_707378

        # Assigning a type to the variable 'ancestors' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'ancestors', ancestors)

        @norecursion
        def register(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'register'
            module_type_store = module_type_store.open_function_context('register', 376, 8, False)
            
            # Passed parameters checking function
            register.stypy_localization = localization
            register.stypy_type_of_self = None
            register.stypy_type_store = module_type_store
            register.stypy_function_name = 'register'
            register.stypy_param_names_list = []
            register.stypy_varargs_param_name = 'types'
            register.stypy_kwargs_param_name = None
            register.stypy_call_defaults = defaults
            register.stypy_call_varargs = varargs
            register.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'register', [], 'types', None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'register', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'register(...)' code ##################

            str_707379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, (-1)), 'str', '\n            Decorator to register an implementation for the given types\n            ')
            
            # Call to check(...): (line 380)
            # Processing the call arguments (line 380)
            # Getting the type of 'types' (line 380)
            types_707381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 18), 'types', False)
            # Processing the call keyword arguments (line 380)
            kwargs_707382 = {}
            # Getting the type of 'check' (line 380)
            check_707380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 12), 'check', False)
            # Calling check(args, kwargs) (line 380)
            check_call_result_707383 = invoke(stypy.reporting.localization.Localization(__file__, 380, 12), check_707380, *[types_707381], **kwargs_707382)
            

            @norecursion
            def dec(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'dec'
                module_type_store = module_type_store.open_function_context('dec', 382, 12, False)
                
                # Passed parameters checking function
                dec.stypy_localization = localization
                dec.stypy_type_of_self = None
                dec.stypy_type_store = module_type_store
                dec.stypy_function_name = 'dec'
                dec.stypy_param_names_list = ['f']
                dec.stypy_varargs_param_name = None
                dec.stypy_kwargs_param_name = None
                dec.stypy_call_defaults = defaults
                dec.stypy_call_varargs = varargs
                dec.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, 'dec', ['f'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'dec', localization, ['f'], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'dec(...)' code ##################

                
                # Call to check(...): (line 383)
                # Processing the call arguments (line 383)
                
                # Call to getfullargspec(...): (line 383)
                # Processing the call arguments (line 383)
                # Getting the type of 'f' (line 383)
                f_707386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 37), 'f', False)
                # Processing the call keyword arguments (line 383)
                kwargs_707387 = {}
                # Getting the type of 'getfullargspec' (line 383)
                getfullargspec_707385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 22), 'getfullargspec', False)
                # Calling getfullargspec(args, kwargs) (line 383)
                getfullargspec_call_result_707388 = invoke(stypy.reporting.localization.Localization(__file__, 383, 22), getfullargspec_707385, *[f_707386], **kwargs_707387)
                
                # Obtaining the member 'args' of a type (line 383)
                args_707389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 22), getfullargspec_call_result_707388, 'args')
                # Getting the type of 'operator' (line 383)
                operator_707390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 46), 'operator', False)
                # Obtaining the member 'lt' of a type (line 383)
                lt_707391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 46), operator_707390, 'lt')
                str_707392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 59), 'str', ' in ')
                # Getting the type of 'f' (line 383)
                f_707393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 68), 'f', False)
                # Obtaining the member '__name__' of a type (line 383)
                name___707394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 68), f_707393, '__name__')
                # Applying the binary operator '+' (line 383)
                result_add_707395 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 59), '+', str_707392, name___707394)
                
                # Processing the call keyword arguments (line 383)
                kwargs_707396 = {}
                # Getting the type of 'check' (line 383)
                check_707384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 16), 'check', False)
                # Calling check(args, kwargs) (line 383)
                check_call_result_707397 = invoke(stypy.reporting.localization.Localization(__file__, 383, 16), check_707384, *[args_707389, lt_707391, result_add_707395], **kwargs_707396)
                
                
                # Assigning a Name to a Subscript (line 384):
                
                # Assigning a Name to a Subscript (line 384):
                # Getting the type of 'f' (line 384)
                f_707398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 33), 'f')
                # Getting the type of 'typemap' (line 384)
                typemap_707399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 16), 'typemap')
                # Getting the type of 'types' (line 384)
                types_707400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 24), 'types')
                # Storing an element on a container (line 384)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 16), typemap_707399, (types_707400, f_707398))
                # Getting the type of 'f' (line 385)
                f_707401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 23), 'f')
                # Assigning a type to the variable 'stypy_return_type' (line 385)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 16), 'stypy_return_type', f_707401)
                
                # ################# End of 'dec(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'dec' in the type store
                # Getting the type of 'stypy_return_type' (line 382)
                stypy_return_type_707402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_707402)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'dec'
                return stypy_return_type_707402

            # Assigning a type to the variable 'dec' (line 382)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'dec', dec)
            # Getting the type of 'dec' (line 386)
            dec_707403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 19), 'dec')
            # Assigning a type to the variable 'stypy_return_type' (line 386)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'stypy_return_type', dec_707403)
            
            # ################# End of 'register(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'register' in the type store
            # Getting the type of 'stypy_return_type' (line 376)
            stypy_return_type_707404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_707404)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'register'
            return stypy_return_type_707404

        # Assigning a type to the variable 'register' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'register', register)

        @norecursion
        def dispatch_info(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'dispatch_info'
            module_type_store = module_type_store.open_function_context('dispatch_info', 388, 8, False)
            
            # Passed parameters checking function
            dispatch_info.stypy_localization = localization
            dispatch_info.stypy_type_of_self = None
            dispatch_info.stypy_type_store = module_type_store
            dispatch_info.stypy_function_name = 'dispatch_info'
            dispatch_info.stypy_param_names_list = []
            dispatch_info.stypy_varargs_param_name = 'types'
            dispatch_info.stypy_kwargs_param_name = None
            dispatch_info.stypy_call_defaults = defaults
            dispatch_info.stypy_call_varargs = varargs
            dispatch_info.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'dispatch_info', [], 'types', None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'dispatch_info', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'dispatch_info(...)' code ##################

            str_707405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, (-1)), 'str', '\n            An utility to introspect the dispatch algorithm\n            ')
            
            # Call to check(...): (line 392)
            # Processing the call arguments (line 392)
            # Getting the type of 'types' (line 392)
            types_707407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 18), 'types', False)
            # Processing the call keyword arguments (line 392)
            kwargs_707408 = {}
            # Getting the type of 'check' (line 392)
            check_707406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'check', False)
            # Calling check(args, kwargs) (line 392)
            check_call_result_707409 = invoke(stypy.reporting.localization.Localization(__file__, 392, 12), check_707406, *[types_707407], **kwargs_707408)
            
            
            # Assigning a List to a Name (line 393):
            
            # Assigning a List to a Name (line 393):
            
            # Obtaining an instance of the builtin type 'list' (line 393)
            list_707410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 18), 'list')
            # Adding type elements to the builtin type 'list' instance (line 393)
            
            # Assigning a type to the variable 'lst' (line 393)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'lst', list_707410)
            
            
            # Call to product(...): (line 394)
            
            # Call to ancestors(...): (line 394)
            # Getting the type of 'types' (line 394)
            types_707414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 53), 'types', False)
            # Processing the call keyword arguments (line 394)
            kwargs_707415 = {}
            # Getting the type of 'ancestors' (line 394)
            ancestors_707413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 42), 'ancestors', False)
            # Calling ancestors(args, kwargs) (line 394)
            ancestors_call_result_707416 = invoke(stypy.reporting.localization.Localization(__file__, 394, 42), ancestors_707413, *[types_707414], **kwargs_707415)
            
            # Processing the call keyword arguments (line 394)
            kwargs_707417 = {}
            # Getting the type of 'itertools' (line 394)
            itertools_707411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 23), 'itertools', False)
            # Obtaining the member 'product' of a type (line 394)
            product_707412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 23), itertools_707411, 'product')
            # Calling product(args, kwargs) (line 394)
            product_call_result_707418 = invoke(stypy.reporting.localization.Localization(__file__, 394, 23), product_707412, *[ancestors_call_result_707416], **kwargs_707417)
            
            # Testing the type of a for loop iterable (line 394)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 394, 12), product_call_result_707418)
            # Getting the type of the for loop variable (line 394)
            for_loop_var_707419 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 394, 12), product_call_result_707418)
            # Assigning a type to the variable 'anc' (line 394)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'anc', for_loop_var_707419)
            # SSA begins for a for statement (line 394)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to append(...): (line 395)
            # Processing the call arguments (line 395)
            
            # Call to tuple(...): (line 395)
            # Processing the call arguments (line 395)
            # Calculating generator expression
            module_type_store = module_type_store.open_function_context('list comprehension expression', 395, 33, True)
            # Calculating comprehension expression
            # Getting the type of 'anc' (line 395)
            anc_707425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 53), 'anc', False)
            comprehension_707426 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 33), anc_707425)
            # Assigning a type to the variable 'a' (line 395)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 33), 'a', comprehension_707426)
            # Getting the type of 'a' (line 395)
            a_707423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 33), 'a', False)
            # Obtaining the member '__name__' of a type (line 395)
            name___707424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 33), a_707423, '__name__')
            list_707427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 33), 'list')
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 33), list_707427, name___707424)
            # Processing the call keyword arguments (line 395)
            kwargs_707428 = {}
            # Getting the type of 'tuple' (line 395)
            tuple_707422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 27), 'tuple', False)
            # Calling tuple(args, kwargs) (line 395)
            tuple_call_result_707429 = invoke(stypy.reporting.localization.Localization(__file__, 395, 27), tuple_707422, *[list_707427], **kwargs_707428)
            
            # Processing the call keyword arguments (line 395)
            kwargs_707430 = {}
            # Getting the type of 'lst' (line 395)
            lst_707420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 16), 'lst', False)
            # Obtaining the member 'append' of a type (line 395)
            append_707421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 16), lst_707420, 'append')
            # Calling append(args, kwargs) (line 395)
            append_call_result_707431 = invoke(stypy.reporting.localization.Localization(__file__, 395, 16), append_707421, *[tuple_call_result_707429], **kwargs_707430)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # Getting the type of 'lst' (line 396)
            lst_707432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 19), 'lst')
            # Assigning a type to the variable 'stypy_return_type' (line 396)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'stypy_return_type', lst_707432)
            
            # ################# End of 'dispatch_info(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'dispatch_info' in the type store
            # Getting the type of 'stypy_return_type' (line 388)
            stypy_return_type_707433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_707433)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'dispatch_info'
            return stypy_return_type_707433

        # Assigning a type to the variable 'dispatch_info' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'dispatch_info', dispatch_info)

        @norecursion
        def _dispatch(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_dispatch'
            module_type_store = module_type_store.open_function_context('_dispatch', 398, 8, False)
            
            # Passed parameters checking function
            _dispatch.stypy_localization = localization
            _dispatch.stypy_type_of_self = None
            _dispatch.stypy_type_store = module_type_store
            _dispatch.stypy_function_name = '_dispatch'
            _dispatch.stypy_param_names_list = ['dispatch_args']
            _dispatch.stypy_varargs_param_name = 'args'
            _dispatch.stypy_kwargs_param_name = 'kw'
            _dispatch.stypy_call_defaults = defaults
            _dispatch.stypy_call_varargs = varargs
            _dispatch.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_dispatch', ['dispatch_args'], 'args', 'kw', defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_dispatch', localization, ['dispatch_args'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_dispatch(...)' code ##################

            
            # Assigning a Call to a Name (line 399):
            
            # Assigning a Call to a Name (line 399):
            
            # Call to tuple(...): (line 399)
            # Processing the call arguments (line 399)
            # Calculating generator expression
            module_type_store = module_type_store.open_function_context('list comprehension expression', 399, 26, True)
            # Calculating comprehension expression
            # Getting the type of 'dispatch_args' (line 399)
            dispatch_args_707439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 47), 'dispatch_args', False)
            comprehension_707440 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 26), dispatch_args_707439)
            # Assigning a type to the variable 'arg' (line 399)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 26), 'arg', comprehension_707440)
            
            # Call to type(...): (line 399)
            # Processing the call arguments (line 399)
            # Getting the type of 'arg' (line 399)
            arg_707436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 31), 'arg', False)
            # Processing the call keyword arguments (line 399)
            kwargs_707437 = {}
            # Getting the type of 'type' (line 399)
            type_707435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 26), 'type', False)
            # Calling type(args, kwargs) (line 399)
            type_call_result_707438 = invoke(stypy.reporting.localization.Localization(__file__, 399, 26), type_707435, *[arg_707436], **kwargs_707437)
            
            list_707441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 26), 'list')
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 26), list_707441, type_call_result_707438)
            # Processing the call keyword arguments (line 399)
            kwargs_707442 = {}
            # Getting the type of 'tuple' (line 399)
            tuple_707434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 20), 'tuple', False)
            # Calling tuple(args, kwargs) (line 399)
            tuple_call_result_707443 = invoke(stypy.reporting.localization.Localization(__file__, 399, 20), tuple_707434, *[list_707441], **kwargs_707442)
            
            # Assigning a type to the variable 'types' (line 399)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'types', tuple_call_result_707443)
            
            
            # SSA begins for try-except statement (line 400)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Subscript to a Name (line 401):
            
            # Assigning a Subscript to a Name (line 401):
            
            # Obtaining the type of the subscript
            # Getting the type of 'types' (line 401)
            types_707444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 28), 'types')
            # Getting the type of 'typemap' (line 401)
            typemap_707445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 20), 'typemap')
            # Obtaining the member '__getitem__' of a type (line 401)
            getitem___707446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 20), typemap_707445, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 401)
            subscript_call_result_707447 = invoke(stypy.reporting.localization.Localization(__file__, 401, 20), getitem___707446, types_707444)
            
            # Assigning a type to the variable 'f' (line 401)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 16), 'f', subscript_call_result_707447)
            # SSA branch for the except part of a try statement (line 400)
            # SSA branch for the except 'KeyError' branch of a try statement (line 400)
            module_type_store.open_ssa_branch('except')
            pass
            # SSA branch for the else branch of a try statement (line 400)
            module_type_store.open_ssa_branch('except else')
            
            # Call to f(...): (line 405)
            # Getting the type of 'args' (line 405)
            args_707449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 26), 'args', False)
            # Processing the call keyword arguments (line 405)
            # Getting the type of 'kw' (line 405)
            kw_707450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 34), 'kw', False)
            kwargs_707451 = {'kw_707450': kw_707450}
            # Getting the type of 'f' (line 405)
            f_707448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 23), 'f', False)
            # Calling f(args, kwargs) (line 405)
            f_call_result_707452 = invoke(stypy.reporting.localization.Localization(__file__, 405, 23), f_707448, *[args_707449], **kwargs_707451)
            
            # Assigning a type to the variable 'stypy_return_type' (line 405)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 16), 'stypy_return_type', f_call_result_707452)
            # SSA join for try-except statement (line 400)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 406):
            
            # Assigning a Call to a Name (line 406):
            
            # Call to product(...): (line 406)
            
            # Call to ancestors(...): (line 406)
            # Getting the type of 'types' (line 406)
            types_707456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 57), 'types', False)
            # Processing the call keyword arguments (line 406)
            kwargs_707457 = {}
            # Getting the type of 'ancestors' (line 406)
            ancestors_707455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 46), 'ancestors', False)
            # Calling ancestors(args, kwargs) (line 406)
            ancestors_call_result_707458 = invoke(stypy.reporting.localization.Localization(__file__, 406, 46), ancestors_707455, *[types_707456], **kwargs_707457)
            
            # Processing the call keyword arguments (line 406)
            kwargs_707459 = {}
            # Getting the type of 'itertools' (line 406)
            itertools_707453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 27), 'itertools', False)
            # Obtaining the member 'product' of a type (line 406)
            product_707454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 27), itertools_707453, 'product')
            # Calling product(args, kwargs) (line 406)
            product_call_result_707460 = invoke(stypy.reporting.localization.Localization(__file__, 406, 27), product_707454, *[ancestors_call_result_707458], **kwargs_707459)
            
            # Assigning a type to the variable 'combinations' (line 406)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 12), 'combinations', product_call_result_707460)
            
            # Call to next(...): (line 407)
            # Processing the call arguments (line 407)
            # Getting the type of 'combinations' (line 407)
            combinations_707462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 17), 'combinations', False)
            # Processing the call keyword arguments (line 407)
            kwargs_707463 = {}
            # Getting the type of 'next' (line 407)
            next_707461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'next', False)
            # Calling next(args, kwargs) (line 407)
            next_call_result_707464 = invoke(stypy.reporting.localization.Localization(__file__, 407, 12), next_707461, *[combinations_707462], **kwargs_707463)
            
            
            # Getting the type of 'combinations' (line 408)
            combinations_707465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 26), 'combinations')
            # Testing the type of a for loop iterable (line 408)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 408, 12), combinations_707465)
            # Getting the type of the for loop variable (line 408)
            for_loop_var_707466 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 408, 12), combinations_707465)
            # Assigning a type to the variable 'types_' (line 408)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'types_', for_loop_var_707466)
            # SSA begins for a for statement (line 408)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 409):
            
            # Assigning a Call to a Name (line 409):
            
            # Call to get(...): (line 409)
            # Processing the call arguments (line 409)
            # Getting the type of 'types_' (line 409)
            types__707469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 32), 'types_', False)
            # Processing the call keyword arguments (line 409)
            kwargs_707470 = {}
            # Getting the type of 'typemap' (line 409)
            typemap_707467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 20), 'typemap', False)
            # Obtaining the member 'get' of a type (line 409)
            get_707468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 20), typemap_707467, 'get')
            # Calling get(args, kwargs) (line 409)
            get_call_result_707471 = invoke(stypy.reporting.localization.Localization(__file__, 409, 20), get_707468, *[types__707469], **kwargs_707470)
            
            # Assigning a type to the variable 'f' (line 409)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 16), 'f', get_call_result_707471)
            
            # Type idiom detected: calculating its left and rigth part (line 410)
            # Getting the type of 'f' (line 410)
            f_707472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 16), 'f')
            # Getting the type of 'None' (line 410)
            None_707473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 28), 'None')
            
            (may_be_707474, more_types_in_union_707475) = may_not_be_none(f_707472, None_707473)

            if may_be_707474:

                if more_types_in_union_707475:
                    # Runtime conditional SSA (line 410)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to f(...): (line 411)
                # Getting the type of 'args' (line 411)
                args_707477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 30), 'args', False)
                # Processing the call keyword arguments (line 411)
                # Getting the type of 'kw' (line 411)
                kw_707478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 38), 'kw', False)
                kwargs_707479 = {'kw_707478': kw_707478}
                # Getting the type of 'f' (line 411)
                f_707476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 27), 'f', False)
                # Calling f(args, kwargs) (line 411)
                f_call_result_707480 = invoke(stypy.reporting.localization.Localization(__file__, 411, 27), f_707476, *[args_707477], **kwargs_707479)
                
                # Assigning a type to the variable 'stypy_return_type' (line 411)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 20), 'stypy_return_type', f_call_result_707480)

                if more_types_in_union_707475:
                    # SSA join for if statement (line 410)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to func(...): (line 414)
            # Getting the type of 'args' (line 414)
            args_707482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 25), 'args', False)
            # Processing the call keyword arguments (line 414)
            # Getting the type of 'kw' (line 414)
            kw_707483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 33), 'kw', False)
            kwargs_707484 = {'kw_707483': kw_707483}
            # Getting the type of 'func' (line 414)
            func_707481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 19), 'func', False)
            # Calling func(args, kwargs) (line 414)
            func_call_result_707485 = invoke(stypy.reporting.localization.Localization(__file__, 414, 19), func_707481, *[args_707482], **kwargs_707484)
            
            # Assigning a type to the variable 'stypy_return_type' (line 414)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'stypy_return_type', func_call_result_707485)
            
            # ################# End of '_dispatch(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_dispatch' in the type store
            # Getting the type of 'stypy_return_type' (line 398)
            stypy_return_type_707486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_707486)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_dispatch'
            return stypy_return_type_707486

        # Assigning a type to the variable '_dispatch' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), '_dispatch', _dispatch)
        
        # Call to create(...): (line 416)
        # Processing the call arguments (line 416)
        # Getting the type of 'func' (line 417)
        func_707489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 12), 'func', False)
        str_707490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 18), 'str', 'return _f_(%s, %%(shortsignature)s)')
        # Getting the type of 'dispatch_str' (line 417)
        dispatch_str_707491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 58), 'dispatch_str', False)
        # Applying the binary operator '%' (line 417)
        result_mod_707492 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 18), '%', str_707490, dispatch_str_707491)
        
        
        # Call to dict(...): (line 418)
        # Processing the call keyword arguments (line 418)
        # Getting the type of '_dispatch' (line 418)
        _dispatch_707494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 21), '_dispatch', False)
        keyword_707495 = _dispatch_707494
        kwargs_707496 = {'_f_': keyword_707495}
        # Getting the type of 'dict' (line 418)
        dict_707493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 12), 'dict', False)
        # Calling dict(args, kwargs) (line 418)
        dict_call_result_707497 = invoke(stypy.reporting.localization.Localization(__file__, 418, 12), dict_707493, *[], **kwargs_707496)
        
        # Processing the call keyword arguments (line 416)
        # Getting the type of 'register' (line 418)
        register_707498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 42), 'register', False)
        keyword_707499 = register_707498
        # Getting the type of 'func' (line 418)
        func_707500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 60), 'func', False)
        keyword_707501 = func_707500
        # Getting the type of 'typemap' (line 419)
        typemap_707502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 20), 'typemap', False)
        keyword_707503 = typemap_707502
        # Getting the type of 'vancestors' (line 419)
        vancestors_707504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 40), 'vancestors', False)
        keyword_707505 = vancestors_707504
        # Getting the type of 'ancestors' (line 419)
        ancestors_707506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 62), 'ancestors', False)
        keyword_707507 = ancestors_707506
        # Getting the type of 'dispatch_info' (line 420)
        dispatch_info_707508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 26), 'dispatch_info', False)
        keyword_707509 = dispatch_info_707508
        # Getting the type of 'func' (line 420)
        func_707510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 53), 'func', False)
        keyword_707511 = func_707510
        kwargs_707512 = {'ancestors': keyword_707507, 'default': keyword_707501, 'register': keyword_707499, '__wrapped__': keyword_707511, 'vancestors': keyword_707505, 'dispatch_info': keyword_707509, 'typemap': keyword_707503}
        # Getting the type of 'FunctionMaker' (line 416)
        FunctionMaker_707487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 15), 'FunctionMaker', False)
        # Obtaining the member 'create' of a type (line 416)
        create_707488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 15), FunctionMaker_707487, 'create')
        # Calling create(args, kwargs) (line 416)
        create_call_result_707513 = invoke(stypy.reporting.localization.Localization(__file__, 416, 15), create_707488, *[func_707489, result_mod_707492, dict_call_result_707497], **kwargs_707512)
        
        # Assigning a type to the variable 'stypy_return_type' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'stypy_return_type', create_call_result_707513)
        
        # ################# End of 'gen_func_dec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'gen_func_dec' in the type store
        # Getting the type of 'stypy_return_type' (line 335)
        stypy_return_type_707514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_707514)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'gen_func_dec'
        return stypy_return_type_707514

    # Assigning a type to the variable 'gen_func_dec' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'gen_func_dec', gen_func_dec)
    
    # Assigning a BinOp to a Attribute (line 422):
    
    # Assigning a BinOp to a Attribute (line 422):
    str_707515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 28), 'str', 'dispatch_on')
    # Getting the type of 'dispatch_str' (line 422)
    dispatch_str_707516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 44), 'dispatch_str')
    # Applying the binary operator '+' (line 422)
    result_add_707517 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 28), '+', str_707515, dispatch_str_707516)
    
    # Getting the type of 'gen_func_dec' (line 422)
    gen_func_dec_707518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'gen_func_dec')
    # Setting the type of the member '__name__' of a type (line 422)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 4), gen_func_dec_707518, '__name__', result_add_707517)
    # Getting the type of 'gen_func_dec' (line 423)
    gen_func_dec_707519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 11), 'gen_func_dec')
    # Assigning a type to the variable 'stypy_return_type' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'stypy_return_type', gen_func_dec_707519)
    
    # ################# End of 'dispatch_on(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dispatch_on' in the type store
    # Getting the type of 'stypy_return_type' (line 321)
    stypy_return_type_707520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_707520)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dispatch_on'
    return stypy_return_type_707520

# Assigning a type to the variable 'dispatch_on' (line 321)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 0), 'dispatch_on', dispatch_on)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

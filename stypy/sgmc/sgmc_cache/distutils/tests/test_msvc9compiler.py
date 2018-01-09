
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.msvc9compiler.'''
2: import sys
3: import unittest
4: import os
5: 
6: from distutils.errors import DistutilsPlatformError
7: from distutils.tests import support
8: from test.test_support import run_unittest
9: 
10: # A manifest with the only assembly reference being the msvcrt assembly, so
11: # should have the assembly completely stripped.  Note that although the
12: # assembly has a <security> reference the assembly is removed - that is
13: # currently a "feature", not a bug :)
14: _MANIFEST_WITH_ONLY_MSVC_REFERENCE = '''\
15: <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
16: <assembly xmlns="urn:schemas-microsoft-com:asm.v1"
17:           manifestVersion="1.0">
18:   <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
19:     <security>
20:       <requestedPrivileges>
21:         <requestedExecutionLevel level="asInvoker" uiAccess="false">
22:         </requestedExecutionLevel>
23:       </requestedPrivileges>
24:     </security>
25:   </trustInfo>
26:   <dependency>
27:     <dependentAssembly>
28:       <assemblyIdentity type="win32" name="Microsoft.VC90.CRT"
29:          version="9.0.21022.8" processorArchitecture="x86"
30:          publicKeyToken="XXXX">
31:       </assemblyIdentity>
32:     </dependentAssembly>
33:   </dependency>
34: </assembly>
35: '''
36: 
37: # A manifest with references to assemblies other than msvcrt.  When processed,
38: # this assembly should be returned with just the msvcrt part removed.
39: _MANIFEST_WITH_MULTIPLE_REFERENCES = '''\
40: <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
41: <assembly xmlns="urn:schemas-microsoft-com:asm.v1"
42:           manifestVersion="1.0">
43:   <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
44:     <security>
45:       <requestedPrivileges>
46:         <requestedExecutionLevel level="asInvoker" uiAccess="false">
47:         </requestedExecutionLevel>
48:       </requestedPrivileges>
49:     </security>
50:   </trustInfo>
51:   <dependency>
52:     <dependentAssembly>
53:       <assemblyIdentity type="win32" name="Microsoft.VC90.CRT"
54:          version="9.0.21022.8" processorArchitecture="x86"
55:          publicKeyToken="XXXX">
56:       </assemblyIdentity>
57:     </dependentAssembly>
58:   </dependency>
59:   <dependency>
60:     <dependentAssembly>
61:       <assemblyIdentity type="win32" name="Microsoft.VC90.MFC"
62:         version="9.0.21022.8" processorArchitecture="x86"
63:         publicKeyToken="XXXX"></assemblyIdentity>
64:     </dependentAssembly>
65:   </dependency>
66: </assembly>
67: '''
68: 
69: _CLEANED_MANIFEST = '''\
70: <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
71: <assembly xmlns="urn:schemas-microsoft-com:asm.v1"
72:           manifestVersion="1.0">
73:   <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
74:     <security>
75:       <requestedPrivileges>
76:         <requestedExecutionLevel level="asInvoker" uiAccess="false">
77:         </requestedExecutionLevel>
78:       </requestedPrivileges>
79:     </security>
80:   </trustInfo>
81:   <dependency>
82: 
83:   </dependency>
84:   <dependency>
85:     <dependentAssembly>
86:       <assemblyIdentity type="win32" name="Microsoft.VC90.MFC"
87:         version="9.0.21022.8" processorArchitecture="x86"
88:         publicKeyToken="XXXX"></assemblyIdentity>
89:     </dependentAssembly>
90:   </dependency>
91: </assembly>'''
92: 
93: if sys.platform=="win32":
94:     from distutils.msvccompiler import get_build_version
95:     if get_build_version()>=8.0:
96:         SKIP_MESSAGE = None
97:     else:
98:         SKIP_MESSAGE = "These tests are only for MSVC8.0 or above"
99: else:
100:     SKIP_MESSAGE = "These tests are only for win32"
101: 
102: @unittest.skipUnless(SKIP_MESSAGE is None, SKIP_MESSAGE)
103: class msvc9compilerTestCase(support.TempdirManager,
104:                             unittest.TestCase):
105: 
106:     def test_no_compiler(self):
107:         # makes sure query_vcvarsall raises
108:         # a DistutilsPlatformError if the compiler
109:         # is not found
110:         from distutils.msvc9compiler import query_vcvarsall
111:         def _find_vcvarsall(version):
112:             return None
113: 
114:         from distutils import msvc9compiler
115:         old_find_vcvarsall = msvc9compiler.find_vcvarsall
116:         msvc9compiler.find_vcvarsall = _find_vcvarsall
117:         try:
118:             self.assertRaises(DistutilsPlatformError, query_vcvarsall,
119:                              'wont find this version')
120:         finally:
121:             msvc9compiler.find_vcvarsall = old_find_vcvarsall
122: 
123:     def test_reg_class(self):
124:         from distutils.msvc9compiler import Reg
125:         self.assertRaises(KeyError, Reg.get_value, 'xxx', 'xxx')
126: 
127:         # looking for values that should exist on all
128:         # windows registry versions.
129:         path = r'Control Panel\Desktop'
130:         v = Reg.get_value(path, u'dragfullwindows')
131:         self.assertIn(v, (u'0', u'1', u'2'))
132: 
133:         import _winreg
134:         HKCU = _winreg.HKEY_CURRENT_USER
135:         keys = Reg.read_keys(HKCU, 'xxxx')
136:         self.assertEqual(keys, None)
137: 
138:         keys = Reg.read_keys(HKCU, r'Control Panel')
139:         self.assertIn('Desktop', keys)
140: 
141:     def test_remove_visual_c_ref(self):
142:         from distutils.msvc9compiler import MSVCCompiler
143:         tempdir = self.mkdtemp()
144:         manifest = os.path.join(tempdir, 'manifest')
145:         f = open(manifest, 'w')
146:         try:
147:             f.write(_MANIFEST_WITH_MULTIPLE_REFERENCES)
148:         finally:
149:             f.close()
150: 
151:         compiler = MSVCCompiler()
152:         compiler._remove_visual_c_ref(manifest)
153: 
154:         # see what we got
155:         f = open(manifest)
156:         try:
157:             # removing trailing spaces
158:             content = '\n'.join([line.rstrip() for line in f.readlines()])
159:         finally:
160:             f.close()
161: 
162:         # makes sure the manifest was properly cleaned
163:         self.assertEqual(content, _CLEANED_MANIFEST)
164: 
165:     def test_remove_entire_manifest(self):
166:         from distutils.msvc9compiler import MSVCCompiler
167:         tempdir = self.mkdtemp()
168:         manifest = os.path.join(tempdir, 'manifest')
169:         f = open(manifest, 'w')
170:         try:
171:             f.write(_MANIFEST_WITH_ONLY_MSVC_REFERENCE)
172:         finally:
173:             f.close()
174: 
175:         compiler = MSVCCompiler()
176:         got = compiler._remove_visual_c_ref(manifest)
177:         self.assertIsNone(got)
178: 
179: 
180: def test_suite():
181:     return unittest.makeSuite(msvc9compilerTestCase)
182: 
183: if __name__ == "__main__":
184:     run_unittest(test_suite())
185: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_41548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.msvc9compiler.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import sys' statement (line 2)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import unittest' statement (line 3)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import os' statement (line 4)
import os

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from distutils.errors import DistutilsPlatformError' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_41549 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.errors')

if (type(import_41549) is not StypyTypeError):

    if (import_41549 != 'pyd_module'):
        __import__(import_41549)
        sys_modules_41550 = sys.modules[import_41549]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.errors', sys_modules_41550.module_type_store, module_type_store, ['DistutilsPlatformError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_41550, sys_modules_41550.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsPlatformError

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.errors', None, module_type_store, ['DistutilsPlatformError'], [DistutilsPlatformError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.errors', import_41549)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from distutils.tests import support' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_41551 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.tests')

if (type(import_41551) is not StypyTypeError):

    if (import_41551 != 'pyd_module'):
        __import__(import_41551)
        sys_modules_41552 = sys.modules[import_41551]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.tests', sys_modules_41552.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_41552, sys_modules_41552.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.tests', import_41551)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from test.test_support import run_unittest' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_41553 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'test.test_support')

if (type(import_41553) is not StypyTypeError):

    if (import_41553 != 'pyd_module'):
        __import__(import_41553)
        sys_modules_41554 = sys.modules[import_41553]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'test.test_support', sys_modules_41554.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_41554, sys_modules_41554.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'test.test_support', import_41553)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')


# Assigning a Str to a Name (line 14):
str_41555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, (-1)), 'str', '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<assembly xmlns="urn:schemas-microsoft-com:asm.v1"\n          manifestVersion="1.0">\n  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">\n    <security>\n      <requestedPrivileges>\n        <requestedExecutionLevel level="asInvoker" uiAccess="false">\n        </requestedExecutionLevel>\n      </requestedPrivileges>\n    </security>\n  </trustInfo>\n  <dependency>\n    <dependentAssembly>\n      <assemblyIdentity type="win32" name="Microsoft.VC90.CRT"\n         version="9.0.21022.8" processorArchitecture="x86"\n         publicKeyToken="XXXX">\n      </assemblyIdentity>\n    </dependentAssembly>\n  </dependency>\n</assembly>\n')
# Assigning a type to the variable '_MANIFEST_WITH_ONLY_MSVC_REFERENCE' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), '_MANIFEST_WITH_ONLY_MSVC_REFERENCE', str_41555)

# Assigning a Str to a Name (line 39):
str_41556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, (-1)), 'str', '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<assembly xmlns="urn:schemas-microsoft-com:asm.v1"\n          manifestVersion="1.0">\n  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">\n    <security>\n      <requestedPrivileges>\n        <requestedExecutionLevel level="asInvoker" uiAccess="false">\n        </requestedExecutionLevel>\n      </requestedPrivileges>\n    </security>\n  </trustInfo>\n  <dependency>\n    <dependentAssembly>\n      <assemblyIdentity type="win32" name="Microsoft.VC90.CRT"\n         version="9.0.21022.8" processorArchitecture="x86"\n         publicKeyToken="XXXX">\n      </assemblyIdentity>\n    </dependentAssembly>\n  </dependency>\n  <dependency>\n    <dependentAssembly>\n      <assemblyIdentity type="win32" name="Microsoft.VC90.MFC"\n        version="9.0.21022.8" processorArchitecture="x86"\n        publicKeyToken="XXXX"></assemblyIdentity>\n    </dependentAssembly>\n  </dependency>\n</assembly>\n')
# Assigning a type to the variable '_MANIFEST_WITH_MULTIPLE_REFERENCES' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), '_MANIFEST_WITH_MULTIPLE_REFERENCES', str_41556)

# Assigning a Str to a Name (line 69):
str_41557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, (-1)), 'str', '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<assembly xmlns="urn:schemas-microsoft-com:asm.v1"\n          manifestVersion="1.0">\n  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">\n    <security>\n      <requestedPrivileges>\n        <requestedExecutionLevel level="asInvoker" uiAccess="false">\n        </requestedExecutionLevel>\n      </requestedPrivileges>\n    </security>\n  </trustInfo>\n  <dependency>\n\n  </dependency>\n  <dependency>\n    <dependentAssembly>\n      <assemblyIdentity type="win32" name="Microsoft.VC90.MFC"\n        version="9.0.21022.8" processorArchitecture="x86"\n        publicKeyToken="XXXX"></assemblyIdentity>\n    </dependentAssembly>\n  </dependency>\n</assembly>')
# Assigning a type to the variable '_CLEANED_MANIFEST' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), '_CLEANED_MANIFEST', str_41557)


# Getting the type of 'sys' (line 93)
sys_41558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 3), 'sys')
# Obtaining the member 'platform' of a type (line 93)
platform_41559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 3), sys_41558, 'platform')
str_41560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 17), 'str', 'win32')
# Applying the binary operator '==' (line 93)
result_eq_41561 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 3), '==', platform_41559, str_41560)

# Testing the type of an if condition (line 93)
if_condition_41562 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 0), result_eq_41561)
# Assigning a type to the variable 'if_condition_41562' (line 93)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'if_condition_41562', if_condition_41562)
# SSA begins for if statement (line 93)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 94, 4))

# 'from distutils.msvccompiler import get_build_version' statement (line 94)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_41563 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 94, 4), 'distutils.msvccompiler')

if (type(import_41563) is not StypyTypeError):

    if (import_41563 != 'pyd_module'):
        __import__(import_41563)
        sys_modules_41564 = sys.modules[import_41563]
        import_from_module(stypy.reporting.localization.Localization(__file__, 94, 4), 'distutils.msvccompiler', sys_modules_41564.module_type_store, module_type_store, ['get_build_version'])
        nest_module(stypy.reporting.localization.Localization(__file__, 94, 4), __file__, sys_modules_41564, sys_modules_41564.module_type_store, module_type_store)
    else:
        from distutils.msvccompiler import get_build_version

        import_from_module(stypy.reporting.localization.Localization(__file__, 94, 4), 'distutils.msvccompiler', None, module_type_store, ['get_build_version'], [get_build_version])

else:
    # Assigning a type to the variable 'distutils.msvccompiler' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'distutils.msvccompiler', import_41563)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')




# Call to get_build_version(...): (line 95)
# Processing the call keyword arguments (line 95)
kwargs_41566 = {}
# Getting the type of 'get_build_version' (line 95)
get_build_version_41565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 7), 'get_build_version', False)
# Calling get_build_version(args, kwargs) (line 95)
get_build_version_call_result_41567 = invoke(stypy.reporting.localization.Localization(__file__, 95, 7), get_build_version_41565, *[], **kwargs_41566)

float_41568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 28), 'float')
# Applying the binary operator '>=' (line 95)
result_ge_41569 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 7), '>=', get_build_version_call_result_41567, float_41568)

# Testing the type of an if condition (line 95)
if_condition_41570 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 4), result_ge_41569)
# Assigning a type to the variable 'if_condition_41570' (line 95)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'if_condition_41570', if_condition_41570)
# SSA begins for if statement (line 95)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Name (line 96):
# Getting the type of 'None' (line 96)
None_41571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 23), 'None')
# Assigning a type to the variable 'SKIP_MESSAGE' (line 96)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'SKIP_MESSAGE', None_41571)
# SSA branch for the else part of an if statement (line 95)
module_type_store.open_ssa_branch('else')

# Assigning a Str to a Name (line 98):
str_41572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 23), 'str', 'These tests are only for MSVC8.0 or above')
# Assigning a type to the variable 'SKIP_MESSAGE' (line 98)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'SKIP_MESSAGE', str_41572)
# SSA join for if statement (line 95)
module_type_store = module_type_store.join_ssa_context()

# SSA branch for the else part of an if statement (line 93)
module_type_store.open_ssa_branch('else')

# Assigning a Str to a Name (line 100):
str_41573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 19), 'str', 'These tests are only for win32')
# Assigning a type to the variable 'SKIP_MESSAGE' (line 100)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'SKIP_MESSAGE', str_41573)
# SSA join for if statement (line 93)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'msvc9compilerTestCase' class
# Getting the type of 'support' (line 103)
support_41574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 28), 'support')
# Obtaining the member 'TempdirManager' of a type (line 103)
TempdirManager_41575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 28), support_41574, 'TempdirManager')
# Getting the type of 'unittest' (line 104)
unittest_41576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 28), 'unittest')
# Obtaining the member 'TestCase' of a type (line 104)
TestCase_41577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 28), unittest_41576, 'TestCase')

class msvc9compilerTestCase(TempdirManager_41575, TestCase_41577, ):

    @norecursion
    def test_no_compiler(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_no_compiler'
        module_type_store = module_type_store.open_function_context('test_no_compiler', 106, 4, False)
        # Assigning a type to the variable 'self' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        msvc9compilerTestCase.test_no_compiler.__dict__.__setitem__('stypy_localization', localization)
        msvc9compilerTestCase.test_no_compiler.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        msvc9compilerTestCase.test_no_compiler.__dict__.__setitem__('stypy_type_store', module_type_store)
        msvc9compilerTestCase.test_no_compiler.__dict__.__setitem__('stypy_function_name', 'msvc9compilerTestCase.test_no_compiler')
        msvc9compilerTestCase.test_no_compiler.__dict__.__setitem__('stypy_param_names_list', [])
        msvc9compilerTestCase.test_no_compiler.__dict__.__setitem__('stypy_varargs_param_name', None)
        msvc9compilerTestCase.test_no_compiler.__dict__.__setitem__('stypy_kwargs_param_name', None)
        msvc9compilerTestCase.test_no_compiler.__dict__.__setitem__('stypy_call_defaults', defaults)
        msvc9compilerTestCase.test_no_compiler.__dict__.__setitem__('stypy_call_varargs', varargs)
        msvc9compilerTestCase.test_no_compiler.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        msvc9compilerTestCase.test_no_compiler.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'msvc9compilerTestCase.test_no_compiler', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_no_compiler', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_no_compiler(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 110, 8))
        
        # 'from distutils.msvc9compiler import query_vcvarsall' statement (line 110)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
        import_41578 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 110, 8), 'distutils.msvc9compiler')

        if (type(import_41578) is not StypyTypeError):

            if (import_41578 != 'pyd_module'):
                __import__(import_41578)
                sys_modules_41579 = sys.modules[import_41578]
                import_from_module(stypy.reporting.localization.Localization(__file__, 110, 8), 'distutils.msvc9compiler', sys_modules_41579.module_type_store, module_type_store, ['query_vcvarsall'])
                nest_module(stypy.reporting.localization.Localization(__file__, 110, 8), __file__, sys_modules_41579, sys_modules_41579.module_type_store, module_type_store)
            else:
                from distutils.msvc9compiler import query_vcvarsall

                import_from_module(stypy.reporting.localization.Localization(__file__, 110, 8), 'distutils.msvc9compiler', None, module_type_store, ['query_vcvarsall'], [query_vcvarsall])

        else:
            # Assigning a type to the variable 'distutils.msvc9compiler' (line 110)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'distutils.msvc9compiler', import_41578)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')
        

        @norecursion
        def _find_vcvarsall(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_find_vcvarsall'
            module_type_store = module_type_store.open_function_context('_find_vcvarsall', 111, 8, False)
            
            # Passed parameters checking function
            _find_vcvarsall.stypy_localization = localization
            _find_vcvarsall.stypy_type_of_self = None
            _find_vcvarsall.stypy_type_store = module_type_store
            _find_vcvarsall.stypy_function_name = '_find_vcvarsall'
            _find_vcvarsall.stypy_param_names_list = ['version']
            _find_vcvarsall.stypy_varargs_param_name = None
            _find_vcvarsall.stypy_kwargs_param_name = None
            _find_vcvarsall.stypy_call_defaults = defaults
            _find_vcvarsall.stypy_call_varargs = varargs
            _find_vcvarsall.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_find_vcvarsall', ['version'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_find_vcvarsall', localization, ['version'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_find_vcvarsall(...)' code ##################

            # Getting the type of 'None' (line 112)
            None_41580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 19), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 112)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'stypy_return_type', None_41580)
            
            # ################# End of '_find_vcvarsall(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_find_vcvarsall' in the type store
            # Getting the type of 'stypy_return_type' (line 111)
            stypy_return_type_41581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_41581)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_find_vcvarsall'
            return stypy_return_type_41581

        # Assigning a type to the variable '_find_vcvarsall' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), '_find_vcvarsall', _find_vcvarsall)
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 114, 8))
        
        # 'from distutils import msvc9compiler' statement (line 114)
        try:
            from distutils import msvc9compiler

        except:
            msvc9compiler = UndefinedType
        import_from_module(stypy.reporting.localization.Localization(__file__, 114, 8), 'distutils', None, module_type_store, ['msvc9compiler'], [msvc9compiler])
        
        
        # Assigning a Attribute to a Name (line 115):
        # Getting the type of 'msvc9compiler' (line 115)
        msvc9compiler_41582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 29), 'msvc9compiler')
        # Obtaining the member 'find_vcvarsall' of a type (line 115)
        find_vcvarsall_41583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 29), msvc9compiler_41582, 'find_vcvarsall')
        # Assigning a type to the variable 'old_find_vcvarsall' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'old_find_vcvarsall', find_vcvarsall_41583)
        
        # Assigning a Name to a Attribute (line 116):
        # Getting the type of '_find_vcvarsall' (line 116)
        _find_vcvarsall_41584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 39), '_find_vcvarsall')
        # Getting the type of 'msvc9compiler' (line 116)
        msvc9compiler_41585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'msvc9compiler')
        # Setting the type of the member 'find_vcvarsall' of a type (line 116)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), msvc9compiler_41585, 'find_vcvarsall', _find_vcvarsall_41584)
        
        # Try-finally block (line 117)
        
        # Call to assertRaises(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'DistutilsPlatformError' (line 118)
        DistutilsPlatformError_41588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 30), 'DistutilsPlatformError', False)
        # Getting the type of 'query_vcvarsall' (line 118)
        query_vcvarsall_41589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 54), 'query_vcvarsall', False)
        str_41590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 29), 'str', 'wont find this version')
        # Processing the call keyword arguments (line 118)
        kwargs_41591 = {}
        # Getting the type of 'self' (line 118)
        self_41586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 118)
        assertRaises_41587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 12), self_41586, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 118)
        assertRaises_call_result_41592 = invoke(stypy.reporting.localization.Localization(__file__, 118, 12), assertRaises_41587, *[DistutilsPlatformError_41588, query_vcvarsall_41589, str_41590], **kwargs_41591)
        
        
        # finally branch of the try-finally block (line 117)
        
        # Assigning a Name to a Attribute (line 121):
        # Getting the type of 'old_find_vcvarsall' (line 121)
        old_find_vcvarsall_41593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 43), 'old_find_vcvarsall')
        # Getting the type of 'msvc9compiler' (line 121)
        msvc9compiler_41594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'msvc9compiler')
        # Setting the type of the member 'find_vcvarsall' of a type (line 121)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 12), msvc9compiler_41594, 'find_vcvarsall', old_find_vcvarsall_41593)
        
        
        # ################# End of 'test_no_compiler(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_no_compiler' in the type store
        # Getting the type of 'stypy_return_type' (line 106)
        stypy_return_type_41595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_41595)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_no_compiler'
        return stypy_return_type_41595


    @norecursion
    def test_reg_class(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_reg_class'
        module_type_store = module_type_store.open_function_context('test_reg_class', 123, 4, False)
        # Assigning a type to the variable 'self' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        msvc9compilerTestCase.test_reg_class.__dict__.__setitem__('stypy_localization', localization)
        msvc9compilerTestCase.test_reg_class.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        msvc9compilerTestCase.test_reg_class.__dict__.__setitem__('stypy_type_store', module_type_store)
        msvc9compilerTestCase.test_reg_class.__dict__.__setitem__('stypy_function_name', 'msvc9compilerTestCase.test_reg_class')
        msvc9compilerTestCase.test_reg_class.__dict__.__setitem__('stypy_param_names_list', [])
        msvc9compilerTestCase.test_reg_class.__dict__.__setitem__('stypy_varargs_param_name', None)
        msvc9compilerTestCase.test_reg_class.__dict__.__setitem__('stypy_kwargs_param_name', None)
        msvc9compilerTestCase.test_reg_class.__dict__.__setitem__('stypy_call_defaults', defaults)
        msvc9compilerTestCase.test_reg_class.__dict__.__setitem__('stypy_call_varargs', varargs)
        msvc9compilerTestCase.test_reg_class.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        msvc9compilerTestCase.test_reg_class.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'msvc9compilerTestCase.test_reg_class', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_reg_class', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_reg_class(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 124, 8))
        
        # 'from distutils.msvc9compiler import Reg' statement (line 124)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
        import_41596 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 124, 8), 'distutils.msvc9compiler')

        if (type(import_41596) is not StypyTypeError):

            if (import_41596 != 'pyd_module'):
                __import__(import_41596)
                sys_modules_41597 = sys.modules[import_41596]
                import_from_module(stypy.reporting.localization.Localization(__file__, 124, 8), 'distutils.msvc9compiler', sys_modules_41597.module_type_store, module_type_store, ['Reg'])
                nest_module(stypy.reporting.localization.Localization(__file__, 124, 8), __file__, sys_modules_41597, sys_modules_41597.module_type_store, module_type_store)
            else:
                from distutils.msvc9compiler import Reg

                import_from_module(stypy.reporting.localization.Localization(__file__, 124, 8), 'distutils.msvc9compiler', None, module_type_store, ['Reg'], [Reg])

        else:
            # Assigning a type to the variable 'distutils.msvc9compiler' (line 124)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'distutils.msvc9compiler', import_41596)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')
        
        
        # Call to assertRaises(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'KeyError' (line 125)
        KeyError_41600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 26), 'KeyError', False)
        # Getting the type of 'Reg' (line 125)
        Reg_41601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 36), 'Reg', False)
        # Obtaining the member 'get_value' of a type (line 125)
        get_value_41602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 36), Reg_41601, 'get_value')
        str_41603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 51), 'str', 'xxx')
        str_41604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 58), 'str', 'xxx')
        # Processing the call keyword arguments (line 125)
        kwargs_41605 = {}
        # Getting the type of 'self' (line 125)
        self_41598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 125)
        assertRaises_41599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), self_41598, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 125)
        assertRaises_call_result_41606 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), assertRaises_41599, *[KeyError_41600, get_value_41602, str_41603, str_41604], **kwargs_41605)
        
        
        # Assigning a Str to a Name (line 129):
        str_41607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 15), 'str', 'Control Panel\\Desktop')
        # Assigning a type to the variable 'path' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'path', str_41607)
        
        # Assigning a Call to a Name (line 130):
        
        # Call to get_value(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'path' (line 130)
        path_41610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 26), 'path', False)
        unicode_41611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 32), 'unicode', u'dragfullwindows')
        # Processing the call keyword arguments (line 130)
        kwargs_41612 = {}
        # Getting the type of 'Reg' (line 130)
        Reg_41608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'Reg', False)
        # Obtaining the member 'get_value' of a type (line 130)
        get_value_41609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 12), Reg_41608, 'get_value')
        # Calling get_value(args, kwargs) (line 130)
        get_value_call_result_41613 = invoke(stypy.reporting.localization.Localization(__file__, 130, 12), get_value_41609, *[path_41610, unicode_41611], **kwargs_41612)
        
        # Assigning a type to the variable 'v' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'v', get_value_call_result_41613)
        
        # Call to assertIn(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'v' (line 131)
        v_41616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 22), 'v', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 131)
        tuple_41617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 131)
        # Adding element type (line 131)
        unicode_41618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 26), 'unicode', u'0')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 26), tuple_41617, unicode_41618)
        # Adding element type (line 131)
        unicode_41619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 32), 'unicode', u'1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 26), tuple_41617, unicode_41619)
        # Adding element type (line 131)
        unicode_41620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 38), 'unicode', u'2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 26), tuple_41617, unicode_41620)
        
        # Processing the call keyword arguments (line 131)
        kwargs_41621 = {}
        # Getting the type of 'self' (line 131)
        self_41614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 131)
        assertIn_41615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), self_41614, 'assertIn')
        # Calling assertIn(args, kwargs) (line 131)
        assertIn_call_result_41622 = invoke(stypy.reporting.localization.Localization(__file__, 131, 8), assertIn_41615, *[v_41616, tuple_41617], **kwargs_41621)
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 133, 8))
        
        # 'import _winreg' statement (line 133)
        import _winreg

        import_module(stypy.reporting.localization.Localization(__file__, 133, 8), '_winreg', _winreg, module_type_store)
        
        
        # Assigning a Attribute to a Name (line 134):
        # Getting the type of '_winreg' (line 134)
        _winreg_41623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 15), '_winreg')
        # Obtaining the member 'HKEY_CURRENT_USER' of a type (line 134)
        HKEY_CURRENT_USER_41624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 15), _winreg_41623, 'HKEY_CURRENT_USER')
        # Assigning a type to the variable 'HKCU' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'HKCU', HKEY_CURRENT_USER_41624)
        
        # Assigning a Call to a Name (line 135):
        
        # Call to read_keys(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'HKCU' (line 135)
        HKCU_41627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 29), 'HKCU', False)
        str_41628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 35), 'str', 'xxxx')
        # Processing the call keyword arguments (line 135)
        kwargs_41629 = {}
        # Getting the type of 'Reg' (line 135)
        Reg_41625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 15), 'Reg', False)
        # Obtaining the member 'read_keys' of a type (line 135)
        read_keys_41626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 15), Reg_41625, 'read_keys')
        # Calling read_keys(args, kwargs) (line 135)
        read_keys_call_result_41630 = invoke(stypy.reporting.localization.Localization(__file__, 135, 15), read_keys_41626, *[HKCU_41627, str_41628], **kwargs_41629)
        
        # Assigning a type to the variable 'keys' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'keys', read_keys_call_result_41630)
        
        # Call to assertEqual(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'keys' (line 136)
        keys_41633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 25), 'keys', False)
        # Getting the type of 'None' (line 136)
        None_41634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 31), 'None', False)
        # Processing the call keyword arguments (line 136)
        kwargs_41635 = {}
        # Getting the type of 'self' (line 136)
        self_41631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 136)
        assertEqual_41632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), self_41631, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 136)
        assertEqual_call_result_41636 = invoke(stypy.reporting.localization.Localization(__file__, 136, 8), assertEqual_41632, *[keys_41633, None_41634], **kwargs_41635)
        
        
        # Assigning a Call to a Name (line 138):
        
        # Call to read_keys(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'HKCU' (line 138)
        HKCU_41639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 29), 'HKCU', False)
        str_41640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 35), 'str', 'Control Panel')
        # Processing the call keyword arguments (line 138)
        kwargs_41641 = {}
        # Getting the type of 'Reg' (line 138)
        Reg_41637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 15), 'Reg', False)
        # Obtaining the member 'read_keys' of a type (line 138)
        read_keys_41638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 15), Reg_41637, 'read_keys')
        # Calling read_keys(args, kwargs) (line 138)
        read_keys_call_result_41642 = invoke(stypy.reporting.localization.Localization(__file__, 138, 15), read_keys_41638, *[HKCU_41639, str_41640], **kwargs_41641)
        
        # Assigning a type to the variable 'keys' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'keys', read_keys_call_result_41642)
        
        # Call to assertIn(...): (line 139)
        # Processing the call arguments (line 139)
        str_41645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 22), 'str', 'Desktop')
        # Getting the type of 'keys' (line 139)
        keys_41646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 33), 'keys', False)
        # Processing the call keyword arguments (line 139)
        kwargs_41647 = {}
        # Getting the type of 'self' (line 139)
        self_41643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 139)
        assertIn_41644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), self_41643, 'assertIn')
        # Calling assertIn(args, kwargs) (line 139)
        assertIn_call_result_41648 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), assertIn_41644, *[str_41645, keys_41646], **kwargs_41647)
        
        
        # ################# End of 'test_reg_class(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_reg_class' in the type store
        # Getting the type of 'stypy_return_type' (line 123)
        stypy_return_type_41649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_41649)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_reg_class'
        return stypy_return_type_41649


    @norecursion
    def test_remove_visual_c_ref(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_remove_visual_c_ref'
        module_type_store = module_type_store.open_function_context('test_remove_visual_c_ref', 141, 4, False)
        # Assigning a type to the variable 'self' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        msvc9compilerTestCase.test_remove_visual_c_ref.__dict__.__setitem__('stypy_localization', localization)
        msvc9compilerTestCase.test_remove_visual_c_ref.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        msvc9compilerTestCase.test_remove_visual_c_ref.__dict__.__setitem__('stypy_type_store', module_type_store)
        msvc9compilerTestCase.test_remove_visual_c_ref.__dict__.__setitem__('stypy_function_name', 'msvc9compilerTestCase.test_remove_visual_c_ref')
        msvc9compilerTestCase.test_remove_visual_c_ref.__dict__.__setitem__('stypy_param_names_list', [])
        msvc9compilerTestCase.test_remove_visual_c_ref.__dict__.__setitem__('stypy_varargs_param_name', None)
        msvc9compilerTestCase.test_remove_visual_c_ref.__dict__.__setitem__('stypy_kwargs_param_name', None)
        msvc9compilerTestCase.test_remove_visual_c_ref.__dict__.__setitem__('stypy_call_defaults', defaults)
        msvc9compilerTestCase.test_remove_visual_c_ref.__dict__.__setitem__('stypy_call_varargs', varargs)
        msvc9compilerTestCase.test_remove_visual_c_ref.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        msvc9compilerTestCase.test_remove_visual_c_ref.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'msvc9compilerTestCase.test_remove_visual_c_ref', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_remove_visual_c_ref', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_remove_visual_c_ref(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 142, 8))
        
        # 'from distutils.msvc9compiler import MSVCCompiler' statement (line 142)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
        import_41650 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 142, 8), 'distutils.msvc9compiler')

        if (type(import_41650) is not StypyTypeError):

            if (import_41650 != 'pyd_module'):
                __import__(import_41650)
                sys_modules_41651 = sys.modules[import_41650]
                import_from_module(stypy.reporting.localization.Localization(__file__, 142, 8), 'distutils.msvc9compiler', sys_modules_41651.module_type_store, module_type_store, ['MSVCCompiler'])
                nest_module(stypy.reporting.localization.Localization(__file__, 142, 8), __file__, sys_modules_41651, sys_modules_41651.module_type_store, module_type_store)
            else:
                from distutils.msvc9compiler import MSVCCompiler

                import_from_module(stypy.reporting.localization.Localization(__file__, 142, 8), 'distutils.msvc9compiler', None, module_type_store, ['MSVCCompiler'], [MSVCCompiler])

        else:
            # Assigning a type to the variable 'distutils.msvc9compiler' (line 142)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'distutils.msvc9compiler', import_41650)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')
        
        
        # Assigning a Call to a Name (line 143):
        
        # Call to mkdtemp(...): (line 143)
        # Processing the call keyword arguments (line 143)
        kwargs_41654 = {}
        # Getting the type of 'self' (line 143)
        self_41652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 18), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 143)
        mkdtemp_41653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 18), self_41652, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 143)
        mkdtemp_call_result_41655 = invoke(stypy.reporting.localization.Localization(__file__, 143, 18), mkdtemp_41653, *[], **kwargs_41654)
        
        # Assigning a type to the variable 'tempdir' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'tempdir', mkdtemp_call_result_41655)
        
        # Assigning a Call to a Name (line 144):
        
        # Call to join(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'tempdir' (line 144)
        tempdir_41659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 32), 'tempdir', False)
        str_41660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 41), 'str', 'manifest')
        # Processing the call keyword arguments (line 144)
        kwargs_41661 = {}
        # Getting the type of 'os' (line 144)
        os_41656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 144)
        path_41657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 19), os_41656, 'path')
        # Obtaining the member 'join' of a type (line 144)
        join_41658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 19), path_41657, 'join')
        # Calling join(args, kwargs) (line 144)
        join_call_result_41662 = invoke(stypy.reporting.localization.Localization(__file__, 144, 19), join_41658, *[tempdir_41659, str_41660], **kwargs_41661)
        
        # Assigning a type to the variable 'manifest' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'manifest', join_call_result_41662)
        
        # Assigning a Call to a Name (line 145):
        
        # Call to open(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'manifest' (line 145)
        manifest_41664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 17), 'manifest', False)
        str_41665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 27), 'str', 'w')
        # Processing the call keyword arguments (line 145)
        kwargs_41666 = {}
        # Getting the type of 'open' (line 145)
        open_41663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'open', False)
        # Calling open(args, kwargs) (line 145)
        open_call_result_41667 = invoke(stypy.reporting.localization.Localization(__file__, 145, 12), open_41663, *[manifest_41664, str_41665], **kwargs_41666)
        
        # Assigning a type to the variable 'f' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'f', open_call_result_41667)
        
        # Try-finally block (line 146)
        
        # Call to write(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of '_MANIFEST_WITH_MULTIPLE_REFERENCES' (line 147)
        _MANIFEST_WITH_MULTIPLE_REFERENCES_41670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 20), '_MANIFEST_WITH_MULTIPLE_REFERENCES', False)
        # Processing the call keyword arguments (line 147)
        kwargs_41671 = {}
        # Getting the type of 'f' (line 147)
        f_41668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'f', False)
        # Obtaining the member 'write' of a type (line 147)
        write_41669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 12), f_41668, 'write')
        # Calling write(args, kwargs) (line 147)
        write_call_result_41672 = invoke(stypy.reporting.localization.Localization(__file__, 147, 12), write_41669, *[_MANIFEST_WITH_MULTIPLE_REFERENCES_41670], **kwargs_41671)
        
        
        # finally branch of the try-finally block (line 146)
        
        # Call to close(...): (line 149)
        # Processing the call keyword arguments (line 149)
        kwargs_41675 = {}
        # Getting the type of 'f' (line 149)
        f_41673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 149)
        close_41674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 12), f_41673, 'close')
        # Calling close(args, kwargs) (line 149)
        close_call_result_41676 = invoke(stypy.reporting.localization.Localization(__file__, 149, 12), close_41674, *[], **kwargs_41675)
        
        
        
        # Assigning a Call to a Name (line 151):
        
        # Call to MSVCCompiler(...): (line 151)
        # Processing the call keyword arguments (line 151)
        kwargs_41678 = {}
        # Getting the type of 'MSVCCompiler' (line 151)
        MSVCCompiler_41677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 19), 'MSVCCompiler', False)
        # Calling MSVCCompiler(args, kwargs) (line 151)
        MSVCCompiler_call_result_41679 = invoke(stypy.reporting.localization.Localization(__file__, 151, 19), MSVCCompiler_41677, *[], **kwargs_41678)
        
        # Assigning a type to the variable 'compiler' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'compiler', MSVCCompiler_call_result_41679)
        
        # Call to _remove_visual_c_ref(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'manifest' (line 152)
        manifest_41682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 38), 'manifest', False)
        # Processing the call keyword arguments (line 152)
        kwargs_41683 = {}
        # Getting the type of 'compiler' (line 152)
        compiler_41680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'compiler', False)
        # Obtaining the member '_remove_visual_c_ref' of a type (line 152)
        _remove_visual_c_ref_41681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 8), compiler_41680, '_remove_visual_c_ref')
        # Calling _remove_visual_c_ref(args, kwargs) (line 152)
        _remove_visual_c_ref_call_result_41684 = invoke(stypy.reporting.localization.Localization(__file__, 152, 8), _remove_visual_c_ref_41681, *[manifest_41682], **kwargs_41683)
        
        
        # Assigning a Call to a Name (line 155):
        
        # Call to open(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'manifest' (line 155)
        manifest_41686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 17), 'manifest', False)
        # Processing the call keyword arguments (line 155)
        kwargs_41687 = {}
        # Getting the type of 'open' (line 155)
        open_41685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'open', False)
        # Calling open(args, kwargs) (line 155)
        open_call_result_41688 = invoke(stypy.reporting.localization.Localization(__file__, 155, 12), open_41685, *[manifest_41686], **kwargs_41687)
        
        # Assigning a type to the variable 'f' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'f', open_call_result_41688)
        
        # Try-finally block (line 156)
        
        # Assigning a Call to a Name (line 158):
        
        # Call to join(...): (line 158)
        # Processing the call arguments (line 158)
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to readlines(...): (line 158)
        # Processing the call keyword arguments (line 158)
        kwargs_41697 = {}
        # Getting the type of 'f' (line 158)
        f_41695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 59), 'f', False)
        # Obtaining the member 'readlines' of a type (line 158)
        readlines_41696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 59), f_41695, 'readlines')
        # Calling readlines(args, kwargs) (line 158)
        readlines_call_result_41698 = invoke(stypy.reporting.localization.Localization(__file__, 158, 59), readlines_41696, *[], **kwargs_41697)
        
        comprehension_41699 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 33), readlines_call_result_41698)
        # Assigning a type to the variable 'line' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 33), 'line', comprehension_41699)
        
        # Call to rstrip(...): (line 158)
        # Processing the call keyword arguments (line 158)
        kwargs_41693 = {}
        # Getting the type of 'line' (line 158)
        line_41691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 33), 'line', False)
        # Obtaining the member 'rstrip' of a type (line 158)
        rstrip_41692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 33), line_41691, 'rstrip')
        # Calling rstrip(args, kwargs) (line 158)
        rstrip_call_result_41694 = invoke(stypy.reporting.localization.Localization(__file__, 158, 33), rstrip_41692, *[], **kwargs_41693)
        
        list_41700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 33), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 33), list_41700, rstrip_call_result_41694)
        # Processing the call keyword arguments (line 158)
        kwargs_41701 = {}
        str_41689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 22), 'str', '\n')
        # Obtaining the member 'join' of a type (line 158)
        join_41690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 22), str_41689, 'join')
        # Calling join(args, kwargs) (line 158)
        join_call_result_41702 = invoke(stypy.reporting.localization.Localization(__file__, 158, 22), join_41690, *[list_41700], **kwargs_41701)
        
        # Assigning a type to the variable 'content' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'content', join_call_result_41702)
        
        # finally branch of the try-finally block (line 156)
        
        # Call to close(...): (line 160)
        # Processing the call keyword arguments (line 160)
        kwargs_41705 = {}
        # Getting the type of 'f' (line 160)
        f_41703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 160)
        close_41704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 12), f_41703, 'close')
        # Calling close(args, kwargs) (line 160)
        close_call_result_41706 = invoke(stypy.reporting.localization.Localization(__file__, 160, 12), close_41704, *[], **kwargs_41705)
        
        
        
        # Call to assertEqual(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'content' (line 163)
        content_41709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 25), 'content', False)
        # Getting the type of '_CLEANED_MANIFEST' (line 163)
        _CLEANED_MANIFEST_41710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 34), '_CLEANED_MANIFEST', False)
        # Processing the call keyword arguments (line 163)
        kwargs_41711 = {}
        # Getting the type of 'self' (line 163)
        self_41707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 163)
        assertEqual_41708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 8), self_41707, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 163)
        assertEqual_call_result_41712 = invoke(stypy.reporting.localization.Localization(__file__, 163, 8), assertEqual_41708, *[content_41709, _CLEANED_MANIFEST_41710], **kwargs_41711)
        
        
        # ################# End of 'test_remove_visual_c_ref(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_remove_visual_c_ref' in the type store
        # Getting the type of 'stypy_return_type' (line 141)
        stypy_return_type_41713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_41713)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_remove_visual_c_ref'
        return stypy_return_type_41713


    @norecursion
    def test_remove_entire_manifest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_remove_entire_manifest'
        module_type_store = module_type_store.open_function_context('test_remove_entire_manifest', 165, 4, False)
        # Assigning a type to the variable 'self' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        msvc9compilerTestCase.test_remove_entire_manifest.__dict__.__setitem__('stypy_localization', localization)
        msvc9compilerTestCase.test_remove_entire_manifest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        msvc9compilerTestCase.test_remove_entire_manifest.__dict__.__setitem__('stypy_type_store', module_type_store)
        msvc9compilerTestCase.test_remove_entire_manifest.__dict__.__setitem__('stypy_function_name', 'msvc9compilerTestCase.test_remove_entire_manifest')
        msvc9compilerTestCase.test_remove_entire_manifest.__dict__.__setitem__('stypy_param_names_list', [])
        msvc9compilerTestCase.test_remove_entire_manifest.__dict__.__setitem__('stypy_varargs_param_name', None)
        msvc9compilerTestCase.test_remove_entire_manifest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        msvc9compilerTestCase.test_remove_entire_manifest.__dict__.__setitem__('stypy_call_defaults', defaults)
        msvc9compilerTestCase.test_remove_entire_manifest.__dict__.__setitem__('stypy_call_varargs', varargs)
        msvc9compilerTestCase.test_remove_entire_manifest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        msvc9compilerTestCase.test_remove_entire_manifest.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'msvc9compilerTestCase.test_remove_entire_manifest', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_remove_entire_manifest', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_remove_entire_manifest(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 166, 8))
        
        # 'from distutils.msvc9compiler import MSVCCompiler' statement (line 166)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
        import_41714 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 166, 8), 'distutils.msvc9compiler')

        if (type(import_41714) is not StypyTypeError):

            if (import_41714 != 'pyd_module'):
                __import__(import_41714)
                sys_modules_41715 = sys.modules[import_41714]
                import_from_module(stypy.reporting.localization.Localization(__file__, 166, 8), 'distutils.msvc9compiler', sys_modules_41715.module_type_store, module_type_store, ['MSVCCompiler'])
                nest_module(stypy.reporting.localization.Localization(__file__, 166, 8), __file__, sys_modules_41715, sys_modules_41715.module_type_store, module_type_store)
            else:
                from distutils.msvc9compiler import MSVCCompiler

                import_from_module(stypy.reporting.localization.Localization(__file__, 166, 8), 'distutils.msvc9compiler', None, module_type_store, ['MSVCCompiler'], [MSVCCompiler])

        else:
            # Assigning a type to the variable 'distutils.msvc9compiler' (line 166)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'distutils.msvc9compiler', import_41714)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')
        
        
        # Assigning a Call to a Name (line 167):
        
        # Call to mkdtemp(...): (line 167)
        # Processing the call keyword arguments (line 167)
        kwargs_41718 = {}
        # Getting the type of 'self' (line 167)
        self_41716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 18), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 167)
        mkdtemp_41717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 18), self_41716, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 167)
        mkdtemp_call_result_41719 = invoke(stypy.reporting.localization.Localization(__file__, 167, 18), mkdtemp_41717, *[], **kwargs_41718)
        
        # Assigning a type to the variable 'tempdir' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'tempdir', mkdtemp_call_result_41719)
        
        # Assigning a Call to a Name (line 168):
        
        # Call to join(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'tempdir' (line 168)
        tempdir_41723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 32), 'tempdir', False)
        str_41724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 41), 'str', 'manifest')
        # Processing the call keyword arguments (line 168)
        kwargs_41725 = {}
        # Getting the type of 'os' (line 168)
        os_41720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 168)
        path_41721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 19), os_41720, 'path')
        # Obtaining the member 'join' of a type (line 168)
        join_41722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 19), path_41721, 'join')
        # Calling join(args, kwargs) (line 168)
        join_call_result_41726 = invoke(stypy.reporting.localization.Localization(__file__, 168, 19), join_41722, *[tempdir_41723, str_41724], **kwargs_41725)
        
        # Assigning a type to the variable 'manifest' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'manifest', join_call_result_41726)
        
        # Assigning a Call to a Name (line 169):
        
        # Call to open(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'manifest' (line 169)
        manifest_41728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 17), 'manifest', False)
        str_41729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 27), 'str', 'w')
        # Processing the call keyword arguments (line 169)
        kwargs_41730 = {}
        # Getting the type of 'open' (line 169)
        open_41727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'open', False)
        # Calling open(args, kwargs) (line 169)
        open_call_result_41731 = invoke(stypy.reporting.localization.Localization(__file__, 169, 12), open_41727, *[manifest_41728, str_41729], **kwargs_41730)
        
        # Assigning a type to the variable 'f' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'f', open_call_result_41731)
        
        # Try-finally block (line 170)
        
        # Call to write(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of '_MANIFEST_WITH_ONLY_MSVC_REFERENCE' (line 171)
        _MANIFEST_WITH_ONLY_MSVC_REFERENCE_41734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 20), '_MANIFEST_WITH_ONLY_MSVC_REFERENCE', False)
        # Processing the call keyword arguments (line 171)
        kwargs_41735 = {}
        # Getting the type of 'f' (line 171)
        f_41732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'f', False)
        # Obtaining the member 'write' of a type (line 171)
        write_41733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 12), f_41732, 'write')
        # Calling write(args, kwargs) (line 171)
        write_call_result_41736 = invoke(stypy.reporting.localization.Localization(__file__, 171, 12), write_41733, *[_MANIFEST_WITH_ONLY_MSVC_REFERENCE_41734], **kwargs_41735)
        
        
        # finally branch of the try-finally block (line 170)
        
        # Call to close(...): (line 173)
        # Processing the call keyword arguments (line 173)
        kwargs_41739 = {}
        # Getting the type of 'f' (line 173)
        f_41737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 173)
        close_41738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 12), f_41737, 'close')
        # Calling close(args, kwargs) (line 173)
        close_call_result_41740 = invoke(stypy.reporting.localization.Localization(__file__, 173, 12), close_41738, *[], **kwargs_41739)
        
        
        
        # Assigning a Call to a Name (line 175):
        
        # Call to MSVCCompiler(...): (line 175)
        # Processing the call keyword arguments (line 175)
        kwargs_41742 = {}
        # Getting the type of 'MSVCCompiler' (line 175)
        MSVCCompiler_41741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 19), 'MSVCCompiler', False)
        # Calling MSVCCompiler(args, kwargs) (line 175)
        MSVCCompiler_call_result_41743 = invoke(stypy.reporting.localization.Localization(__file__, 175, 19), MSVCCompiler_41741, *[], **kwargs_41742)
        
        # Assigning a type to the variable 'compiler' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'compiler', MSVCCompiler_call_result_41743)
        
        # Assigning a Call to a Name (line 176):
        
        # Call to _remove_visual_c_ref(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'manifest' (line 176)
        manifest_41746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 44), 'manifest', False)
        # Processing the call keyword arguments (line 176)
        kwargs_41747 = {}
        # Getting the type of 'compiler' (line 176)
        compiler_41744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 14), 'compiler', False)
        # Obtaining the member '_remove_visual_c_ref' of a type (line 176)
        _remove_visual_c_ref_41745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 14), compiler_41744, '_remove_visual_c_ref')
        # Calling _remove_visual_c_ref(args, kwargs) (line 176)
        _remove_visual_c_ref_call_result_41748 = invoke(stypy.reporting.localization.Localization(__file__, 176, 14), _remove_visual_c_ref_41745, *[manifest_41746], **kwargs_41747)
        
        # Assigning a type to the variable 'got' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'got', _remove_visual_c_ref_call_result_41748)
        
        # Call to assertIsNone(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'got' (line 177)
        got_41751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 26), 'got', False)
        # Processing the call keyword arguments (line 177)
        kwargs_41752 = {}
        # Getting the type of 'self' (line 177)
        self_41749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'self', False)
        # Obtaining the member 'assertIsNone' of a type (line 177)
        assertIsNone_41750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), self_41749, 'assertIsNone')
        # Calling assertIsNone(args, kwargs) (line 177)
        assertIsNone_call_result_41753 = invoke(stypy.reporting.localization.Localization(__file__, 177, 8), assertIsNone_41750, *[got_41751], **kwargs_41752)
        
        
        # ################# End of 'test_remove_entire_manifest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_remove_entire_manifest' in the type store
        # Getting the type of 'stypy_return_type' (line 165)
        stypy_return_type_41754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_41754)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_remove_entire_manifest'
        return stypy_return_type_41754


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 102, 0, False)
        # Assigning a type to the variable 'self' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'msvc9compilerTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'msvc9compilerTestCase' (line 102)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), 'msvc9compilerTestCase', msvc9compilerTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 180, 0, False)
    
    # Passed parameters checking function
    test_suite.stypy_localization = localization
    test_suite.stypy_type_of_self = None
    test_suite.stypy_type_store = module_type_store
    test_suite.stypy_function_name = 'test_suite'
    test_suite.stypy_param_names_list = []
    test_suite.stypy_varargs_param_name = None
    test_suite.stypy_kwargs_param_name = None
    test_suite.stypy_call_defaults = defaults
    test_suite.stypy_call_varargs = varargs
    test_suite.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_suite', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_suite', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_suite(...)' code ##################

    
    # Call to makeSuite(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'msvc9compilerTestCase' (line 181)
    msvc9compilerTestCase_41757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 30), 'msvc9compilerTestCase', False)
    # Processing the call keyword arguments (line 181)
    kwargs_41758 = {}
    # Getting the type of 'unittest' (line 181)
    unittest_41755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 181)
    makeSuite_41756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 11), unittest_41755, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 181)
    makeSuite_call_result_41759 = invoke(stypy.reporting.localization.Localization(__file__, 181, 11), makeSuite_41756, *[msvc9compilerTestCase_41757], **kwargs_41758)
    
    # Assigning a type to the variable 'stypy_return_type' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'stypy_return_type', makeSuite_call_result_41759)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 180)
    stypy_return_type_41760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_41760)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_41760

# Assigning a type to the variable 'test_suite' (line 180)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 184)
    # Processing the call arguments (line 184)
    
    # Call to test_suite(...): (line 184)
    # Processing the call keyword arguments (line 184)
    kwargs_41763 = {}
    # Getting the type of 'test_suite' (line 184)
    test_suite_41762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 184)
    test_suite_call_result_41764 = invoke(stypy.reporting.localization.Localization(__file__, 184, 17), test_suite_41762, *[], **kwargs_41763)
    
    # Processing the call keyword arguments (line 184)
    kwargs_41765 = {}
    # Getting the type of 'run_unittest' (line 184)
    run_unittest_41761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 184)
    run_unittest_call_result_41766 = invoke(stypy.reporting.localization.Localization(__file__, 184, 4), run_unittest_41761, *[test_suite_call_result_41764], **kwargs_41765)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

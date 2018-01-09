
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Test possibility of patching fftpack with pyfftw.
2: 
3: No module source outside of scipy.fftpack should contain an import of
4: the form `from scipy.fftpack import ...`, so that a simple replacement
5: of scipy.fftpack by the corresponding fftw interface completely swaps
6: the two FFT implementations.
7: 
8: Because this simply inspects source files, we only need to run the test
9: on one version of Python.
10: '''
11: 
12: 
13: import sys
14: if sys.version_info >= (3, 4):
15:     from pathlib import Path
16:     import re
17:     import tokenize
18:     from numpy.testing import assert_
19:     import scipy
20: 
21:     class TestFFTPackImport(object):
22:         def test_fftpack_import(self):
23:             base = Path(scipy.__file__).parent
24:             regexp = r"\s*from.+\.fftpack import .*\n"
25:             for path in base.rglob("*.py"):
26:                 if base / "fftpack" in path.parents:
27:                     continue
28:                 # use tokenize to auto-detect encoding on systems where no
29:                 # default encoding is defined (e.g. LANG='C')
30:                 with tokenize.open(str(path)) as file:
31:                     assert_(all(not re.fullmatch(regexp, line)
32:                                 for line in file),
33:                             "{0} contains an import from fftpack".format(path))
34: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_24191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, (-1)), 'str', 'Test possibility of patching fftpack with pyfftw.\n\nNo module source outside of scipy.fftpack should contain an import of\nthe form `from scipy.fftpack import ...`, so that a simple replacement\nof scipy.fftpack by the corresponding fftw interface completely swaps\nthe two FFT implementations.\n\nBecause this simply inspects source files, we only need to run the test\non one version of Python.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import sys' statement (line 13)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'sys', sys, module_type_store)



# Getting the type of 'sys' (line 14)
sys_24192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 3), 'sys')
# Obtaining the member 'version_info' of a type (line 14)
version_info_24193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 3), sys_24192, 'version_info')

# Obtaining an instance of the builtin type 'tuple' (line 14)
tuple_24194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 14)
# Adding element type (line 14)
int_24195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 24), tuple_24194, int_24195)
# Adding element type (line 14)
int_24196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 24), tuple_24194, int_24196)

# Applying the binary operator '>=' (line 14)
result_ge_24197 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 3), '>=', version_info_24193, tuple_24194)

# Testing the type of an if condition (line 14)
if_condition_24198 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 0), result_ge_24197)
# Assigning a type to the variable 'if_condition_24198' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'if_condition_24198', if_condition_24198)
# SSA begins for if statement (line 14)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 4))

# 'from pathlib import Path' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/tests/')
import_24199 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 4), 'pathlib')

if (type(import_24199) is not StypyTypeError):

    if (import_24199 != 'pyd_module'):
        __import__(import_24199)
        sys_modules_24200 = sys.modules[import_24199]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 4), 'pathlib', sys_modules_24200.module_type_store, module_type_store, ['Path'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 4), __file__, sys_modules_24200, sys_modules_24200.module_type_store, module_type_store)
    else:
        from pathlib import Path

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 4), 'pathlib', None, module_type_store, ['Path'], [Path])

else:
    # Assigning a type to the variable 'pathlib' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'pathlib', import_24199)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 4))

# 'import re' statement (line 16)
import re

import_module(stypy.reporting.localization.Localization(__file__, 16, 4), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 4))

# 'import tokenize' statement (line 17)
import tokenize

import_module(stypy.reporting.localization.Localization(__file__, 17, 4), 'tokenize', tokenize, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 4))

# 'from numpy.testing import assert_' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/tests/')
import_24201 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'numpy.testing')

if (type(import_24201) is not StypyTypeError):

    if (import_24201 != 'pyd_module'):
        __import__(import_24201)
        sys_modules_24202 = sys.modules[import_24201]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'numpy.testing', sys_modules_24202.module_type_store, module_type_store, ['assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 4), __file__, sys_modules_24202, sys_modules_24202.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'numpy.testing', None, module_type_store, ['assert_'], [assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'numpy.testing', import_24201)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 4))

# 'import scipy' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/tests/')
import_24203 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 4), 'scipy')

if (type(import_24203) is not StypyTypeError):

    if (import_24203 != 'pyd_module'):
        __import__(import_24203)
        sys_modules_24204 = sys.modules[import_24203]
        import_module(stypy.reporting.localization.Localization(__file__, 19, 4), 'scipy', sys_modules_24204.module_type_store, module_type_store)
    else:
        import scipy

        import_module(stypy.reporting.localization.Localization(__file__, 19, 4), 'scipy', scipy, module_type_store)

else:
    # Assigning a type to the variable 'scipy' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'scipy', import_24203)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/tests/')

# Declaration of the 'TestFFTPackImport' class

class TestFFTPackImport(object, ):

    @norecursion
    def test_fftpack_import(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_fftpack_import'
        module_type_store = module_type_store.open_function_context('test_fftpack_import', 22, 8, False)
        # Assigning a type to the variable 'self' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFFTPackImport.test_fftpack_import.__dict__.__setitem__('stypy_localization', localization)
        TestFFTPackImport.test_fftpack_import.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFFTPackImport.test_fftpack_import.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFFTPackImport.test_fftpack_import.__dict__.__setitem__('stypy_function_name', 'TestFFTPackImport.test_fftpack_import')
        TestFFTPackImport.test_fftpack_import.__dict__.__setitem__('stypy_param_names_list', [])
        TestFFTPackImport.test_fftpack_import.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFFTPackImport.test_fftpack_import.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFFTPackImport.test_fftpack_import.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFFTPackImport.test_fftpack_import.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFFTPackImport.test_fftpack_import.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFFTPackImport.test_fftpack_import.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFFTPackImport.test_fftpack_import', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_fftpack_import', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_fftpack_import(...)' code ##################

        
        # Assigning a Attribute to a Name (line 23):
        
        # Call to Path(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'scipy' (line 23)
        scipy_24206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 24), 'scipy', False)
        # Obtaining the member '__file__' of a type (line 23)
        file___24207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 24), scipy_24206, '__file__')
        # Processing the call keyword arguments (line 23)
        kwargs_24208 = {}
        # Getting the type of 'Path' (line 23)
        Path_24205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 19), 'Path', False)
        # Calling Path(args, kwargs) (line 23)
        Path_call_result_24209 = invoke(stypy.reporting.localization.Localization(__file__, 23, 19), Path_24205, *[file___24207], **kwargs_24208)
        
        # Obtaining the member 'parent' of a type (line 23)
        parent_24210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 19), Path_call_result_24209, 'parent')
        # Assigning a type to the variable 'base' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'base', parent_24210)
        
        # Assigning a Str to a Name (line 24):
        str_24211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 21), 'str', '\\s*from.+\\.fftpack import .*\\n')
        # Assigning a type to the variable 'regexp' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'regexp', str_24211)
        
        
        # Call to rglob(...): (line 25)
        # Processing the call arguments (line 25)
        str_24214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 35), 'str', '*.py')
        # Processing the call keyword arguments (line 25)
        kwargs_24215 = {}
        # Getting the type of 'base' (line 25)
        base_24212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 24), 'base', False)
        # Obtaining the member 'rglob' of a type (line 25)
        rglob_24213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 24), base_24212, 'rglob')
        # Calling rglob(args, kwargs) (line 25)
        rglob_call_result_24216 = invoke(stypy.reporting.localization.Localization(__file__, 25, 24), rglob_24213, *[str_24214], **kwargs_24215)
        
        # Testing the type of a for loop iterable (line 25)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 25, 12), rglob_call_result_24216)
        # Getting the type of the for loop variable (line 25)
        for_loop_var_24217 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 25, 12), rglob_call_result_24216)
        # Assigning a type to the variable 'path' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'path', for_loop_var_24217)
        # SSA begins for a for statement (line 25)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'base' (line 26)
        base_24218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 19), 'base')
        str_24219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 26), 'str', 'fftpack')
        # Applying the binary operator 'div' (line 26)
        result_div_24220 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 19), 'div', base_24218, str_24219)
        
        # Getting the type of 'path' (line 26)
        path_24221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 39), 'path')
        # Obtaining the member 'parents' of a type (line 26)
        parents_24222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 39), path_24221, 'parents')
        # Applying the binary operator 'in' (line 26)
        result_contains_24223 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 19), 'in', result_div_24220, parents_24222)
        
        # Testing the type of an if condition (line 26)
        if_condition_24224 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 26, 16), result_contains_24223)
        # Assigning a type to the variable 'if_condition_24224' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 16), 'if_condition_24224', if_condition_24224)
        # SSA begins for if statement (line 26)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 26)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to open(...): (line 30)
        # Processing the call arguments (line 30)
        
        # Call to str(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'path' (line 30)
        path_24228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 39), 'path', False)
        # Processing the call keyword arguments (line 30)
        kwargs_24229 = {}
        # Getting the type of 'str' (line 30)
        str_24227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 35), 'str', False)
        # Calling str(args, kwargs) (line 30)
        str_call_result_24230 = invoke(stypy.reporting.localization.Localization(__file__, 30, 35), str_24227, *[path_24228], **kwargs_24229)
        
        # Processing the call keyword arguments (line 30)
        kwargs_24231 = {}
        # Getting the type of 'tokenize' (line 30)
        tokenize_24225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 21), 'tokenize', False)
        # Obtaining the member 'open' of a type (line 30)
        open_24226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 21), tokenize_24225, 'open')
        # Calling open(args, kwargs) (line 30)
        open_call_result_24232 = invoke(stypy.reporting.localization.Localization(__file__, 30, 21), open_24226, *[str_call_result_24230], **kwargs_24231)
        
        with_24233 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 30, 21), open_call_result_24232, 'with parameter', '__enter__', '__exit__')

        if with_24233:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 30)
            enter___24234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 21), open_call_result_24232, '__enter__')
            with_enter_24235 = invoke(stypy.reporting.localization.Localization(__file__, 30, 21), enter___24234)
            # Assigning a type to the variable 'file' (line 30)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 21), 'file', with_enter_24235)
            
            # Call to assert_(...): (line 31)
            # Processing the call arguments (line 31)
            
            # Call to all(...): (line 31)
            # Processing the call arguments (line 31)
            # Calculating generator expression
            module_type_store = module_type_store.open_function_context('list comprehension expression', 31, 32, True)
            # Calculating comprehension expression
            # Getting the type of 'file' (line 32)
            file_24245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 44), 'file', False)
            comprehension_24246 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 32), file_24245)
            # Assigning a type to the variable 'line' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 32), 'line', comprehension_24246)
            
            
            # Call to fullmatch(...): (line 31)
            # Processing the call arguments (line 31)
            # Getting the type of 'regexp' (line 31)
            regexp_24240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 49), 'regexp', False)
            # Getting the type of 'line' (line 31)
            line_24241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 57), 'line', False)
            # Processing the call keyword arguments (line 31)
            kwargs_24242 = {}
            # Getting the type of 're' (line 31)
            re_24238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 36), 're', False)
            # Obtaining the member 'fullmatch' of a type (line 31)
            fullmatch_24239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 36), re_24238, 'fullmatch')
            # Calling fullmatch(args, kwargs) (line 31)
            fullmatch_call_result_24243 = invoke(stypy.reporting.localization.Localization(__file__, 31, 36), fullmatch_24239, *[regexp_24240, line_24241], **kwargs_24242)
            
            # Applying the 'not' unary operator (line 31)
            result_not__24244 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 32), 'not', fullmatch_call_result_24243)
            
            list_24247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 32), 'list')
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 32), list_24247, result_not__24244)
            # Processing the call keyword arguments (line 31)
            kwargs_24248 = {}
            # Getting the type of 'all' (line 31)
            all_24237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 28), 'all', False)
            # Calling all(args, kwargs) (line 31)
            all_call_result_24249 = invoke(stypy.reporting.localization.Localization(__file__, 31, 28), all_24237, *[list_24247], **kwargs_24248)
            
            
            # Call to format(...): (line 33)
            # Processing the call arguments (line 33)
            # Getting the type of 'path' (line 33)
            path_24252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 73), 'path', False)
            # Processing the call keyword arguments (line 33)
            kwargs_24253 = {}
            str_24250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 28), 'str', '{0} contains an import from fftpack')
            # Obtaining the member 'format' of a type (line 33)
            format_24251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 28), str_24250, 'format')
            # Calling format(args, kwargs) (line 33)
            format_call_result_24254 = invoke(stypy.reporting.localization.Localization(__file__, 33, 28), format_24251, *[path_24252], **kwargs_24253)
            
            # Processing the call keyword arguments (line 31)
            kwargs_24255 = {}
            # Getting the type of 'assert_' (line 31)
            assert__24236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'assert_', False)
            # Calling assert_(args, kwargs) (line 31)
            assert__call_result_24256 = invoke(stypy.reporting.localization.Localization(__file__, 31, 20), assert__24236, *[all_call_result_24249, format_call_result_24254], **kwargs_24255)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 30)
            exit___24257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 21), open_call_result_24232, '__exit__')
            with_exit_24258 = invoke(stypy.reporting.localization.Localization(__file__, 30, 21), exit___24257, None, None, None)

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_fftpack_import(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_fftpack_import' in the type store
        # Getting the type of 'stypy_return_type' (line 22)
        stypy_return_type_24259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24259)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_fftpack_import'
        return stypy_return_type_24259


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 21, 4, False)
        # Assigning a type to the variable 'self' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFFTPackImport.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestFFTPackImport' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'TestFFTPackImport', TestFFTPackImport)
# SSA join for if statement (line 14)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

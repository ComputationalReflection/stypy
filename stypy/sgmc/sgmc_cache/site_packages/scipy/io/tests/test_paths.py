
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Ensure that we can use pathlib.Path objects in all relevant IO functions.
3: '''
4: import sys
5: 
6: try:
7:     from pathlib import Path
8: except ImportError:
9:     # Not available. No fallback import, since we'll skip the entire
10:     # test suite for Python < 3.6.
11:     pass
12: 
13: import numpy as np
14: from numpy.testing import assert_
15: import pytest
16: 
17: import scipy.io
18: import scipy.io.wavfile
19: from scipy._lib._tmpdirs import tempdir
20: import scipy.sparse
21: 
22: 
23: @pytest.mark.skipif(sys.version_info < (3, 6),
24:                     reason='Passing path-like objects to IO functions requires Python >= 3.6')
25: class TestPaths(object):
26:     data = np.arange(5).astype(np.int64)
27: 
28:     def test_savemat(self):
29:         with tempdir() as temp_dir:
30:             path = Path(temp_dir) / 'data.mat'
31:             scipy.io.savemat(path, {'data': self.data})
32:             assert_(path.is_file())
33: 
34:     def test_loadmat(self):
35:         # Save data with string path, load with pathlib.Path
36:         with tempdir() as temp_dir:
37:             path = Path(temp_dir) / 'data.mat'
38:             scipy.io.savemat(str(path), {'data': self.data})
39: 
40:             mat_contents = scipy.io.loadmat(path)
41:             assert_((mat_contents['data'] == self.data).all())
42: 
43:     def test_whosmat(self):
44:         # Save data with string path, load with pathlib.Path
45:         with tempdir() as temp_dir:
46:             path = Path(temp_dir) / 'data.mat'
47:             scipy.io.savemat(str(path), {'data': self.data})
48: 
49:             contents = scipy.io.whosmat(path)
50:             assert_(contents[0] == ('data', (1, 5), 'int64'))
51: 
52:     def test_readsav(self):
53:         path = Path(__file__).parent / 'data/scalar_string.sav'
54:         scipy.io.readsav(path)
55: 
56:     def test_hb_read(self):
57:         # Save data with string path, load with pathlib.Path
58:         with tempdir() as temp_dir:
59:             data = scipy.sparse.csr_matrix(scipy.sparse.eye(3))
60:             path = Path(temp_dir) / 'data.hb'
61:             scipy.io.harwell_boeing.hb_write(str(path), data)
62: 
63:             data_new = scipy.io.harwell_boeing.hb_read(path)
64:             assert_((data_new != data).nnz == 0)
65: 
66:     def test_hb_write(self):
67:         with tempdir() as temp_dir:
68:             data = scipy.sparse.csr_matrix(scipy.sparse.eye(3))
69:             path = Path(temp_dir) / 'data.hb'
70:             scipy.io.harwell_boeing.hb_write(path, data)
71:             assert_(path.is_file())
72: 
73:     def test_netcdf_file(self):
74:         path = Path(__file__).parent / 'data/example_1.nc'
75:         scipy.io.netcdf.netcdf_file(path)
76: 
77:     def test_wavfile_read(self):
78:         path = Path(__file__).parent / 'data/test-8000Hz-le-2ch-1byteu.wav'
79:         scipy.io.wavfile.read(path)
80: 
81:     def test_wavfile_write(self):
82:         # Read from str path, write to Path
83:         input_path = Path(__file__).parent / 'data/test-8000Hz-le-2ch-1byteu.wav'
84:         rate, data = scipy.io.wavfile.read(str(input_path))
85: 
86:         with tempdir() as temp_dir:
87:             output_path = Path(temp_dir) / input_path.name
88:             scipy.io.wavfile.write(output_path, rate, data)
89: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_9016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', '\nEnsure that we can use pathlib.Path objects in all relevant IO functions.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)



# SSA begins for try-except statement (line 6)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 4))

# 'from pathlib import Path' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_9017 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'pathlib')

if (type(import_9017) is not StypyTypeError):

    if (import_9017 != 'pyd_module'):
        __import__(import_9017)
        sys_modules_9018 = sys.modules[import_9017]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'pathlib', sys_modules_9018.module_type_store, module_type_store, ['Path'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 4), __file__, sys_modules_9018, sys_modules_9018.module_type_store, module_type_store)
    else:
        from pathlib import Path

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'pathlib', None, module_type_store, ['Path'], [Path])

else:
    # Assigning a type to the variable 'pathlib' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'pathlib', import_9017)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

# SSA branch for the except part of a try statement (line 6)
# SSA branch for the except 'ImportError' branch of a try statement (line 6)
module_type_store.open_ssa_branch('except')
pass
# SSA join for try-except statement (line 6)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import numpy' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_9019 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy')

if (type(import_9019) is not StypyTypeError):

    if (import_9019 != 'pyd_module'):
        __import__(import_9019)
        sys_modules_9020 = sys.modules[import_9019]
        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'np', sys_modules_9020.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy', import_9019)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from numpy.testing import assert_' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_9021 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing')

if (type(import_9021) is not StypyTypeError):

    if (import_9021 != 'pyd_module'):
        __import__(import_9021)
        sys_modules_9022 = sys.modules[import_9021]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing', sys_modules_9022.module_type_store, module_type_store, ['assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_9022, sys_modules_9022.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing', None, module_type_store, ['assert_'], [assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing', import_9021)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'import pytest' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_9023 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'pytest')

if (type(import_9023) is not StypyTypeError):

    if (import_9023 != 'pyd_module'):
        __import__(import_9023)
        sys_modules_9024 = sys.modules[import_9023]
        import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'pytest', sys_modules_9024.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'pytest', import_9023)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'import scipy.io' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_9025 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.io')

if (type(import_9025) is not StypyTypeError):

    if (import_9025 != 'pyd_module'):
        __import__(import_9025)
        sys_modules_9026 = sys.modules[import_9025]
        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.io', sys_modules_9026.module_type_store, module_type_store)
    else:
        import scipy.io

        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.io', scipy.io, module_type_store)

else:
    # Assigning a type to the variable 'scipy.io' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.io', import_9025)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'import scipy.io.wavfile' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_9027 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.io.wavfile')

if (type(import_9027) is not StypyTypeError):

    if (import_9027 != 'pyd_module'):
        __import__(import_9027)
        sys_modules_9028 = sys.modules[import_9027]
        import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.io.wavfile', sys_modules_9028.module_type_store, module_type_store)
    else:
        import scipy.io.wavfile

        import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.io.wavfile', scipy.io.wavfile, module_type_store)

else:
    # Assigning a type to the variable 'scipy.io.wavfile' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.io.wavfile', import_9027)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from scipy._lib._tmpdirs import tempdir' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_9029 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy._lib._tmpdirs')

if (type(import_9029) is not StypyTypeError):

    if (import_9029 != 'pyd_module'):
        __import__(import_9029)
        sys_modules_9030 = sys.modules[import_9029]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy._lib._tmpdirs', sys_modules_9030.module_type_store, module_type_store, ['tempdir'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_9030, sys_modules_9030.module_type_store, module_type_store)
    else:
        from scipy._lib._tmpdirs import tempdir

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy._lib._tmpdirs', None, module_type_store, ['tempdir'], [tempdir])

else:
    # Assigning a type to the variable 'scipy._lib._tmpdirs' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy._lib._tmpdirs', import_9029)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import scipy.sparse' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_9031 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.sparse')

if (type(import_9031) is not StypyTypeError):

    if (import_9031 != 'pyd_module'):
        __import__(import_9031)
        sys_modules_9032 = sys.modules[import_9031]
        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.sparse', sys_modules_9032.module_type_store, module_type_store)
    else:
        import scipy.sparse

        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.sparse', scipy.sparse, module_type_store)

else:
    # Assigning a type to the variable 'scipy.sparse' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.sparse', import_9031)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

# Declaration of the 'TestPaths' class

class TestPaths(object, ):
    
    # Assigning a Call to a Name (line 26):

    @norecursion
    def test_savemat(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_savemat'
        module_type_store = module_type_store.open_function_context('test_savemat', 28, 4, False)
        # Assigning a type to the variable 'self' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPaths.test_savemat.__dict__.__setitem__('stypy_localization', localization)
        TestPaths.test_savemat.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPaths.test_savemat.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPaths.test_savemat.__dict__.__setitem__('stypy_function_name', 'TestPaths.test_savemat')
        TestPaths.test_savemat.__dict__.__setitem__('stypy_param_names_list', [])
        TestPaths.test_savemat.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPaths.test_savemat.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPaths.test_savemat.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPaths.test_savemat.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPaths.test_savemat.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPaths.test_savemat.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPaths.test_savemat', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_savemat', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_savemat(...)' code ##################

        
        # Call to tempdir(...): (line 29)
        # Processing the call keyword arguments (line 29)
        kwargs_9034 = {}
        # Getting the type of 'tempdir' (line 29)
        tempdir_9033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 13), 'tempdir', False)
        # Calling tempdir(args, kwargs) (line 29)
        tempdir_call_result_9035 = invoke(stypy.reporting.localization.Localization(__file__, 29, 13), tempdir_9033, *[], **kwargs_9034)
        
        with_9036 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 29, 13), tempdir_call_result_9035, 'with parameter', '__enter__', '__exit__')

        if with_9036:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 29)
            enter___9037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 13), tempdir_call_result_9035, '__enter__')
            with_enter_9038 = invoke(stypy.reporting.localization.Localization(__file__, 29, 13), enter___9037)
            # Assigning a type to the variable 'temp_dir' (line 29)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 13), 'temp_dir', with_enter_9038)
            
            # Assigning a BinOp to a Name (line 30):
            
            # Assigning a BinOp to a Name (line 30):
            
            # Call to Path(...): (line 30)
            # Processing the call arguments (line 30)
            # Getting the type of 'temp_dir' (line 30)
            temp_dir_9040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 24), 'temp_dir', False)
            # Processing the call keyword arguments (line 30)
            kwargs_9041 = {}
            # Getting the type of 'Path' (line 30)
            Path_9039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 19), 'Path', False)
            # Calling Path(args, kwargs) (line 30)
            Path_call_result_9042 = invoke(stypy.reporting.localization.Localization(__file__, 30, 19), Path_9039, *[temp_dir_9040], **kwargs_9041)
            
            str_9043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 36), 'str', 'data.mat')
            # Applying the binary operator 'div' (line 30)
            result_div_9044 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 19), 'div', Path_call_result_9042, str_9043)
            
            # Assigning a type to the variable 'path' (line 30)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'path', result_div_9044)
            
            # Call to savemat(...): (line 31)
            # Processing the call arguments (line 31)
            # Getting the type of 'path' (line 31)
            path_9048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 29), 'path', False)
            
            # Obtaining an instance of the builtin type 'dict' (line 31)
            dict_9049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 35), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 31)
            # Adding element type (key, value) (line 31)
            str_9050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 36), 'str', 'data')
            # Getting the type of 'self' (line 31)
            self_9051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 44), 'self', False)
            # Obtaining the member 'data' of a type (line 31)
            data_9052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 44), self_9051, 'data')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 35), dict_9049, (str_9050, data_9052))
            
            # Processing the call keyword arguments (line 31)
            kwargs_9053 = {}
            # Getting the type of 'scipy' (line 31)
            scipy_9045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'scipy', False)
            # Obtaining the member 'io' of a type (line 31)
            io_9046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 12), scipy_9045, 'io')
            # Obtaining the member 'savemat' of a type (line 31)
            savemat_9047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 12), io_9046, 'savemat')
            # Calling savemat(args, kwargs) (line 31)
            savemat_call_result_9054 = invoke(stypy.reporting.localization.Localization(__file__, 31, 12), savemat_9047, *[path_9048, dict_9049], **kwargs_9053)
            
            
            # Call to assert_(...): (line 32)
            # Processing the call arguments (line 32)
            
            # Call to is_file(...): (line 32)
            # Processing the call keyword arguments (line 32)
            kwargs_9058 = {}
            # Getting the type of 'path' (line 32)
            path_9056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 20), 'path', False)
            # Obtaining the member 'is_file' of a type (line 32)
            is_file_9057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 20), path_9056, 'is_file')
            # Calling is_file(args, kwargs) (line 32)
            is_file_call_result_9059 = invoke(stypy.reporting.localization.Localization(__file__, 32, 20), is_file_9057, *[], **kwargs_9058)
            
            # Processing the call keyword arguments (line 32)
            kwargs_9060 = {}
            # Getting the type of 'assert_' (line 32)
            assert__9055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'assert_', False)
            # Calling assert_(args, kwargs) (line 32)
            assert__call_result_9061 = invoke(stypy.reporting.localization.Localization(__file__, 32, 12), assert__9055, *[is_file_call_result_9059], **kwargs_9060)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 29)
            exit___9062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 13), tempdir_call_result_9035, '__exit__')
            with_exit_9063 = invoke(stypy.reporting.localization.Localization(__file__, 29, 13), exit___9062, None, None, None)

        
        # ################# End of 'test_savemat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_savemat' in the type store
        # Getting the type of 'stypy_return_type' (line 28)
        stypy_return_type_9064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9064)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_savemat'
        return stypy_return_type_9064


    @norecursion
    def test_loadmat(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_loadmat'
        module_type_store = module_type_store.open_function_context('test_loadmat', 34, 4, False)
        # Assigning a type to the variable 'self' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPaths.test_loadmat.__dict__.__setitem__('stypy_localization', localization)
        TestPaths.test_loadmat.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPaths.test_loadmat.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPaths.test_loadmat.__dict__.__setitem__('stypy_function_name', 'TestPaths.test_loadmat')
        TestPaths.test_loadmat.__dict__.__setitem__('stypy_param_names_list', [])
        TestPaths.test_loadmat.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPaths.test_loadmat.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPaths.test_loadmat.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPaths.test_loadmat.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPaths.test_loadmat.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPaths.test_loadmat.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPaths.test_loadmat', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_loadmat', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_loadmat(...)' code ##################

        
        # Call to tempdir(...): (line 36)
        # Processing the call keyword arguments (line 36)
        kwargs_9066 = {}
        # Getting the type of 'tempdir' (line 36)
        tempdir_9065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 13), 'tempdir', False)
        # Calling tempdir(args, kwargs) (line 36)
        tempdir_call_result_9067 = invoke(stypy.reporting.localization.Localization(__file__, 36, 13), tempdir_9065, *[], **kwargs_9066)
        
        with_9068 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 36, 13), tempdir_call_result_9067, 'with parameter', '__enter__', '__exit__')

        if with_9068:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 36)
            enter___9069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 13), tempdir_call_result_9067, '__enter__')
            with_enter_9070 = invoke(stypy.reporting.localization.Localization(__file__, 36, 13), enter___9069)
            # Assigning a type to the variable 'temp_dir' (line 36)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 13), 'temp_dir', with_enter_9070)
            
            # Assigning a BinOp to a Name (line 37):
            
            # Assigning a BinOp to a Name (line 37):
            
            # Call to Path(...): (line 37)
            # Processing the call arguments (line 37)
            # Getting the type of 'temp_dir' (line 37)
            temp_dir_9072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 24), 'temp_dir', False)
            # Processing the call keyword arguments (line 37)
            kwargs_9073 = {}
            # Getting the type of 'Path' (line 37)
            Path_9071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 19), 'Path', False)
            # Calling Path(args, kwargs) (line 37)
            Path_call_result_9074 = invoke(stypy.reporting.localization.Localization(__file__, 37, 19), Path_9071, *[temp_dir_9072], **kwargs_9073)
            
            str_9075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 36), 'str', 'data.mat')
            # Applying the binary operator 'div' (line 37)
            result_div_9076 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 19), 'div', Path_call_result_9074, str_9075)
            
            # Assigning a type to the variable 'path' (line 37)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'path', result_div_9076)
            
            # Call to savemat(...): (line 38)
            # Processing the call arguments (line 38)
            
            # Call to str(...): (line 38)
            # Processing the call arguments (line 38)
            # Getting the type of 'path' (line 38)
            path_9081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 33), 'path', False)
            # Processing the call keyword arguments (line 38)
            kwargs_9082 = {}
            # Getting the type of 'str' (line 38)
            str_9080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 29), 'str', False)
            # Calling str(args, kwargs) (line 38)
            str_call_result_9083 = invoke(stypy.reporting.localization.Localization(__file__, 38, 29), str_9080, *[path_9081], **kwargs_9082)
            
            
            # Obtaining an instance of the builtin type 'dict' (line 38)
            dict_9084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 40), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 38)
            # Adding element type (key, value) (line 38)
            str_9085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 41), 'str', 'data')
            # Getting the type of 'self' (line 38)
            self_9086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 49), 'self', False)
            # Obtaining the member 'data' of a type (line 38)
            data_9087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 49), self_9086, 'data')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 40), dict_9084, (str_9085, data_9087))
            
            # Processing the call keyword arguments (line 38)
            kwargs_9088 = {}
            # Getting the type of 'scipy' (line 38)
            scipy_9077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'scipy', False)
            # Obtaining the member 'io' of a type (line 38)
            io_9078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 12), scipy_9077, 'io')
            # Obtaining the member 'savemat' of a type (line 38)
            savemat_9079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 12), io_9078, 'savemat')
            # Calling savemat(args, kwargs) (line 38)
            savemat_call_result_9089 = invoke(stypy.reporting.localization.Localization(__file__, 38, 12), savemat_9079, *[str_call_result_9083, dict_9084], **kwargs_9088)
            
            
            # Assigning a Call to a Name (line 40):
            
            # Assigning a Call to a Name (line 40):
            
            # Call to loadmat(...): (line 40)
            # Processing the call arguments (line 40)
            # Getting the type of 'path' (line 40)
            path_9093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 44), 'path', False)
            # Processing the call keyword arguments (line 40)
            kwargs_9094 = {}
            # Getting the type of 'scipy' (line 40)
            scipy_9090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 27), 'scipy', False)
            # Obtaining the member 'io' of a type (line 40)
            io_9091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 27), scipy_9090, 'io')
            # Obtaining the member 'loadmat' of a type (line 40)
            loadmat_9092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 27), io_9091, 'loadmat')
            # Calling loadmat(args, kwargs) (line 40)
            loadmat_call_result_9095 = invoke(stypy.reporting.localization.Localization(__file__, 40, 27), loadmat_9092, *[path_9093], **kwargs_9094)
            
            # Assigning a type to the variable 'mat_contents' (line 40)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'mat_contents', loadmat_call_result_9095)
            
            # Call to assert_(...): (line 41)
            # Processing the call arguments (line 41)
            
            # Call to all(...): (line 41)
            # Processing the call keyword arguments (line 41)
            kwargs_9105 = {}
            
            
            # Obtaining the type of the subscript
            str_9097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 34), 'str', 'data')
            # Getting the type of 'mat_contents' (line 41)
            mat_contents_9098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 21), 'mat_contents', False)
            # Obtaining the member '__getitem__' of a type (line 41)
            getitem___9099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 21), mat_contents_9098, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 41)
            subscript_call_result_9100 = invoke(stypy.reporting.localization.Localization(__file__, 41, 21), getitem___9099, str_9097)
            
            # Getting the type of 'self' (line 41)
            self_9101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 45), 'self', False)
            # Obtaining the member 'data' of a type (line 41)
            data_9102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 45), self_9101, 'data')
            # Applying the binary operator '==' (line 41)
            result_eq_9103 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 21), '==', subscript_call_result_9100, data_9102)
            
            # Obtaining the member 'all' of a type (line 41)
            all_9104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 21), result_eq_9103, 'all')
            # Calling all(args, kwargs) (line 41)
            all_call_result_9106 = invoke(stypy.reporting.localization.Localization(__file__, 41, 21), all_9104, *[], **kwargs_9105)
            
            # Processing the call keyword arguments (line 41)
            kwargs_9107 = {}
            # Getting the type of 'assert_' (line 41)
            assert__9096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'assert_', False)
            # Calling assert_(args, kwargs) (line 41)
            assert__call_result_9108 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), assert__9096, *[all_call_result_9106], **kwargs_9107)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 36)
            exit___9109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 13), tempdir_call_result_9067, '__exit__')
            with_exit_9110 = invoke(stypy.reporting.localization.Localization(__file__, 36, 13), exit___9109, None, None, None)

        
        # ################# End of 'test_loadmat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_loadmat' in the type store
        # Getting the type of 'stypy_return_type' (line 34)
        stypy_return_type_9111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9111)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_loadmat'
        return stypy_return_type_9111


    @norecursion
    def test_whosmat(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_whosmat'
        module_type_store = module_type_store.open_function_context('test_whosmat', 43, 4, False)
        # Assigning a type to the variable 'self' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPaths.test_whosmat.__dict__.__setitem__('stypy_localization', localization)
        TestPaths.test_whosmat.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPaths.test_whosmat.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPaths.test_whosmat.__dict__.__setitem__('stypy_function_name', 'TestPaths.test_whosmat')
        TestPaths.test_whosmat.__dict__.__setitem__('stypy_param_names_list', [])
        TestPaths.test_whosmat.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPaths.test_whosmat.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPaths.test_whosmat.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPaths.test_whosmat.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPaths.test_whosmat.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPaths.test_whosmat.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPaths.test_whosmat', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_whosmat', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_whosmat(...)' code ##################

        
        # Call to tempdir(...): (line 45)
        # Processing the call keyword arguments (line 45)
        kwargs_9113 = {}
        # Getting the type of 'tempdir' (line 45)
        tempdir_9112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 13), 'tempdir', False)
        # Calling tempdir(args, kwargs) (line 45)
        tempdir_call_result_9114 = invoke(stypy.reporting.localization.Localization(__file__, 45, 13), tempdir_9112, *[], **kwargs_9113)
        
        with_9115 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 45, 13), tempdir_call_result_9114, 'with parameter', '__enter__', '__exit__')

        if with_9115:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 45)
            enter___9116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 13), tempdir_call_result_9114, '__enter__')
            with_enter_9117 = invoke(stypy.reporting.localization.Localization(__file__, 45, 13), enter___9116)
            # Assigning a type to the variable 'temp_dir' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 13), 'temp_dir', with_enter_9117)
            
            # Assigning a BinOp to a Name (line 46):
            
            # Assigning a BinOp to a Name (line 46):
            
            # Call to Path(...): (line 46)
            # Processing the call arguments (line 46)
            # Getting the type of 'temp_dir' (line 46)
            temp_dir_9119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 24), 'temp_dir', False)
            # Processing the call keyword arguments (line 46)
            kwargs_9120 = {}
            # Getting the type of 'Path' (line 46)
            Path_9118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 19), 'Path', False)
            # Calling Path(args, kwargs) (line 46)
            Path_call_result_9121 = invoke(stypy.reporting.localization.Localization(__file__, 46, 19), Path_9118, *[temp_dir_9119], **kwargs_9120)
            
            str_9122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 36), 'str', 'data.mat')
            # Applying the binary operator 'div' (line 46)
            result_div_9123 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 19), 'div', Path_call_result_9121, str_9122)
            
            # Assigning a type to the variable 'path' (line 46)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'path', result_div_9123)
            
            # Call to savemat(...): (line 47)
            # Processing the call arguments (line 47)
            
            # Call to str(...): (line 47)
            # Processing the call arguments (line 47)
            # Getting the type of 'path' (line 47)
            path_9128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 33), 'path', False)
            # Processing the call keyword arguments (line 47)
            kwargs_9129 = {}
            # Getting the type of 'str' (line 47)
            str_9127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 29), 'str', False)
            # Calling str(args, kwargs) (line 47)
            str_call_result_9130 = invoke(stypy.reporting.localization.Localization(__file__, 47, 29), str_9127, *[path_9128], **kwargs_9129)
            
            
            # Obtaining an instance of the builtin type 'dict' (line 47)
            dict_9131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 40), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 47)
            # Adding element type (key, value) (line 47)
            str_9132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 41), 'str', 'data')
            # Getting the type of 'self' (line 47)
            self_9133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 49), 'self', False)
            # Obtaining the member 'data' of a type (line 47)
            data_9134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 49), self_9133, 'data')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 40), dict_9131, (str_9132, data_9134))
            
            # Processing the call keyword arguments (line 47)
            kwargs_9135 = {}
            # Getting the type of 'scipy' (line 47)
            scipy_9124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'scipy', False)
            # Obtaining the member 'io' of a type (line 47)
            io_9125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), scipy_9124, 'io')
            # Obtaining the member 'savemat' of a type (line 47)
            savemat_9126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), io_9125, 'savemat')
            # Calling savemat(args, kwargs) (line 47)
            savemat_call_result_9136 = invoke(stypy.reporting.localization.Localization(__file__, 47, 12), savemat_9126, *[str_call_result_9130, dict_9131], **kwargs_9135)
            
            
            # Assigning a Call to a Name (line 49):
            
            # Assigning a Call to a Name (line 49):
            
            # Call to whosmat(...): (line 49)
            # Processing the call arguments (line 49)
            # Getting the type of 'path' (line 49)
            path_9140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 40), 'path', False)
            # Processing the call keyword arguments (line 49)
            kwargs_9141 = {}
            # Getting the type of 'scipy' (line 49)
            scipy_9137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 23), 'scipy', False)
            # Obtaining the member 'io' of a type (line 49)
            io_9138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 23), scipy_9137, 'io')
            # Obtaining the member 'whosmat' of a type (line 49)
            whosmat_9139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 23), io_9138, 'whosmat')
            # Calling whosmat(args, kwargs) (line 49)
            whosmat_call_result_9142 = invoke(stypy.reporting.localization.Localization(__file__, 49, 23), whosmat_9139, *[path_9140], **kwargs_9141)
            
            # Assigning a type to the variable 'contents' (line 49)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'contents', whosmat_call_result_9142)
            
            # Call to assert_(...): (line 50)
            # Processing the call arguments (line 50)
            
            
            # Obtaining the type of the subscript
            int_9144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 29), 'int')
            # Getting the type of 'contents' (line 50)
            contents_9145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 20), 'contents', False)
            # Obtaining the member '__getitem__' of a type (line 50)
            getitem___9146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 20), contents_9145, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 50)
            subscript_call_result_9147 = invoke(stypy.reporting.localization.Localization(__file__, 50, 20), getitem___9146, int_9144)
            
            
            # Obtaining an instance of the builtin type 'tuple' (line 50)
            tuple_9148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 36), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 50)
            # Adding element type (line 50)
            str_9149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 36), 'str', 'data')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 36), tuple_9148, str_9149)
            # Adding element type (line 50)
            
            # Obtaining an instance of the builtin type 'tuple' (line 50)
            tuple_9150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 45), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 50)
            # Adding element type (line 50)
            int_9151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 45), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 45), tuple_9150, int_9151)
            # Adding element type (line 50)
            int_9152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 48), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 45), tuple_9150, int_9152)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 36), tuple_9148, tuple_9150)
            # Adding element type (line 50)
            str_9153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 52), 'str', 'int64')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 36), tuple_9148, str_9153)
            
            # Applying the binary operator '==' (line 50)
            result_eq_9154 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 20), '==', subscript_call_result_9147, tuple_9148)
            
            # Processing the call keyword arguments (line 50)
            kwargs_9155 = {}
            # Getting the type of 'assert_' (line 50)
            assert__9143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'assert_', False)
            # Calling assert_(args, kwargs) (line 50)
            assert__call_result_9156 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), assert__9143, *[result_eq_9154], **kwargs_9155)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 45)
            exit___9157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 13), tempdir_call_result_9114, '__exit__')
            with_exit_9158 = invoke(stypy.reporting.localization.Localization(__file__, 45, 13), exit___9157, None, None, None)

        
        # ################# End of 'test_whosmat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_whosmat' in the type store
        # Getting the type of 'stypy_return_type' (line 43)
        stypy_return_type_9159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9159)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_whosmat'
        return stypy_return_type_9159


    @norecursion
    def test_readsav(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_readsav'
        module_type_store = module_type_store.open_function_context('test_readsav', 52, 4, False)
        # Assigning a type to the variable 'self' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPaths.test_readsav.__dict__.__setitem__('stypy_localization', localization)
        TestPaths.test_readsav.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPaths.test_readsav.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPaths.test_readsav.__dict__.__setitem__('stypy_function_name', 'TestPaths.test_readsav')
        TestPaths.test_readsav.__dict__.__setitem__('stypy_param_names_list', [])
        TestPaths.test_readsav.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPaths.test_readsav.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPaths.test_readsav.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPaths.test_readsav.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPaths.test_readsav.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPaths.test_readsav.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPaths.test_readsav', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_readsav', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_readsav(...)' code ##################

        
        # Assigning a BinOp to a Name (line 53):
        
        # Assigning a BinOp to a Name (line 53):
        
        # Call to Path(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of '__file__' (line 53)
        file___9161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 20), '__file__', False)
        # Processing the call keyword arguments (line 53)
        kwargs_9162 = {}
        # Getting the type of 'Path' (line 53)
        Path_9160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 15), 'Path', False)
        # Calling Path(args, kwargs) (line 53)
        Path_call_result_9163 = invoke(stypy.reporting.localization.Localization(__file__, 53, 15), Path_9160, *[file___9161], **kwargs_9162)
        
        # Obtaining the member 'parent' of a type (line 53)
        parent_9164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 15), Path_call_result_9163, 'parent')
        str_9165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 39), 'str', 'data/scalar_string.sav')
        # Applying the binary operator 'div' (line 53)
        result_div_9166 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 15), 'div', parent_9164, str_9165)
        
        # Assigning a type to the variable 'path' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'path', result_div_9166)
        
        # Call to readsav(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'path' (line 54)
        path_9170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 25), 'path', False)
        # Processing the call keyword arguments (line 54)
        kwargs_9171 = {}
        # Getting the type of 'scipy' (line 54)
        scipy_9167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'scipy', False)
        # Obtaining the member 'io' of a type (line 54)
        io_9168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), scipy_9167, 'io')
        # Obtaining the member 'readsav' of a type (line 54)
        readsav_9169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), io_9168, 'readsav')
        # Calling readsav(args, kwargs) (line 54)
        readsav_call_result_9172 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), readsav_9169, *[path_9170], **kwargs_9171)
        
        
        # ################# End of 'test_readsav(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_readsav' in the type store
        # Getting the type of 'stypy_return_type' (line 52)
        stypy_return_type_9173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9173)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_readsav'
        return stypy_return_type_9173


    @norecursion
    def test_hb_read(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_hb_read'
        module_type_store = module_type_store.open_function_context('test_hb_read', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPaths.test_hb_read.__dict__.__setitem__('stypy_localization', localization)
        TestPaths.test_hb_read.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPaths.test_hb_read.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPaths.test_hb_read.__dict__.__setitem__('stypy_function_name', 'TestPaths.test_hb_read')
        TestPaths.test_hb_read.__dict__.__setitem__('stypy_param_names_list', [])
        TestPaths.test_hb_read.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPaths.test_hb_read.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPaths.test_hb_read.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPaths.test_hb_read.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPaths.test_hb_read.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPaths.test_hb_read.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPaths.test_hb_read', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_hb_read', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_hb_read(...)' code ##################

        
        # Call to tempdir(...): (line 58)
        # Processing the call keyword arguments (line 58)
        kwargs_9175 = {}
        # Getting the type of 'tempdir' (line 58)
        tempdir_9174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 13), 'tempdir', False)
        # Calling tempdir(args, kwargs) (line 58)
        tempdir_call_result_9176 = invoke(stypy.reporting.localization.Localization(__file__, 58, 13), tempdir_9174, *[], **kwargs_9175)
        
        with_9177 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 58, 13), tempdir_call_result_9176, 'with parameter', '__enter__', '__exit__')

        if with_9177:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 58)
            enter___9178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 13), tempdir_call_result_9176, '__enter__')
            with_enter_9179 = invoke(stypy.reporting.localization.Localization(__file__, 58, 13), enter___9178)
            # Assigning a type to the variable 'temp_dir' (line 58)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 13), 'temp_dir', with_enter_9179)
            
            # Assigning a Call to a Name (line 59):
            
            # Assigning a Call to a Name (line 59):
            
            # Call to csr_matrix(...): (line 59)
            # Processing the call arguments (line 59)
            
            # Call to eye(...): (line 59)
            # Processing the call arguments (line 59)
            int_9186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 60), 'int')
            # Processing the call keyword arguments (line 59)
            kwargs_9187 = {}
            # Getting the type of 'scipy' (line 59)
            scipy_9183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 43), 'scipy', False)
            # Obtaining the member 'sparse' of a type (line 59)
            sparse_9184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 43), scipy_9183, 'sparse')
            # Obtaining the member 'eye' of a type (line 59)
            eye_9185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 43), sparse_9184, 'eye')
            # Calling eye(args, kwargs) (line 59)
            eye_call_result_9188 = invoke(stypy.reporting.localization.Localization(__file__, 59, 43), eye_9185, *[int_9186], **kwargs_9187)
            
            # Processing the call keyword arguments (line 59)
            kwargs_9189 = {}
            # Getting the type of 'scipy' (line 59)
            scipy_9180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 19), 'scipy', False)
            # Obtaining the member 'sparse' of a type (line 59)
            sparse_9181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 19), scipy_9180, 'sparse')
            # Obtaining the member 'csr_matrix' of a type (line 59)
            csr_matrix_9182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 19), sparse_9181, 'csr_matrix')
            # Calling csr_matrix(args, kwargs) (line 59)
            csr_matrix_call_result_9190 = invoke(stypy.reporting.localization.Localization(__file__, 59, 19), csr_matrix_9182, *[eye_call_result_9188], **kwargs_9189)
            
            # Assigning a type to the variable 'data' (line 59)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'data', csr_matrix_call_result_9190)
            
            # Assigning a BinOp to a Name (line 60):
            
            # Assigning a BinOp to a Name (line 60):
            
            # Call to Path(...): (line 60)
            # Processing the call arguments (line 60)
            # Getting the type of 'temp_dir' (line 60)
            temp_dir_9192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 24), 'temp_dir', False)
            # Processing the call keyword arguments (line 60)
            kwargs_9193 = {}
            # Getting the type of 'Path' (line 60)
            Path_9191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 19), 'Path', False)
            # Calling Path(args, kwargs) (line 60)
            Path_call_result_9194 = invoke(stypy.reporting.localization.Localization(__file__, 60, 19), Path_9191, *[temp_dir_9192], **kwargs_9193)
            
            str_9195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 36), 'str', 'data.hb')
            # Applying the binary operator 'div' (line 60)
            result_div_9196 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 19), 'div', Path_call_result_9194, str_9195)
            
            # Assigning a type to the variable 'path' (line 60)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'path', result_div_9196)
            
            # Call to hb_write(...): (line 61)
            # Processing the call arguments (line 61)
            
            # Call to str(...): (line 61)
            # Processing the call arguments (line 61)
            # Getting the type of 'path' (line 61)
            path_9202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 49), 'path', False)
            # Processing the call keyword arguments (line 61)
            kwargs_9203 = {}
            # Getting the type of 'str' (line 61)
            str_9201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 45), 'str', False)
            # Calling str(args, kwargs) (line 61)
            str_call_result_9204 = invoke(stypy.reporting.localization.Localization(__file__, 61, 45), str_9201, *[path_9202], **kwargs_9203)
            
            # Getting the type of 'data' (line 61)
            data_9205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 56), 'data', False)
            # Processing the call keyword arguments (line 61)
            kwargs_9206 = {}
            # Getting the type of 'scipy' (line 61)
            scipy_9197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'scipy', False)
            # Obtaining the member 'io' of a type (line 61)
            io_9198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 12), scipy_9197, 'io')
            # Obtaining the member 'harwell_boeing' of a type (line 61)
            harwell_boeing_9199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 12), io_9198, 'harwell_boeing')
            # Obtaining the member 'hb_write' of a type (line 61)
            hb_write_9200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 12), harwell_boeing_9199, 'hb_write')
            # Calling hb_write(args, kwargs) (line 61)
            hb_write_call_result_9207 = invoke(stypy.reporting.localization.Localization(__file__, 61, 12), hb_write_9200, *[str_call_result_9204, data_9205], **kwargs_9206)
            
            
            # Assigning a Call to a Name (line 63):
            
            # Assigning a Call to a Name (line 63):
            
            # Call to hb_read(...): (line 63)
            # Processing the call arguments (line 63)
            # Getting the type of 'path' (line 63)
            path_9212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 55), 'path', False)
            # Processing the call keyword arguments (line 63)
            kwargs_9213 = {}
            # Getting the type of 'scipy' (line 63)
            scipy_9208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 23), 'scipy', False)
            # Obtaining the member 'io' of a type (line 63)
            io_9209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 23), scipy_9208, 'io')
            # Obtaining the member 'harwell_boeing' of a type (line 63)
            harwell_boeing_9210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 23), io_9209, 'harwell_boeing')
            # Obtaining the member 'hb_read' of a type (line 63)
            hb_read_9211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 23), harwell_boeing_9210, 'hb_read')
            # Calling hb_read(args, kwargs) (line 63)
            hb_read_call_result_9214 = invoke(stypy.reporting.localization.Localization(__file__, 63, 23), hb_read_9211, *[path_9212], **kwargs_9213)
            
            # Assigning a type to the variable 'data_new' (line 63)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'data_new', hb_read_call_result_9214)
            
            # Call to assert_(...): (line 64)
            # Processing the call arguments (line 64)
            
            
            # Getting the type of 'data_new' (line 64)
            data_new_9216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 21), 'data_new', False)
            # Getting the type of 'data' (line 64)
            data_9217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 33), 'data', False)
            # Applying the binary operator '!=' (line 64)
            result_ne_9218 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 21), '!=', data_new_9216, data_9217)
            
            # Obtaining the member 'nnz' of a type (line 64)
            nnz_9219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 21), result_ne_9218, 'nnz')
            int_9220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 46), 'int')
            # Applying the binary operator '==' (line 64)
            result_eq_9221 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 20), '==', nnz_9219, int_9220)
            
            # Processing the call keyword arguments (line 64)
            kwargs_9222 = {}
            # Getting the type of 'assert_' (line 64)
            assert__9215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'assert_', False)
            # Calling assert_(args, kwargs) (line 64)
            assert__call_result_9223 = invoke(stypy.reporting.localization.Localization(__file__, 64, 12), assert__9215, *[result_eq_9221], **kwargs_9222)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 58)
            exit___9224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 13), tempdir_call_result_9176, '__exit__')
            with_exit_9225 = invoke(stypy.reporting.localization.Localization(__file__, 58, 13), exit___9224, None, None, None)

        
        # ################# End of 'test_hb_read(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_hb_read' in the type store
        # Getting the type of 'stypy_return_type' (line 56)
        stypy_return_type_9226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9226)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_hb_read'
        return stypy_return_type_9226


    @norecursion
    def test_hb_write(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_hb_write'
        module_type_store = module_type_store.open_function_context('test_hb_write', 66, 4, False)
        # Assigning a type to the variable 'self' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPaths.test_hb_write.__dict__.__setitem__('stypy_localization', localization)
        TestPaths.test_hb_write.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPaths.test_hb_write.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPaths.test_hb_write.__dict__.__setitem__('stypy_function_name', 'TestPaths.test_hb_write')
        TestPaths.test_hb_write.__dict__.__setitem__('stypy_param_names_list', [])
        TestPaths.test_hb_write.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPaths.test_hb_write.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPaths.test_hb_write.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPaths.test_hb_write.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPaths.test_hb_write.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPaths.test_hb_write.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPaths.test_hb_write', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_hb_write', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_hb_write(...)' code ##################

        
        # Call to tempdir(...): (line 67)
        # Processing the call keyword arguments (line 67)
        kwargs_9228 = {}
        # Getting the type of 'tempdir' (line 67)
        tempdir_9227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 13), 'tempdir', False)
        # Calling tempdir(args, kwargs) (line 67)
        tempdir_call_result_9229 = invoke(stypy.reporting.localization.Localization(__file__, 67, 13), tempdir_9227, *[], **kwargs_9228)
        
        with_9230 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 67, 13), tempdir_call_result_9229, 'with parameter', '__enter__', '__exit__')

        if with_9230:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 67)
            enter___9231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 13), tempdir_call_result_9229, '__enter__')
            with_enter_9232 = invoke(stypy.reporting.localization.Localization(__file__, 67, 13), enter___9231)
            # Assigning a type to the variable 'temp_dir' (line 67)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 13), 'temp_dir', with_enter_9232)
            
            # Assigning a Call to a Name (line 68):
            
            # Assigning a Call to a Name (line 68):
            
            # Call to csr_matrix(...): (line 68)
            # Processing the call arguments (line 68)
            
            # Call to eye(...): (line 68)
            # Processing the call arguments (line 68)
            int_9239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 60), 'int')
            # Processing the call keyword arguments (line 68)
            kwargs_9240 = {}
            # Getting the type of 'scipy' (line 68)
            scipy_9236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 43), 'scipy', False)
            # Obtaining the member 'sparse' of a type (line 68)
            sparse_9237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 43), scipy_9236, 'sparse')
            # Obtaining the member 'eye' of a type (line 68)
            eye_9238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 43), sparse_9237, 'eye')
            # Calling eye(args, kwargs) (line 68)
            eye_call_result_9241 = invoke(stypy.reporting.localization.Localization(__file__, 68, 43), eye_9238, *[int_9239], **kwargs_9240)
            
            # Processing the call keyword arguments (line 68)
            kwargs_9242 = {}
            # Getting the type of 'scipy' (line 68)
            scipy_9233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 19), 'scipy', False)
            # Obtaining the member 'sparse' of a type (line 68)
            sparse_9234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 19), scipy_9233, 'sparse')
            # Obtaining the member 'csr_matrix' of a type (line 68)
            csr_matrix_9235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 19), sparse_9234, 'csr_matrix')
            # Calling csr_matrix(args, kwargs) (line 68)
            csr_matrix_call_result_9243 = invoke(stypy.reporting.localization.Localization(__file__, 68, 19), csr_matrix_9235, *[eye_call_result_9241], **kwargs_9242)
            
            # Assigning a type to the variable 'data' (line 68)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'data', csr_matrix_call_result_9243)
            
            # Assigning a BinOp to a Name (line 69):
            
            # Assigning a BinOp to a Name (line 69):
            
            # Call to Path(...): (line 69)
            # Processing the call arguments (line 69)
            # Getting the type of 'temp_dir' (line 69)
            temp_dir_9245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 24), 'temp_dir', False)
            # Processing the call keyword arguments (line 69)
            kwargs_9246 = {}
            # Getting the type of 'Path' (line 69)
            Path_9244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 19), 'Path', False)
            # Calling Path(args, kwargs) (line 69)
            Path_call_result_9247 = invoke(stypy.reporting.localization.Localization(__file__, 69, 19), Path_9244, *[temp_dir_9245], **kwargs_9246)
            
            str_9248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 36), 'str', 'data.hb')
            # Applying the binary operator 'div' (line 69)
            result_div_9249 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 19), 'div', Path_call_result_9247, str_9248)
            
            # Assigning a type to the variable 'path' (line 69)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'path', result_div_9249)
            
            # Call to hb_write(...): (line 70)
            # Processing the call arguments (line 70)
            # Getting the type of 'path' (line 70)
            path_9254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 45), 'path', False)
            # Getting the type of 'data' (line 70)
            data_9255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 51), 'data', False)
            # Processing the call keyword arguments (line 70)
            kwargs_9256 = {}
            # Getting the type of 'scipy' (line 70)
            scipy_9250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'scipy', False)
            # Obtaining the member 'io' of a type (line 70)
            io_9251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), scipy_9250, 'io')
            # Obtaining the member 'harwell_boeing' of a type (line 70)
            harwell_boeing_9252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), io_9251, 'harwell_boeing')
            # Obtaining the member 'hb_write' of a type (line 70)
            hb_write_9253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), harwell_boeing_9252, 'hb_write')
            # Calling hb_write(args, kwargs) (line 70)
            hb_write_call_result_9257 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), hb_write_9253, *[path_9254, data_9255], **kwargs_9256)
            
            
            # Call to assert_(...): (line 71)
            # Processing the call arguments (line 71)
            
            # Call to is_file(...): (line 71)
            # Processing the call keyword arguments (line 71)
            kwargs_9261 = {}
            # Getting the type of 'path' (line 71)
            path_9259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 20), 'path', False)
            # Obtaining the member 'is_file' of a type (line 71)
            is_file_9260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 20), path_9259, 'is_file')
            # Calling is_file(args, kwargs) (line 71)
            is_file_call_result_9262 = invoke(stypy.reporting.localization.Localization(__file__, 71, 20), is_file_9260, *[], **kwargs_9261)
            
            # Processing the call keyword arguments (line 71)
            kwargs_9263 = {}
            # Getting the type of 'assert_' (line 71)
            assert__9258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'assert_', False)
            # Calling assert_(args, kwargs) (line 71)
            assert__call_result_9264 = invoke(stypy.reporting.localization.Localization(__file__, 71, 12), assert__9258, *[is_file_call_result_9262], **kwargs_9263)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 67)
            exit___9265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 13), tempdir_call_result_9229, '__exit__')
            with_exit_9266 = invoke(stypy.reporting.localization.Localization(__file__, 67, 13), exit___9265, None, None, None)

        
        # ################# End of 'test_hb_write(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_hb_write' in the type store
        # Getting the type of 'stypy_return_type' (line 66)
        stypy_return_type_9267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9267)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_hb_write'
        return stypy_return_type_9267


    @norecursion
    def test_netcdf_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_netcdf_file'
        module_type_store = module_type_store.open_function_context('test_netcdf_file', 73, 4, False)
        # Assigning a type to the variable 'self' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPaths.test_netcdf_file.__dict__.__setitem__('stypy_localization', localization)
        TestPaths.test_netcdf_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPaths.test_netcdf_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPaths.test_netcdf_file.__dict__.__setitem__('stypy_function_name', 'TestPaths.test_netcdf_file')
        TestPaths.test_netcdf_file.__dict__.__setitem__('stypy_param_names_list', [])
        TestPaths.test_netcdf_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPaths.test_netcdf_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPaths.test_netcdf_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPaths.test_netcdf_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPaths.test_netcdf_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPaths.test_netcdf_file.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPaths.test_netcdf_file', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_netcdf_file', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_netcdf_file(...)' code ##################

        
        # Assigning a BinOp to a Name (line 74):
        
        # Assigning a BinOp to a Name (line 74):
        
        # Call to Path(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of '__file__' (line 74)
        file___9269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), '__file__', False)
        # Processing the call keyword arguments (line 74)
        kwargs_9270 = {}
        # Getting the type of 'Path' (line 74)
        Path_9268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 15), 'Path', False)
        # Calling Path(args, kwargs) (line 74)
        Path_call_result_9271 = invoke(stypy.reporting.localization.Localization(__file__, 74, 15), Path_9268, *[file___9269], **kwargs_9270)
        
        # Obtaining the member 'parent' of a type (line 74)
        parent_9272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 15), Path_call_result_9271, 'parent')
        str_9273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 39), 'str', 'data/example_1.nc')
        # Applying the binary operator 'div' (line 74)
        result_div_9274 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 15), 'div', parent_9272, str_9273)
        
        # Assigning a type to the variable 'path' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'path', result_div_9274)
        
        # Call to netcdf_file(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'path' (line 75)
        path_9279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 36), 'path', False)
        # Processing the call keyword arguments (line 75)
        kwargs_9280 = {}
        # Getting the type of 'scipy' (line 75)
        scipy_9275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'scipy', False)
        # Obtaining the member 'io' of a type (line 75)
        io_9276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), scipy_9275, 'io')
        # Obtaining the member 'netcdf' of a type (line 75)
        netcdf_9277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), io_9276, 'netcdf')
        # Obtaining the member 'netcdf_file' of a type (line 75)
        netcdf_file_9278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), netcdf_9277, 'netcdf_file')
        # Calling netcdf_file(args, kwargs) (line 75)
        netcdf_file_call_result_9281 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), netcdf_file_9278, *[path_9279], **kwargs_9280)
        
        
        # ################# End of 'test_netcdf_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_netcdf_file' in the type store
        # Getting the type of 'stypy_return_type' (line 73)
        stypy_return_type_9282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9282)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_netcdf_file'
        return stypy_return_type_9282


    @norecursion
    def test_wavfile_read(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_wavfile_read'
        module_type_store = module_type_store.open_function_context('test_wavfile_read', 77, 4, False)
        # Assigning a type to the variable 'self' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPaths.test_wavfile_read.__dict__.__setitem__('stypy_localization', localization)
        TestPaths.test_wavfile_read.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPaths.test_wavfile_read.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPaths.test_wavfile_read.__dict__.__setitem__('stypy_function_name', 'TestPaths.test_wavfile_read')
        TestPaths.test_wavfile_read.__dict__.__setitem__('stypy_param_names_list', [])
        TestPaths.test_wavfile_read.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPaths.test_wavfile_read.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPaths.test_wavfile_read.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPaths.test_wavfile_read.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPaths.test_wavfile_read.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPaths.test_wavfile_read.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPaths.test_wavfile_read', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_wavfile_read', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_wavfile_read(...)' code ##################

        
        # Assigning a BinOp to a Name (line 78):
        
        # Assigning a BinOp to a Name (line 78):
        
        # Call to Path(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of '__file__' (line 78)
        file___9284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 20), '__file__', False)
        # Processing the call keyword arguments (line 78)
        kwargs_9285 = {}
        # Getting the type of 'Path' (line 78)
        Path_9283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 15), 'Path', False)
        # Calling Path(args, kwargs) (line 78)
        Path_call_result_9286 = invoke(stypy.reporting.localization.Localization(__file__, 78, 15), Path_9283, *[file___9284], **kwargs_9285)
        
        # Obtaining the member 'parent' of a type (line 78)
        parent_9287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 15), Path_call_result_9286, 'parent')
        str_9288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 39), 'str', 'data/test-8000Hz-le-2ch-1byteu.wav')
        # Applying the binary operator 'div' (line 78)
        result_div_9289 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 15), 'div', parent_9287, str_9288)
        
        # Assigning a type to the variable 'path' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'path', result_div_9289)
        
        # Call to read(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'path' (line 79)
        path_9294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 30), 'path', False)
        # Processing the call keyword arguments (line 79)
        kwargs_9295 = {}
        # Getting the type of 'scipy' (line 79)
        scipy_9290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'scipy', False)
        # Obtaining the member 'io' of a type (line 79)
        io_9291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), scipy_9290, 'io')
        # Obtaining the member 'wavfile' of a type (line 79)
        wavfile_9292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), io_9291, 'wavfile')
        # Obtaining the member 'read' of a type (line 79)
        read_9293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), wavfile_9292, 'read')
        # Calling read(args, kwargs) (line 79)
        read_call_result_9296 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), read_9293, *[path_9294], **kwargs_9295)
        
        
        # ################# End of 'test_wavfile_read(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_wavfile_read' in the type store
        # Getting the type of 'stypy_return_type' (line 77)
        stypy_return_type_9297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9297)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_wavfile_read'
        return stypy_return_type_9297


    @norecursion
    def test_wavfile_write(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_wavfile_write'
        module_type_store = module_type_store.open_function_context('test_wavfile_write', 81, 4, False)
        # Assigning a type to the variable 'self' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPaths.test_wavfile_write.__dict__.__setitem__('stypy_localization', localization)
        TestPaths.test_wavfile_write.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPaths.test_wavfile_write.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPaths.test_wavfile_write.__dict__.__setitem__('stypy_function_name', 'TestPaths.test_wavfile_write')
        TestPaths.test_wavfile_write.__dict__.__setitem__('stypy_param_names_list', [])
        TestPaths.test_wavfile_write.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPaths.test_wavfile_write.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPaths.test_wavfile_write.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPaths.test_wavfile_write.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPaths.test_wavfile_write.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPaths.test_wavfile_write.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPaths.test_wavfile_write', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_wavfile_write', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_wavfile_write(...)' code ##################

        
        # Assigning a BinOp to a Name (line 83):
        
        # Assigning a BinOp to a Name (line 83):
        
        # Call to Path(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of '__file__' (line 83)
        file___9299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 26), '__file__', False)
        # Processing the call keyword arguments (line 83)
        kwargs_9300 = {}
        # Getting the type of 'Path' (line 83)
        Path_9298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 21), 'Path', False)
        # Calling Path(args, kwargs) (line 83)
        Path_call_result_9301 = invoke(stypy.reporting.localization.Localization(__file__, 83, 21), Path_9298, *[file___9299], **kwargs_9300)
        
        # Obtaining the member 'parent' of a type (line 83)
        parent_9302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 21), Path_call_result_9301, 'parent')
        str_9303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 45), 'str', 'data/test-8000Hz-le-2ch-1byteu.wav')
        # Applying the binary operator 'div' (line 83)
        result_div_9304 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 21), 'div', parent_9302, str_9303)
        
        # Assigning a type to the variable 'input_path' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'input_path', result_div_9304)
        
        # Assigning a Call to a Tuple (line 84):
        
        # Assigning a Subscript to a Name (line 84):
        
        # Obtaining the type of the subscript
        int_9305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 8), 'int')
        
        # Call to read(...): (line 84)
        # Processing the call arguments (line 84)
        
        # Call to str(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'input_path' (line 84)
        input_path_9311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 47), 'input_path', False)
        # Processing the call keyword arguments (line 84)
        kwargs_9312 = {}
        # Getting the type of 'str' (line 84)
        str_9310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 43), 'str', False)
        # Calling str(args, kwargs) (line 84)
        str_call_result_9313 = invoke(stypy.reporting.localization.Localization(__file__, 84, 43), str_9310, *[input_path_9311], **kwargs_9312)
        
        # Processing the call keyword arguments (line 84)
        kwargs_9314 = {}
        # Getting the type of 'scipy' (line 84)
        scipy_9306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 21), 'scipy', False)
        # Obtaining the member 'io' of a type (line 84)
        io_9307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 21), scipy_9306, 'io')
        # Obtaining the member 'wavfile' of a type (line 84)
        wavfile_9308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 21), io_9307, 'wavfile')
        # Obtaining the member 'read' of a type (line 84)
        read_9309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 21), wavfile_9308, 'read')
        # Calling read(args, kwargs) (line 84)
        read_call_result_9315 = invoke(stypy.reporting.localization.Localization(__file__, 84, 21), read_9309, *[str_call_result_9313], **kwargs_9314)
        
        # Obtaining the member '__getitem__' of a type (line 84)
        getitem___9316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), read_call_result_9315, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 84)
        subscript_call_result_9317 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), getitem___9316, int_9305)
        
        # Assigning a type to the variable 'tuple_var_assignment_9014' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'tuple_var_assignment_9014', subscript_call_result_9317)
        
        # Assigning a Subscript to a Name (line 84):
        
        # Obtaining the type of the subscript
        int_9318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 8), 'int')
        
        # Call to read(...): (line 84)
        # Processing the call arguments (line 84)
        
        # Call to str(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'input_path' (line 84)
        input_path_9324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 47), 'input_path', False)
        # Processing the call keyword arguments (line 84)
        kwargs_9325 = {}
        # Getting the type of 'str' (line 84)
        str_9323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 43), 'str', False)
        # Calling str(args, kwargs) (line 84)
        str_call_result_9326 = invoke(stypy.reporting.localization.Localization(__file__, 84, 43), str_9323, *[input_path_9324], **kwargs_9325)
        
        # Processing the call keyword arguments (line 84)
        kwargs_9327 = {}
        # Getting the type of 'scipy' (line 84)
        scipy_9319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 21), 'scipy', False)
        # Obtaining the member 'io' of a type (line 84)
        io_9320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 21), scipy_9319, 'io')
        # Obtaining the member 'wavfile' of a type (line 84)
        wavfile_9321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 21), io_9320, 'wavfile')
        # Obtaining the member 'read' of a type (line 84)
        read_9322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 21), wavfile_9321, 'read')
        # Calling read(args, kwargs) (line 84)
        read_call_result_9328 = invoke(stypy.reporting.localization.Localization(__file__, 84, 21), read_9322, *[str_call_result_9326], **kwargs_9327)
        
        # Obtaining the member '__getitem__' of a type (line 84)
        getitem___9329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), read_call_result_9328, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 84)
        subscript_call_result_9330 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), getitem___9329, int_9318)
        
        # Assigning a type to the variable 'tuple_var_assignment_9015' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'tuple_var_assignment_9015', subscript_call_result_9330)
        
        # Assigning a Name to a Name (line 84):
        # Getting the type of 'tuple_var_assignment_9014' (line 84)
        tuple_var_assignment_9014_9331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'tuple_var_assignment_9014')
        # Assigning a type to the variable 'rate' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'rate', tuple_var_assignment_9014_9331)
        
        # Assigning a Name to a Name (line 84):
        # Getting the type of 'tuple_var_assignment_9015' (line 84)
        tuple_var_assignment_9015_9332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'tuple_var_assignment_9015')
        # Assigning a type to the variable 'data' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 14), 'data', tuple_var_assignment_9015_9332)
        
        # Call to tempdir(...): (line 86)
        # Processing the call keyword arguments (line 86)
        kwargs_9334 = {}
        # Getting the type of 'tempdir' (line 86)
        tempdir_9333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 13), 'tempdir', False)
        # Calling tempdir(args, kwargs) (line 86)
        tempdir_call_result_9335 = invoke(stypy.reporting.localization.Localization(__file__, 86, 13), tempdir_9333, *[], **kwargs_9334)
        
        with_9336 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 86, 13), tempdir_call_result_9335, 'with parameter', '__enter__', '__exit__')

        if with_9336:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 86)
            enter___9337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 13), tempdir_call_result_9335, '__enter__')
            with_enter_9338 = invoke(stypy.reporting.localization.Localization(__file__, 86, 13), enter___9337)
            # Assigning a type to the variable 'temp_dir' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 13), 'temp_dir', with_enter_9338)
            
            # Assigning a BinOp to a Name (line 87):
            
            # Assigning a BinOp to a Name (line 87):
            
            # Call to Path(...): (line 87)
            # Processing the call arguments (line 87)
            # Getting the type of 'temp_dir' (line 87)
            temp_dir_9340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 31), 'temp_dir', False)
            # Processing the call keyword arguments (line 87)
            kwargs_9341 = {}
            # Getting the type of 'Path' (line 87)
            Path_9339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 26), 'Path', False)
            # Calling Path(args, kwargs) (line 87)
            Path_call_result_9342 = invoke(stypy.reporting.localization.Localization(__file__, 87, 26), Path_9339, *[temp_dir_9340], **kwargs_9341)
            
            # Getting the type of 'input_path' (line 87)
            input_path_9343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 43), 'input_path')
            # Obtaining the member 'name' of a type (line 87)
            name_9344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 43), input_path_9343, 'name')
            # Applying the binary operator 'div' (line 87)
            result_div_9345 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 26), 'div', Path_call_result_9342, name_9344)
            
            # Assigning a type to the variable 'output_path' (line 87)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'output_path', result_div_9345)
            
            # Call to write(...): (line 88)
            # Processing the call arguments (line 88)
            # Getting the type of 'output_path' (line 88)
            output_path_9350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 35), 'output_path', False)
            # Getting the type of 'rate' (line 88)
            rate_9351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 48), 'rate', False)
            # Getting the type of 'data' (line 88)
            data_9352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 54), 'data', False)
            # Processing the call keyword arguments (line 88)
            kwargs_9353 = {}
            # Getting the type of 'scipy' (line 88)
            scipy_9346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'scipy', False)
            # Obtaining the member 'io' of a type (line 88)
            io_9347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 12), scipy_9346, 'io')
            # Obtaining the member 'wavfile' of a type (line 88)
            wavfile_9348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 12), io_9347, 'wavfile')
            # Obtaining the member 'write' of a type (line 88)
            write_9349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 12), wavfile_9348, 'write')
            # Calling write(args, kwargs) (line 88)
            write_call_result_9354 = invoke(stypy.reporting.localization.Localization(__file__, 88, 12), write_9349, *[output_path_9350, rate_9351, data_9352], **kwargs_9353)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 86)
            exit___9355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 13), tempdir_call_result_9335, '__exit__')
            with_exit_9356 = invoke(stypy.reporting.localization.Localization(__file__, 86, 13), exit___9355, None, None, None)

        
        # ################# End of 'test_wavfile_write(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_wavfile_write' in the type store
        # Getting the type of 'stypy_return_type' (line 81)
        stypy_return_type_9357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9357)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_wavfile_write'
        return stypy_return_type_9357


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 23, 0, False)
        # Assigning a type to the variable 'self' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPaths.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestPaths' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'TestPaths', TestPaths)

# Assigning a Call to a Name (line 26):

# Call to astype(...): (line 26)
# Processing the call arguments (line 26)
# Getting the type of 'np' (line 26)
np_9364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 31), 'np', False)
# Obtaining the member 'int64' of a type (line 26)
int64_9365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 31), np_9364, 'int64')
# Processing the call keyword arguments (line 26)
kwargs_9366 = {}

# Call to arange(...): (line 26)
# Processing the call arguments (line 26)
int_9360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 21), 'int')
# Processing the call keyword arguments (line 26)
kwargs_9361 = {}
# Getting the type of 'np' (line 26)
np_9358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'np', False)
# Obtaining the member 'arange' of a type (line 26)
arange_9359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 11), np_9358, 'arange')
# Calling arange(args, kwargs) (line 26)
arange_call_result_9362 = invoke(stypy.reporting.localization.Localization(__file__, 26, 11), arange_9359, *[int_9360], **kwargs_9361)

# Obtaining the member 'astype' of a type (line 26)
astype_9363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 11), arange_call_result_9362, 'astype')
# Calling astype(args, kwargs) (line 26)
astype_call_result_9367 = invoke(stypy.reporting.localization.Localization(__file__, 26, 11), astype_9363, *[int64_9365], **kwargs_9366)

# Getting the type of 'TestPaths'
TestPaths_9368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestPaths')
# Setting the type of the member 'data' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestPaths_9368, 'data', astype_call_result_9367)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

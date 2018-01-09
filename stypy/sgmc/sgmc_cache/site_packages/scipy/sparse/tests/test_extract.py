
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''test sparse matrix construction functions'''
2: 
3: from __future__ import division, print_function, absolute_import
4: 
5: from numpy.testing import assert_equal
6: from scipy.sparse import csr_matrix
7: 
8: import numpy as np
9: from scipy.sparse import extract
10: 
11: 
12: class TestExtract(object):
13:     def setup_method(self):
14:         self.cases = [
15:             csr_matrix([[1,2]]),
16:             csr_matrix([[1,0]]),
17:             csr_matrix([[0,0]]),
18:             csr_matrix([[1],[2]]),
19:             csr_matrix([[1],[0]]),
20:             csr_matrix([[0],[0]]),
21:             csr_matrix([[1,2],[3,4]]),
22:             csr_matrix([[0,1],[0,0]]),
23:             csr_matrix([[0,0],[1,0]]),
24:             csr_matrix([[0,0],[0,0]]),
25:             csr_matrix([[1,2,0,0,3],[4,5,0,6,7],[0,0,8,9,0]]),
26:             csr_matrix([[1,2,0,0,3],[4,5,0,6,7],[0,0,8,9,0]]).T,
27:         ]
28: 
29:     def find(self):
30:         for A in self.cases:
31:             I,J,V = extract.find(A)
32:             assert_equal(A.toarray(), csr_matrix(((I,J),V), shape=A.shape))
33: 
34:     def test_tril(self):
35:         for A in self.cases:
36:             B = A.toarray()
37:             for k in [-3,-2,-1,0,1,2,3]:
38:                 assert_equal(extract.tril(A,k=k).toarray(), np.tril(B,k=k))
39: 
40:     def test_triu(self):
41:         for A in self.cases:
42:             B = A.toarray()
43:             for k in [-3,-2,-1,0,1,2,3]:
44:                 assert_equal(extract.triu(A,k=k).toarray(), np.triu(B,k=k))
45: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_459885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'test sparse matrix construction functions')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy.testing import assert_equal' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_459886 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing')

if (type(import_459886) is not StypyTypeError):

    if (import_459886 != 'pyd_module'):
        __import__(import_459886)
        sys_modules_459887 = sys.modules[import_459886]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', sys_modules_459887.module_type_store, module_type_store, ['assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_459887, sys_modules_459887.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', None, module_type_store, ['assert_equal'], [assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', import_459886)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.sparse import csr_matrix' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_459888 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse')

if (type(import_459888) is not StypyTypeError):

    if (import_459888 != 'pyd_module'):
        __import__(import_459888)
        sys_modules_459889 = sys.modules[import_459888]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse', sys_modules_459889.module_type_store, module_type_store, ['csr_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_459889, sys_modules_459889.module_type_store, module_type_store)
    else:
        from scipy.sparse import csr_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse', None, module_type_store, ['csr_matrix'], [csr_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse', import_459888)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import numpy' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_459890 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy')

if (type(import_459890) is not StypyTypeError):

    if (import_459890 != 'pyd_module'):
        __import__(import_459890)
        sys_modules_459891 = sys.modules[import_459890]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', sys_modules_459891.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', import_459890)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.sparse import extract' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_459892 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse')

if (type(import_459892) is not StypyTypeError):

    if (import_459892 != 'pyd_module'):
        __import__(import_459892)
        sys_modules_459893 = sys.modules[import_459892]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse', sys_modules_459893.module_type_store, module_type_store, ['extract'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_459893, sys_modules_459893.module_type_store, module_type_store)
    else:
        from scipy.sparse import extract

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse', None, module_type_store, ['extract'], [extract])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.sparse', import_459892)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

# Declaration of the 'TestExtract' class

class TestExtract(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 13, 4, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExtract.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestExtract.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExtract.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExtract.setup_method.__dict__.__setitem__('stypy_function_name', 'TestExtract.setup_method')
        TestExtract.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestExtract.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExtract.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExtract.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExtract.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExtract.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExtract.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExtract.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a List to a Attribute (line 14):
        
        # Assigning a List to a Attribute (line 14):
        
        # Obtaining an instance of the builtin type 'list' (line 14)
        list_459894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 14)
        # Adding element type (line 14)
        
        # Call to csr_matrix(...): (line 15)
        # Processing the call arguments (line 15)
        
        # Obtaining an instance of the builtin type 'list' (line 15)
        list_459896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 15)
        # Adding element type (line 15)
        
        # Obtaining an instance of the builtin type 'list' (line 15)
        list_459897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 15)
        # Adding element type (line 15)
        int_459898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 24), list_459897, int_459898)
        # Adding element type (line 15)
        int_459899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 24), list_459897, int_459899)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 23), list_459896, list_459897)
        
        # Processing the call keyword arguments (line 15)
        kwargs_459900 = {}
        # Getting the type of 'csr_matrix' (line 15)
        csr_matrix_459895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 15)
        csr_matrix_call_result_459901 = invoke(stypy.reporting.localization.Localization(__file__, 15, 12), csr_matrix_459895, *[list_459896], **kwargs_459900)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 21), list_459894, csr_matrix_call_result_459901)
        # Adding element type (line 14)
        
        # Call to csr_matrix(...): (line 16)
        # Processing the call arguments (line 16)
        
        # Obtaining an instance of the builtin type 'list' (line 16)
        list_459903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 16)
        # Adding element type (line 16)
        
        # Obtaining an instance of the builtin type 'list' (line 16)
        list_459904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 16)
        # Adding element type (line 16)
        int_459905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 24), list_459904, int_459905)
        # Adding element type (line 16)
        int_459906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 24), list_459904, int_459906)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 23), list_459903, list_459904)
        
        # Processing the call keyword arguments (line 16)
        kwargs_459907 = {}
        # Getting the type of 'csr_matrix' (line 16)
        csr_matrix_459902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 16)
        csr_matrix_call_result_459908 = invoke(stypy.reporting.localization.Localization(__file__, 16, 12), csr_matrix_459902, *[list_459903], **kwargs_459907)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 21), list_459894, csr_matrix_call_result_459908)
        # Adding element type (line 14)
        
        # Call to csr_matrix(...): (line 17)
        # Processing the call arguments (line 17)
        
        # Obtaining an instance of the builtin type 'list' (line 17)
        list_459910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 17)
        # Adding element type (line 17)
        
        # Obtaining an instance of the builtin type 'list' (line 17)
        list_459911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 17)
        # Adding element type (line 17)
        int_459912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 24), list_459911, int_459912)
        # Adding element type (line 17)
        int_459913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 24), list_459911, int_459913)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 23), list_459910, list_459911)
        
        # Processing the call keyword arguments (line 17)
        kwargs_459914 = {}
        # Getting the type of 'csr_matrix' (line 17)
        csr_matrix_459909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 17)
        csr_matrix_call_result_459915 = invoke(stypy.reporting.localization.Localization(__file__, 17, 12), csr_matrix_459909, *[list_459910], **kwargs_459914)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 21), list_459894, csr_matrix_call_result_459915)
        # Adding element type (line 14)
        
        # Call to csr_matrix(...): (line 18)
        # Processing the call arguments (line 18)
        
        # Obtaining an instance of the builtin type 'list' (line 18)
        list_459917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 18)
        # Adding element type (line 18)
        
        # Obtaining an instance of the builtin type 'list' (line 18)
        list_459918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 18)
        # Adding element type (line 18)
        int_459919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 24), list_459918, int_459919)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 23), list_459917, list_459918)
        # Adding element type (line 18)
        
        # Obtaining an instance of the builtin type 'list' (line 18)
        list_459920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 18)
        # Adding element type (line 18)
        int_459921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 28), list_459920, int_459921)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 23), list_459917, list_459920)
        
        # Processing the call keyword arguments (line 18)
        kwargs_459922 = {}
        # Getting the type of 'csr_matrix' (line 18)
        csr_matrix_459916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 12), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 18)
        csr_matrix_call_result_459923 = invoke(stypy.reporting.localization.Localization(__file__, 18, 12), csr_matrix_459916, *[list_459917], **kwargs_459922)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 21), list_459894, csr_matrix_call_result_459923)
        # Adding element type (line 14)
        
        # Call to csr_matrix(...): (line 19)
        # Processing the call arguments (line 19)
        
        # Obtaining an instance of the builtin type 'list' (line 19)
        list_459925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 19)
        # Adding element type (line 19)
        
        # Obtaining an instance of the builtin type 'list' (line 19)
        list_459926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 19)
        # Adding element type (line 19)
        int_459927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 24), list_459926, int_459927)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_459925, list_459926)
        # Adding element type (line 19)
        
        # Obtaining an instance of the builtin type 'list' (line 19)
        list_459928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 19)
        # Adding element type (line 19)
        int_459929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 28), list_459928, int_459929)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_459925, list_459928)
        
        # Processing the call keyword arguments (line 19)
        kwargs_459930 = {}
        # Getting the type of 'csr_matrix' (line 19)
        csr_matrix_459924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 19)
        csr_matrix_call_result_459931 = invoke(stypy.reporting.localization.Localization(__file__, 19, 12), csr_matrix_459924, *[list_459925], **kwargs_459930)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 21), list_459894, csr_matrix_call_result_459931)
        # Adding element type (line 14)
        
        # Call to csr_matrix(...): (line 20)
        # Processing the call arguments (line 20)
        
        # Obtaining an instance of the builtin type 'list' (line 20)
        list_459933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 20)
        # Adding element type (line 20)
        
        # Obtaining an instance of the builtin type 'list' (line 20)
        list_459934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 20)
        # Adding element type (line 20)
        int_459935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 24), list_459934, int_459935)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), list_459933, list_459934)
        # Adding element type (line 20)
        
        # Obtaining an instance of the builtin type 'list' (line 20)
        list_459936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 20)
        # Adding element type (line 20)
        int_459937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 28), list_459936, int_459937)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), list_459933, list_459936)
        
        # Processing the call keyword arguments (line 20)
        kwargs_459938 = {}
        # Getting the type of 'csr_matrix' (line 20)
        csr_matrix_459932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 20)
        csr_matrix_call_result_459939 = invoke(stypy.reporting.localization.Localization(__file__, 20, 12), csr_matrix_459932, *[list_459933], **kwargs_459938)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 21), list_459894, csr_matrix_call_result_459939)
        # Adding element type (line 14)
        
        # Call to csr_matrix(...): (line 21)
        # Processing the call arguments (line 21)
        
        # Obtaining an instance of the builtin type 'list' (line 21)
        list_459941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 21)
        # Adding element type (line 21)
        
        # Obtaining an instance of the builtin type 'list' (line 21)
        list_459942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 21)
        # Adding element type (line 21)
        int_459943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 24), list_459942, int_459943)
        # Adding element type (line 21)
        int_459944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 24), list_459942, int_459944)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 23), list_459941, list_459942)
        # Adding element type (line 21)
        
        # Obtaining an instance of the builtin type 'list' (line 21)
        list_459945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 21)
        # Adding element type (line 21)
        int_459946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 30), list_459945, int_459946)
        # Adding element type (line 21)
        int_459947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 30), list_459945, int_459947)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 23), list_459941, list_459945)
        
        # Processing the call keyword arguments (line 21)
        kwargs_459948 = {}
        # Getting the type of 'csr_matrix' (line 21)
        csr_matrix_459940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 21)
        csr_matrix_call_result_459949 = invoke(stypy.reporting.localization.Localization(__file__, 21, 12), csr_matrix_459940, *[list_459941], **kwargs_459948)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 21), list_459894, csr_matrix_call_result_459949)
        # Adding element type (line 14)
        
        # Call to csr_matrix(...): (line 22)
        # Processing the call arguments (line 22)
        
        # Obtaining an instance of the builtin type 'list' (line 22)
        list_459951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 22)
        # Adding element type (line 22)
        
        # Obtaining an instance of the builtin type 'list' (line 22)
        list_459952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 22)
        # Adding element type (line 22)
        int_459953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 24), list_459952, int_459953)
        # Adding element type (line 22)
        int_459954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 24), list_459952, int_459954)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 23), list_459951, list_459952)
        # Adding element type (line 22)
        
        # Obtaining an instance of the builtin type 'list' (line 22)
        list_459955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 22)
        # Adding element type (line 22)
        int_459956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 30), list_459955, int_459956)
        # Adding element type (line 22)
        int_459957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 30), list_459955, int_459957)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 23), list_459951, list_459955)
        
        # Processing the call keyword arguments (line 22)
        kwargs_459958 = {}
        # Getting the type of 'csr_matrix' (line 22)
        csr_matrix_459950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 22)
        csr_matrix_call_result_459959 = invoke(stypy.reporting.localization.Localization(__file__, 22, 12), csr_matrix_459950, *[list_459951], **kwargs_459958)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 21), list_459894, csr_matrix_call_result_459959)
        # Adding element type (line 14)
        
        # Call to csr_matrix(...): (line 23)
        # Processing the call arguments (line 23)
        
        # Obtaining an instance of the builtin type 'list' (line 23)
        list_459961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 23)
        # Adding element type (line 23)
        
        # Obtaining an instance of the builtin type 'list' (line 23)
        list_459962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 23)
        # Adding element type (line 23)
        int_459963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 24), list_459962, int_459963)
        # Adding element type (line 23)
        int_459964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 24), list_459962, int_459964)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 23), list_459961, list_459962)
        # Adding element type (line 23)
        
        # Obtaining an instance of the builtin type 'list' (line 23)
        list_459965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 23)
        # Adding element type (line 23)
        int_459966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 30), list_459965, int_459966)
        # Adding element type (line 23)
        int_459967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 30), list_459965, int_459967)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 23), list_459961, list_459965)
        
        # Processing the call keyword arguments (line 23)
        kwargs_459968 = {}
        # Getting the type of 'csr_matrix' (line 23)
        csr_matrix_459960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 23)
        csr_matrix_call_result_459969 = invoke(stypy.reporting.localization.Localization(__file__, 23, 12), csr_matrix_459960, *[list_459961], **kwargs_459968)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 21), list_459894, csr_matrix_call_result_459969)
        # Adding element type (line 14)
        
        # Call to csr_matrix(...): (line 24)
        # Processing the call arguments (line 24)
        
        # Obtaining an instance of the builtin type 'list' (line 24)
        list_459971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 24)
        # Adding element type (line 24)
        
        # Obtaining an instance of the builtin type 'list' (line 24)
        list_459972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 24)
        # Adding element type (line 24)
        int_459973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 24), list_459972, int_459973)
        # Adding element type (line 24)
        int_459974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 24), list_459972, int_459974)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 23), list_459971, list_459972)
        # Adding element type (line 24)
        
        # Obtaining an instance of the builtin type 'list' (line 24)
        list_459975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 24)
        # Adding element type (line 24)
        int_459976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 30), list_459975, int_459976)
        # Adding element type (line 24)
        int_459977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 30), list_459975, int_459977)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 23), list_459971, list_459975)
        
        # Processing the call keyword arguments (line 24)
        kwargs_459978 = {}
        # Getting the type of 'csr_matrix' (line 24)
        csr_matrix_459970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 24)
        csr_matrix_call_result_459979 = invoke(stypy.reporting.localization.Localization(__file__, 24, 12), csr_matrix_459970, *[list_459971], **kwargs_459978)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 21), list_459894, csr_matrix_call_result_459979)
        # Adding element type (line 14)
        
        # Call to csr_matrix(...): (line 25)
        # Processing the call arguments (line 25)
        
        # Obtaining an instance of the builtin type 'list' (line 25)
        list_459981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 25)
        # Adding element type (line 25)
        
        # Obtaining an instance of the builtin type 'list' (line 25)
        list_459982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 25)
        # Adding element type (line 25)
        int_459983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 24), list_459982, int_459983)
        # Adding element type (line 25)
        int_459984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 24), list_459982, int_459984)
        # Adding element type (line 25)
        int_459985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 24), list_459982, int_459985)
        # Adding element type (line 25)
        int_459986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 24), list_459982, int_459986)
        # Adding element type (line 25)
        int_459987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 24), list_459982, int_459987)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 23), list_459981, list_459982)
        # Adding element type (line 25)
        
        # Obtaining an instance of the builtin type 'list' (line 25)
        list_459988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 25)
        # Adding element type (line 25)
        int_459989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 36), list_459988, int_459989)
        # Adding element type (line 25)
        int_459990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 36), list_459988, int_459990)
        # Adding element type (line 25)
        int_459991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 36), list_459988, int_459991)
        # Adding element type (line 25)
        int_459992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 36), list_459988, int_459992)
        # Adding element type (line 25)
        int_459993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 36), list_459988, int_459993)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 23), list_459981, list_459988)
        # Adding element type (line 25)
        
        # Obtaining an instance of the builtin type 'list' (line 25)
        list_459994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 25)
        # Adding element type (line 25)
        int_459995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 48), list_459994, int_459995)
        # Adding element type (line 25)
        int_459996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 48), list_459994, int_459996)
        # Adding element type (line 25)
        int_459997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 48), list_459994, int_459997)
        # Adding element type (line 25)
        int_459998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 48), list_459994, int_459998)
        # Adding element type (line 25)
        int_459999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 48), list_459994, int_459999)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 23), list_459981, list_459994)
        
        # Processing the call keyword arguments (line 25)
        kwargs_460000 = {}
        # Getting the type of 'csr_matrix' (line 25)
        csr_matrix_459980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 25)
        csr_matrix_call_result_460001 = invoke(stypy.reporting.localization.Localization(__file__, 25, 12), csr_matrix_459980, *[list_459981], **kwargs_460000)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 21), list_459894, csr_matrix_call_result_460001)
        # Adding element type (line 14)
        
        # Call to csr_matrix(...): (line 26)
        # Processing the call arguments (line 26)
        
        # Obtaining an instance of the builtin type 'list' (line 26)
        list_460003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 26)
        # Adding element type (line 26)
        
        # Obtaining an instance of the builtin type 'list' (line 26)
        list_460004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 26)
        # Adding element type (line 26)
        int_460005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 24), list_460004, int_460005)
        # Adding element type (line 26)
        int_460006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 24), list_460004, int_460006)
        # Adding element type (line 26)
        int_460007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 24), list_460004, int_460007)
        # Adding element type (line 26)
        int_460008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 24), list_460004, int_460008)
        # Adding element type (line 26)
        int_460009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 24), list_460004, int_460009)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 23), list_460003, list_460004)
        # Adding element type (line 26)
        
        # Obtaining an instance of the builtin type 'list' (line 26)
        list_460010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 26)
        # Adding element type (line 26)
        int_460011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 36), list_460010, int_460011)
        # Adding element type (line 26)
        int_460012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 36), list_460010, int_460012)
        # Adding element type (line 26)
        int_460013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 36), list_460010, int_460013)
        # Adding element type (line 26)
        int_460014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 36), list_460010, int_460014)
        # Adding element type (line 26)
        int_460015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 36), list_460010, int_460015)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 23), list_460003, list_460010)
        # Adding element type (line 26)
        
        # Obtaining an instance of the builtin type 'list' (line 26)
        list_460016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 26)
        # Adding element type (line 26)
        int_460017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 48), list_460016, int_460017)
        # Adding element type (line 26)
        int_460018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 48), list_460016, int_460018)
        # Adding element type (line 26)
        int_460019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 48), list_460016, int_460019)
        # Adding element type (line 26)
        int_460020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 48), list_460016, int_460020)
        # Adding element type (line 26)
        int_460021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 48), list_460016, int_460021)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 23), list_460003, list_460016)
        
        # Processing the call keyword arguments (line 26)
        kwargs_460022 = {}
        # Getting the type of 'csr_matrix' (line 26)
        csr_matrix_460002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 26)
        csr_matrix_call_result_460023 = invoke(stypy.reporting.localization.Localization(__file__, 26, 12), csr_matrix_460002, *[list_460003], **kwargs_460022)
        
        # Obtaining the member 'T' of a type (line 26)
        T_460024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 12), csr_matrix_call_result_460023, 'T')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 21), list_459894, T_460024)
        
        # Getting the type of 'self' (line 14)
        self_460025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'self')
        # Setting the type of the member 'cases' of a type (line 14)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), self_460025, 'cases', list_459894)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 13)
        stypy_return_type_460026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_460026)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_460026


    @norecursion
    def find(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'find'
        module_type_store = module_type_store.open_function_context('find', 29, 4, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExtract.find.__dict__.__setitem__('stypy_localization', localization)
        TestExtract.find.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExtract.find.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExtract.find.__dict__.__setitem__('stypy_function_name', 'TestExtract.find')
        TestExtract.find.__dict__.__setitem__('stypy_param_names_list', [])
        TestExtract.find.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExtract.find.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExtract.find.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExtract.find.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExtract.find.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExtract.find.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExtract.find', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'find', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'find(...)' code ##################

        
        # Getting the type of 'self' (line 30)
        self_460027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 17), 'self')
        # Obtaining the member 'cases' of a type (line 30)
        cases_460028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 17), self_460027, 'cases')
        # Testing the type of a for loop iterable (line 30)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 30, 8), cases_460028)
        # Getting the type of the for loop variable (line 30)
        for_loop_var_460029 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 30, 8), cases_460028)
        # Assigning a type to the variable 'A' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'A', for_loop_var_460029)
        # SSA begins for a for statement (line 30)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 31):
        
        # Assigning a Subscript to a Name (line 31):
        
        # Obtaining the type of the subscript
        int_460030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 12), 'int')
        
        # Call to find(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'A' (line 31)
        A_460033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 33), 'A', False)
        # Processing the call keyword arguments (line 31)
        kwargs_460034 = {}
        # Getting the type of 'extract' (line 31)
        extract_460031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'extract', False)
        # Obtaining the member 'find' of a type (line 31)
        find_460032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 20), extract_460031, 'find')
        # Calling find(args, kwargs) (line 31)
        find_call_result_460035 = invoke(stypy.reporting.localization.Localization(__file__, 31, 20), find_460032, *[A_460033], **kwargs_460034)
        
        # Obtaining the member '__getitem__' of a type (line 31)
        getitem___460036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 12), find_call_result_460035, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 31)
        subscript_call_result_460037 = invoke(stypy.reporting.localization.Localization(__file__, 31, 12), getitem___460036, int_460030)
        
        # Assigning a type to the variable 'tuple_var_assignment_459882' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'tuple_var_assignment_459882', subscript_call_result_460037)
        
        # Assigning a Subscript to a Name (line 31):
        
        # Obtaining the type of the subscript
        int_460038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 12), 'int')
        
        # Call to find(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'A' (line 31)
        A_460041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 33), 'A', False)
        # Processing the call keyword arguments (line 31)
        kwargs_460042 = {}
        # Getting the type of 'extract' (line 31)
        extract_460039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'extract', False)
        # Obtaining the member 'find' of a type (line 31)
        find_460040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 20), extract_460039, 'find')
        # Calling find(args, kwargs) (line 31)
        find_call_result_460043 = invoke(stypy.reporting.localization.Localization(__file__, 31, 20), find_460040, *[A_460041], **kwargs_460042)
        
        # Obtaining the member '__getitem__' of a type (line 31)
        getitem___460044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 12), find_call_result_460043, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 31)
        subscript_call_result_460045 = invoke(stypy.reporting.localization.Localization(__file__, 31, 12), getitem___460044, int_460038)
        
        # Assigning a type to the variable 'tuple_var_assignment_459883' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'tuple_var_assignment_459883', subscript_call_result_460045)
        
        # Assigning a Subscript to a Name (line 31):
        
        # Obtaining the type of the subscript
        int_460046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 12), 'int')
        
        # Call to find(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'A' (line 31)
        A_460049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 33), 'A', False)
        # Processing the call keyword arguments (line 31)
        kwargs_460050 = {}
        # Getting the type of 'extract' (line 31)
        extract_460047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'extract', False)
        # Obtaining the member 'find' of a type (line 31)
        find_460048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 20), extract_460047, 'find')
        # Calling find(args, kwargs) (line 31)
        find_call_result_460051 = invoke(stypy.reporting.localization.Localization(__file__, 31, 20), find_460048, *[A_460049], **kwargs_460050)
        
        # Obtaining the member '__getitem__' of a type (line 31)
        getitem___460052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 12), find_call_result_460051, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 31)
        subscript_call_result_460053 = invoke(stypy.reporting.localization.Localization(__file__, 31, 12), getitem___460052, int_460046)
        
        # Assigning a type to the variable 'tuple_var_assignment_459884' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'tuple_var_assignment_459884', subscript_call_result_460053)
        
        # Assigning a Name to a Name (line 31):
        # Getting the type of 'tuple_var_assignment_459882' (line 31)
        tuple_var_assignment_459882_460054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'tuple_var_assignment_459882')
        # Assigning a type to the variable 'I' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'I', tuple_var_assignment_459882_460054)
        
        # Assigning a Name to a Name (line 31):
        # Getting the type of 'tuple_var_assignment_459883' (line 31)
        tuple_var_assignment_459883_460055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'tuple_var_assignment_459883')
        # Assigning a type to the variable 'J' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'J', tuple_var_assignment_459883_460055)
        
        # Assigning a Name to a Name (line 31):
        # Getting the type of 'tuple_var_assignment_459884' (line 31)
        tuple_var_assignment_459884_460056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'tuple_var_assignment_459884')
        # Assigning a type to the variable 'V' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'V', tuple_var_assignment_459884_460056)
        
        # Call to assert_equal(...): (line 32)
        # Processing the call arguments (line 32)
        
        # Call to toarray(...): (line 32)
        # Processing the call keyword arguments (line 32)
        kwargs_460060 = {}
        # Getting the type of 'A' (line 32)
        A_460058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 25), 'A', False)
        # Obtaining the member 'toarray' of a type (line 32)
        toarray_460059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 25), A_460058, 'toarray')
        # Calling toarray(args, kwargs) (line 32)
        toarray_call_result_460061 = invoke(stypy.reporting.localization.Localization(__file__, 32, 25), toarray_460059, *[], **kwargs_460060)
        
        
        # Call to csr_matrix(...): (line 32)
        # Processing the call arguments (line 32)
        
        # Obtaining an instance of the builtin type 'tuple' (line 32)
        tuple_460063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 32)
        # Adding element type (line 32)
        
        # Obtaining an instance of the builtin type 'tuple' (line 32)
        tuple_460064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 51), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 32)
        # Adding element type (line 32)
        # Getting the type of 'I' (line 32)
        I_460065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 51), 'I', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 51), tuple_460064, I_460065)
        # Adding element type (line 32)
        # Getting the type of 'J' (line 32)
        J_460066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 53), 'J', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 51), tuple_460064, J_460066)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 50), tuple_460063, tuple_460064)
        # Adding element type (line 32)
        # Getting the type of 'V' (line 32)
        V_460067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 56), 'V', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 50), tuple_460063, V_460067)
        
        # Processing the call keyword arguments (line 32)
        # Getting the type of 'A' (line 32)
        A_460068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 66), 'A', False)
        # Obtaining the member 'shape' of a type (line 32)
        shape_460069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 66), A_460068, 'shape')
        keyword_460070 = shape_460069
        kwargs_460071 = {'shape': keyword_460070}
        # Getting the type of 'csr_matrix' (line 32)
        csr_matrix_460062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 38), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 32)
        csr_matrix_call_result_460072 = invoke(stypy.reporting.localization.Localization(__file__, 32, 38), csr_matrix_460062, *[tuple_460063], **kwargs_460071)
        
        # Processing the call keyword arguments (line 32)
        kwargs_460073 = {}
        # Getting the type of 'assert_equal' (line 32)
        assert_equal_460057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 32)
        assert_equal_call_result_460074 = invoke(stypy.reporting.localization.Localization(__file__, 32, 12), assert_equal_460057, *[toarray_call_result_460061, csr_matrix_call_result_460072], **kwargs_460073)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'find(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find' in the type store
        # Getting the type of 'stypy_return_type' (line 29)
        stypy_return_type_460075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_460075)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find'
        return stypy_return_type_460075


    @norecursion
    def test_tril(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tril'
        module_type_store = module_type_store.open_function_context('test_tril', 34, 4, False)
        # Assigning a type to the variable 'self' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExtract.test_tril.__dict__.__setitem__('stypy_localization', localization)
        TestExtract.test_tril.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExtract.test_tril.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExtract.test_tril.__dict__.__setitem__('stypy_function_name', 'TestExtract.test_tril')
        TestExtract.test_tril.__dict__.__setitem__('stypy_param_names_list', [])
        TestExtract.test_tril.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExtract.test_tril.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExtract.test_tril.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExtract.test_tril.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExtract.test_tril.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExtract.test_tril.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExtract.test_tril', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tril', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tril(...)' code ##################

        
        # Getting the type of 'self' (line 35)
        self_460076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 17), 'self')
        # Obtaining the member 'cases' of a type (line 35)
        cases_460077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 17), self_460076, 'cases')
        # Testing the type of a for loop iterable (line 35)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 35, 8), cases_460077)
        # Getting the type of the for loop variable (line 35)
        for_loop_var_460078 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 35, 8), cases_460077)
        # Assigning a type to the variable 'A' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'A', for_loop_var_460078)
        # SSA begins for a for statement (line 35)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 36):
        
        # Assigning a Call to a Name (line 36):
        
        # Call to toarray(...): (line 36)
        # Processing the call keyword arguments (line 36)
        kwargs_460081 = {}
        # Getting the type of 'A' (line 36)
        A_460079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'A', False)
        # Obtaining the member 'toarray' of a type (line 36)
        toarray_460080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 16), A_460079, 'toarray')
        # Calling toarray(args, kwargs) (line 36)
        toarray_call_result_460082 = invoke(stypy.reporting.localization.Localization(__file__, 36, 16), toarray_460080, *[], **kwargs_460081)
        
        # Assigning a type to the variable 'B' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'B', toarray_call_result_460082)
        
        
        # Obtaining an instance of the builtin type 'list' (line 37)
        list_460083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 37)
        # Adding element type (line 37)
        int_460084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 21), list_460083, int_460084)
        # Adding element type (line 37)
        int_460085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 21), list_460083, int_460085)
        # Adding element type (line 37)
        int_460086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 21), list_460083, int_460086)
        # Adding element type (line 37)
        int_460087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 21), list_460083, int_460087)
        # Adding element type (line 37)
        int_460088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 21), list_460083, int_460088)
        # Adding element type (line 37)
        int_460089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 21), list_460083, int_460089)
        # Adding element type (line 37)
        int_460090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 21), list_460083, int_460090)
        
        # Testing the type of a for loop iterable (line 37)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 37, 12), list_460083)
        # Getting the type of the for loop variable (line 37)
        for_loop_var_460091 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 37, 12), list_460083)
        # Assigning a type to the variable 'k' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'k', for_loop_var_460091)
        # SSA begins for a for statement (line 37)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_equal(...): (line 38)
        # Processing the call arguments (line 38)
        
        # Call to toarray(...): (line 38)
        # Processing the call keyword arguments (line 38)
        kwargs_460101 = {}
        
        # Call to tril(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'A' (line 38)
        A_460095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 42), 'A', False)
        # Processing the call keyword arguments (line 38)
        # Getting the type of 'k' (line 38)
        k_460096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 46), 'k', False)
        keyword_460097 = k_460096
        kwargs_460098 = {'k': keyword_460097}
        # Getting the type of 'extract' (line 38)
        extract_460093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 29), 'extract', False)
        # Obtaining the member 'tril' of a type (line 38)
        tril_460094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 29), extract_460093, 'tril')
        # Calling tril(args, kwargs) (line 38)
        tril_call_result_460099 = invoke(stypy.reporting.localization.Localization(__file__, 38, 29), tril_460094, *[A_460095], **kwargs_460098)
        
        # Obtaining the member 'toarray' of a type (line 38)
        toarray_460100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 29), tril_call_result_460099, 'toarray')
        # Calling toarray(args, kwargs) (line 38)
        toarray_call_result_460102 = invoke(stypy.reporting.localization.Localization(__file__, 38, 29), toarray_460100, *[], **kwargs_460101)
        
        
        # Call to tril(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'B' (line 38)
        B_460105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 68), 'B', False)
        # Processing the call keyword arguments (line 38)
        # Getting the type of 'k' (line 38)
        k_460106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 72), 'k', False)
        keyword_460107 = k_460106
        kwargs_460108 = {'k': keyword_460107}
        # Getting the type of 'np' (line 38)
        np_460103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 60), 'np', False)
        # Obtaining the member 'tril' of a type (line 38)
        tril_460104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 60), np_460103, 'tril')
        # Calling tril(args, kwargs) (line 38)
        tril_call_result_460109 = invoke(stypy.reporting.localization.Localization(__file__, 38, 60), tril_460104, *[B_460105], **kwargs_460108)
        
        # Processing the call keyword arguments (line 38)
        kwargs_460110 = {}
        # Getting the type of 'assert_equal' (line 38)
        assert_equal_460092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 38)
        assert_equal_call_result_460111 = invoke(stypy.reporting.localization.Localization(__file__, 38, 16), assert_equal_460092, *[toarray_call_result_460102, tril_call_result_460109], **kwargs_460110)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_tril(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tril' in the type store
        # Getting the type of 'stypy_return_type' (line 34)
        stypy_return_type_460112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_460112)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tril'
        return stypy_return_type_460112


    @norecursion
    def test_triu(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_triu'
        module_type_store = module_type_store.open_function_context('test_triu', 40, 4, False)
        # Assigning a type to the variable 'self' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExtract.test_triu.__dict__.__setitem__('stypy_localization', localization)
        TestExtract.test_triu.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExtract.test_triu.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExtract.test_triu.__dict__.__setitem__('stypy_function_name', 'TestExtract.test_triu')
        TestExtract.test_triu.__dict__.__setitem__('stypy_param_names_list', [])
        TestExtract.test_triu.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExtract.test_triu.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExtract.test_triu.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExtract.test_triu.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExtract.test_triu.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExtract.test_triu.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExtract.test_triu', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_triu', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_triu(...)' code ##################

        
        # Getting the type of 'self' (line 41)
        self_460113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 17), 'self')
        # Obtaining the member 'cases' of a type (line 41)
        cases_460114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 17), self_460113, 'cases')
        # Testing the type of a for loop iterable (line 41)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 41, 8), cases_460114)
        # Getting the type of the for loop variable (line 41)
        for_loop_var_460115 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 41, 8), cases_460114)
        # Assigning a type to the variable 'A' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'A', for_loop_var_460115)
        # SSA begins for a for statement (line 41)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 42):
        
        # Assigning a Call to a Name (line 42):
        
        # Call to toarray(...): (line 42)
        # Processing the call keyword arguments (line 42)
        kwargs_460118 = {}
        # Getting the type of 'A' (line 42)
        A_460116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 16), 'A', False)
        # Obtaining the member 'toarray' of a type (line 42)
        toarray_460117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 16), A_460116, 'toarray')
        # Calling toarray(args, kwargs) (line 42)
        toarray_call_result_460119 = invoke(stypy.reporting.localization.Localization(__file__, 42, 16), toarray_460117, *[], **kwargs_460118)
        
        # Assigning a type to the variable 'B' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'B', toarray_call_result_460119)
        
        
        # Obtaining an instance of the builtin type 'list' (line 43)
        list_460120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 43)
        # Adding element type (line 43)
        int_460121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 21), list_460120, int_460121)
        # Adding element type (line 43)
        int_460122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 21), list_460120, int_460122)
        # Adding element type (line 43)
        int_460123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 21), list_460120, int_460123)
        # Adding element type (line 43)
        int_460124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 21), list_460120, int_460124)
        # Adding element type (line 43)
        int_460125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 21), list_460120, int_460125)
        # Adding element type (line 43)
        int_460126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 21), list_460120, int_460126)
        # Adding element type (line 43)
        int_460127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 21), list_460120, int_460127)
        
        # Testing the type of a for loop iterable (line 43)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 43, 12), list_460120)
        # Getting the type of the for loop variable (line 43)
        for_loop_var_460128 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 43, 12), list_460120)
        # Assigning a type to the variable 'k' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'k', for_loop_var_460128)
        # SSA begins for a for statement (line 43)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_equal(...): (line 44)
        # Processing the call arguments (line 44)
        
        # Call to toarray(...): (line 44)
        # Processing the call keyword arguments (line 44)
        kwargs_460138 = {}
        
        # Call to triu(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'A' (line 44)
        A_460132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 42), 'A', False)
        # Processing the call keyword arguments (line 44)
        # Getting the type of 'k' (line 44)
        k_460133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 46), 'k', False)
        keyword_460134 = k_460133
        kwargs_460135 = {'k': keyword_460134}
        # Getting the type of 'extract' (line 44)
        extract_460130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 29), 'extract', False)
        # Obtaining the member 'triu' of a type (line 44)
        triu_460131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 29), extract_460130, 'triu')
        # Calling triu(args, kwargs) (line 44)
        triu_call_result_460136 = invoke(stypy.reporting.localization.Localization(__file__, 44, 29), triu_460131, *[A_460132], **kwargs_460135)
        
        # Obtaining the member 'toarray' of a type (line 44)
        toarray_460137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 29), triu_call_result_460136, 'toarray')
        # Calling toarray(args, kwargs) (line 44)
        toarray_call_result_460139 = invoke(stypy.reporting.localization.Localization(__file__, 44, 29), toarray_460137, *[], **kwargs_460138)
        
        
        # Call to triu(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'B' (line 44)
        B_460142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 68), 'B', False)
        # Processing the call keyword arguments (line 44)
        # Getting the type of 'k' (line 44)
        k_460143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 72), 'k', False)
        keyword_460144 = k_460143
        kwargs_460145 = {'k': keyword_460144}
        # Getting the type of 'np' (line 44)
        np_460140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 60), 'np', False)
        # Obtaining the member 'triu' of a type (line 44)
        triu_460141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 60), np_460140, 'triu')
        # Calling triu(args, kwargs) (line 44)
        triu_call_result_460146 = invoke(stypy.reporting.localization.Localization(__file__, 44, 60), triu_460141, *[B_460142], **kwargs_460145)
        
        # Processing the call keyword arguments (line 44)
        kwargs_460147 = {}
        # Getting the type of 'assert_equal' (line 44)
        assert_equal_460129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 44)
        assert_equal_call_result_460148 = invoke(stypy.reporting.localization.Localization(__file__, 44, 16), assert_equal_460129, *[toarray_call_result_460139, triu_call_result_460146], **kwargs_460147)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_triu(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_triu' in the type store
        # Getting the type of 'stypy_return_type' (line 40)
        stypy_return_type_460149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_460149)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_triu'
        return stypy_return_type_460149


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 12, 0, False)
        # Assigning a type to the variable 'self' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExtract.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestExtract' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'TestExtract', TestExtract)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

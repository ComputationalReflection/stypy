
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Eigenvalue solver using iterative methods.
3: 
4: Find k eigenvectors and eigenvalues of a matrix A using the
5: Arnoldi/Lanczos iterative methods from ARPACK [1]_,[2]_.
6: 
7: These methods are most useful for large sparse matrices.
8: 
9:   - eigs(A,k)
10:   - eigsh(A,k)
11: 
12: References
13: ----------
14: .. [1] ARPACK Software, http://www.caam.rice.edu/software/ARPACK/
15: .. [2] R. B. Lehoucq, D. C. Sorensen, and C. Yang,  ARPACK USERS GUIDE:
16:    Solution of Large Scale Eigenvalue Problems by Implicitly Restarted
17:    Arnoldi Methods. SIAM, Philadelphia, PA, 1998.
18: 
19: '''
20: from __future__ import division, print_function, absolute_import
21: 
22: from .arpack import *
23: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_401304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, (-1)), 'str', '\nEigenvalue solver using iterative methods.\n\nFind k eigenvectors and eigenvalues of a matrix A using the\nArnoldi/Lanczos iterative methods from ARPACK [1]_,[2]_.\n\nThese methods are most useful for large sparse matrices.\n\n  - eigs(A,k)\n  - eigsh(A,k)\n\nReferences\n----------\n.. [1] ARPACK Software, http://www.caam.rice.edu/software/ARPACK/\n.. [2] R. B. Lehoucq, D. C. Sorensen, and C. Yang,  ARPACK USERS GUIDE:\n   Solution of Large Scale Eigenvalue Problems by Implicitly Restarted\n   Arnoldi Methods. SIAM, Philadelphia, PA, 1998.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from scipy.sparse.linalg.eigen.arpack.arpack import ' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/arpack/')
import_401305 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.sparse.linalg.eigen.arpack.arpack')

if (type(import_401305) is not StypyTypeError):

    if (import_401305 != 'pyd_module'):
        __import__(import_401305)
        sys_modules_401306 = sys.modules[import_401305]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.sparse.linalg.eigen.arpack.arpack', sys_modules_401306.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_401306, sys_modules_401306.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.eigen.arpack.arpack import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.sparse.linalg.eigen.arpack.arpack', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.eigen.arpack.arpack' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.sparse.linalg.eigen.arpack.arpack', import_401305)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/arpack/')


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

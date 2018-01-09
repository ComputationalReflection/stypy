
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''\
2: Core Linear Algebra Tools
3: -------------------------
4: Linear algebra basics:
5: 
6: - norm            Vector or matrix norm
7: - inv             Inverse of a square matrix
8: - solve           Solve a linear system of equations
9: - det             Determinant of a square matrix
10: - lstsq           Solve linear least-squares problem
11: - pinv            Pseudo-inverse (Moore-Penrose) calculated using a singular
12:                   value decomposition
13: - matrix_power    Integer power of a square matrix
14: 
15: Eigenvalues and decompositions:
16: 
17: - eig             Eigenvalues and vectors of a square matrix
18: - eigh            Eigenvalues and eigenvectors of a Hermitian matrix
19: - eigvals         Eigenvalues of a square matrix
20: - eigvalsh        Eigenvalues of a Hermitian matrix
21: - qr              QR decomposition of a matrix
22: - svd             Singular value decomposition of a matrix
23: - cholesky        Cholesky decomposition of a matrix
24: 
25: Tensor operations:
26: 
27: - tensorsolve     Solve a linear tensor equation
28: - tensorinv       Calculate an inverse of a tensor
29: 
30: Exceptions:
31: 
32: - LinAlgError     Indicates a failed linear algebra operation
33: 
34: '''
35: from __future__ import division, absolute_import, print_function
36: 
37: depends = ['core']
38: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_134264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, (-1)), 'str', 'Core Linear Algebra Tools\n-------------------------\nLinear algebra basics:\n\n- norm            Vector or matrix norm\n- inv             Inverse of a square matrix\n- solve           Solve a linear system of equations\n- det             Determinant of a square matrix\n- lstsq           Solve linear least-squares problem\n- pinv            Pseudo-inverse (Moore-Penrose) calculated using a singular\n                  value decomposition\n- matrix_power    Integer power of a square matrix\n\nEigenvalues and decompositions:\n\n- eig             Eigenvalues and vectors of a square matrix\n- eigh            Eigenvalues and eigenvectors of a Hermitian matrix\n- eigvals         Eigenvalues of a square matrix\n- eigvalsh        Eigenvalues of a Hermitian matrix\n- qr              QR decomposition of a matrix\n- svd             Singular value decomposition of a matrix\n- cholesky        Cholesky decomposition of a matrix\n\nTensor operations:\n\n- tensorsolve     Solve a linear tensor equation\n- tensorinv       Calculate an inverse of a tensor\n\nExceptions:\n\n- LinAlgError     Indicates a failed linear algebra operation\n\n')

# Assigning a List to a Name (line 37):

# Obtaining an instance of the builtin type 'list' (line 37)
list_134265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 37)
# Adding element type (line 37)
str_134266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 11), 'str', 'core')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 10), list_134265, str_134266)

# Assigning a type to the variable 'depends' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'depends', list_134265)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

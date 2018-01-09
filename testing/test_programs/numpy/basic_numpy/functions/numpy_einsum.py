# http://www.labri.fr/perso/nrougier/teaching/numpy.100/

import numpy as np

a = np.arange(25).reshape(5, 5)
b = np.arange(5)
c = np.arange(6).reshape(2, 3)

r = np.einsum('ii', a)

r2 = np.einsum(a, [0, 0])

r3 = np.trace(a)

r4 = np.einsum('ii->i', a)

r5 = np.einsum(a, [0, 0], [0])

r6 = np.diag(a)

r7 = np.einsum('ij,j', a, b)

r8 = np.einsum(a, [0, 1], b, [1])

r9 = np.dot(a, b)

r10 = np.einsum('...j,j', a, b)

r11 = np.einsum('ji', c)

r12 = np.einsum(c, [1, 0])

r13 = c.T

r14 = np.einsum('..., ...', 3, c)

r15 = np.einsum(3, [Ellipsis], c, [Ellipsis])

r16 = np.multiply(3, c)

r17 = np.einsum('i,i', b, b)

r18 = np.einsum(b, [0], b, [0])

r19 = np.inner(b, b)

r20 = np.einsum('i,j', np.arange(2) + 1, b)

r21 = np.einsum(np.arange(2) + 1, [0], b, [1])

r22 = np.outer(np.arange(2) + 1, b)

r23 = np.einsum('i...->...', a)

r24 = np.einsum(a, [0, Ellipsis], [Ellipsis])

r25 = np.sum(a, axis=0)

a2 = np.arange(60.).reshape(3, 4, 5)
b2 = np.arange(24.).reshape(4, 3, 2)
r26 = np.einsum('ijk,jil->kl', a2, b2)

r27 = np.einsum(a2, [0, 1, 2], b2, [1, 0, 3], [2, 3])

r28 = np.tensordot(a2, b2, axes=([1, 0], [0, 1]))

a3 = np.arange(6).reshape((3, 2))
b3 = np.arange(12).reshape((4, 3))
r29 = np.einsum('ki,jk->ij', a3, b3)

r30 = np.einsum('ki,...k->i...', a3, b3)

r31 = np.einsum('k...,jk', a3, b3)

# since version 1.10.0
a4 = np.zeros((3, 3))
r32 = np.einsum('ii->i', a4)[:] = 1

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")

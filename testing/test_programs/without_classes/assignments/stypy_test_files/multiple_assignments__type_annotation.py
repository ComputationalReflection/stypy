
# r: str
r = 'hi'
# a: int; b: int
(a, b) = (5, 3)
# c: tuple[int]
c = (4, 5)
# m: int; o: int; n: int
(m, n, o) = (4, 5, 6)
# m: int; o: int; n: int
(m, n, o) = [4, 5, 6]
# y: int; x: int; r: int; z: int
[x, y, z, r] = [1, 2, 3, 4]
# y: int; x: int; r: int; z: int
(x, y, z, r) = [1, 2, 3, 4]
# x2: int; x3: int; x1: int
x1 = x2 = x3 = 5
# r4: int; r1: int; r2: int; r3: int
(r1, r2) = (r3, r4) = (8, 9)
# lr1: int; lr3: int; lr2: int; lr4: int
[lr1, lr2] = [lr3, lr4] = (13, 14)
# lr1: int; lr3: int; lr2: int; lr4: int
(lr1, lr2) = [lr3, lr4] = (113, 114)
# func: function
# <Dead code detected>

def func():
    r = 'hi'
    (a, b) = (5, 3)
    c = (4, 5)
    (m, n, o) = (4, 5, 6)
    (m, n, o) = [4, 5, 6]
    [x, y, z, r] = [1, 2, 3, 4]
    (x, y, z, r) = [1, 2, 3, 4]
    x1 = x2 = x3 = 5
    (r1, r2) = (r3, r4) = (8, 9)
    [lr1, lr2] = [lr3, lr4] = (13, 14)
    (lr1, lr2) = [lr3, lr4] = (113, 114)

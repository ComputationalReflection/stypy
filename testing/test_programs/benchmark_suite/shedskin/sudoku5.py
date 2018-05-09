"""
Author: Ali Assaf <ali.assaf.mail@gmail.com>
Copyright: (C) 2010 Ali Assaf
License: GNU General Public License <http://www.gnu.org/licenses/>
"""

from itertools import product


def solve_sudoku(size, grid):
    """An efficient Sudoku solver using Algorithm X."""
    R, C = size
    N = R * C
    X1 = ([("rc", rc) for rc in product(xrange(N), xrange(N))] +
          [("rn", rn) for rn in product(xrange(N), xrange(1, N + 1))] +
          [("cn", cn) for cn in product(xrange(N), xrange(1, N + 1))] +
          [("bn", bn) for bn in product(xrange(N), xrange(1, N + 1))])
    Y = dict()
    for r, c, n in product(xrange(N), xrange(N), xrange(1, N + 1)):
        b = (r // R) * R + (c // C)  # Box number
        Y[(r, c, n)] = [
            ("rc", (r, c)),
            ("rn", (r, n)),
            ("cn", (c, n)),
            ("bn", (b, n))]
    X, Y = exact_cover(X1, Y)
    for i, row in enumerate(grid):
        for j, n in enumerate(row):
            if n:
                select(X, Y, (i, j, n))
    for solution in solve(X, Y, []):
        for (r, c, n) in solution:
            grid[r][c] = n
        yield grid


def exact_cover(X1, Y):
    X = dict((j, set()) for j in X1)
    for i, row in Y.iteritems():
        for j in row:
            X[j].add(i)
    return X, Y


def solve(X, Y, solution):
    if not X:
        yield list(solution)
    else:
        # c = min(X, key=lambda c: len(X[c])) # shedskin doesn't support closures!
        c = min([(len(X[c]), c) for c in X])[1]
        for r in list(X[c]):
            solution.append(r)
            cols = select(X, Y, r)
            for solution in solve(X, Y, solution):
                yield solution
            deselect(X, Y, r, cols)
            v = solution.pop()


def select(X, Y, r):
    cols = []
    for j in Y[r]:
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].remove(i)
        cols.append(X.pop(j))
    return cols


def deselect(X, Y, r, cols):
    for j in reversed(Y[r]):
        X[j] = cols.pop()
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].add(i)


def main():
    grid = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]
    for solution in solve_sudoku((3, 3), grid):
        a = "\n".join(str(s) for s in solution)
        # pass#print "\n".join(str(s) for s in solution)


def run():
    for i in range(100):
        main()
    return True


run()


# math: module
import math

# words: list[str]
words = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
# normal_list: list[int]
normal_list = range(5)
# r1: TypeError
r1 = math.fsum(words)
# r2: TypeError
r2 = math.fsum(list)
# r3: TypeError
r3 = len(3)
# r4: TypeError
r4 = len(list)
# r5: float
r5 = math.fsum(normal_list)
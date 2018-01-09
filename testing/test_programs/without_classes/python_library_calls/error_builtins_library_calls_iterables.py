import math

words = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
normal_list = range(5)

r1 = math.fsum(words)  # Reported
r2 = math.fsum(list) # Not reported
r3 = len(3)  # Reported
r4 = len(list)  # Not reported

r5 = math.fsum(normal_list)


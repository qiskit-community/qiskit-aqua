\ENCODING=ISO-8859-1
\Problem name: op_ip2

Maximize
 obj: x1 + 2 x2 + 3 x3 + [ - 33 x1 ^2 + 12 x1 * x2 - 22 x2 ^2 + 23 x2 * x3
 - 11 x3 ^2 ] / 2
Subject To
 c1: - x1 + x2 + x3 <= 20
 c2: x1 - 3 x2 + x3 <= 30
Bounds
 0 <= x1 <= 40
End
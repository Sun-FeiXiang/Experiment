

grades = [(91,1),(74,2),(80,4),(90,3),(80,3),(90,3),(93,3),(79,3),(89,2),(90,2),(86,2)]

a,b = 0,0
for grade in grades:
    a = a + grade[0]*grade[1]
    b = b + grade[1]

print(a,b)
print("GPA:",a/b)
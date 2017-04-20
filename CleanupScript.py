file = open("Working_Images.txt")
column = []
for line in file:
    column.append(int(line.split()[0]))
column.sort()
print column
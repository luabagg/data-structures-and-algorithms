def hanoi(n, source, aux, destination):
    count = 1
    if n == 1:
        print ("Move disk", n, "from source", source, "to destination", destination)
    else:
        count += hanoi(n-1, source, destination, aux)
        print ("Move disk", n, "from source", source, "to destination", destination)
        count += hanoi(n-1, aux, source, destination)
    return count

disks = int(input("Enter the amout of disks:"))

print(f"steps: {hanoi(disks, 'A', 'B', 'C')}")

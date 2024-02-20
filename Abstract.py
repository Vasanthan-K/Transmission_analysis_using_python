import Function

glob_input = input("Enter the input(space seperated): ")
fault, inputs = Function.one(glob_input)
busNo, th_inputs = Function.two(inputs)
distance = Function.three(th_inputs)

if fault[0] == 1:
    print("The fault is A-G")
elif fault[1] == 1:
    print("The fault is B-G")
elif fault[2] == 1:
    print("The fault is C-G")
else:
    print("There is no fault in lines")
if busNo != 0:
    if not fault[0] and not fault[1] and not fault[2]:
        print('The BUS No.', busNo, 'is having a fault')
    else:
        print("The fault occur in the area of", busNo)
        if distance == 0:
            print("The fault occurs near 10km")
        elif distance == 1:
            print("The fault occurs near 25km")
        elif distance == 2:
            print("The fault occurs near 50km")
        elif distance == 3:
            print("The fault occurs near 75km")
        elif distance == 4:
            print("The fault occurs near 100km")


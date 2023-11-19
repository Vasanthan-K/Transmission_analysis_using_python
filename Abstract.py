import Function

glob_input = input("Enter the input(space seperated): ")
fault, inputs = Function.one(glob_input)
busNo = Function.two(inputs)


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
        print('The PMU No.', busNo, 'is having a fault')
    else:
        print("The fault occur in the area of", busNo)

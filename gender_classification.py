from sklearn import tree

#Height(Inches),Weight(Kilo), Shoe-Size
data = [
    [170, 65, 42],
    [160, 55, 38],
    [175, 70, 43],
    [180, 75, 44],
    [165, 60, 40],
    [172, 68, 41],
    [178, 72, 43],
    [182, 78, 45],
    [168, 58, 39],
    [176, 74, 42],
    [167, 63, 40],
    [174, 69, 41],
    [181, 80, 44],
    [169, 66, 40],
    [177, 73, 42],
    [185, 82, 46],
    [162, 56, 38],
    [171, 67, 41],
    [179, 76, 44],
    [166, 61, 39]
]

labels = [
    'M', 'F', 'M', 'M', 'F',
    'M', 'M', 'M', 'F', 'M',
    'F', 'M', 'M', 'F', 'M',
    'M', 'F', 'M', 'M', 'F'
]

clf = tree.DecisionTreeClassifier()
clf.fit(data,labels)

def user_input():
    while True:
        try:
            # Prompt the user for input
            x = input("Input your predictive values in the format: Height,Weight,Shoe-Size: ")
            # Split the input string by commas
            x = x.split(",")
            # Convert each value to an integer
            x = [int(i) for i in x]
            return x
        except ValueError:
            print("Invalid input. Please enter integers in the format: Height,Weight,Shoe-Size")

# Assuming clf is your trained classifier
print(clf.predict([user_input()]))

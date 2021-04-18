file = open("seinfeld.txt", "r")
output = open("seinfeld_stripped.txt", "w")

for x in file:
    a_string = x
    alpha = " "
    for character in x:
        if character.isalphanumeric() or character.isspace() character.:
            alpha += character
    print(alpha)
    output.write(alpha + " ")
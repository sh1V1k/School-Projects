import sys

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Error. Incorrect number of arguments.")
        exit()
    
    first_arg = sys.argv[1]  # cipher text
    second_arg = sys.argv[2] # key
    third_arg = sys.argv[3]  # where we want to store solution
    sol = ""                 # holds the solution as we parse the cipher text

    with open(first_arg) as f:
        file_content = f.read().strip()
    
    with open(second_arg) as f:
        key = f.read().strip()
    
    for c in file_content:
        ascii_value = ord(c)
        if 65 <= ascii_value <= 96: # determine if we have an upper case alphabet
            i = key.index(c)
            sol += chr(65+i)
        else:
            sol += c

    with open(third_arg, 'w') as f:
        f.write(str(sol))



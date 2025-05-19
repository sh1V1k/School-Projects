import sys

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Error. Incorrect number of arguments.")
        exit()
    
    first_arg = sys.argv[1]  # cipher text
    second_arg = sys.argv[2] # key
    third_arg = sys.argv[3]  # modulo
    fourth_arg = sys.argv[4] # output file

    with open(first_arg) as f:
        cipher_text = int(f.read().strip(), 16)
    
    with open(second_arg) as f:
        key = int(f.read().strip(), 16)
   
    with open(third_arg) as f:
        modulo = int(f.read().strip(), 16)

    sol = hex(pow(cipher_text, key, modulo))[2:]

    with open(fourth_arg, 'w') as f:
        f.write(str(sol))

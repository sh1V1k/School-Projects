import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Error. Incorrect number of arguments.")
        exit()
    
    first_arg = sys.argv[1]  # message to encode
    second_arg = sys.argv[2] # output file

    with open(first_arg) as f:
        message = f.read().strip().encode()
    
    mask = 0x3FFFFFFF
    outHash = 0
    
    for b in message:
        print(b)
        intermediate_value = ((b ^ 0xCC) << 24) |  ((b ^ 0x33) << 16) | ((b ^ 0xAA) << 8) | (b ^ 0x55)
        print(intermediate_value)
        outHash = (outHash & mask) + (intermediate_value & mask)

    with open(second_arg, 'w') as f:
        f.write(str(hex(outHash)))

import sys
from Crypto.Cipher import AES

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Error. Incorrect number of arguments.")
        exit()
    
    first_arg = sys.argv[1]  # cipher text
    second_arg = sys.argv[2] # key
    third_arg = sys.argv[3]  # iv
    fourth_arg = sys.argv[4] # output file

    with open(first_arg) as f:
        cipher_text = bytes.fromhex(f.read().strip())
    
    with open(second_arg) as f:
        key = bytes.fromhex(f.read().strip())
   
    with open(third_arg) as f:
        iv = bytes.fromhex(f.read().strip())

    cipher = AES.new(key, AES.MODE_CBC, iv)
    plaintext = cipher.decrypt(cipher_text)

    with open(fourth_arg, 'w') as f:
        f.write(plaintext.decode())

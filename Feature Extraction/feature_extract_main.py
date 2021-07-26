#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import re
import joblib
from sklearn.metrics import *
import sys, getopt
from Feature_Extract import FQDN,FLINT,ACCESS,JaccardFeature,WHOIS,IP,WHOISJaccard,FEATUREUNION,FEATURESELECT
argv = sys.argv[1:]
dir_fqdn = r'Data/fqdn.csv'
dir_flint = r'Data/flint.csv'
dir_access = r'Data/access.csv'
dir_ip = r'Data/ip.csv'
dir_ipv6 = r'Data/ipv6.csv'
dir_whois = r'Data/whois.json'
dir_Label = r'Label/label.csv'
dir_label = r'Feature_Extract/Label_For_Feature_Extract/label.csv'
dir_save = r'./'
if(argv):

    try:
        opts, args = getopt.getopt(argv,"h1:2:3:4:",["fqdn=","flint=","access=","ip=","ipv6=","whois=","Label=","label=","outputModel="])
    except getopt.GetoptError:
        print( 'test.py -1 <dir_fqdn> -2 <dir_flint> -3 <dir_access> -4 <dir_ip> -5 <dir_ipv6> -6 <dir_whois> -7 <dir_Label> -8 <dir_label> -9 <output_path>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print( 'test.py -1 <dir_fqdn> -2 <dir_flint> -3 <dir_access> -4 <dir_ip> -5 <dir_ipv6> -6 <dir_whois> -7 <dir_Label> -8 <dir_label> -9 <output_path>')
            sys.exit()
        elif opt in ("-1", "--fqdn"):
            dir_fqdn = arg
        
        elif opt in ("-2", "--flint"):
            dir_flint = arg
            
        elif opt in ("-3", "--access"):
            dir_access = arg
            
        elif opt in ("-4", "--dir_ip"):
            dir_ip = arg

        elif opt in ("-5", "--dir_ipv6"):
            dir_ipv6 = arg

        elif opt in ("-6", "--dir_whois"):
            dir_whois = arg

        elif opt in ("-7", "--dir_Label"):
            dir_Label = arg

        elif opt in ("-8", "--dir_label"):
            dir_label = arg

        elif opt in ("-8", "--output_path"):
            dir_save = arg
#Feature Extract--------------------------------------------------------
print("Feature Extract:")
FQDN()
FLINT()
ACCESS()

WHOIS()
IP()
JaccardFeature()

# WHOISJaccard()
# WHOISJaccard是JaccardFeature一个子集不要重复提取
FEATUREUNION()

#Model Apply--------------------------------------------------------
# print("Model Apply:")

# FQDN(dir_fqdn)
# FLINT(path_train_flint=dir_flint,path_train_fqdn=dir_fqdn)
# ACCESS(dir_fqdn=dir_fqdn,dir_access=dir_access)
#
# WHOIS(whois_path=dir_whois,path_train_fqdn=dir_fqdn)
# IP(path_df_label=dir_label,
#                  path_train_fqdn=dir_fqdn,path_train_flint = dir_flint,
#                  path_train_access=dir_access,path_train_ip=dir_ip,path_train_ipv6=dir_ipv6)
# JaccardFeature(path_df_label=dir_label,
#                  path_train_fqdn=dir_fqdn,path_train_flint = dir_flint,
#                  path_train_access=dir_access,path_train_ip=dir_ip,path_train_ipv6=dir_ipv6)
#
# WHOISJaccard(whois_path=dir_whois,path_train_fqdn=dir_fqdn,path_df_label=dir_label)
# # WHOISJaccard是JaccardFeature一个子集不要重复提取
# FEATUREUNION()

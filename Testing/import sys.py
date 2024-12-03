import sys

def add(i,j):
    return int(i) + int(j)

if __name__ == '__main__':
    print(sys.argv)
    print(add(sys.argv[1], sys.argv[2]))
    
    
    
# import subprocess
# from concurrent.futures import ProcessPoolExecutor
# import json
# import os

# # File paths for worker scripts
# WORKER_SCRIPT = "fetch_worker.py"
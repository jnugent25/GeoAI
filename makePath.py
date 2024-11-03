import os, json
from pyexpat.errors import codes

f=open("C:\\Users\\Jack\\Downloads\\ISO-3166-Countries-with-Regional-Codes-master\\ISO-3166-Countries-with-Regional-Codes-master\\all\\all.json")
codes = json.load(f)
for code in codes:
    newpath="C:\\Users\\Jack\\Downloads\\data\\train\\"+code['alpha-2']
    if not os.path.exists(newpath):
        os.makedirs(newpath)
#for code in codes:
#    newpath="C:\\Users\\Jack\\Downloads\\data2\\val\\"+code['alpha-2']
#    if not os.path.exists(newpath):
 #       os.makedirs(newpath)

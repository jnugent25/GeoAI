import os

root = "C:\\Users\\Jack\\Downloads\\data\\train"
folders = list(os.walk(root))[1:]

for folder in folders:
    if not folder[2]:
        os.rmdir(folder[0])

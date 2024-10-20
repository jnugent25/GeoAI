import os,random,shutil

source = "C:\\Users\\Jack\\Downloads\\data\\val"
dest = "C:\\Users\\Jack\\Downloads\\data\\train"
files = os.listdir(source)
for file in files:
    file2=os.listdir(os.path.join(source,file))
    no_of_files = round(len(file2) * 0.8)
    print(no_of_files)
    print(len(file2))
    for file_name in random.sample(file2, no_of_files):
        print(file_name)
        shutil.move(os.path.join(source,file,file_name), os.path.join(dest,file))
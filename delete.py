import os


def remove_empty_folders(path_abs):
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.remove(path)

if __name__ == '__main__':
    remove_empty_folders(r"C:\Users\Jack\Downloads\data\train")
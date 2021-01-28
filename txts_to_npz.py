#IREM CELIK 151180015 BM455 YAPAY ZEKAYA GIRIS UYGULAMA ODEVI 2
#this code is making the dataset (npz form) from the txt files.

import numpy as np
import os
import os.path as op


def get_files_path_list_on_dir(*paths) -> list:
    ls = []
    for path in paths:
        for path_file in os.listdir(path):
            ls.append(op.join(path, path_file))
    return ls


def get_txt_data(path: str) -> str:
    with open(path, 'r') as f:
        data = f.read()
    return data


def get_email_class_from_path(path: str) -> bool:
    file_name = path.split('\\')[-1]
    email_class = file_name[:-4].split('.')[-1]
    email_class = True if email_class == "spam" else False
    return email_class

#giving the wanted txt files data path 
if __name__ == "__main__":
    path_scan = "../../../dataset/training"
    path_npz_output = "../res/training.npz"

    ls_file_paths = get_files_path_list_on_dir(op.join(path_scan, "ham"), op.join(path_scan, "spam"))
    ls_texts = []
    ls_classes = []
    for path_file in ls_file_paths:
        email_class = get_email_class_from_path(path_file)
        txt = get_txt_data(path_file)
        ls_texts.append(txt)
        ls_classes.append(email_class)

    texts = np.array(ls_texts)
    classes = np.array(ls_classes, dtype=np.bool)
    np.savez_compressed(path_npz_output, texts=texts, classes=classes)
    
#IREM CELIK 151180015 BM455 YAPAY ZEKAYA GIRIS UYGULAMA ODEVI 2
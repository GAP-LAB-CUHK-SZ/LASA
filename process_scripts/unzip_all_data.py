import os
import glob
import argparse

parser = argparse.ArgumentParser("unzip the prepared data")
parser.add_argument("--occ_root", type=str, default="../submodules/DisCo/data/occ_data")
parser.add_argument("--other_root", type=str,default="../submodules/DisCo/data/other_data")
parser.add_argument("--unzip_occ",default=False,action="store_true")
parser.add_argument("--unzip_other",default=False,action="store_true")

args=parser.parse_args()
if args.unzip_occ:
    filelist=os.listdir(args.occ_root)
    for filename in filelist:
        filepath=os.path.join(args.occ_root,filename)
        if ".rar" in filename:
            unrar_command="unrar x %s %s"%(filepath,args.occ_root)
            os.system(unrar_command)
        elif ".zip" in filename:
            unzip_command="7z x %s -o%s"%(filepath,args.occ_root)
            os.system(unzip_command)


if args.unzip_other:
    category_list=os.listdir(args.other_root)
    for category in category_list:
        category_folder=os.path.join(args.other_root,category)
        #print(category_folder)
        rar_filelist=glob.glob(category_folder+"/*.rar")
        zip_filelist=glob.glob(category_folder+"/*.zip")

        for rar_filepath in rar_filelist:
            unrar_command="unrar x %s %s"%(rar_filepath,category_folder)
            os.system(unrar_command)
        for zip_filepath in zip_filelist:
            unzip_command="7z x %s -o%s"%(zip_filepath,category_folder)
            os.system(unzip_command)
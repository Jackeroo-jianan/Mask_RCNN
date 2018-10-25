import os
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("images_dir")
args = parser.parse_args()

index = 0
for file_name in os.listdir(args.images_dir):
    full_path = os.path.join(args.images_dir, file_name)
    if os.path.isfile(full_path):
        # 正则,re.I忽略大小写
        match_pattern = r"^(img \([0-9]+\))(\.[0-9a-z]+)$"
        matchObj = re.match(match_pattern, file_name, re.I)
        # print("matchObj", matchObj)
        # 找到未重命名的文件
        if matchObj == None:
            print("file_name", file_name)
            # 生成新文件名，跳过已存在的名称
            new_name = "img ({0})".format(index)
            # 用于标记新文件名是否存在
            is_new = False
            while is_new == False:
                is_new = True
                for file_name2 in os.listdir(args.images_dir):
                    matchObj2 = re.match(match_pattern, file_name2, re.I)
                    # 名称重复
                    if matchObj2 != None and matchObj2.group(1) == new_name:
                        index += 1
                        new_name = "img ({0})".format(index)
                        is_new = False
                        break
            # 加上后缀
            match_pattern3 = r"^(.+)(\.[0-9a-z]+)$"
            matchObj3 = re.match(match_pattern3, file_name, re.I)
            new_name = "{0}{1}".format(new_name, matchObj3.group(2))
            print("new_name", new_name)
            new_full_path = os.path.join(args.images_dir, new_name)
            os.rename(full_path, new_full_path)
            index += 1

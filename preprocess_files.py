from pathlib import Path
import json
import itertools
import random
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate json file for training')
    parser.add_argument('-d','--dataset', help='Path of dataset folder', required=True)
    parser.add_argument('-o','--output', help='Name of json output file', required=True)
    args = vars(parser.parse_args())

    folder_name = args['dataset']
    json_file_name = args['output']
    keys_dict = {}
    sub_folders = os.listdir(folder_name)
    key_dict = {}
    pos = 0
    neg = 0
    for key in sub_folders:
        sub_folder_name = os.path.join(folder_name, key)
        uuid_file_map = {}
        file_uuid_map = {}
        sub_pos = 0
        for path in Path(sub_folder_name).rglob('*.jpg'):
            items = path.name.split('__')
            if "Crop" not in items[4] or "BBox" in items[4]:
                continue
            else:
                tracker_uuid = items[3]
                file_path = os.path.abspath(str(path))        
                file_uuid_map.update({file_path:tracker_uuid})
                if tracker_uuid not in uuid_file_map.keys():
                    uuid_file_map.update({tracker_uuid: [file_path]})
                else:
                    uuid_file_map[tracker_uuid].append(file_path)
        set_list = []
        # test_set = []
        for tracker_uuid, files in uuid_file_map.items():
            combination = list(itertools.combinations(files, 2))
            for pair in combination:
                # if key == 'bellevue' and not ('ParkStart' in pair[0] and 'ParkEnd' in pair [1]) and not ('ParkStart' in pair[1] and 'ParkEnd' in pair [0]):
                #     if ('ParkStart' in pair[0] and 'ParkStart' in pair[1]) or ('ParkEnd' in pair [0] and 'ParkEnd' in pair [1]):
                #         continue
                #     test_set.append([pair[0], pair[1], 1])
                set_list.append([pair[0], pair[1], 1])
                pos+=1
                sub_pos+=1
        test_files = [*file_uuid_map]
        combination = list(itertools.combinations(test_files, 2))
        random_combination = random.sample(combination, sub_pos)
        for path1, path2 in random_combination:
            if file_uuid_map[path1] == file_uuid_map[path2]:
                continue
            else:
                set_list.append([path1, path2, 0])
                # if key == 'bellevue':
                #     test_set.append([path1, path2, 0])
                neg+=1
            random.shuffle(set_list)
            # if key == 'bellevue':
            #     random.shuffle(test_set)
            #     test_dict = {'test_data':test_set}
            key_dict[key] = set_list

    # with open('test_dataset.json', 'w') as fp:
    #     json.dump(test_dict, fp, indent=4)
    with open(json_file_name, 'w') as fp:
        json.dump(key_dict, fp, indent=4)
    print(json_file_name, key_dict.keys(), "positive: {} negative: {}".format(pos, neg))
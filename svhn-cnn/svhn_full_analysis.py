import os
import h5py

TRAIN_FULL_DIR = 'train_full_data'
TEST_FULL_DIR = 'test_full_data'

# Each element in digitStruct has the following fields: 'name' which is a string containing the filename of the
# corresponding image. 'bbox' which is a struct array that contains the position, size and label of each digit
# bounding box in the image. Eg: digitStruct(300).bbox(2).height gives height of the 2nd digit bounding box
# in the 300th image.

class create_data:
    def __init__(self, input):
        self.content = h5py.File(input, 'r')
        for key in self.content.keys():
            print(key)
        # Get the HDF5 group
        group = self.content[key]
        # Checkout what keys are inside that group.
        for key in group.keys():
            print(key)
        print(self.content)
        self.name = self.content['digitStruct']['name']
        self.bbox = self.content['digitStruct']['bbox']

    def helper(self,arr):
        if (len(arr) > 1):
            arr_list = []
            for i in range(len(arr)):
                arr = self.content[arr.value[i].item()].value[0][0]
                arr_list.append(arr)
        else:
            arr_list = [arr.value[0][0]]
        return arr_list

    def fetch_name(self,n):
        return ''.join([chr(char[0]) for char in self.content[self.name[n][0]].value])

    def fetch_bbox(self,n):
        box = {}
        bb = self.name[n].item()
        con = self.content[bb]
        box['label'] = self.helper(["label"])
        box['left'] = self.helper(con["left"])
        box['top'] = self.helper(con["top"])
        box['width'] = self.helper(con["width"])
        box['height'] = self.helper(con["height"])
        return box

    def get_digit(self,n):
        box_dict = self.fetch_bbox(n)
        box_dict['name'] = self.fetch_name(n)
        return box_dict

    def get_full_data(self):
        digits = []
        for i in range(len(self.name)):
            dig_data = self.get_digit(i)
            digits.append(dig_data)
        final_list = []
        c = 1
        for i in range(len(digits)):
            val = {'filename': digits[i]["name"] }
            list_of_dict = []
            for j in range(len(digits[i]['height'])):
               single_dict = {}
               dig = digits[i]
               single_dict['label'] = dig['label'][j]  # label could be from 1 to 10, where 10 represents 0
               single_dict['left'] = dig['left'][j]
               single_dict['top'] = dig['top'][j]
               single_dict['width'] = dig['width'][j]
               single_dict['height'] = dig['height'][j]
               list_of_dict.append(single_dict)
            c = c + 1
            val['boxes'] = list_of_dict
            val['number_of_digits'] = len(list_of_dict)
            final_list.append(val)
        return final_list


if __name__ == '__main__':
    path = os.path.join(TEST_FULL_DIR, 'digitStruct.mat')  # first argument: TRAIN_FULL_DIR
    dsf = create_data(path)
    data = dsf.get_full_data()
    # print('traindata', train_data[0])
    # json_data = {"results": data}  # JSON {"results": [asdasd, asdads]}
    # print("Writing to JSON file")
    # with open('test_data.json', 'w') as fp:  # for train data use: train_data, for test use test_data
    #     json.dump(json_data, fp)

# reference: http://www.a2ialab.com/lib/exe/fetch.php?media=public:scripts:svhn_dataextract_tojson.py.txt
# reference: https://stackoverflow.com/questions/41176258/h5py-access-data-in-datasets-in-svhn

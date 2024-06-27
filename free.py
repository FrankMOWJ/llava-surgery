# ##############################avi转图片################################

import cv2
import os
import re
import json
import random

# def vedio2image():
#     mission = ["Knot_Tying", "Needle_Passing", "Suturing"]
#     for mission_name in mission:
#         avi_root = f'data/JIGSAWS/{mission_name}/video/'
#         avi_videos = os.listdir(avi_root)
#         for avi_video in avi_videos:
#             if 'capture2' not in avi_video:
#                 videoFile = avi_root + avi_video
#                 if mission_name == "Suturing":
#                     outputFile = avi_root + '_'.join(avi_video.split('_')[:2]) + '/'
#                 else:
#                     outputFile = avi_root + '_'.join(avi_video.split('_')[:3]) + '/'
#                 if not os.path.exists(outputFile):
#                     os.makedirs(outputFile)
#                 vc = cv2.VideoCapture(videoFile)
#                 c = 1
#                 print(avi_video)
#                 if vc.isOpened():
#                     rval, frame = vc.read()
#                 else:
#                     print(f'{avi_video} openerror!')
#                     rval = False

#                 timeF = 1  #视频帧计数间隔次数
#                 while rval:
#                     # print(1)
#                     #print(c)
#                     rval, frame = vc.read()
#                     if c % timeF == 0 and frame is not None:
#                         # print(2)
#                         cv2.imwrite(outputFile + str(int(c / timeF)) + '.jpg', frame)
#                     c += 1
#                     cv2.waitKey(1)
#                 vc.release()

#         ##############################图片汇总################################

#         avi_root = f'data/JIGSAWS/{mission_name}/video/'
#         avi_images = os.listdir(avi_root)

#         for avi_image in avi_images:
#             if '.avi' not in avi_image:
#                 image_root = avi_root + avi_image + '/'
#                 images = os.listdir(image_root)
#                 for image in images:
#                     image_ori = image_root + image
#                     image_dis = f'data/JIGSAWS/{mission_name}/images/' + avi_image + '_' + "{:04d}.jpg".format(int(image.replace('.jpg', '')))
#                     os.rename(image_ori, image_dis)



# ############################## 生成json文件 ##############################
json_list = []
num_img = 1
# mission_name = "Needle_Passing"
# mission_name = "Suturing"
mission_name = "Knot_Tying"
transcription_path = f'data/JIGSAWS/{mission_name}/transcriptions/'
image_path = f'JIGSAWS/{mission_name}/images/'
kinematics_path = f'data/JIGSAWS/{mission_name}/kinematics/AllGestures/'
json_path = f'data/JIGSAWS/{mission_name}/{mission_name}_{num_img}images.json'
num_dict = {
    1: "first",
    2: "second",
    3: "third",
    4: "forth"
}
"""
json format:
id(name) :  Knot_Tying_B001_0001
image(path_list) : [Three consecutive images path if i_n-2, i_n-1, i_n]
conversations(list) : 
[
    {
        "from": "human",
        "value": "<image>\n<image>\n<image>\nThe gesture of the three frames is (g1).The kinematics information of the first frame is [...]. 
            The kinematics information of the second frame is [...]. Predict the kinematics information of the third frame in 1 X 76 tensor."
            
        "value": "Here are three frames of surgeric gesture of (gesture). <image>\nThe kinematics information of the first frame is [...]. 
            <image>\nThe kinematics information of the second frame is [...]. <image>\nPredict the kinematics information of the third frame in 1 X 76 tensor."
    },
    {
        "from": "gpt",
        "value": "[Kinematics information of i_n]"
    }
]
"""
gesture_descriptions = {
    'G1': 'Reaching for needle with right hand',
    'G2': 'Positioning needle',
    'G3': 'Pushing needle through tissue',
    'G4': 'Transferring needle from left to right',
    'G5': 'Moving to center with needle in grip',
    'G6': 'Pulling suture with left hand',
    'G7': 'Pulling suture with right hand',
    'G8': 'Orienting needle',
    'G9': 'Using right hand to help tighten suture',
    'G10': 'Loosening more suture',
    'G11': 'Dropping suture at end and moving to end points',
    'G12': 'Reaching for needle with left hand',
    'G13': 'Making C loop around right hand',
    'G14': 'Reaching for suture with right hand',
    'G15': 'Pulling suture with both hands'
}

# 获取运动学信息
def get_kinematic_information(filename) -> list:
    kinematics_info = []
    with open(os.path.join(kinematics_path,  filename)) as file:
        cnt = 0
        line = file.readline()
        while line:
            cnt += 1
            data = [re.split(r'\s+', line.strip()) for line in line.strip().split('\n')][0]
            data_str = ','.join(data)
            # 将每一帧的运动学信息方法数组
            kinematics_info.append(data_str)
            if len(data) != 76:
                print(data)
                print(line)
                # kinematics_info.append(data_str)
                # continue
                print(f"the dimension of the kinematic data from {filename} line {cnt} is not equal to 76, is {len(data)}")
            line = file.readline()
        print(f"file {filename} have line {cnt}")     
    file.close()
    return kinematics_info

def create_json_file():
    global json_list
    # 遍历transcription文件夹
    file_list = os.listdir(transcription_path)
    file_list.sort(key=lambda x: x.split(".")[0].split("_")[-1])
    cnt = 0
    for filename in file_list:
        # print(filename)
        # cnt += 1
        # if cnt > 1:
        #     return
        if filename.endswith('.txt'):
            file_path = os.path.join(transcription_path, filename)
            # 获取运动学信息
            kinematics_info = get_kinematic_information(filename=filename)
            print(f"len of kinematicx information: {len(kinematics_info)}")
            frame = []
            # 获取有效帧的序号
            with open(file_path, 'r') as file:
                for line in file: 
                    start_f, end_f, gesture = line.strip().split()
                    # print(start_f, end_f, gesture)
                    for i in range(int(start_f), int(end_f)+1):
                        frame.append((i, gesture))

            # 每连续的num_img张图片组成一个sample
            for i in range(len(frame)-num_img+1):
                # image_list
                image_lst = []
                id : str = ""
                human_prompt: str = "Here are three consecutive frames of surgical activity."
                gpt_prompt: str = ""
                for j in range(num_img):
                    postfix = filename.replace('.txt', '') + '_' + "{:04d}.jpg".format(int(frame[i+j][0]))
                    image = image_path + postfix
                    image_lst.append(image)

                    # id : the name of the predict image
                    id = postfix.replace('.jpg', '')

                    # human prompt
                    if j != num_img -1:
                        human_prompt += f" <image>\nThe kinematics information and gesture of the {num_dict[j+1]} frame is [{kinematics_info[frame[i+j][0]-1]}] and {gesture_descriptions[frame[i+j][1]]}."
                    else:
                        human_prompt += f" <image>\nThe gesture of the {num_dict[j+1]} frame is {gesture_descriptions[frame[i+j][1]]}, please predict kinematics information of the {num_dict[j+1]} frame in 1 X 76 tensor."
                        gpt_prompt += f"The 1 X 76  kinematics information tensor is [{kinematics_info[frame[i+j][0]-1]}]"
                   
      
                dict = {
                    "id": id,
                    "image": image_lst,
                    "conversations": [
                        {
                            "from": "human",
                            "value": human_prompt
                        },
                        {
                            "from": "gpt",
                            "value": gpt_prompt
                        }
                    ]
                }
                json_list.append(dict)

def create_single_image_dataset():
    global json_list
    # 遍历transcription文件夹
    file_list = os.listdir(transcription_path)
    file_list.sort(key=lambda x: x.split(".")[0].split("_")[-1])
    for filename in file_list:
        if filename.endswith('.txt'):
            file_path = os.path.join(transcription_path, filename)
            # 获取运动学信息
            kinematics_info = get_kinematic_information(filename=filename)
            print(f"len of kinematicx information: {len(kinematics_info)}")
            frame = []
            # 获取有效帧的序号
            with open(file_path, 'r') as file:
                for line in file: 
                    start_f, end_f, gesture = line.strip().split()
                    # print(start_f, end_f, gesture)
                    for i in range(int(start_f), int(end_f)+1):
                        frame.append((i, gesture))

            # 每连续的num_img张图片组成一个sample
            for i in range(len(frame)-num_img+1):
                # image_list
                image_lst = []
                id : str = ""
                human_prompt: str = "Here is a frame of surgical activity."
                gpt_prompt: str = ""
                for j in range(num_img):
                    postfix = filename.replace('.txt', '') + '_' + "{:04d}.jpg".format(int(frame[i+j][0]))
                    image = image_path + postfix
                    image_lst.append(image)

                    # id : the name of the predict image
                    id = postfix.replace('.jpg', '')

                    # human prompt
                    human_prompt += f" <image>\nThe gesture of the frame is {gesture_descriptions[frame[i+j][1]]}, please predict kinematics information of this frame in 1 X 76 tensor."
                    gpt_prompt += f"{kinematics_info[frame[i+j][0]-1]}"
                   
      
                dict = {
                    "id": id,
                    "image": image_lst,
                    "conversations": [
                        {
                            "from": "human",
                            "value": human_prompt
                        },
                        {
                            "from": "gpt",
                            "value": gpt_prompt
                        }
                    ]
                }
                json_list.append(dict)
def aggregate(num_img=num_img):
    mission = ["Knot_Tying", "Needle_Passing", "Suturing"]
    total_sample = []
    for mission_name in mission:
        json_path = f'data/JIGSAWS/{mission_name}/{mission_name}_{num_img}images.json'
        with open(json_path, 'r') as f:
            json_file = json.load(f)
        total_sample.extend(json_file)
    
    print("Number of sample: ", len(total_sample))
    total_len = len(total_sample)
    num_test = total_len // 5
    num_train = total_len - num_test

    test_sam = []
    train_sam = []
    random.shuffle(total_sample)
    for i in range(total_len):
        if i < num_test:
            test_sam.append(total_sample[i])
        else:
            train_sam.append(total_sample[i])

    with open(f"data/JIGSAWS/train_{num_img}images.json", 'w') as f:
        json.dump(train_sam, f, indent=2)

    with open(f"data/JIGSAWS/test_{num_img}images.json", 'w') as f:
        json.dump(test_sam, f, indent=2)
        
def create_small_train_set(num_img=num_img):
    train_set = []
    small_train_set = []
    with open(f"data/JIGSAWS/train_{num_img}images.json", 'r') as f:
            train_set = json.load(f)
    
    small_train_set = train_set[:50000]
    with open(f"data/JIGSAWS/train_{num_img}images_small.json", 'w') as f:
        json.dump(small_train_set, f, indent=2)


if __name__ == "__main__":
    # create_single_image_dataset()
    # save the json file
    # with open(json_path, 'w') as file:
        # json.dump(json_list, file, indent=2)
    # print(f"total number of {mission_name} image: {len(json_list)}")
    # aggregate(num_img=num_img)
    create_small_train_set(num_img=num_img)
    # pass

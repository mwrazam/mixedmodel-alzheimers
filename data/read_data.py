# try to figure out how to unzip through gzip
# try on one file
# then open and write with pandas through terminal hmm
import os
from matplotlib.image import imread

file_names = [];
for i in range(12):
    zip_number = str(i+1)
    file_names.append('disc'+zip_number)

# print(file_names)

data_path = os.getcwd() # where all the folders are stored

def get_img_array(interested_path, interested_id):
    """
    interestd path starts after case folder
    i.e. OAS1_0044_MR1 + '/PROCESSED/MPRAGE/SUBJ_111',
    '/PROCESSED/MPRAGE/SUBJ_111' is the interested path

    for SUBJ - interested_id = 'gif'
    for T88_110 - interested_id = 't88_gfc_cor_110'
    for T88_95 - interested_id = 'gfc_sag_95'
    for T88_90 't88_gfc_tra_90.gif'
    for T88_M90 'masked'
    for FSL_SEG - T88_M90 interested_id = 'gif'
     """
    img_array=[]
    img_count = 0
    case_ID_list = []
    for file_name in file_names: # just try the first folder and then we expand
    #file_name = file_names[2]
        #print('/n')
        #print('AHHHH')
        #print(file_name)
        disc_path = data_path + '/' + file_name
        case_list = [actual_case for actual_case in os.listdir(disc_path) if 'OAS' in actual_case]
        # print(case_list)
        sorted_case_list = sorted(case_list, key = lambda x: x.split('_')[1]) # sort the list to match the classification label list
        # print('sorted')
        #print(sorted_case_list)

        for case in sorted_case_list:
            case_path = disc_path + '/' + case
            #print(case_path)
            interested_f_path = case_path + interested_path #(where the image is stored)
            interestd_image = [f_name for f_name in os.listdir(interested_f_path) if interested_id in f_name][0]
            #print(type(SUBJ_image))
            #print(SUBJ_path)
            img_path = interested_f_path+'/'+interestd_image
            #print(img_path)
            img = imread(img_path)
            img_array.append(img)
            case_ID_list.append(case)
        img_count += len(sorted_case_list)
    print(img_count)
    return (img_array,case_ID_list)

# print('Following is our SUBJ_111 list of arrary')
# print(img_array)
# print('list of image arrary length', len(img_array))
# print('img count',img_count)
# print('manual check', 39+38+36+35+38+37+38+35+35+35+34+36)
# print(case_ID_list)

# ////// TO DO LIST ///// AM
# then create other x input data for others = every dataset one image - make sure it's the same orientation

#### SUBJ ####
SUBJ_array = get_img_array(interested_path = '/PROCESSED/MPRAGE/SUBJ_111',interested_id='gif')[0]
case_ID_list = get_img_array(interested_path = '/PROCESSED/MPRAGE/SUBJ_111',interested_id='gif')[1]

import pickle
# universial
outfile = open('case_ID_list','wb')
pickle.dump(case_ID_list,outfile)
outfile.close()

outfile = open('SUBJ_image_array','wb')
pickle.dump(SUBJ_array,outfile)
outfile.close()

#### T88_110 ####
T88_110_array = get_img_array(interested_path = '/PROCESSED/MPRAGE/T88_111',interested_id='t88_gfc_cor_110')[0]
outfile = open('T88_110_image_array','wb')
pickle.dump(T88_110_array,outfile)
outfile.close()

T88_95_array = get_img_array(interested_path = '/PROCESSED/MPRAGE/T88_111',interested_id='gfc_sag_95')[0]
outfile = open('T88_95_image_array','wb')
pickle.dump(T88_95_array,outfile)
outfile.close()

T88_90_array = get_img_array(interested_path = '/PROCESSED/MPRAGE/T88_111',interested_id='t88_gfc_tra_90.gif')[0]
outfile = open('T88_90_image_array','wb')
pickle.dump(T88_90_array,outfile)
outfile.close()

T88_M90_array = get_img_array(interested_path = '/PROCESSED/MPRAGE/T88_111',interested_id='t88_masked_gfc_tra_90.gif')[0]
outfile = open('T88_M90_image_array','wb')
pickle.dump(T88_M90_array,outfile)
outfile.close()

FSL_SEG_M90_array = get_img_array(interested_path = '/FSL_SEG',interested_id='gif')[0]
outfile = open('FSL_SEG_M90_image_array','wb')
pickle.dump(FSL_SEG_M90_array,outfile)
outfile.close()

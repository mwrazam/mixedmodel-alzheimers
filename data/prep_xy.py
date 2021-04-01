import pickle
import numpy as np
from numpy import save

infile = open('case_ID_list','rb')
case_ID_list = pickle.load(infile)
infile.close()
#
# infile = open('SUBJ_image_array','rb')
# SUBJ_image_array = pickle.load(infile)
# infile.close()

#print(SUBJ_image_array)
#print(case_ID_list)

import pandas as pd
class_info = pd.read_csv('classification_info.txt')
print(class_info)
print(class_info.info())
print('So we only have 216 data')

y_candidate = class_info[['ID','SES']]
#print(y_candidate)
y = y_candidate.dropna()
#print(y)

index_list =[]
for ID in y['ID'].tolist():
    index_list.append(case_ID_list.index(ID))

#print(index_list)

#print(type(y['SES'].tolist()[1]))

class_y = y['SES'].tolist()
class_y = np.array(class_y)
print(class_y.shape)
save('y.npy', class_y)

#
#
# SUBJ_x = []
# for index in index_list:
#     SUBJ_x.append(SUBJ_image_array[index])
#
# print(len(SUBJ_x))
# print(SUBJ_x[1].shape)
#
# outfile = open('SUBJ_x','wb')
# pickle.dump(SUBJ_x,outfile)
# outfile.close()
#

def create_x(pickled_image_array,array_filename):

    """
    pickled_image_array - already pickled from last step - e.g. SUBJ_image_array
    array_filename = string - 'SUBJ_x.npy'
    """


    infile = open(pickled_image_array,'rb')
    image_array = pickle.load(infile)
    infile.close()

    x = []
    for index in index_list:
        x.append(image_array[index])

    x = np.array(x)
    print('shape of ' + pickled_image_array+': ', x.shape)
    save(array_filename, x)

# our X for SUBJ111 IMAGE
create_x('SUBJ_image_array','SUBJ_x.npy')
create_x('T88_110_image_array','T88_110_x.npy')
create_x('T88_95_image_array','T88_95_x.npy')
create_x('T88_90_image_array','T88_90_x.npy')
create_x('T88_M90_image_array','T88_M90_x.npy')
create_x('FSL_SEG_M90_image_array','FSEG_T88_M90_x.npy')

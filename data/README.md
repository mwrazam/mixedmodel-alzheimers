# Data used for network development

216 datapoint in each x_list; 1 y_list
## X:
x_list: Taken from the processed image data folder in each case.
- SUBJ 111 (SUBJ_x)
- T88 111 - cor_110 (T88_110_x)
- T88 111 - sag_95 (T88_95_x)
- T88 111 - tra_90 (T88_90_x)
- T88 111 - masked_tra_90 (T88_M90_x)
- FSEG T88 111 - masked-gfc-fseg-tra-90 (FSEG_T88_M90_x)

Each array shape (216, 256, 256)

## Y:
y is taken from the SES column of the CSV sheet;
(float data type -> converted to int in load_data.py)

import os
import urllib.request
import multiprocessing

if not os.path.exists('train_img_slices'):
    os.makedirs('train_img_slices')
# if not os.path.exists('valid_img_slices'):
#     os.makedirs('valid_img_slices')
# if not os.path.exists('test_img_slices'):
#     os.makedirs('test_img_slices')


def download_volume(i):
    if i < 17:
        folder = 'train_img_slices'
    # elif i < 19:
    #     folder = 'valid_img_slices'
    # else:
    #     folder = 'test_img_slices'
        
    for j in range(320):
        
        url = "https://s3-us-west-2.amazonaws.com/knee-slices/{}/{}_{}.ra".format(folder, i, j)
        filename = "{}_{}.ra".format(i, j)
        print('Downloading {}'.format(filename))
        
        urllib.request.urlretrieve (url, os.path.join(folder, filename))

    
with multiprocessing.Pool() as pool:

    pool.map(download_volume, range(1, 2))

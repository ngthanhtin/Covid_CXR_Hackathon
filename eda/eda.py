import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

path = '../TrainSet/'
image_path = path + 'TrainSet/'
metadata_path = path + 'trainClinData.xls'
metadata_df = pd.read_excel(metadata_path)
# for col_name in metadata_df.columns: 
#     print(col_name, metadata_df[col_name].count())
metadata_df['ImageFile'] = image_path+metadata_df['ImageFile']
selected_df = metadata_df[['ImageFile', 'Prognosis']]
mapping = {'SEVERE': 0, 'MILD': 1}

# Check label distribution
print(selected_df.Prognosis.value_counts())

# check im size distribution
def check_im_size(df):
    im_shape_severe_x = []
    im_shape_severe_y = []
    im_shape_mild_x = []
    im_shape_mild_y = []
    
    grouped = df.groupby(df.Prognosis)
    severe_df = grouped.get_group("SEVERE")
    mild_df = grouped.get_group("MILD")

    for i, img in enumerate(severe_df.ImageFile):
        sample_img = Image.open(img)
        w, h = sample_img.size
        im_shape_severe_x.append(w)
        im_shape_severe_y.append(h)

    for i, img in enumerate(mild_df.ImageFile):
        sample_img = Image.open(img)
        w, h = sample_img.size
        im_shape_mild_x.append(w)
        im_shape_mild_y.append(h)
        
    fig = plt.figure(figsize=(12, 8))

    fig.add_subplot(231)
    plt.hist(im_shape_severe_x)
    plt.title('X size: Severe')
    fig.add_subplot(232)
    plt.hist(im_shape_severe_y)
    plt.title('Y size: Severe')
    fig.add_subplot(233)
    plt.hist(im_shape_mild_x)
    plt.title('X size: Mild')
    fig.add_subplot(234)
    plt.hist(im_shape_mild_y)
    plt.title('Y size: Mild')

    plt.tight_layout()
    plt.show()
    # print(im_shape_severe_w, im_shape_severe_h, im_shape_mild_w, im_shape_mild_h)

def standardization(df, npics= 12):
    
    fig = plt.figure(figsize=(15, 15))
    count=1
    for i, img in enumerate(df.ImageFile):
        sample_img = Image.open(img)   
        sample_img = np.array(sample_img)
        sample_img = sample_img/255.0
        sample_img_mean = np.mean(sample_img)
        sample_img_std = np.std(sample_img)
        new_sample_img = (sample_img - sample_img_mean)/sample_img_std
        # ax = fig.add_subplot(int(npics/2) , 3, count, yticks=[])
        sns.histplot(new_sample_img.ravel(), 
                label=f'Pixel Mean A {np.mean(new_sample_img):.2f} & Std. A {np.std(new_sample_img):.2f}', kde=False, color='blue', bins=35, alpha=0.8)
        sns.histplot(sample_img.ravel(), 
                label=f'Pixel Mean B {np.mean(sample_img):.2f} & Std. B {np.std(sample_img):.2f}', kde=False, color='red', bins=35, alpha=0.8)
        plt.legend(loc='upper center', fontsize=9)
        plt.title('Image Num: %s'% (img))
        plt.xlabel('Pixel Intensity')
        plt.ylabel('# Pixels in Image')
        count +=1
    fig.suptitle('Pixel Intensity Distribution (Before & After Std.)')
    plt.tight_layout()
    plt.show()


check_im_size(selected_df)
# standardization(selected_df, npics=1)

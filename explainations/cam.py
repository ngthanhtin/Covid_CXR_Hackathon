#Here, I have used Gradient-weighted Class Activation Maps.It uses the gradients of any target concept (say logits for ‘cat’), 
# flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept.

#So, to explain in simple terms, we simply take the final convolutional feature map and then we weigh every channel in that feature with 
# the gradient of the class with respect to the channel. It’s just nothing but how intensely the input image activates different channels by 
# how important each channel is with regard to the class. The best part is it doesn’t require any re-training or change in the 
# existing architecture unlike CAM where a Global Average Pooling layer is needed to generate activations.

import numpy as np
import cv2
from skimage import data, color, io, img_as_float
import matplotlib.pyplot as plt

def get_heatmap(processed_image, class_idx):
    # we want the activations for the predicted label
    class_output = vgg_conv.output[:, class_idx]
    
    # choose the last conv layer in your model
    last_conv_layer = vgg_conv.get_layer('block5_conv3')
    
    # get the gradients wrt to the last conv layer
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    
   # we pool the gradients over all the axes leaving out the channel dimension
    pooled_grads = K.mean(grads, axis=(0,1,2))
    
    # Define a function that generates the values for the output and gradients
    iterate = K.function([vgg_conv.input], [pooled_grads, last_conv_layer.output[0]])
    
    # get the values
    grads_values, conv_ouput_values = iterate([processed_image])
    
    # iterate over each feature map in your conv output and multiply
    # the gradient values with the conv output values. This gives an 
    # indication of "how important a feature is"
    for i in range(512): # we have 512 features in our last conv layer
        conv_ouput_values[:,:,i] *= grads_values[i]
    
    # create a heatmap
    heatmap = np.mean(conv_ouput_values, axis=-1)
    
    # remove negative values
    heatmap = np.maximum(heatmap, 0)
    
    # normalize
    heatmap /= heatmap.max()
    
    return heatmap

# select the sample and read the corresponding image and label
sample_image = cv2.imread('/home/hci-a4000/TIN/covid2022/TestSet/TestSet/P_1_107.png')
# pre-process the image
sample_image = cv2.resize(sample_image, (256,256))
if sample_image.shape[2] ==1:
            sample_image = np.dstack([sample_image, sample_image, sample_image])
sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
sample_image = sample_image.astype(np.float32)/255.
sample_label = 1
    
    
sample_image_processed = np.expand_dims(sample_image, axis=0)#since we pass only one image,we expand dim to include
                                                             #batch size 1
    
# get the label predicted by our original model
pred_label = np.argmax(vgg_conv.predict(sample_image_processed), axis=-1)[0]
    
    
# get the heatmap for class activation map(CAM)
heatmap = get_heatmap(sample_image_processed, pred_label)
heatmap = cv2.resize(heatmap, (sample_image.shape[0], sample_image.shape[1]))
heatmap = heatmap *255
heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#superimpose the heatmap on the image    

sample_image_hsv = color.rgb2hsv(sample_image)
heatmap = color.rgb2hsv(heatmap)

alpha=0.7
sample_image_hsv[..., 0] = heatmap[..., 0]
sample_image_hsv[..., 1] = heatmap[..., 1] * alpha

img_masked = color.hsv2rgb(sample_image_hsv)

f,ax = plt.subplots(1,2, figsize=(16,6))
ax[0].imshow(sample_image)
ax[0].set_title(f"True label: {sample_label} \n Predicted label: {pred_label}")
ax[0].axis('off')
    
ax[1].imshow(img_masked)
ax[1].set_title("Class Activation Map")
ax[1].axis('off')

plt.show()
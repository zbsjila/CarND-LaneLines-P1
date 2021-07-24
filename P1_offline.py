#!/usr/bin/env python
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# 
# ## Project: **Finding Lane Lines on the Road** 
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 
# 
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
# 
# In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.
# 
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
# 
# **Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**
# 
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
# 
# ---
# 
# <figure>
#  <img src="examples/line-segments-example.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
#  </figcaption>
# </figure>
#  <p></p> 
# <figure>
#  <img src="examples/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
#  </figcaption>
# </figure>

# **Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, consult the forums for more troubleshooting tips.**  

# ## Import Packages

# In[ ]:


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import logging
from matplotlib.widgets import Slider
#get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read in an Image

# In[ ]:


#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
# plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


# ## Ideas for Lane Detection Pipeline

# **Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**
# 
# `cv2.inRange()` for color selection  
# `cv2.fillPoly()` for regions selection  
# `cv2.line()` to draw lines on an image given endpoints  
# `cv2.addWeighted()` to coadd / overlay two images  
# `cv2.cvtColor()` to grayscale or change color  
# `cv2.imwrite()` to output images to file  
# `cv2.bitwise_and()` to apply a mask to an image
# 
# **Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

# ## Helper Functions

# Below are some helper functions to help get you started. They should look familiar from the lesson!

# In[ ]:


import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=None, thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            #cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            #cv2.line(img, (x1, y1), (x2, y2), np.random.randint(0, 256), thickness);
            #cv2.line(img, (x1, y1), (x2, y2), list(np.random.randint(0, 256, size=3, dtype=img.dtype)), thickness); # np.uint8
            #color = tuple ([int(x) for x in np.random.randint(0, 256, size=3, dtype=img.dtype)]);
            #color = np.random.randint(0, 256, size=3, dtype=int);
            if color is None:
                color = [int(x) for x in np.random.randint(0, 256, size=3, dtype=img.dtype)];
            cv2.line(img, (x1, y1), (x2, y2), color, thickness); # np.uint8

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap): # TODO: now take out the process_lines & draw_lines_to. update the caller
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    #print("len(lines) = %d"%(len(lines)));
    return lines;

def lines_process(lines, ymin, ymax, logging_str=''):
    # lines_process.dot
    # lines -> {k, b, length} 
    # k -> + - each sign category: 
    #   length sort two longest: keep 2x2 lines
    # : kb average
    # return two lines: kb -> xxyy with given yy; or ymin ymax of xxyy
    # if missing lanes: do nothing <- no lines returned

    # prepare
    lines_out = [];
    nline = len(lines);
    if (nline == 0):
        logging.warning("%s: nline = 0"%(logging_str));
        return;

    # k b length compute:
    k_b_length = np.zeros((3, nline)); # 3xn: 3 row matrix: k b length
    #print("lines = ");
    #print(lines);

    for iline in range(nline):
        #print("lines[iline] = " );
        #print(lines[iline]);

        for x1,y1,x2,y2 in lines[iline]:
            #print("[x1,y1,x2,y2] = [%d %d %d %d]"%(x1,y1,x2,y2));
            k = (y2-y1)/(x2-x1);
            b = y1-k*x1;
            length = np.sqrt((y2-y1)**2 + (x2-x1)**2);
            k_b_length[:, iline] = [k, b, length];

    # left:
    def longest_mean(index_selected, k_b_length):
        line = np.array([[]]);
        nselected = np.sum(index_selected);
        if (nselected == 0):
            logging.warning("%s: selected lane not detected"%(logging_str));
            return line;

        k_b_length_selected = k_b_length[:, index_selected];
        if nselected > 0 and nselected <= 2:
            k_b_length_selected_longest_mean = np.mean(k_b_length_selected, axis=1);
        else: # nline > 2
            index_longest = np.argsort(k_b_length_selected[2, :])[[-2, -1]];
            k_b_length_selected_longest_mean = np.mean(k_b_length_selected[:, index_longest], axis=1); # longest 2 lines

        line = np.array([[(ymin - k_b_length_selected_longest_mean[1])/k_b_length_selected_longest_mean[0], ymin, (ymax - k_b_length_selected_longest_mean[1])/k_b_length_selected_longest_mean[0], ymax]], dtype='int32');
        return line;

    # y = kx + b; x = (y-b)/k
    # slope classifying: left: [0] right [0]
    #print("slope = ", np.sort(k_b_length[0, :]));
    index_left = np.logical_and(k_b_length[0, :] >= -0.95, k_b_length[0, :] <= -0.60);
    index_right =  np.logical_and(k_b_length[0, :] >= 0.45, k_b_length[0, :] <= 0.75); 

    line_left = longest_mean(index_left, k_b_length);
    if line_left.size > 0:
        lines_out.append(line_left);

    line_right = longest_mean(index_right, k_b_length);
    if line_right.size > 0:
        lines_out.append(line_right);

    lines_out = np.array(lines_out);
    return lines_out; 

def lines_to_image(lines, imshape, color=None, thickness=2):
    line_img = np.zeros((imshape[0], imshape[1], 3), dtype=np.uint8); # img.shape
    draw_lines(line_img, lines, color=color, thickness=thickness);
    return line_img;

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


# ## Test Images
# 
# Build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[ ]:


import os
image_list = [];
images_dir = 'test_images_mixed/'; # test_images
image_basename_list = [filename for filename in sorted(os.listdir(images_dir)) if '.jpg' in filename];
image_file_list = [images_dir + image_basename for image_basename in image_basename_list];
#print("image_file_list =", image_file_list);
n_image = len(image_file_list);
print("n_image = %d"%n_image);
for i_image in range(n_image):
    image_list.append(mpimg.imread(image_file_list[i_image]));
imshape = image_list[0].shape;

def plot_image_pair_4x3(image_list, image_canny_list):
    # cmap ignored for RGB(A) data so I use 'gray'
    AxesImage_canny_list = [];
    nx_sp = 3; # n_image/2;
    fig_canny_gui, ax = plt.subplots(4, 3, figsize=(15, 8), sharex=True, sharey=True);
    for i_image in range(n_image):
        #ax[0][i_image].imshow(image_blur_list[i_image], cmap='gray'); 
        ax[0+int(i_image/3)][i_image%3].set_title(image_file_list[i_image]);

        ax[0+int(i_image/3)][i_image%3].imshow(image_list[i_image], cmap='gray'); 
        AxesImage_canny_list.append(ax[2+int(i_image/3)][i_image%3].imshow(image_canny_list[i_image], cmap='gray')); # 
    return fig_canny_gui, AxesImage_canny_list;

# ## Build a Lane Finding Pipeline
# 
# 

# Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.
# 
# Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.

# In[ ]:


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.
# since later there is a process_image funcion, this can be the parameter tuning section

# ### Blur
image_blur_list = [];
for i_image in range(n_image):
    image_blur_list.append(gaussian_blur(grayscale(image_list[i_image]), 5));

# ### Canny: GUI
low_threshold_tuned, high_threshold_tuned = 42, 85; # 110, 215;
image_canny_list = image_blur_list.copy(); 
fig_canny_gui, AxesImage_canny_list = plot_image_pair_4x3(image_list, image_canny_list);
fig_canny_gui.suptitle('canny_gui');


def update_canny(val):
    low_threshold = slider_low_threshold.val;
    high_threshold = slider_high_threshold.val;
    for i_image in range(n_image):
        image_canny_list[i_image] = canny(image_blur_list[i_image], low_threshold, high_threshold); # 50, 150
        AxesImage_canny_list[i_image].set_data(image_canny_list[i_image]);

    fig_canny_gui.canvas.draw_idle();

plt.subplots_adjust(bottom=0.10+0.05*2); # [0.10 0.15 0.20]

ax_low_threshold = plt.axes([0.25, 0.15, 0.65, 0.03]);
slider_low_threshold = Slider(ax_low_threshold, 'low_threshold', 0, 400, valinit=low_threshold_tuned, valstep=5);
slider_low_threshold.on_changed(update_canny);

ax_high_threshold = plt.axes([0.25, 0.10, 0.65, 0.03]);
slider_high_threshold = Slider(ax_high_threshold, 'high_threshold', 0, 400, valinit=high_threshold_tuned, valstep=5);
slider_high_threshold.on_changed(update_canny);

""" parameter tuning findings:
low: < 55; otherwise P1 left lane right side will be missed
high: <280; otherwise P2 left lane right side will be missed
[55 280]
[65 195 or 215 < 225]
"""

# ### Canny: apply tuned parameter
for i_image in range(n_image):
    image_canny_list[i_image] = canny(image_blur_list[i_image], low_threshold_tuned, high_threshold_tuned); # 65, 215 # 50, 150 #  110, 215

# ### region_of_interest: vertices
# 445 -> 405: 40 pixels clearance
x_ROI_normalized = np.array([172, 405, 592, 854])/960.0;
y_ROI_normalized = np.array([347.0, 500.0])/540.0;
x_ROI = (imshape[1]*x_ROI_normalized).astype(int); 
y_ROI = (imshape[0]*y_ROI_normalized).astype(int); 

#y_up = 347; # 322; 
#y_down = 500; # imshape[0] - 1;  # TODO: normalized
# start: lower left corner; clockwise
#vertices = np.array([[(123,y_down),(427, y_up), (550, y_up), (918,y_down)]], dtype=np.int32) 
vertices = np.array([[(x_ROI[0],y_ROI[1]),(x_ROI[1], y_ROI[0]), (x_ROI[2], y_ROI[0]), (x_ROI[3],y_ROI[1])]], dtype=np.int32) 
print("vertices = ", vertices);

image_vertices = image_list[2].copy();
cv2.polylines(image_vertices,[vertices],True,(0,255,255), thickness=5);
fig_v, ax_v = plt.subplots(1,1); 
plt.imshow(image_vertices);
ax_v.set_title('image_vertices');

masked_edges_list = [];
for i_image in range(n_image):
    masked_edges_list.append(region_of_interest(image_canny_list[i_image], vertices));

fig_masked_edges, AxesImage_masked_list = plot_image_pair_4x3(image_list, masked_edges_list);
fig_masked_edges.suptitle('masked_edges');

# ### Hough
# #### init & plot
threshold_tuned, min_line_len_tuned, max_line_gap_tuned = 15, 15, 300; # 135; # 36, 120, 135;
image_hough_list = masked_edges_list.copy();
fig_hough_gui, AxesImage_hough_gui_list = plot_image_pair_4x3(masked_edges_list, image_hough_list);
fig_hough_gui.suptitle('hough_gui');

# #### def update_hough(val):
def update_hough(val):
    threshold = int(slider_threshold.val);
    min_line_len = slider_min_line_len.val;
    max_line_gap = slider_max_line_gap.val;

    for i_image in range(n_image):
        image_hough_list[i_image] = lines_to_image(hough_lines(masked_edges_list[i_image], 1, np.pi/180, threshold, min_line_len, max_line_gap), imshape, color=None); # 40, 20, 10); // 
        AxesImage_hough_gui_list[i_image].set_data(image_hough_list[i_image]);

    fig_hough_gui.canvas.draw_idle();

plt.subplots_adjust(bottom=0.10+0.05*3); # [0.10 0.15 0.20 0.25]

ax_threshold = plt.axes([0.25, 0.20, 0.65, 0.03]);
slider_threshold = Slider(ax_threshold, 'threshold', 1, 400, valinit=threshold_tuned, valstep=5);
slider_threshold.on_changed(update_hough);

ax_min_line_len = plt.axes([0.25, 0.15, 0.65, 0.03]);
slider_min_line_len = Slider(ax_min_line_len, 'min_line_len', 0, 400, valinit=min_line_len_tuned, valstep=5);
slider_min_line_len.on_changed(update_hough);

ax_max_line_gap = plt.axes([0.25, 0.10, 0.65, 0.03]);
slider_max_line_gap = Slider(ax_max_line_gap, 'max_line_gap', 0, 300, valinit=max_line_gap_tuned, valstep=5);
slider_max_line_gap.on_changed(update_hough);

""" parameter tuning findings:
* start broad: [threshold len gap] = [1, 5(instead 1 which gives too many and distracting), longest gap between line segments ~200]
* narrow down: 
        * len: increase until some line filterd out. 170
        * th: increase until critical lines get filtered out
        * gap: reduce until continuity -> broken
"""

# #### apply tuned parameters
lines_list = [];
image_line_list = []; 
for i_image in range(n_image):
    lines_list.append(hough_lines(masked_edges_list[i_image], 1, np.pi/180, threshold_tuned, min_line_len_tuned, max_line_gap_tuned)); 
    image_line_list.append(lines_to_image(lines_list[i_image], imshape, color=None));

print("type(lines_list) = ", type(lines_list));
print("type(lines_list[0]) = ", type(lines_list[0]));
print("lines_list[0].shape = ", lines_list[0].shape); 

"""
type(lines_list) =  <class 'list'>
type(lines_list[0]) =  <class 'numpy.ndarray'>
lines_list[0].shape =  (6, 1, 4)
lines_list = [ array([[[154, 538, 456, 326]], [[175, 539, 442, 338]], [[155, 538, 449, 332]], [[175, 538, 446, 334]], [[193, 510, 455, 326]], [[515, 332, 780, 491]]], dtype=int32), ...]
"""
#print("lines_list = ");
#print(lines_list);


# ### line processing
"""
lines -> [length slope:+-] -> max 2: average: [kb1 kb2] -> draw

* original
* extended to ROI top/bottom
* slope: + -: 2 categories
* reduce: 
    * average 
    * median of longest 2 
* longest 2 averaging: -> [k b]
* draw: ROI top/bottom
"""
logging.basicConfig(format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO);
lines_processed_list = [];
for i_image in range(n_image): # [3]: # 
    lines_processed_i = lines_process(lines_list[i_image], y_ROI[0], y_ROI[1], logging_str="i_image=%d"%i_image);
    print("lines_list[%d] = "%(i_image), lines_list[i_image]);
    print("lines_processed_i = ", lines_processed_i);
    lines_processed_list.append(lines_processed_i);

#print("lines_processed_list = ");
#print(lines_processed_list);

# plot image_lines_processed_list
image_lines_processed_list = []; 
for i_image in range(n_image):
    image_lines_processed_list.append(lines_to_image(lines_processed_list[i_image], imshape, color=[255, 0, 0], thickness=20)); # # lines_list
fig_line_process, AxesImage_line_process_list = plot_image_pair_4x3(image_line_list, image_lines_processed_list);
fig_line_process.suptitle('line_process');

# plot and save final overlayed output: line_image
image_annotated_list = [];
images_output_dir = 'test_images_output/';
image_output_file_list = [images_output_dir + image_basename for image_basename in image_basename_list];
print("image_output_file_list = ", image_output_file_list);

for i_image in range(n_image):
    image_annotated_list.append(weighted_img(image_list[i_image], image_lines_processed_list[i_image], 0.8, 1.0, 0.));

    logging.info("i_image = %d, imsave %s"%(i_image, image_output_file_list[i_image]));
    mpimg.imsave(image_output_file_list[i_image], image_annotated_list[i_image]);
fig_annotated, _ = plot_image_pair_4x3(image_list, image_annotated_list);
fig_annotated.suptitle('annotated');

# ### check single image across pipeline
i_image = 3;
fig_pipeline, ax_pipeline = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey=True);

ax_pipeline = ax_pipeline.ravel();

ax_pipeline[0].imshow(image_list[i_image]);
ax_pipeline[0].set_title('image');

ax_pipeline[1].imshow(image_canny_list[i_image]);
ax_pipeline[1].set_title('image_canny_list');

ax_pipeline[2].imshow(image_line_list[i_image]);
ax_pipeline[2].set_title('image_line_list');

ax_pipeline[3].imshow(image_annotated_list[i_image]);
ax_pipeline[3].set_title('image_annotated_list');

# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# 
# We can test our solution on two provided videos:
# 
# `solidWhiteRight.mp4`
# 
# `solidYellowLeft.mp4`
# 
# **Note: if you get an import error when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel). Still have problems? Try relaunching Jupyter Notebook from the terminal prompt. Also, consult the forums for more troubleshooting tips.**
# 
# **If you get an error that looks like this:**
# ```
# NeedDownloadError: Need ffmpeg exe. 
# You can download it by calling: 
# imageio.plugins.ffmpeg.download()
# ```
# **Follow the instructions in the error message and check out [this forum post](https://discussions.udacity.com/t/project-error-of-test-on-videos/274082) for more troubleshooting tips across operating systems.**

# In[ ]:


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[ ]:


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # you should return the final output (image where lines are drawn on lanes)

    imshape = image.shape;
    #x_ROI = (imshape[1]*np.array([172, 405, 592, 854])/960.0).astype(int); # 445 -> 405: 40 pixels clearance
    #y_ROI = (imshape[0]*np.array([347.0, 500.0])/540.0).astype(int); 

    x_ROI = (imshape[1]*x_ROI_normalized).astype(int); 
    y_ROI = (imshape[0]*y_ROI_normalized).astype(int); 

    image_blur = gaussian_blur(grayscale(image), 5);
    image_canny = canny(image_blur, low_threshold_tuned, high_threshold_tuned); 
    vertices = np.array([[(x_ROI[0],y_ROI[1]),(x_ROI[1], y_ROI[0]), (x_ROI[2], y_ROI[0]), (x_ROI[3],y_ROI[1])]], dtype=np.int32) 
    masked_edges = region_of_interest(image_canny, vertices);
    lines = hough_lines(masked_edges, 1, np.pi/180, threshold_tuned, min_line_len_tuned, max_line_gap_tuned); 
    lines_processed = lines_process(lines, y_ROI[0], y_ROI[1]);
    image_lines_processed = lines_to_image(lines_processed, imshape, color=[255, 0, 0], thickness=20); # # lines_list
    image_annotated = weighted_img(image, image_lines_processed, 0.8, 1.0, 0.);
    return image_annotated

# Let's try the one with the solid white lane on the right first ...

# In[ ]:

white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!

logging.info('write_videofile to %s start ...'%white_output);
white_clip.write_videofile(white_output, audio=False);
#get_ipython().run_line_magic('time', 'white_clip.write_videofile(white_output, audio=False)')
logging.info('write_videofile to %s end'%white_output);


# Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.

# In[ ]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


# ## Improve the draw_lines() function
# 
# **At this point, if you were successful with making the pipeline and tuning parameters, you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform. As mentioned previously, try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video "P1_example.mp4".**
# 
# **Go back and modify your draw_lines function accordingly and try re-running your pipeline. The new output should draw a single, solid line over the left lane line and a single, solid line over the right lane line. The lines should start from the bottom of the image and extend out to the top of the region of interest.**

# Now for the one with the solid yellow lane on the left. This one's more tricky!

# In[ ]:


yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)

#get_ipython().run_line_magic('time', 'yellow_clip.write_videofile(yellow_output, audio=False)')

logging.info('write_videofile to %s start ...'%yellow_output);
yellow_clip.write_videofile(yellow_output, audio=False);
#get_ipython().run_line_magic('time', 'yellow_clip.write_videofile(yellow_output, audio=False)')
logging.info('write_videofile to %s end'%yellow_output);

# In[ ]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))


# ## Writeup and Submission
# 
# If you're satisfied with your video outputs, it's time to make the report writeup in a pdf or markdown file. Once you have this Ipython notebook ready along with the writeup, it's time to submit for review! Here is a [link](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) to the writeup template file.
# 

# ## Optional Challenge
# 
# Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!

# In[ ]:


challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
#get_ipython().run_line_magic('time', 'challenge_clip.write_videofile(challenge_output, audio=False)')
logging.info('write_videofile to %s start ...'%challenge_output);
challenge_clip.write_videofile(challenge_output, audio=False);
logging.info('write_videofile to %s end'%challenge_output);


# In[ ]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))

# ## End: show
plt.show();

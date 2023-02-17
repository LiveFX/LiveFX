![Banner](/assets/banner.png)
# Project Description
This project contains a Python script that applies various image processing effects to a video stream. The script uses several computer vision libraries including OpenCV and CuPy.

The effects that are currently implemented in the script include:

- Adjusting the brightness and contrast of the video
- Detecting edges in the video
- Applying Gaussian blur to the video
- Rotating the video by a given angle

The script reads in a video stream from the default camera and applies the selected effect on each frame before displaying it in a virtual camera.

The effects are implemented using a combination of OpenCV and CuPy libraries. OpenCV is used for standard image processing tasks, such as color conversion, image splitting and merging, and kernel filtering. CuPy is used to accelerate matrix computations using the power of GPUs.

# Technical Details
The script first reads in the video stream from the default camera and displays it in a window using OpenCV. Then, it starts a virtual camera using the pyvirtualcam library. This virtual camera allows the script to output the processed video stream as a webcam source, which can be used in video conferencing applications and other similar programs.
## Brightness and Contrast Adjustment
The __brightness and contrast adjustment__ effect is implemented by first converting the input frame from RGB color space to YUV color space. The brightness of the frame is then adjusted by scaling the Y channel by the brightness value and adding it to the Y channel. The contrast of the frame is adjusted by scaling the Y channel by the contrast value. Finally, the frame is converted back to RGB color space before being returned.

## Edge Detection and Gaussian Blur
The __edge detection__ effect is implemented by applying a 3x3 kernel filter to the Y channel of the frame after converting it to YUV color space. The kernel used for this effect is a simple edge detection kernel. After applying the kernel, the Y channel is merged with the U and V channels, and the frame is converted back to RGB color space.

The __Gaussian blur__ effect is implemented using a Gaussian kernel function. The kernel function is used to generate a 2D Gaussian distribution matrix of size kernel_size x kernel_size. The sigma parameter controls the standard deviation of the Gaussian distribution. The 2D Gaussian distribution matrix is then convolved with the Y channel of the input frame using a 2D convolution operation. The U and V channels are then merged with the processed Y channel and the frame is converted back to RGB color space.

## Rotation
The __rotation effect__ is implemented using a rotation matrix. The rotation matrix is generated using the angle parameter and a pre-defined formula. The input frame is then transformed using the rotation matrix to achieve the desired rotation. The frame is then returned as the output.

All of the matrix operations in the script are performed using the CuPy library to take advantage of the power of GPUs. This allows for much faster processing of the video stream and smoother output.

# Conclusion
This script demonstrates how to apply various image processing effects to a video stream using OpenCV and CuPy libraries. The script can be used to create virtual cameras with various effects applied to them. With some additional tweaking, the script could be modified to apply additional effects or even recognize and track objects in the video stream.
import numpy as np
from PIL import Image

# Read the raw file
with open('sample-image.raw', 'rb') as f:
    data = f.read()

# Extract height and width from the file
height = int.from_bytes(data[0:4], byteorder='little')
width = int.from_bytes(data[4:8], byteorder='little')

# Extract the pixel data array
pixel_data = np.frombuffer(data[8:], dtype=np.uint16)

# Reshape the pixel data array to match the image dimensions
pixel_data = pixel_data.reshape((height, width))

# Initialize an empty image
image = np.zeros((height, width, 3), dtype=np.uint16)


red_channel = np.zeros((height, width))
green_channel = np.zeros((height, width))
blue_channel = np.zeros((height, width))

# Color the pixels based on the Bayer filter pattern
# Assuming RGGB Bayer filter pattern
for i in range(height):
    for j in range(width):
        if (i % 2 == 0 and j % 2 == 0):
            # Red pixel
            image[i, j, 0] = pixel_data[i, j]
            red_channel[i,j] = pixel_data[i, j]
        elif (i % 2 == 0 and j % 2 == 1) or (i % 2 == 1) and (j % 2 == 0):
            # Green pixel in blue row
            image[i, j, 1] = pixel_data[i, j]
            green_channel[i,j] = pixel_data[i, j]
        else:
            # Blue pixel
            image[i, j, 2] = pixel_data[i, j]
            blue_channel[i,j] = pixel_data[i, j]

# Convert the image to 8-bit per channel for visualization
image = np.uint8(image / 256 * image.clip(0, 1)**(1/2.2))
    
# Create a PIL image from the numpy array
pil_image = Image.fromarray(image)

# Save the image
pil_image.save('q4.png')


# Perform per-channel bilinear interpolation
# Red channel
for i in range(1, height - 1, 2):
    for j in range(1, width - 1, 2):
        red_channel[i, j] = (red_channel[i-1, j-1] + red_channel[i-1, j+1] +
                             red_channel[i+1, j-1] + red_channel[i+1, j+1]) // 4

# Green channel
for i in range(0, height - 1, 2):
    for j in range(1, width - 1, 2):
        green_channel[i, j] = (green_channel[i, j-1] + green_channel[i, j+1] +
                               green_channel[i-1, j] + green_channel[i+1, j]) // 4

# Blue channel
for i in range(1, height - 1, 2):
    for j in range(0, width - 1, 2):
        blue_channel[i, j] = (blue_channel[i-1, j] + blue_channel[i+1, j] +
                              blue_channel[i, j-1] + blue_channel[i, j+1]) // 4

# Combine color channels to create RGB image
rgb_image = np.stack((red_channel, green_channel, blue_channel), axis=-1)


# Clip values to ensure they are within valid range
rgb_image = np.clip(rgb_image, 0, 65535)

# Perform gamma encoding
gamma = 1/2.2
rgb_image = ((rgb_image / 65535) ** gamma) * 255

# Convert pixel values to uint8 for saving as image
rgb_image = rgb_image.astype(np.uint8)

# Create PIL image from numpy array
image = Image.fromarray(rgb_image, 'RGB')

# Save the image
image.save("q5.png")


from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image

__all__ = ['read_image', 'write_image', 'pad', 'resize']

#------------------------------------------------------------------------------

def read_image(path: Union[Path, str]) -> np.ndarray:
    '''
    Read a PNG or JPG image an array of linear RGB radiance values ∈ [0,1].
    '''
    return (np.float32(Image.open(path)) / 255)**2.2


def write_image(path: Union[Path, str], image: np.ndarray) -> None:
    '''
    Write an array of linear RGB radiance values ∈ [0,1] as a PNG or JPG image.
    '''
    Image.fromarray(np.uint8(255 * image.clip(0, 1)**(1/2.2))).save(path)


def pad(image: np.ndarray, u_pad: int, v_pad: int) -> np.ndarray:
    '''
    Pad an image using nearest-neighbor extrapolation.

    Parameters:
        image: A single-channel (2D) or multichannel (3D) image
        u_pad: The number of pixels to add to each side, vertically
        v_pad: The number of pixels to add to each side, horizontally

    Returns:
        An image of size `(image.shape[0] + 2×u_pad, image.shape[1] + 2×v_pad)`
    '''
    # Compute measurements.
    u0 = u_pad # Top edge of the original image
    v0 = v_pad # Left edge of the original image
    u1 = u_pad + image.shape[0] # Bottom edge of the original image
    v1 = v_pad + image.shape[1] # Right edge of the original image
    h_padded = image.shape[0] + 2 * u_pad # Padded image height
    w_padded = image.shape[1] + 2 * v_pad # Padded image width

    # Copy the input image into the center of a larger array.
    padded_image = np.zeros((h_padded, w_padded, *image.shape[2:]), np.float32)
    padded_image[u0:u1, v0:v1] = image

    # Fill in the edges.
    padded_image[:u0, v0:v1] = image[:1, :]
    padded_image[u1:, v0:v1] = image[-1:, :]
    padded_image[u0:u1, :v0] = image[:, :1]
    padded_image[u0:u1, v1:] = image[:, -1:]

    # Fill in the corners.
    padded_image[:u0, :v0] = image[0, 0]
    padded_image[:u0, v1:] = image[0, -1]
    padded_image[u1:, :v0] = image[-1, 0]
    padded_image[u1:, v1:] = image[-1, -1]

    # Return the result.
    return padded_image


def resize(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    '''
    Convert an image to the specified size (height, width) using bicubic
    interpolation.
    '''
    return _resize_channel(image, size) if image.ndim == 2 else np.dstack([
        _resize_channel(chan, size) for chan in image.transpose(2, 0, 1)
    ])


def _resize_channel(chan: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(chan).resize(size[::-1], Image.CUBIC))
image = read_image('q6-original.png')

# perform gamma decoding
image = image ** 2.2
height = image.shape[0]
width = image.shape[1]

#Y = 0.5*R + 0.6*G + 0.1*B 
#C_b = (B-Y)/1.8
#C_r = (R-Y)/1.4
Y_pixels = np.zeros((height, width))
C_b_pixels = np.zeros((height, width))
C_r_pixels = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        Y_pixels[i,j] = 0.5 * image[i,j,0] + 0.6 * image[i, j, 1] + 0.1 * image[i, j, 2]
        C_b_pixels[i, j] = (image[i, j, 1] - Y_pixels[i,j]) / 1.8
        C_r_pixels[i, j] = (image[i, j, 0] - Y_pixels[i,j]) / 1.4
        
        #color adaptation image
color_adapt_Y = (np.ones((height,width)) * 0.5)
color_adapt_C_b = -C_b_pixels
color_adapt_C_r = -C_r_pixels
# Clip values to ensure they are within valid range
color_adapt_C_b = np.clip(color_adapt_C_b, -0.5, 0.5)
color_adapt_C_r = np.clip(color_adapt_C_r, -0.5, 0.5)


# doing some calculations
# R = Y + 1.4 * C_r
# B = Y + 1.8 * C_b
# G = (Y - 0.3 * R - 0.1 * B) / 0.6

adapt_R = color_adapt_Y + 1.4 * color_adapt_C_r
adapt_B = color_adapt_Y + 1.8*color_adapt_C_b
adapt_G = (color_adapt_Y - 0.3 * adapt_R - 0.1 * adapt_B)/0.6


# Merge YCbCr components
adaptation_image_array = np.stack((adapt_R, adapt_G, adapt_B), axis=-1)
write_image("q6-adaptation.png", adaptation_image_array)

Y_pixels = np.clip(Y_pixels, 0, 1)
gray_C = np.zeros((height, width))

gray_R = Y_pixels + 1.4 * gray_C
gray_B = Y_pixels + 1.8 * gray_C
gray_G = (Y_pixels - 0.3 * gray_R - 0.1 * gray_B)/0.6
grayscale_image_array = np.stack((gray_R, gray_G, gray_B), axis = -1)
                                 
write_image("q6-gray.png", grayscale_image_array)
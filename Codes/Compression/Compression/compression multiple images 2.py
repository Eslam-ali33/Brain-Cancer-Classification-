import numpy as np
import zlib
from PIL import Image
import io
import os

# Define the compression quality (0-100)
compression_quality = 100

# Define a list of image file names to compress
image_files = ['fusion_dg_mr.png', 'fusion_mr_tc.png']

# Loop over each image and compress it
for image_file in image_files:
    # Load the image data
    img = Image.open(image_file)
    img_data = np.array(img)

    # Compress the image using JPEG
    compressed_img = img.convert('RGB').save('compressed_image.jpg', format='JPEG', quality=compression_quality)

    # Load the compressed JPEG image data
    with open('compressed_image.jpg', 'rb') as f:
        lossy_compressed_data = f.read()

    # Compress the lossy JPEG data using DEFLATE
    lossless_compressed_data = zlib.compress(lossy_compressed_data)

    # Save the hybrid compressed image data to a file
    compressed_file = os.path.splitext(image_file)[0] + '_compressed.hybrid'
    with open(compressed_file, 'wb') as f:
        # Write the DEFLATE compressed data
        f.write(lossless_compressed_data)

    # Load the hybrid compressed image data from the file
    with open(compressed_file, 'rb') as f:
        # Read the DEFLATE compressed data
        lossless_compressed_data = f.read()

    # Decompress the lossy JPEG data from the hybrid compressed data
    lossy_decompressed_data = zlib.decompress(lossless_compressed_data)

    # Create a new image object from the lossy JPEG data
    compressed_img_data = Image.open(io.BytesIO(lossy_decompressed_data))

    # Save the compressed image data to a file
    compressed_image_file = os.path.splitext(image_file)[0] + '_compressed.jpg'
    compressed_img_data.save(compressed_image_file)

    # Clean up the temporary compressed JPEG file
    os.remove('compressed_image.jpg')
import numpy as np
from PIL import Image
from skimage import color
from skimage.draw import disk
from skimage.transform import resize

def create_dotted_image(input_path, output_path, dot_color, dot_size=100, scale_factor=0.1):
    # Load the image
    image = Image.open(input_path)
    image = image.convert('RGB')
    
    # Resize the image to make the process faster and adjust the dot effect
    image = image.resize((int(image.width * scale_factor), int(image.height * scale_factor)), Image.LANCZOS)
    
    # Convert image to grayscale and then to numpy array
    gray_image = color.rgb2gray(np.array(image))
    
    # Prepare canvas with a black background
    canvas = np.zeros((gray_image.shape[0] * dot_size, gray_image.shape[1] * dot_size, 3))
    
    # Define a fixed radius for uniform dot size
    fixed_radius = dot_size // 2  # You can adjust this value to make the dots larger or smaller

    # Place dots based on the grayscale intensity, modifying the color's brightness
    for y in range(gray_image.shape[0]):
        for x in range(gray_image.shape[1]):
            intensity = gray_image[y, x]
            shade = (intensity) * np.array(dot_color)  # Darker for brighter areas
            rr, cc = disk((y * dot_size + dot_size // 2, x * dot_size + dot_size // 2), fixed_radius)
            rr = np.clip(rr, 0, canvas.shape[0] - 1)
            cc = np.clip(cc, 0, canvas.shape[1] - 1)
            canvas[rr, cc] = shade / 255.0  # Apply color shade
    
    # Convert array to image and save
    output_image = Image.fromarray((canvas * 255).astype(np.uint8))
    output_image.save(output_path)

# # Example usage
# input_image_path = 'your_input_image_path_here.jpg'
# output_image_path = 'your_output_image_path_here.jpg'
# main_color = [255, 0, 0]  # Red in RGB
# create_dotted_image(input_image_path, output_image_path, main_color)

# # Example usage
input_image_path = 'yoder.png'
# input_image_path = 'BernieFinancial.png'
# output image will be saved as input image name with '_dotted' suffix and same filetype extension
# get the input image path up to the last dot
output_image_path = input_image_path[:input_image_path.rfind('.')] + '_dotted' + input_image_path[input_image_path.rfind('.'):]
# output_image_path = input_image_path + '_dotted' + input_image_path[-4:]
main_color = [255, 0, 0]  # Red in RGB
create_dotted_image(input_image_path, output_image_path, main_color)


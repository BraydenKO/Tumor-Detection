from PIL import Image
from helper import data_dir, get_image_paths 
import os

def convert_files(images_paths, image_format):
  for path in images_paths:
    image = Image.open(str(path))
    image = image.convert('RGB')
    image.save(str(path)[:str(path).index(".")+1] + image_format)
    if str(path) != str(path)[:str(path).index(".")+1] + image_format:
      os.remove(str(path))

def convert_all_images():
  images_paths = get_image_paths(data_dir)
  convert_files(images_paths[0], "png")
  convert_files(images_paths[1], "png")

if __name__ == "__main__":
  convert_all_images()
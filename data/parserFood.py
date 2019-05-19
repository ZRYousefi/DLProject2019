import os
import fnmatch


def get_image_paths(source, extension='jpg'):
    images_path, class_names = [], []
    for root, dirnames, filenames in os.walk(source):            
        filenames = [filename for filename in filenames]
        for filename in fnmatch.filter(filenames, '*.'+extension):
            images_path.append(os.path.join(root, filename))
            class_name = root.split('/')
            class_name = class_name[len(class_name)-1:]
            class_name = ''.join(class_name)
            class_names.append(class_name)
    return class_names, images_path

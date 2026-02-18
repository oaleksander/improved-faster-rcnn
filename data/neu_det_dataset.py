import os
import xml.etree.ElementTree as ET

import numpy as np

from .util import read_image


class NeuDetDataset:

    def __init__(self, data_dir, split, use_difficult=False):

        id_list_file = os.path.join(data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.label_names = NEU_DET_LABEL_NAMES

    def __len__(self):
        return len(self.ids)
    
    def get_sample(self, i):
        id_ = self.ids[i]
        # parse annotation
        anno = ET.parse(os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = []
        label = []
        difficult = []
        for obj in anno.findall('object'):


            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([int(bndbox_anno.find(tag).text) - 1 for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(NEU_DET_LABEL_NAMES.index(name))

        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

        # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_file, color=True)

        return img, bbox, label, difficult
    
    __getitem__= get_sample

NEU_DET_LABEL_NAMES = (
    'crazing',
    'rolled-in_scale',
    'inclusion',
    'patches',
    'scratches',
    'pitted_surface')
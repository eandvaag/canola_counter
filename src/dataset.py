
import os
import cv2


class Dataset(object):

	def __init__(self, img_paths, name):

		self.imgs = []
		for img_path in img_paths:
			self.imgs.append(Img(img_path))

		self.name = name



class Img(object):

	def __init__(self, img_path):

		self.img_path = img_path
		self.xml_path = img_path[:-3] + "xml"
		self.is_annotated = os.path.exists(self.xml_path)


	def load_img_array(self):
		return cv2.imread(self.img_path)
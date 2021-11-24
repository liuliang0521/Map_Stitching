import numpy as np
import cv2
import sys
from matchers import matchers
import time

class Stitch:
	def __init__(self, args):
		filenames = args
		print(filenames)
		self.images = [cv2.resize(cv2.imread(each),(480, 320)) for each in filenames]
		self.count = len(self.images)
		self.left_list, self.right_list, self.center_im = [], [],None
		self.matcher_obj = matchers()
		self.prepare_lists()

	def prepare_lists(self):
		print("一共需要拼接图片的数量为：%d"%self.count)
		self.centerIdx = self.count/2 
		print("中间下标的图片为：%d"%self.centerIdx)
		self.center_im = self.images[int(self.centerIdx)]
		for i in range(self.count):
			if(i<=self.centerIdx):
				self.left_list.append(self.images[i])
			else:
				self.right_list.append(self.images[i])
		print("图片已经拓扑排序完成")

	def leftshift(self):
		# self.left_list = reversed(self.left_list)
		a = self.left_list[0]
		for b in self.left_list[1:]:
			H = self.matcher_obj.match(a, b, 'left')
			print("单应性：",H)
			xh = np.linalg.inv(H)
			print("逆单应性：",xh)
			ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]));
			ds = ds / ds[-1]
			f1 = np.dot(xh, np.array([0, 0, 1]))
			f1 = f1 / f1[-1]
			xh[0][-1] += abs(f1[0])
			xh[1][-1] += abs(f1[1])
			ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
			offsety = abs(int(f1[1]))
			offsetx = abs(int(f1[0]))
			dsize = (int(ds[0]) + offsetx, int(ds[1]) + offsety)
			tmp = cv2.warpPerspective(a, xh, dsize)
			tmp[offsety:b.shape[0] + offsety, offsetx:b.shape[1] + offsetx] = b
			a = tmp
		self.leftImage = tmp

		
	def rightshift(self):
		for each in self.right_list:
			H = self.matcher_obj.match(self.leftImage, each, 'right')
			print("单应性：", H)
			txyz = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
			txyz = txyz/txyz[-1]
			dsize = (int(txyz[0])+self.leftImage.shape[1], int(txyz[1])+self.leftImage.shape[0])
			tmp = cv2.warpPerspective(each, H, dsize)
			cv2.imshow("tp", tmp)
			cv2.waitKey()
			tmp = self.mix_and_match(self.leftImage, tmp)
			print("tmp的shape",tmp.shape)
			print("左图的shape=", self.leftImage.shape)
			self.leftImage = tmp



	def mix_and_match(self, leftImage, warpedImage):
		i1y, i1x = leftImage.shape[:2]
		i2y, i2x = warpedImage.shape[:2]
		print(leftImage[-1,-1])

		t = time.time()
		black_l = np.where(leftImage == np.array([0,0,0]))
		black_wi = np.where(warpedImage == np.array([0,0,0]))
		print(time.time() - t)
		print(black_l[-1])
		for i in range(0, i1x):
			for j in range(0, i1y):
				try:
					if(np.array_equal(leftImage[j,i],np.array([0,0,0])) and  np.array_equal(warpedImage[j,i],np.array([0,0,0]))):
						warpedImage[j,i] = [0, 0, 0]
					else:
						if(np.array_equal(warpedImage[j,i],[0,0,0])):
							warpedImage[j,i] = leftImage[j,i]
						else:
							if not np.array_equal(leftImage[j,i], [0,0,0]):
								bw, gw, rw = warpedImage[j,i]
								bl,gl,rl = leftImage[j,i]
								warpedImage[j, i] = [bl,gl,rl]
				except:
					pass
		return warpedImage

	def trim_left(self):
		pass
	def showImage(self, string=None):
		if string == 'left':
			cv2.imshow("左图", self.leftImage)
		elif string == "right":
			cv2.imshow("右图", self.rightImage)
		cv2.waitKey()


if __name__ == '__main__':
	path = ["../images/pic1.jpg","../images/pic2.jpg"]
	s = Stitch(path)
	s.leftshift()
	s.rightshift()
	print("图像合成完成")
	cv2.imwrite("../images/answer.jpg", s.leftImage)
	print("图片成功保存")
	cv2.destroyAllWindows()
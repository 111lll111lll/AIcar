# 模型推理文件
import os
import cv2
from detect_xiugai import detect

class Inference(object):
	def __init__(self, model_name, model_path):
		self.model_name = model_name
		self.model_path = model_path
		#self.model = Model(self.model_path)
	
	def _preprocess(self, data):
		# 输入为图片文件夹
		preprocessed_data = {}
		for file_path in os.listdir(data):
			preprocessed_data[file_path] = cv2.imread(os.path.join(data, file_path))
		return preprocessed_data
	
	def _inference(self, data):
		# 输入为_preprocess的输出
		infered_data = {}
		for k, v in data.items():
			#print(k)
			infered_data[k] = detect(v)
		#print(infered_data)
		return infered_data
	
	def _postprocess(self, data):
		# 输入为_inference的输出
		postprocessed_data = []
		#print(data)

		for k in data:
			#print(len(data[k]))
			predict_per_img = {"detection_classes": [], "detection_boxes": [], "detection_scores": []}

			# “detection_classes”指图片中目标的类别,示例中类别分别为人行横道、红灯
			# “detection_boxes”指图片中目标位置的水平矩形框坐标，坐标顺序依次是ymin、xmin、ymax、 xmax
			# “detection_scores”指检测结果的置信度

			#print('----------------')
			#print(result[0])

			for i in range(0, len(data[k])):
				predicted_class = data[k][i]['class']
				score = data[k][i]['conf']
				labels_n0 = data[k][i]['position']
				#print(labels_n0)
				labels_n1 = [labels_n0[1],labels_n0[0],labels_n0[3],labels_n0[2]]
				#labels_n1 = [labels_n0[1]-labels_n0[3],labels_n0[0],labels_n0[1],labels_n0[0]+labels_n0[2]]
				#print(labels_n1)
				predict_per_img["detection_classes"].append(str(predicted_class))
				predict_per_img["detection_boxes"].append([str(labels_n1[0]), str(labels_n1[1]), str(labels_n1[2]), str(labels_n1[3])])
				predict_per_img["detection_scores"].append(str(float(score)))
			postprocessed_data.append(predict_per_img)
		return postprocessed_data




"""
infer.py附加说明
1，训练数据
训练数据集包含红灯、绿灯、黄灯、人行横道、限速标志、解除限速标志六种类型图片，需使用ModelArts数据管理模块完成以上六种检测目标的标注，标签按照如下规则命名：
红灯：red_stop
绿灯：green_go
黄灯：yellow_back
人行横道：pedestrian_crossing
限速标志：speed_limited
解除限速标志：speed_unlimited
"""
"""
data=r'D:\university\2021-2022大二下\人工智能导论\大作业\yolov5-master\picture'
server = Inference('', '')
data = server._preprocess(data)
result = server._inference(data)
result = server._postprocess(result)
print(result)
"""


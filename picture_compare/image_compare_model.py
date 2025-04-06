# import time
# import cv2
# import numpy as np
# from transformers import CLIPProcessor, CLIPModel
# import torch
# from PIL import Image
# from transformers.utils import logging as hf_logging
# import torch.quantization
#
# hf_logging.set_verbosity_error()
#
#
# class ImageComparer:
#     def __init__(self, model_path="D:/test_case/ai"):
#         """初始化模型和处理器（仅加载一次）"""
#         self.model_path = model_path
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model = self._load_model()
#         self.processor = self._load_processor()
#
#     def _load_model(self):
#         """加载并量化模型（移动到指定设备）"""
#         model = CLIPModel.from_pretrained(self.model_path, local_files_only=True)
#         # 动态量化（可选，根据需求开启/关闭）
#         # model = torch.quantization.quantize_dynamic(
#         #     model, {torch.nn.Linear}, dtype=torch.qint8
#         # )
#         return model.to(self.device)
#
#     def _load_processor(self):
#         """加载处理器"""
#         return CLIPProcessor.from_pretrained(self.model_path, local_files_only=True)
#
#     def detect_abnormalities(self, image_path):
#         """检测图像异常（黑屏/颜色异常）"""
#         img = cv2.imread(image_path)
#         if img is None:
#             return True  # 读取失败视为异常
#
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
#         brightness = np.mean(img_rgb)
#         if brightness < 0.1:  # 黑屏检测
#             return True
#
#         r_mean, g_mean, b_mean = np.mean(img_rgb, axis=(0, 1))
#         if abs(max(r_mean, g_mean, b_mean) - min(r_mean, g_mean, b_mean)) > 0.5:  # 颜色异常检测
#             return True
#
#         return False  # 正常
#
#     def get_image_features(self, image_path):
#         """提取图像特征（支持本地路径）"""
#         image = Image.open(image_path)
#         inputs = self.processor(images=image, return_tensors="pt", padding=True).to(self.device)
#         with torch.no_grad():
#             features = self.model.get_image_features(**inputs)
#         return features[0].cpu().numpy()  # 返回CPU上的numpy数组
#
#     def cosine_similarity(self, feature_a, feature_b):
#         """计算余弦相似度"""
#         a_tensor = torch.tensor(feature_a)
#         b_tensor = torch.tensor(feature_b)
#         return (a_tensor @ b_tensor) / (torch.norm(a_tensor) * torch.norm(b_tensor))
#
#     def compare_images(self, image_path1, image_path2, threshold=0.85):
#         """
#         比较两张图像的相似度（主接口）
#         返回：(是否相同, 相似度值, 状态, 运行时间)
#         """
#         start_time = time.perf_counter()  # 高精度计时开始
#
#         # 异常检测
#         if self.detect_abnormalities(image_path1) or self.detect_abnormalities(image_path2):
#             end_time = time.perf_counter()
#             return (False, 0.0, "图像异常", end_time - start_time)
#
#         # 特征提取
#         feature1 = self.get_image_features(image_path1)
#         feature2 = self.get_image_features(image_path2)
#
#         # 相似度计算
#         similarity = self.cosine_similarity(feature1, feature2).item()
#         is_same = similarity > threshold
#
#         end_time = time.perf_counter()  # 计时结束
#         run_time = end_time - start_time  # 计算运行时间
#
#         return (is_same, similarity, "正常", run_time)
#
#
# # --------------------- 示例运行 ---------------------
# if __name__ == "__main__":
#     comparer = ImageComparer()
#     image1 = "11.jpg"
#     image2 = "22.jpg"
#
#     is_same, similarity, status, run_time = comparer.compare_images(image1, image2)
#
#     print(f"检测状态：{status}")
#     print(f"特征相似度：{similarity:.4f}")
#     print(f"图像是否相同：{'是' if is_same else '否'}")
#     print(f"运行时间：{run_time:.4f}秒 ({run_time * 1000:.1f}毫秒)")



import time
import cv2
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
from transformers.utils import logging as hf_logging
import torch.quantization

hf_logging.set_verbosity_error()


class ImageCompare:
    def __init__(self, model_path=r"E:\ATRI(1)\ai_picture"):
        """初始化模型和处理器（仅加载一次）"""
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model()
        self.processor = self._load_processor()

    def _load_model(self):
        """加载并量化模型（移动到指定设备）"""
        model = CLIPModel.from_pretrained(self.model_path, local_files_only=True)
        # 动态量化（可选，根据需求开启/关闭）
        # model = torch.quantization.quantize_dynamic(
        #     model, {torch.nn.Linear}, dtype=torch.qint8
        # )
        return model.to(self.device)

    def _load_processor(self):
        """加载处理器"""
        return CLIPProcessor.from_pretrained(self.model_path, local_files_only=True)

    def get_image_features(self, image_path):
        """提取图像特征（支持本地路径）"""
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features[0].cpu().numpy()  # 返回CPU上的numpy数组

    def cosine_similarity(self, feature_a, feature_b):
        """计算余弦相似度"""
        a_tensor = torch.tensor(feature_a)
        b_tensor = torch.tensor(feature_b)
        return (a_tensor @ b_tensor) / (torch.norm(a_tensor) * torch.norm(b_tensor))

    def compare_images(self, image_path1, image_path2, threshold=0.85):
        """
        比较两张图像的相似度（主接口）
        返回：(比较结果, 相似度值, 运行时间)
        """
        start_time = time.perf_counter()  # 高精度计时开始

        # 特征提取
        feature1 = self.get_image_features(image_path1)
        feature2 = self.get_image_features(image_path2)

        # 相似度计算
        similarity = self.cosine_similarity(feature1, feature2).item()
        result = "一致" if similarity > threshold else "不一致"

        end_time = time.perf_counter()  # 计时结束
        run_time = end_time - start_time  # 计算运行时间

        return result, similarity, run_time


# --------------------- 示例运行 ---------------------
if __name__ == "__main__":
    comparer = ImageCompare()
    # image1 = r"./all_image/11.jpg"
    image1 = r"E:\ATRI(1)\picture_compare\all_image\binoculus\real_left_test_real.jpg"
    # image2 = r"./all_image/12.jpg"
    image2 = r"E:\ATRI(1)\picture_compare\all_image\binoculus\render_right_test_render.jpg"

    result, similarity, run_time = comparer.compare_images(image1, image2)

    print(f"比较结果：{result}")
    print(f"相似度：{similarity:.4f}")
    print(f"运行时间：{run_time:.4f}秒 ({run_time * 1000:.1f}毫秒)")
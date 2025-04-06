import os
import subprocess
import time
from datetime import datetime
from typing import Tuple, Optional, Dict
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

class ImageProcessor:
    @staticmethod
    def cut_double_render(image_path: str, output_folder: str) -> Tuple[bool, str, str]:
        """处理render图像，分割为左右两部分"""
        try:
            img = Image.open(image_path)
            rotated_img = img.rotate(180)
            width, height = rotated_img.size
            
            left_half = rotated_img.crop((0, 0, width // 2, height))
            right_half = rotated_img.crop((width // 2, 0, width, height))
            
            os.makedirs(output_folder, exist_ok=True)
            base_name = os.path.basename(image_path)
            
            left_path = os.path.join(output_folder, f"render_left_{base_name}")
            right_path = os.path.join(output_folder, f"render_right_{base_name}")
            
            left_half.save(left_path)
            right_half.save(right_path)
            
            return True, left_path, right_path
        except Exception as e:
            print(f"Render图像处理失败: {str(e)}")
            return False, "", ""

    @staticmethod
    def process_real_image(input_path: str, output_path: str) -> Tuple[bool, str]:
        """处理real图像，裁剪屏幕内容"""
        try:
            img = cv2.imread(input_path)
            if img is None:
                return False, ""
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return False, ""
                
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            cropped = img[y:y+h, x:x+w]
            cv2.imwrite(output_path, cropped)
            return True, output_path
        except Exception as e:
            print(f"Real图像处理失败: {str(e)}")
            return False, ""

class ImageCompare:
    def __init__(self, model_path=r"E:\ATRI(1)\ai_picture"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model()
        self.processor = self._load_processor()

    def _load_model(self):
        model = CLIPModel.from_pretrained(self.model_path, local_files_only=True)
        return model.to(self.device)

    def _load_processor(self):
        return CLIPProcessor.from_pretrained(self.model_path, local_files_only=True)

    def compare_images(self, image_path1, image_path2, threshold=0.85):
        """比较两张图像的相似度"""
        try:
            image1 = Image.open(image_path1)
            image2 = Image.open(image_path2)
            
            inputs1 = self.processor(images=image1, return_tensors="pt").to(self.device)
            inputs2 = self.processor(images=image2, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                features1 = self.model.get_image_features(**inputs1)
                features2 = self.model.get_image_features(**inputs2)
            
            similarity = torch.cosine_similarity(features1, features2).item()
            result = "一致" if similarity > threshold else "不一致"
            return result, similarity
        except Exception as e:
            print(f"图像比对失败: {str(e)}")
            return "错误", 0.0

def main():
    # 设置执行时长(1分钟=60秒)
    duration = 60
    start_time = time.time()
    
    while time.time() - start_time < duration:
        try:
            loop_start = time.time()
            
            # 1. 截图模块
            print(f"开始截图 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 2. 图像处理
            render_success, render_left, render_right = ImageProcessor.cut_double_render(
                r"E:\ATRI(1)\PythonProject\test_render.jpg",
                r"E:\ATRI(1)\picture_compare\all_image\binoculus"
            )
            
            real_success, real_left, real_right = ImageProcessor.process_real_image(
                r"E:\ATRI(1)\PythonProject\test_real.jpg",
                r"E:\ATRI(1)\picture_compare\all_image\binoculus"
            )
            
            # 3. 比对模块
            comparer = ImageCompare()
            left_result, left_sim = comparer.compare_images(real_left, render_left)
            right_result, right_sim = comparer.compare_images(real_right, render_right)
            
            print(f"左眼比对结果: {left_result} (相似度: {left_sim:.4f})")
            print(f"右眼比对结果: {right_result} (相似度: {right_sim:.4f})")
            
            # 4. 结果处理
            loop_time = time.time() - loop_start
            print(f"当前轮次耗时: {loop_time:.2f}秒")
            
        except Exception as e:
            print(f"主程序发生错误: {str(e)}")
            time.sleep(5)

if __name__ == "__main__":
    main()
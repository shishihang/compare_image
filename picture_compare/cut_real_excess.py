import cv2
import numpy as np
from PIL import Image
from typing import Tuple
import os 
from datetime import datetime 

def process_screen_image(input_path: str, output_cropped_path: str) -> Tuple[bool, str]:
    """
    处理摄像头拍摄的小屏幕图像，裁剪屏幕内容。
    :param input_path: 输入图像路径
    :param output_cropped_path: 裁剪后的图像保存路径
    :return: (是否成功, 输出文件名)
    """
    try:
        # 读取图像
        img = cv2.imread(input_path)
        if img is None:
            print("无法读取图像，请检查路径。")
            return False, ""
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # 高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
        # OTSU 二值化
        # 合并连续操作
        thresh = cv2.threshold(
            cv2.GaussianBlur(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 
                (9, 9), 0
            ), 
            0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
        
        # 直接使用numpy计算
        img_center_x, img_center_y = np.array(img.shape[1::-1]) // 2
    
        # 加强形态学操作：膨胀和腐蚀，合并分裂的轮廓
        kernel = np.ones((7, 7), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=3)
        thresh = cv2.erode(thresh, kernel, iterations=2)
        
        # 保存二值化结果以供调试
        debug_path = r"E:\ATRI(1)\picture_compare\all_image\binoculus\thresh_debug.jpg"
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        cv2.imwrite(debug_path, thresh)
        
        # print("二值化结果已保存至 thresh_debug.jpg。")
    
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        if not contours:
            print("未检测到任何轮廓，请调整预处理参数。")
            return
    
        # 计算图像总面积和中心
        img_area = img.shape[0] * img.shape[1]
        img_center_x, img_center_y = img.shape[1] // 2, img.shape[0] // 2
    
        # 选择面积最大的轮廓
        best_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best_contour)
        perimeter = cv2.arcLength(best_contour, True)
        if perimeter == 0:
            print("选中的轮廓无效（周长为 0），请调整预处理参数。")
            return
    
        # 计算轮廓的面积占比、圆形度和宽高比
        area_ratio = area / img_area
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        x, y, w, h = cv2.boundingRect(best_contour)
        aspect_ratio = w / h if h != 0 else 0
        # 计算轮廓中心到图像中心的距离
        contour_center_x, contour_center_y = x + w // 2, y + h // 2
        distance_to_center = ((contour_center_x - img_center_x) ** 2 + (contour_center_y - img_center_y) ** 2) ** 0.5
    
        # # 打印轮廓信息
        # print(
        #     f"选择面积最大的轮廓：面积占比={area_ratio:.4f}, 圆形度={circularity:.4f}, 宽高比={aspect_ratio:.4f}, 距中心距离={distance_to_center:.2f}")
    
        # 验证条件：宽高比接近 1，距离中心不超过图像宽度的 1/2
        if not (0.5 < aspect_ratio < 2.0 and distance_to_center < img.shape[1] // 2):
            print("选中的轮廓可能不是屏幕区域（宽高比或位置不符合预期），请检查二值化结果或调整预处理参数。")
            # 仍继续处理，但提示用户可能有问题
            print("继续处理，但结果可能不准确。")
    
        # 找到最左和最右点
        # 使用numpy一次性计算最左最右点
        expand_pixel = 5  # 先定义expand_pixel变量
        x_coords = best_contour[:,:,0]
        leftmost_x, rightmost_x = np.min(x_coords), np.max(x_coords)
        leftmost = (max(0, leftmost_x - expand_pixel), np.mean(best_contour[x_coords.argmin()][0][1]))
        rightmost = (min(img.shape[1], rightmost_x + expand_pixel), np.mean(best_contour[x_coords.argmax()][0][1]))
    
        # 移除下面重复的expand_pixel定义和使用
        # expand_pixel = 5
        # leftmost = (max(0, leftmost[0] - expand_pixel), leftmost[1])
        # rightmost = (min(img.shape[1], rightmost[0] + expand_pixel), rightmost[1])
        width = rightmost[0] - leftmost[0]
    
        # 计算轮廓垂直方向中点
        y_coords = best_contour[:, :, 1]
        center_y = int(np.mean(y_coords))
    
        # 按宽度比例确定高度（可调整比例，如1.1倍）
        height_ratio = 1.1
        height = int(width * height_ratio)
        top = max(0, center_y - height // 2)
        bottom = min(img.shape[0], center_y + height // 2)
    
        # # 绘制矩形框（调试用）
        # framed_img = img.copy()
        # cv2.rectangle(framed_img, (leftmost[0], top), (rightmost[0], bottom), (0, 255, 0), 2)
        # cv2.imwrite(output_framed_path, framed_img)
        # print(f"框选结果已保存至 {output_framed_path}")
    
        # 使用 Pillow 裁剪图像
        pil_img = Image.open(input_path)
        cropped_img = pil_img.crop((leftmost[0], top, rightmost[0], bottom))
        # 修改返回路径为完整路径
        cropped_img.save(output_cropped_path)
        # print(f"裁剪结果已保存至 {os.path.abspath(output_cropped_path)}")
        return True, os.path.abspath(output_cropped_path)
        
    except Exception as e:
        print(f"处理失败: {str(e)}")
        return False, ""






def cut_double_real(input_path: str, output_folder: str) -> Tuple[bool, str, str]:
    """
    将输入图像分割为左右两部分并分别裁剪
    :param input_path: 输入图像路径
    :param output_folder: 输出文件夹路径
    :return: (是否成功, 左眼路径, 右眼路径)
    """
    try:
        # 确保输出目录存在
        os.makedirs(output_folder, exist_ok=True)
        
        # 1. 分割图像为左右两部分
        img = Image.open(input_path)
        width, height = img.size
        
        # 获取原始文件名并生成完整输出路径
        base_name = os.path.basename(input_path)
        left_path = os.path.abspath(os.path.join(output_folder, f"real_left_{base_name}"))
        right_path = os.path.abspath(os.path.join(output_folder, f"real_right_{base_name}"))
        
        # 保存分割后的图片
        img.crop((0, 0, width//2, height)).save(left_path)
        img.crop((width//2, 0, width, height)).save(right_path)
        
        # 2. 分别裁剪左右图像
        success_left, left_cropped = process_screen_image(left_path, left_path)
        success_right, right_cropped = process_screen_image(right_path, right_path)
        
        if success_left and success_right:
            return True, left_cropped, right_cropped
        return False, "", ""
        
    except Exception as e:
        print(f"分割并裁剪图像失败: {str(e)}")
        return False, "", ""


# 单张real图片的剪裁
if __name__ == "__main11__":
    input_path = r"./all_image/13.jpg"
    output_cropped_path = r"./all_image/cropped_output.jpg"
    success, filename = process_screen_image(input_path, output_cropped_path)
    if success:
        print(f"处理成功，生成文件: {filename}")
    else:
        print("处理失败")




# 一张整的先分割两份，分别剪裁
if __name__ == "__main__":
    input_path = r"./all_image/103.jpg"
    output_folder = r"./all_image/binoculus"
    success, left_path, right_path = cut_double_real(input_path, output_folder)
    if success:
        print(f"处理成功，生成文件:\n左眼: {left_path}\n右眼: {right_path}")
    else:
        print("处理失败")

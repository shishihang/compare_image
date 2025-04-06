from PIL import Image
import os
from typing import Tuple, Optional


def cut_double_render(image_path: str, output_folder: str) -> Tuple[bool, str, str]:
    # 保持原样，方法名与文件名完全一致
    try:
        # 打开并旋转图片
        img = Image.open(image_path)
        rotated_img = img.rotate(180)
        width, height = rotated_img.size

        # 分割图片
        left_half = rotated_img.crop((0, 0, width // 2, height))
        right_half = rotated_img.crop((width // 2, 0, width, height))

        # 确保输出文件夹存在
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 获取原文件名
        base_name = os.path.basename(image_path)

        # 生成保存路径(添加render_left和render_right前缀)
        left_path = os.path.join(output_folder, f"render_left_{base_name}")
        right_path = os.path.join(output_folder, f"render_right_{base_name}")

        # 保存分割后的图片
        left_half.save(left_path)
        right_half.save(right_path)
        
        return True, left_path, right_path

    except Exception as e:
        print(f"处理失败: {str(e)}")
        return False, None, None


if __name__ == "__main__":
    input_path = r"E:\ATRI(1)\picture_compare\all_image\img_v3_02l0_406db17c-4864-4998-8f4f-75e39872167g-right.jpg"
    output_folder = r"E:\ATRI(1)\picture_compare\all_image\binoculus"
    success, left_path, right_path = cut_double_render(input_path, output_folder)
    if success:
        print(f"处理成功，生成文件:\n左眼: {left_path}\n右眼: {right_path}")
    else:
        print("处理失败")
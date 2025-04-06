import os
import subprocess
import time
from datetime import datetime
from typing import Tuple, Optional

def main():
    # 设置截图间隔(秒)
    capture_interval = 60
    
    while True:
        try:
            # 1. 截图模块
            print(f"开始截图 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Render截图(需要获取图片路径)
            render_success, render_image_path = capture_render_image()
            if not render_success:
                print("Render截图命令发送失败")
                continue
                
            # 处理Render图像(新增步骤)
            render_success, render_left, render_right = cut_double_render(render_image_path)
            if not render_success:
                print("Render图像处理失败")
                continue
                
            # Real截图(只需获取单张图像)
            real_success, real_image_path = capture_real_image()
            if not real_success:
                print("Real截图失败，等待重试...")
                time.sleep(5)
                real_success, real_image_path = capture_real_image()
                if not real_success:
                    print("Real截图重试失败，程序终止")
                    return False
            
            # 处理Real图像(直接传入单张图像路径)
            real_success, real_left, real_right = cut_real_excess(real_image_path)
            if not real_success:
                print("Real图像处理失败")
                continue
            
            # 3. 比对模块
            from image_compare_model import ImageCompare
            comparer = ImageCompare(model_path=r"E:\ATRI(1)\ai_picture")
            
            # 左眼比对
            print(f"\n开始比对:\nA图(左眼Real): {real_left}\nB图(左眼Render): {render_left}")
            left_result, left_similarity, _ = comparer.compare_images(
                real_left,
                render_left
            )
            
            # 右眼比对
            print(f"\n开始比对:\nA图(右眼Real): {real_right}\nB图(右眼Render): {render_right}")
            right_result, right_similarity, _ = comparer.compare_images(
                real_right,
                render_right
            )
            
            # 4. 结果处理
            if left_result == "一致" and right_result == "一致":
                print("双眼比对全部通过")
                return True
            else:
                failed_files = {
                    "real_left": real_left if left_result != "一致" else None,
                    "render_left": render_left if left_result != "一致" else None,
                    "real_right": real_right if right_result != "一致" else None,
                    "render_right": render_right if right_result != "一致" else None
                }
                print(f"比对失败，失败文件: {failed_files}")
                return False
                
        except Exception as e:
            print(f"主程序发生错误: {str(e)}")
            time.sleep(5)  # 等待5秒后重试
            continue


def capture_render_image() -> Tuple[bool, str]:
    """发送render截图命令并返回最新图片路径"""
    try:
        # 1. 发送截图命令
        subprocess.run(["render_screenshot_cmd"], check=True, timeout=10)
        
        # 2. 获取最新生成的jpg文件
        render_dir = r"E:\ATRI(1)\picture_compare\all_image"
        # 只获取.jpg文件并按修改时间排序
        jpg_files = [
            f for f in os.listdir(render_dir) 
            if f.lower().endswith('.jpg')
        ]
        if not jpg_files:
            return (False, "")
            
        # 获取修改时间最新的文件
        latest_file = max(
            jpg_files,
            key=lambda f: os.path.getmtime(os.path.join(render_dir, f))
        )
        return (True, os.path.join(render_dir, latest_file))
        
    except Exception as e:
        print(f"Render截图命令发送失败: {str(e)}")
        return (False, "")

def capture_real_image() -> Tuple[bool, str]:
    """使用USB摄像头截图，返回单张图像"""
    timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S-%f")[:-3]
    output_path = os.path.join("all_image", f"real_{timestamp}.jpg")
    
    try:
        # 发送摄像头截图命令
        subprocess.run(["cam_capture_cmd", "-o", output_path], check=True, timeout=15)
        return (True, output_path) if os.path.exists(output_path) else (False, "")
    except Exception as e:
        print(f"Real截图失败: {str(e)}")
        return (False, "")

def cut_double_render(image_path: str) -> Tuple[bool, str, str]:
    """处理render图像，分割为左右两部分"""
    from cut_double_render import cut_double_render as render_processor
    try:
        success, left, right = render_processor(image_path, "all_image")
        return success, left, right
    except Exception as e:
        print(f"Render图像处理失败: {str(e)}")
        return False, "", ""

def cut_real_excess(image_path: str) -> Tuple[bool, str, str]:
    """处理real图像，分割为两部分并修剪"""
    from cut_real_excess import cut_double_real as real_processor
    try:
        # 现在只需要传入单张图像路径
        success, left_processed, right_processed = real_processor(image_path, "all_image")
        return success, left_processed, right_processed
    except Exception as e:
        print(f"Real图像处理失败: {str(e)}")
        return False, "", ""

if __name__ == "__main__":
    main()
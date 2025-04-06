import os
import subprocess
import time
from datetime import datetime
from typing import Tuple, Optional

def main():
    # 设置执行时长(1分钟=60秒)
    duration = 60  # 1分钟
    start_time = time.time()
    
    while time.time() - start_time < duration:
        try:
            loop_start = time.time()  # 记录单次循环开始时间
            
            # 1. 截图模块
            print(f"开始截图 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Render截图(需要获取图片路径)
            render_success, render_image_path = capture_render_image()
            if not render_success:
                print("Render截图命令发送失败")
                continue
            # print(f"Render截取成功")
            # Real截图(确保这步执行)
            real_success, real_image_path = capture_real_image()
            if not real_success:
                print("Real截图失败")
                continue
            print(f"图像截取成功")
            # 处理Render图像
            render_success, render_left, render_right = cut_double_render(render_image_path)
            if not render_success:
                print("Render图像处理失败")
                continue
            # print(f"Render处理成功")
              
            # 处理Real图像
            real_success, real_left, real_right = cut_real_excess(real_image_path)
            if not real_success:
                print("Real图像处理失败")
                continue
            print(f"图像处理成功")
            
            print(f"开始左右眼对比")
            # 3. 比对模块
            compare_result, failed_files = compare_eyes(real_left, real_right, render_left, render_right)
            if not compare_result:
                return False
                
            # 4. 结果处理
            loop_time = time.time() - loop_start
            print(f"当前轮次耗时: {loop_time:.2f}秒")
            continue
            
        except Exception as e:
            print(f"主程序发生错误: {str(e)}")
            time.sleep(5)  # 等待5秒后重试
            continue
            
            # 计算本次循环耗时
            loop_time = time.time() - loop_start
            print(f"本次循环耗时: {loop_time:.2f}秒")
                
        except Exception as e:
            print(f"主程序发生错误: {str(e)}")
            time.sleep(5)  # 错误后等待5秒再重试
            continue


def capture_render_image() -> Tuple[bool, str]:
    #     """发送render截图命令并返回最新图片路径"""
    # try:
    #     # 1. 发送截图命令
    #     subprocess.run(["render_screenshot_cmd"], check=True, timeout=10)
        
    #     # 2. 获取最新生成的jpg文件
    #     render_dir = r"E:\ATRI(1)\picture_compare\all_image"
    #     # 只获取.jpg文件并按修改时间排序
    #     jpg_files = [
    #         f for f in os.listdir(render_dir) 
    #         if f.lower().endswith('.jpg')
    #     ]
    #     if not jpg_files:
    #         return (False, "")
            
    #     # 获取修改时间最新的文件
    #     latest_file = max(
    #         jpg_files,
    #         key=lambda f: os.path.getmtime(os.path.join(render_dir, f))
    #     )
    #     return (True, os.path.join(render_dir, latest_file))
        
    # except Exception as e:
    #     print(f"Render截图命令发送失败: {str(e)}")
    #     return (False, "")


    """改为直接返回固定Render图片路径"""
    fixed_render_path = r"E:\ATRI(1)\PythonProject\test_render.jpg"
    return (True, fixed_render_path) if os.path.exists(fixed_render_path) else (False, "")

def capture_real_image() -> Tuple[bool, str]:
    # """使用USB摄像头截图，返回单张图像"""
    # timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S-%f")[:-3]
    # output_path = os.path.join("all_image", f"real_{timestamp}.jpg")
    
    # try:
    #     # 发送摄像头截图命令
    #     subprocess.run(["cam_capture_cmd", "-o", output_path], check=True, timeout=15)
    #     return (True, output_path) if os.path.exists(output_path) else (False, "")
    # except Exception as e:
    #     print(f"Real截图失败: {str(e)}")
    #     return (False, "")
    """改为直接返回固定Real图片路径"""
    fixed_real_path = r"E:\ATRI(1)\PythonProject\test_real.jpg"
    return (True, fixed_real_path) if os.path.exists(fixed_real_path) else (False, "")

def cut_double_render(image_path: str) -> Tuple[bool, str, str]:
    """处理render图像，分割为左右两部分"""
    from cut_double_render import cut_double_render as render_processor
    try:
        output_dir = r"E:\ATRI(1)\picture_compare\all_image\binoculus"
        os.makedirs(output_dir, exist_ok=True)
        success, left, right = render_processor(image_path, output_dir)
        return success, os.path.basename(left), os.path.basename(right)
    except Exception as e:
        print(f"Render图像处理失败: {str(e)}")
        return False, "", ""

def cut_real_excess(image_path: str) -> Tuple[bool, str, str]:
    """处理real图像，分割为两部分并修剪"""
    from cut_real_excess import cut_double_real as real_processor
    try:
        output_dir = r"E:\ATRI(1)\picture_compare\all_image\binoculus"
        os.makedirs(output_dir, exist_ok=True)
        success, left_processed, right_processed = real_processor(image_path, output_dir)
        return success, os.path.basename(left_processed), os.path.basename(right_processed)
    except Exception as e:
        print(f"Real图像处理失败: {str(e)}")
        return False, "", ""

def compare_eyes(real_left: str, real_right: str, render_left: str, render_right: str) -> Tuple[bool, dict]:
    """比对左右眼图片"""
    from image_compare_model import ImageCompare
    
    try:
        base_dir = r"E:\ATRI(1)\picture_compare\all_image\binoculus"
        
        # 调试：打印所有输入参数
        # print(f"\n[调试] 输入参数:")
        # print(f"real_left: {real_left}")
        # print(f"real_right: {real_right}") 
        # print(f"render_left: {render_left}")
        # print(f"render_right: {render_right}")

        # 获取完整路径
        def get_full_path(filename):
            full_path = os.path.join(base_dir, filename)
            # print(f"[路径处理] {filename} -> {full_path}")
            return full_path

        # 初始化比较器
        comparer = ImageCompare(model_path=r"E:\ATRI(1)\ai_picture")
        
        # 左眼比对
        left_full = get_full_path(real_left)
        render_left_full = get_full_path(render_left)
        # print(f"\n[左眼比对] 实际路径验证:")
        # print(f"real_left exists: {os.path.exists(left_full)}")
        # print(f"render_left exists: {os.path.exists(render_left_full)}")
        
        left_result, left_sim, _ = comparer.compare_images(left_full, render_left_full)
        print(f"左眼比对结果: {left_result} (相似度: {left_sim:.4f})")
        
        # 右眼比对
        right_full = get_full_path(real_right) 
        render_right_full = get_full_path(render_right)
        # print(f"\n[右眼比对] 实际路径验证:")
        # print(f"real_right exists: {os.path.exists(right_full)}")
        # print(f"render_right exists: {os.path.exists(render_right_full)}")
        
        right_result, right_sim, _ = comparer.compare_images(right_full, render_right_full)
        print(f"右眼比对结果: {right_result} (相似度: {right_sim:.4f})")

        # 统一结果处理
        if left_result == "一致" and right_result == "一致":
            # print("双眼比对全部通过")
            return True, {}
        else:
            failed_files = {
                "real_left": real_left if left_result != "一致" else None,
                "render_left": render_left if left_result != "一致" else None,
                "real_right": real_right if right_result != "一致" else None,
                "render_right": render_right if right_result != "一致" else None
            }
            print(f"比对失败，失败文件: {failed_files}")
            return False, failed_files
            
    except Exception as e:
        print(f"比对过程中发生错误: {str(e)}")
        return False, {}

if __name__ == "__main__":
    main()
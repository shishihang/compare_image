# from image_comparer import ImageComparer
#
# # 初始化比较器
# comparer = ImageComparer(model_path="D:/test_case/ai")
#
# # 获取的图片路径
# image_path_a = "11.jpg"
# image_path_b = "33.jpg"
# # 调用比较接口
#
# # 处理结果
# result, similarity, run_time = comparer.compare_images(image_path_a, image_path_b)
#
# print(f"比较结果：{result}")
# print(f"相似度：{similarity:.4f}")
# print(f"运行时间：{run_time:.4f}秒 ({run_time * 1000:.1f}毫秒)")
from image_compare_model import ImageCompare

# 初始化比较器
comparer = ImageCompare(model_path=r"E:\ATRI(1)\ai_picture")

# 定义多个图片路径对列表，每一个元组代表一对需要比较的图片
image_pairs = [
    (r"../all_image/11.jpg", r"../all_image/11.jpg"),
    (r"../all_image/11.jpg", r"../all_image/22.jpg"),
    (r"../all_image/11.jpg", r"../all_image/33.jpg"),
    (r"../all_image/11.jpg", r"../all_image/44.jpg"),
    (r"../all_image/11.jpg", r"../all_image/103.jpg"),
    (r"../all_image/11.jpg", r"../all_image/103.jpg"),
    (r"../all_image/display-25-04-03-17-01-12-1-14992.jpg", r"../all_image/img_v3_02l0_b6b5820f-5505-42e7-a401-42cd21ad8fdg.jpg"),
    # 你可以根据需要添加更多的图片对
]

# 循环遍历图片对列表
for image_path_a, image_path_b in image_pairs:
    # 调用比较接口
    result, similarity, run_time = comparer.compare_images(image_path_a, image_path_b)

    # 处理并输出每一对图片的比较结果
    print(f"图片 {image_path_a} 和 {image_path_b} 的比较结果：")
    print(f"比较结果：{result}")
    print(f"相似度：{similarity:.4f}")
    print(f"运行时间：{run_time:.4f}秒 ({run_time * 1000:.1f}毫秒)")
    print("-" * 30)

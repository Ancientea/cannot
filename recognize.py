import os

import cv2
import numpy as np
import pytesseract
from PIL import ImageGrab

# 配置Tesseract路径
pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\tesseract.exe'

# 鼠标交互全局变量
drawing = False
roi_box = []

# 预定义相对坐标（基于选定的大区域）
# relative_regions = [
#     (0.334, 0.783, 0.407, 0.911),
#     (0.416, 0.781, 0.490, 0.918),
#     (0.498, 0.785, 0.573, 0.920),
#     (0.661, 0.785, 0.737, 0.916),
#     (0.742, 0.779, 0.818, 0.916),
#     (0.826, 0.783, 0.900, 0.916)
# ]
relative_regions = [
    (0.0, 0.0, 0.131, 1),
    (0.1462, 0.0, 0.2762, 1),
    (0.2923, 0.0, 0.4214, 1),
    (0.5786, 0.0, 0.7087, 1),
    (0.7248, 0.0, 0.8538, 1),
    (0.8679, 0.0, 1, 1)
]


def save_number_image(number, processed, mon_id):
    """保存数字图片到对应文件夹
    Args:
        number: 识别出的数字
        processed: 处理后的图片
        mon_id: 怪物ID
    """
    if number and mon_id != 0:
        # 创建数字对应的文件夹
        num_folder = os.path.join("images", "nums", str(number))
        if not os.path.exists(num_folder):
            os.makedirs(num_folder)

        # 获取文件夹中已有的图片数量
        existing_files = [f for f in os.listdir(num_folder) if f.endswith('.png')]
        next_index = len(existing_files) + 1

        # 保存图片，命名为 id_序号.png
        save_path = os.path.join(num_folder, f"{mon_id}_{next_index}.png")
        cv2.imwrite(save_path, processed)

def mouse_callback(event, x, y, flags, param):
    global roi_box, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_box = [(x, y)]
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_copy = param.copy()
        cv2.rectangle(img_copy, roi_box[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("Select ROI", img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        roi_box.append((x, y))
        drawing = False


def select_roi():
    """改进的交互式区域选择"""
    global roi_box  # 声明为全局变量
    while True:
        # 获取初始截图
        screenshot = np.array(ImageGrab.grab())
        img = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

        # 添加操作提示
        cv2.putText(img, "Drag to select area | ENTER:confirm | ESC:retry",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 添加示例图片
        example_img = cv2.imread("images/eg.png")
        # 显示示例图片在单独的窗口中
        cv2.imshow("example", example_img)

        # 显示窗口
        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select ROI", 1280, 720)
        cv2.setMouseCallback("Select ROI", mouse_callback, img)
        cv2.imshow("Select ROI", img)

        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key == 13 and len(roi_box) == 2:  # Enter确认
            # 标准化坐标 (x1,y1)为左上角，(x2,y2)为右下角
            x1, y1 = min(roi_box[0][0], roi_box[1][0]), min(roi_box[0][1], roi_box[1][1])
            x2, y2 = max(roi_box[0][0], roi_box[1][0]), max(roi_box[0][1], roi_box[1][1])
            return [(x1, y1), (x2, y2)]
        elif key == 27:  # ESC重试
            roi_box = []
            continue


def preprocess(img):
    """优化的预处理流程"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh


def find_best_match(target, ref_images):
    """
    结合灰度匹配和RGB通道匹配，找到最佳匹配的参考图像
    :param target: 目标图像
    :param ref_images: 参考图像字典 {id: image}
    :return: (最佳匹配的id, 最小差异值)
    """
    min_diff = float('inf')
    best_id = -1

    # 确保目标图像是RGB格式
    if len(target.shape) == 2:
        target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)

    # 1. 灰度匹配
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.GaussianBlur(target_gray, (3, 3), 0)
    _, target_gray = cv2.threshold(target_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. RGB通道匹配
    target_r = target[:, :, 0]
    target_g = target[:, :, 1]
    target_b = target[:, :, 2]

    # 对每个通道进行高斯模糊和二值化
    for channel in [target_r, target_g, target_b]:
        channel = cv2.GaussianBlur(channel, (3, 3), 0)
        _, channel = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    for img_id, ref_img in ref_images.items():
        try:
            # 确保参考图像是RGB格式
            if len(ref_img.shape) == 2:
                ref_img = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)

            # 调整参考图像大小以匹配目标图像
            ref_resized = cv2.resize(ref_img, (target.shape[1], target.shape[0]))

            # 1. 灰度匹配
            ref_gray = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2GRAY)
            ref_gray = cv2.GaussianBlur(ref_gray, (3, 3), 0)
            _, ref_gray = cv2.threshold(ref_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            diff_gray = cv2.absdiff(target_gray, ref_gray)
            diff_value_gray = np.sum(diff_gray) / target.size

            # 2. RGB通道匹配
            ref_r = ref_resized[:, :, 0]
            ref_g = ref_resized[:, :, 1]
            ref_b = ref_resized[:, :, 2]

            # 对每个通道进行高斯模糊和二值化
            for channel in [ref_r, ref_g, ref_b]:
                channel = cv2.GaussianBlur(channel, (3, 3), 0)
                _, channel = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # 分别计算RGB三个通道的差异
            diff_r = cv2.absdiff(target_r, ref_r)
            diff_g = cv2.absdiff(target_g, ref_g)
            diff_b = cv2.absdiff(target_b, ref_b)

            # 计算每个通道的差异值
            diff_value_r = np.sum(diff_r) / target.size
            diff_value_g = np.sum(diff_g) / target.size
            diff_value_b = np.sum(diff_b) / target.size

            # 综合所有差异值（灰度匹配权重0.4，RGB通道匹配权重各0.2）
            total_diff = (diff_value_gray * 0.4 +
                          diff_value_r * 0.2 +
                          diff_value_g * 0.2 +
                          diff_value_b * 0.2)

            if total_diff < min_diff:
                min_diff = total_diff
                best_id = img_id

        except Exception as e:
            print(f"处理参考图像 {img_id} 时出错: {str(e)}")
            continue

    return best_id, min_diff


def process_regions(main_roi, ref_images, screenshot=None):
    results = []
    (x1, y1), (x2, y2) = main_roi
    main_width = x2 - x1
    main_height = y2 - y1

    # 如果没有提供screenshot，则获取最新截图（仅截取主区域）
    if screenshot is None:
        screenshot = np.array(ImageGrab.grab(bbox=(x1, y1, x2, y2)))
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    else:
        # 从当前screenshot中提取主区域
        screenshot = screenshot[y1:y2, x1:x2]

    for idx, rel in enumerate(relative_regions):
        try:
            # 计算子区域坐标
            rx1 = int(rel[0] * main_width)
            ry1 = int(rel[1] * main_height)
            rx2 = int(rel[2] * main_width)
            ry2 = int(rel[3] * main_height)

            sub_roi = screenshot[ry1:ry2, rx1:rx2]

            # 图像匹配
            matched_id, confidence = find_best_match(sub_roi, ref_images)

            # OCR识别（优化区域截取）
            number_roi = sub_roi[-sub_roi.shape[0] // 4:, sub_roi.shape[1] // 3:]
            processed = preprocess(number_roi)

            # cv2.imshow("Processed", processed)
            # cv2.waitKey(0)  # 等待用户按键
            # cv2.destroyAllWindows()  # 关闭所有窗口


            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789x×X'
            number = pytesseract.image_to_string(processed, config=custom_config).strip()
            number = number.replace('×', 'x').lower()  # 统一符号
            # print(f"区域{idx} OCR识别结果: {number}")
            # 找到第一个x的位置并截取后续内容
            x_pos = number.find('x')
            if x_pos != -1:
                number = number[x_pos + 1:]  # 截取x之后的字符串
            # 只保留数字
            number = ''.join(filter(str.isdigit, number))

            # 保存有数字的图片到images/nums中的对应文件夹
            if number:
                save_number_image(number, processed, matched_id)

            results.append({
                "region_id": idx,
                "matched_id": matched_id,
                "number": number if number else "N/A",
                "confidence": round(confidence, 2)
            })
        except Exception as e:
            print(f"区域{idx}处理失败: {str(e)}")
            results.append({
                "region_id": idx,
                "error": str(e)
            })

    return results

def load_ref_images(ref_dir="images"):
    """加载参考图片库"""
    ref_images = {}
    for i in range(27):
        path = os.path.join(ref_dir, f"{i}.png")
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                ref_images[i] = img
    return ref_images

if __name__ == "__main__":
    print("请用鼠标拖拽选择主区域...")
    main_roi = select_roi()
    ref_images = load_ref_images()
    results = process_regions(main_roi, ref_images)
    # 输出结果
    print("\n识别结果：")
    for res in results:
        if 'error' in res:
            print(f"区域{res['region_id']}: 错误 - {res['error']}")
        else:
            if res['matched_id'] != 0:
                print(
                    f"区域{res['region_id']} => 匹配ID:{res['matched_id']} 数字:{res['number']} 置信度:{res['confidence']}")

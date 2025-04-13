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
    min_diff = float('inf')
    best_id = -1
    target_pre = preprocess(target)

    for img_id, ref_img in ref_images.items():
        try:
            ref_resized = cv2.resize(ref_img, (target.shape[1], target.shape[0]))
            ref_pre = preprocess(ref_resized)
            diff = cv2.absdiff(target_pre, ref_pre)
            diff_value = np.sum(diff) / target.size
            if diff_value < min_diff:
                min_diff = diff_value
                best_id = img_id
        except:
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

            # 保存有数字的图片到images/nums，命名为数字，重复的直接覆盖
            if number:
                save_path = f"images/nums/{number}.png"
                if not os.path.exists("images/nums"):
                    os.makedirs("images/nums")
                cv2.imwrite(save_path, processed)

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

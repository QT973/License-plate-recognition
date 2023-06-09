import cv2
import openpyxl
import pytesseract
from ultralytics import YOLO
# import easyocr 

# Object detection
model = YOLO("best.pt")
video = cv2.VideoCapture("test1.mp4")
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
# reader  = easyocr.Reader(["en"],gpu = false)
xe = 0
line_position = 400 # Khoảng cách đường line
# Tạo một workbook mới
workbook = openpyxl.Workbook()

# Chọn sheet hiện tại
sheet = workbook.active

# Ghi tiêu đề cho các cột
sheet['A1'] = 'STT'
sheet['B1'] = 'Biển số xe'

while True:
    ret, frame = video.read()
    if not ret:
        break
    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 255, 0), 2)    # vẽ đường line 
    results = model(frame, stream=True)
    for result in results:
        resulst_cpu = result.cpu()
        boxes = resulst_cpu.boxes.numpy()
        for box in boxes:
            if box.cls == 0 and box.conf > 0.85:
                image_result = frame[int(box.xyxy[0][1]): int(
                    box.xyxy[0][3]), int(box.xyxy[0][0]):int(box.xyxy[0][2])]
                img_gray = cv2.cvtColor(image_result, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
                threash = cv2.threshold(
                    blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                cv2.imshow("crop ", threash)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                openping = cv2.morphologyEx(
                    threash, cv2.MORPH_OPEN, kernel, iterations=1)
                invert = 255 - openping
                text = pytesseract.image_to_string(
                    invert, lang="eng", config="--psm 6")
                # text = reader.readtext(invert,detail =0)
                if box.xyxy[0][3]>line_position:    # nều điểm y_max của bouding box lớn hơn đường line thì thực hiện 
                    xe += 1
                    # Ghi dữ liệu vào các ô tương ứng
                    sheet[f"A{xe+1}"] = xe
                    sheet[f"B{xe+1}"] = text

                    print("xe số ", xe, ":", text)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

# Lưu workbook vào file
workbook.save('data1.xlsx')
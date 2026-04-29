import os
import cv2

SRC_DIR = "./animal_data/train"
DST_DIR = "./animal_data/cropped"
CLASSES = ["bear", "cat", "dog", "fox"]
MAX_DISPLAY = 900  # 화면에 보여줄 최대 변 길이 (px)


def label_class(cls):
    src = os.path.join(SRC_DIR, cls)
    dst = os.path.join(DST_DIR, cls)
    os.makedirs(dst, exist_ok=True)

    if not os.path.isdir(src):
        print(f"[{cls}] 폴더 없음: {src}")
        return True

    files = sorted(f for f in os.listdir(src)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")))
    total = len(files)

    for i, fname in enumerate(files, 1):
        out_path = os.path.join(dst, fname)
        if os.path.exists(out_path):
            continue  # 이미 라벨링한 건 스킵

        img = cv2.imread(os.path.join(src, fname))
        if img is None:
            print(f"  [skip] 못 읽음: {fname}")
            continue

        h, w = img.shape[:2]
        scale = min(MAX_DISPLAY / w, MAX_DISPLAY / h, 1.0)
        disp = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1.0 else img

        win = f"[{cls}] {i}/{total}  {fname}   (drag=box, ENTER=save, C=skip, ESC=quit)"
        roi = cv2.selectROI(win, disp, showCrosshair=False, fromCenter=False)
        key = cv2.waitKey(1) & 0xFF
        cv2.destroyAllWindows()

        if key == 27:  # ESC → 전체 종료
            print("\n중단됨")
            return False

        x, y, rw, rh = roi
        if rw == 0 or rh == 0:
            print(f"  [skip] {fname}")
            continue

        if scale < 1.0:
            x, y, rw, rh = [int(v / scale) for v in (x, y, rw, rh)]

        crop = img[y:y + rh, x:x + rw]
        cv2.imwrite(out_path, crop)
        print(f"  [save] {out_path}  ({rw}x{rh})")

    print(f"[{cls}] 완료\n")
    return True


if __name__ == "__main__":
    for cls in CLASSES:
        if not label_class(cls):
            break
    print("라벨링 종료")

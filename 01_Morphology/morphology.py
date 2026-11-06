"""
Morphological Image Processing: Dilation / Erosion / Opening / Closing
- 네 가지 연산을 NumPy로 직접 구현 (OpenCV 함수는 결과 검증용으로만 사용)
- 효과가 잘 드러나도록 설계한 이진 테스트 이미지 사용
    * 배경의 흰 점 노이즈(salt)      -> Opening으로 제거되는지 확인
    * 도형 내부의 검은 구멍(pepper)  -> Closing으로 채워지는지 확인
    * 글자의 끊어진 틈(gap)           -> Closing / Dilation으로 메워지는지 확인
    * 도형을 잇는 얇은 선             -> Erosion / Opening으로 끊어지는지 확인
- 결과: images/ 폴더에 원본, 각 결과, 비교 그림 저장

사용법:
    python morphology.py                 # 테스트 이미지 생성 후 처리
    python morphology.py --input my.png  # 다른 이미지 사용 (자동 이진화)
"""
import argparse
import os

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# 1. 형태학 연산 직접 구현
# ----------------------------------------------------------------------------
def make_kernel(size=5, shape="rect"):
    """구조 요소(structuring element) 생성"""
    if shape == "rect":
        return np.ones((size, size), dtype=np.uint8)
    if shape == "cross":
        k = np.zeros((size, size), dtype=np.uint8)
        k[size // 2, :] = 1
        k[:, size // 2] = 1
        return k
    if shape == "ellipse":
        r = size // 2
        y, x = np.ogrid[-r:r + 1, -r:r + 1]
        return (x * x + y * y <= r * r).astype(np.uint8)
    raise ValueError(shape)


def _apply(img, kernel, op):
    """구조 요소의 각 원소 위치만큼 이미지를 이동시켜 max(팽창) / min(침식)을 누적.
    이진 이미지에서는 max = OR, min = AND 와 같고, 그레이스케일에도 그대로 동작한다."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    # 경계 처리: 팽창은 0, 침식은 최대값으로 패딩 -> 경계 밖이 결과에 영향 주지 않음
    pad_val = 0 if op == "dilate" else 255
    padded = np.pad(img, ((ph, ph), (pw, pw)), constant_values=pad_val)
    H, W = img.shape
    out = np.full_like(img, 0 if op == "dilate" else 255)
    for dy in range(kh):
        for dx in range(kw):
            if kernel[dy, dx] == 0:
                continue
            # 반사(reflection)된 구조 요소를 사용 (정의: A ⊕ B = ∪ (A)_b)
            window = padded[kh - 1 - dy: kh - 1 - dy + H, kw - 1 - dx: kw - 1 - dx + W]
            out = np.maximum(out, window) if op == "dilate" else np.minimum(out, window)
    return out


def dilation(img, kernel):
    """팽창: 커널 범위 안에 전경(255)이 하나라도 있으면 전경 -> 객체 확장, 틈·구멍 메움"""
    return _apply(img, kernel, "dilate")


def erosion(img, kernel):
    """침식: 커널 범위 전체가 전경일 때만 전경 -> 객체 축소, 작은 점·얇은 선 제거"""
    return _apply(img, kernel, "erode")


def opening(img, kernel):
    """열림 = 침식 후 팽창: 작은 노이즈 제거, 큰 객체의 크기는 대체로 유지"""
    return dilation(erosion(img, kernel), kernel)


def closing(img, kernel):
    """닫힘 = 팽창 후 침식: 작은 구멍·틈 메움, 큰 객체의 크기는 대체로 유지"""
    return erosion(dilation(img, kernel), kernel)


# ----------------------------------------------------------------------------
# 2. 테스트 이미지 생성
# ----------------------------------------------------------------------------
def make_test_image(seed=0):
    rng = np.random.default_rng(seed)
    H, W = 320, 560
    img = np.zeros((H, W), dtype=np.uint8)

    # 굵은 글자 "MORPH"
    cv2.putText(img, "MORPH", (30, 130), cv2.FONT_HERSHEY_DUPLEX, 3.6, 255, 14, cv2.LINE_8)
    # 글자에 세로 틈(gap) 만들기 -> Closing이 메우는지 관찰
    for x in [70, 160, 250, 350, 440, 500]:
        img[40:140, x:x + 3] = 0

    # 도형: 원, 사각형, 그리고 이들을 잇는 얇은 선(2px)
    cv2.circle(img, (110, 240), 55, 255, -1)
    cv2.rectangle(img, (260, 190), (400, 290), 255, -1)
    cv2.line(img, (165, 240), (260, 240), 255, 2)          # 얇은 다리 -> Erosion이 끊는지 관찰
    cv2.rectangle(img, (450, 190), (530, 290), 255, 10)     # 두꺼운 테두리 사각형
    img[238:242, 440:460] = 0                                # 테두리의 작은 틈 -> Closing이 잇는지 관찰

    # Pepper 노이즈: 도형 내부의 작은 검은 구멍
    fg = np.argwhere(img == 255)
    for y, x in fg[rng.choice(len(fg), 150, replace=False)]:
        img[y:y + 2, x:x + 2] = 0

    # Salt 노이즈: 배경의 작은 흰 점
    bg = np.argwhere(img == 0)
    for y, x in bg[rng.choice(len(bg), 450, replace=False)]:
        s = rng.integers(1, 3)
        img[y:y + s, x:x + s] = 255
    return img


# ----------------------------------------------------------------------------
# 3. 결과 분석
# ----------------------------------------------------------------------------
def stats(img):
    """전경 픽셀 비율, 연결 요소(객체) 수, 내부 구멍 수"""
    fg_ratio = (img > 127).mean()
    n_obj = cv2.connectedComponents((img > 127).astype(np.uint8), connectivity=8)[0] - 1
    n_bg = cv2.connectedComponents((img <= 127).astype(np.uint8), connectivity=4)[0] - 1
    return fg_ratio, n_obj, n_bg - 1  # 바깥 배경 1개를 제외한 나머지 = 구멍 수


def save_comparison(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, (name, r) in zip(axes.ravel(), results.items()):
        ax.imshow(r, cmap="gray", vmin=0, vmax=255)
        fr, no, nh = stats(r)
        ax.set_title(f"{name}\nfg {fr * 100:.1f}% · objects {no} · holes {nh}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="입력 이미지 (없으면 테스트 이미지 생성)")
    ap.add_argument("--out-dir", default="./images")
    ap.add_argument("--ksize", type=int, default=5)
    ap.add_argument("--shape", default="rect", choices=["rect", "cross", "ellipse"])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.input:
        gray = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        _, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        img = make_test_image()

    k = make_kernel(args.ksize, args.shape)
    results = {
        "Original": img,
        "Dilation": dilation(img, k),
        "Erosion": erosion(img, k),
        "Opening": opening(img, k),
        "Closing": closing(img, k),
    }
    # 노이즈 제거 + 구멍 메우기를 함께 적용한 실용적 조합
    results["Open → Close"] = closing(opening(img, k), k)

    # 직접 구현이 OpenCV와 일치하는지 검증
    ref = {
        "Dilation": cv2.dilate(img, k),
        "Erosion": cv2.erode(img, k),
        "Opening": cv2.morphologyEx(img, cv2.MORPH_OPEN, k),
        "Closing": cv2.morphologyEx(img, cv2.MORPH_CLOSE, k),
    }
    print(f"구조 요소: {args.shape} {args.ksize}x{args.ksize}")
    for name, r in ref.items():
        print(f"  {name:8s} OpenCV와 일치: {np.array_equal(results[name], r)}")

    print("\n결과 요약 (전경 비율 / 객체 수 / 구멍 수)")
    for name, r in results.items():
        fr, no, nh = stats(r)
        print(f"  {name:13s} {fr * 100:5.1f}%  {no:4d}  {nh:4d}")
        fname = name.lower().replace(" → ", "_").replace(" ", "_") + ".png"
        cv2.imwrite(os.path.join(args.out_dir, fname), r)

    save_comparison({kk: v for kk, v in results.items()}, os.path.join(args.out_dir, "comparison.png"))
    print(f"\n결과 저장: {args.out_dir}/")


if __name__ == "__main__":
    main()

import numpy as np
import cv2


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def get_label(label_file):
    with open(label_file, 'r') as f:
        lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
    return lb


def vis_label(img, label, save_name="1.png" ):
    h, w = img.shape[:2]
    labe = xywhn2xyxy(label[:, 1:], w, h)
    cls = label[:, 0]
    colors = [(0, 0, 255), (255, 0, 255), (0, 255, 0), (100, 255, 255), (255, 0, 0)]
    for i, x in enumerate(labe):
        cv2.rectangle(img, (int(x[0]), int(x[1])), (int(x[2]), int(x[3])), colors[int(cls[i])], 3)

    cv2.imshow('show', img)
    cv2.imwrite(save_name, img)
    cv2.waitKey(1000)


def crop_label(label, h0, w0, h, w, offset_h, offset_w):
    label = np.copy(label)
    label[:, 1] = (label[:, 1] * w0 - offset_w)
    label[:, 2] = (label[:, 2] * h0 - offset_h)
    label[:, 3] = label[:, 3] * w0
    label[:, 4] = label[:, 4] * h0

    valid_inds = (label[:, 1] + label[:, 3]/2 < w) & (label[:, 2] + label[:, 4]/2 < h) & \
                 (label[:, 1] - label[:, 3]/2 > 0) & (label[:, 2] - label[:, 4]/2 > 0)

    valid_label = label[valid_inds, :]
    valid_label[:, 1::2] = valid_label[:, 1::2]/w
    valid_label[:, 2::2] = valid_label[:, 2::2]/h
    return valid_label


def crop_image(img, size=640):
    h0, w0 = img.shape[:2]  # orig hw
    margin_h = h0 - size
    margin_w = w0 - size
    offset_h = np.random.randint(0, margin_h + 1)
    offset_w = np.random.randint(0, margin_w + 1)
    crop_y1, crop_y2 = offset_h, offset_h + size
    crop_x1, crop_x2 = offset_w, offset_w + size
    return img[crop_y1:crop_y2, crop_x1:crop_x2, :], (h0, w0), (size, size), (offset_h, offset_w)


if __name__ == "__main__":
    img_file = "00001_FV.png"
    img = cv2.imread(img_file)

    label_file = "00001_FV.txt"
    label = get_label(label_file)

    # vis_label(img, label)

    img_crop, (h0, w0), (h, w), (offset_h, offset_w) = crop_image(img, size=960)
    img = cv2.resize(img_crop, (960, 960), interpolation=cv2.INTER_LINEAR)

    label_crop = crop_label(label, h0, w0, h, w, offset_h, offset_w)

    vis_label(img, label_crop, "crop_.png")

    print(" ")
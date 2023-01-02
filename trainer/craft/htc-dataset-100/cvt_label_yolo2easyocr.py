import os
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm

if __name__=="__main__":
    for fn in tqdm(glob("full_images/*txt")):
        df_r = pd.read_csv(fn, sep=" ", header=None)
        img = cv2.imread(fn.replace(".txt", ".png"))
        H, W, C = img.shape
        Frames = []
        for _, row in df_r.iterrows():
            l, x_left, y_top, w_, h_ = row
            x_right = int((x_left + w_) * W)
            y_bottom = int((y_top + h_) * H)
            x_left = int(x_left * W)
            y_top = int(y_top * H)
            Frames.append([x_left, y_top, x_right, y_top, x_right, y_bottom, x_left, y_bottom, "ANoi"])

        df_w = pd.DataFrame(Frames)
        bn = os.path.basename(fn)
        df_w.to_csv(fn.replace("full_images", "full_gt").replace(bn, f"gt_{bn}"), header=None, index=None)

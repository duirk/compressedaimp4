import cv2
import torch
import tkinter as tk
from tkinter import ttk
from tqdm import tqdm

def process_frame(frame, model):
    tensor = torch.from_numpy(frame.transpose((2,0,1))).unsqueeze(0).float()
    with torch.no_grad():
        output = model(tensor)
    compressed_frame = (255 * output.cpu().numpy().squeeze()).astype('uint8')
    return compressed_frame

def generate_compressed_video():
    cap = cv2.VideoCapture('compressed_video.mp4')
    # Define num_channels variable
    num_channels = 3

    model = torch.nn.Sequential(
        torch.nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
        torch.nn.ReLU()
    )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    compressed_frames = []
    with tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                compressed_frame = process_frame(frame, model)
                compressed_frames.append(compressed_frame)
                pbar.update(1)
            else:
                break

    out = cv2.VideoWriter('compressed_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, (300,250))
    for frame in tqdm(compressed_frames):
        out.write(frame)
    out.release()
    progress_bar.stop()
    
def generate_button_click():
    progress_bar.start()
    generate_compressed_video()
    
# Creating the Tkinter window and widgets
root = tk.Tk()
root.title("Compressed Video Generator")

frame = tk.Frame(root, padx=10, pady=10)
frame.pack()

generate_button = tk.Button(frame, text="Generate", bg="orange", command=generate_button_click)
generate_button.pack(pady=10)

progress_bar = ttk.Progressbar(frame, orient="horizontal", length=400, mode="determinate")
progress_bar.pack(pady=10)

root.mainloop()

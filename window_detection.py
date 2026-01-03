import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
import pyautogui
import cv2
from playsound import playsound
from plyer import notification

coords_df = pd.read_csv("points.csv")
coords = list(coords_df.itertuples(index=False, name=None))

class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(2, 2)
            )
        self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, 1, 1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2, 2)
            )
        self.conv3 = nn.Sequential(
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(2, 2)
            )
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

device = torch.device("cpu")
model = CNN(num_classes=2)
model.load_state_dict(torch.load("waldo_cnn.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

def sliding_window(image, step_size, window_size):
    for y in range(0, image.height - window_size + 1, step_size):
        for x in range(0, image.width - window_size + 1, step_size):
            patch = image.crop((x, y, x + window_size, y + window_size))
            yield (x, y, patch)


def ask():
    while True:
        try:
            choice = int(input("Do you want to 1) use a pre-downloaded image, 2) take a screenshot, or 3) take a webcam picture? "))

            # 1) Use pre-downloaded image
            if choice == 1:
                path = input("Enter image path: ").strip()
                path = path.replace("\\", "/")
                path = path.replace('"', '')
                try:
                    full_image = Image.open(path)
                    full_image2 = Image.open(path)
                    print("Image loaded successfully! Starting search now...")
                    return full_image, full_image2
                except FileNotFoundError:
                    print("The file path does not exist. Try again.")
                except OSError:
                    print("The file could not be opened. Make sure it's a valid image.")
                except Exception as e:
                    print(f"Unexpected error: {e}")

            # 2) Take screenshot
            elif choice == 2:
                print("Navigate to the webpage. Screenshot will be taken in 5 seconds...")
                time.sleep(5)
                screenshot = pyautogui.screenshot("screenshot.jpg")
                playsound("C:/Users/Ezra/Downloads/Games/Guitar_Player/waldo/camera-shutter-314056.mp3")
                notification.notify(
                title='Screenshot Taken',
                message='A screenshot was successfully taken! Watch the program run in the terminal.',
                app_name='Python Notifier',
                timeout=5
            )
                print("Screenshot saved as 'screenshot.jpg'.")
                full_image = screenshot
                full_image2 = screenshot
                return full_image, full_image2

            # 3) Take webcam picture
            elif choice == 3:
                camera = cv2.VideoCapture(0)
                if not camera.isOpened():
                    print("Could not open camera.")
                    continue

                print("Picture will be taken in 5 seconds...")
                time.sleep(5)

                ret, frame = camera.read()
                playsound("C:/Users/Ezra/Downloads/Games/Guitar_Player/waldo/camera-shutter-314056.mp3")
                notification.notify(
                title='Picture Taken',
                message='A webcam image was successfully taken! Watch the program run in the terminal.',
                app_name='Python Notifier',
                timeout=5
            )
                camera.release()

                if not ret:
                    print("Could not read from webcam.")
                    continue

                cv2.imwrite("webcam.jpg", frame)
                print("Image saved as 'webcam.jpg'.")
                full_image = full_image2 = Image.open("webcam.jpg")
                return full_image, full_image2

            else:
                print("Please enter a number from 1-3. ")
        except ValueError:
            print("Enter a valid number from 1-3.")
        except Exception as e:
            print(f"Unexpected error: {e}")


full_image, full_image2 = ask()
full_image = full_image.convert("RGB")
full_image2 = full_image2.convert("RGB")

# full_image = Image.open("C:/Users/Ezra/Downloads/Games/Guitar_Player/waldo/full_waldo_page.jpeg").convert("RGB")
# full_image2 = Image.open("C:/Users/Ezra/Downloads/Games/Guitar_Player/waldo/full_waldo_page.jpeg").convert("RGB")
window_size = 64
step_size = 4
instances = 500 #instances of waldo's face in the picture
nums = []

xs = (coords_df['X'] / coords_df['X'].max() * full_image.width).tolist()
ys = (coords_df['Y'] / coords_df['Y'].max() * full_image.height).tolist()

coords = list(zip(xs, ys))

def get_surrounding_points(x, y, radius, step):
    points = []
    for dy in range(-radius, radius + 1, step):
        for dx in range(-radius, radius + 1, step):
            points.append((x + dx, y + dy))
    return points

#optimal path
start = time.perf_counter()
detections = []
for (px, py) in tqdm(coords, desc="Optimal Path Search"):
    for (x, y) in get_surrounding_points(px, py, radius=20, step=4):
        if 0 <= x <= full_image.width - window_size and 0 <= y <= full_image.height - window_size:
            patch = full_image.crop((x, y, x + window_size, y + window_size))
            input_tensor = transform(patch).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                waldo_prob = probs[0, 1].item()
                if waldo_prob > 0.4:
                    detections.append((x, y, waldo_prob))


nums.append(len(detections))
end = time.perf_counter()
elapsed = end-start
print(f"Optimal search path got {nums[0]}/{instances} instances and took {elapsed} seconds.")

draw = ImageDraw.Draw(full_image)
for (x, y, prob) in detections:
    draw.rectangle([x, y, x + window_size, y + window_size], outline="green", width=3)
    draw.text((x, y - 16), f"{prob:.2f}", fill="green")

for i in range(len(coords) - 1):
    x1, y1 = coords[i]
    x2, y2 = coords[i+1]
    y1_flipped = full_image.height - y1
    y2_flipped = full_image.height - y2
    draw.line([(x1, y1_flipped), (x2, y2_flipped)], fill="blue", width=3)

for (x, y) in coords:
    y_flipped = full_image.height - y
    draw.ellipse([x-2, y_flipped-2, x+2, y_flipped+2], fill="red")

full_image.show() 

#everything
if len(detections) < instances:
    
    start2 = time.perf_counter()
    detections = [] 
    
    for x, y, patch in tqdm(sliding_window(full_image, step_size, window_size), desc="Everything Search"):
        input_tensor = transform(patch).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            waldo_prob = probs[0, 1].item()
            if waldo_prob > 0.1:
                detections.append((x, y, waldo_prob))

    nums.append(len(detections))
    end2 = time.perf_counter()
    elapsed2 = end2-start2
    faster = elapsed2 - elapsed
    print(f"Optimal search path got {nums[0]}/{instances} instances and took {elapsed} seconds. Everything search got {nums[1]}/{instances} instances and took {elapsed2} seconds ({faster} seconds faster).") 

    draw = ImageDraw.Draw(full_image2)
    for (x, y, prob) in detections:
        draw.rectangle([x, y, x + window_size, y + window_size], outline="green", width=3)
        draw.text((x, y - 16), f"{prob:.2f}", fill="green")


    full_image2.show() 


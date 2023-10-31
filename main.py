from __future__ import annotations
import os

from itertools import cycle

from torch.optim import Optimizer
from torch.utils.data import (DataLoader, Dataset)
from torchvision.datasets.mnist import EMNIST
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt

class EMNISTDataset(Dataset):
    def __init__(self, train: bool, path: str, device: torch.device) -> None:
        super().__init__()
        self.path = path
        self.prefix = 'train' if train else 'test'
        self.path_xs = os.path.join(self.path, f'emnist_{self.prefix}_xs.pt')
        self.path_ys = os.path.join(self.path, f'emnist_{self.prefix}_ys.pt')
        
        self.transform = T.Compose([T.ToTensor()])

        if not os.path.exists(self.path_xs) or not os.path.exists(self.path_ys):
            set = EMNIST(path, split= "letters",train=train, download=True, transform=self.transform)
            loader = DataLoader(set, batch_size=batch_size, shuffle=train)
            n = len(set)

            xs = torch.empty((n, *set[0][0].shape), dtype=torch.float32)
            ys = torch.empty((n, ), dtype=torch.long)
            desc = f'Preparing {self.prefix.capitalize()} Set'
            for i, (x, y) in enumerate(tqdm(loader, desc=desc)):
                xs[i * batch_size:min((i + 1) * batch_size, n)] = x
                ys[i * batch_size:min((i + 1) * batch_size, n)] = y
            torch.save(xs, self.path_xs)
            torch.save(ys, self.path_ys)
        
        self.device = device
        self.xs = torch.load(self.path_xs, map_location=self.device)
        self.ys = torch.load(self.path_ys, map_location=self.device)


        # Retrieve an image and label
        index = 20
        image = self.xs[index]
        label = self.ys[index]
        # Convert tensor image to PIL image for displaying
        img_flip = T.functional.rotate(image, -90)
        corrected_image = T.functional.hflip(img_flip)
        pil_image = T.ToPILImage()(corrected_image) # "L" mode is for grayscale
        # Display the image
        plt.imshow(pil_image, cmap='gray')  # Ensure the colormap is set to gray
        plt.title(f'Label: {chr(label + 64)}')
        plt.show()
        
    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.xs[idx], self.ys[idx]


class FeaturesDetector(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 24, 3, 1)
        self.conv3 = nn.Conv2d(24, out_channels, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(torch.max_pool2d(self.conv3(x), 2))
        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = FeaturesDetector(1, 32)
        self.fc = MLP(3872, 52)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def fit(self, loader: DataLoader, optimizer: Optimizer, scheduler, epochs: int) -> None:
        self.train()
        array_loss = []
        batches = iter(cycle(loader))
        for _ in tqdm(range(epochs * len(loader)), desc='Training'):
            x, l = next(batches)
            optimizer.zero_grad(set_to_none=True)
            logits = self(x)
            loss = F.nll_loss(torch.log_softmax(logits, dim=1), l)
            loss.backward()
            optimizer.step()
            scheduler.step()
            array_loss.append(loss.item())

        plt.plot(array_loss)
        plt.show()

    @torch.inference_mode()
    def test(self, loader: DataLoader) -> None:
        self.eval()
        loss, acc = 0, 0.0
        for x, l in tqdm(loader, total=len(loader), desc='Testing'):
            logits = self(x)
            preds = torch.argmax(logits, dim=1, keepdim=True)
            loss += F.nll_loss(torch.log_softmax(logits, dim=1), l, reduction='sum').item()
            acc += (preds == l.view_as(preds)).sum().item()
        print()
        print(f'Loss: {loss / len(loader.dataset):.2e}')
        print(f'Accuracy: {acc / len(loader.dataset) * 100:.2f}%')
        print()

if __name__ == '__main__':
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    from PIL import Image
    import matplotlib.pyplot as plt
    import torch
    import torch.onnx
    import numpy as np

    def plot_image(image_path):
        img = Image.open(image_path)
        plt.imshow(img, cmap='gray')
        plt.title(image_path)
        plt.show()



    choice = input("Train or Else: ").lower()
    if choice == "train":
        device = torch.device('cuda')
        epochs = 3
        batch_size = 512
        lr = 1e-2
        train_set = EMNISTDataset(True, './data', device)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        
        test_set = EMNISTDataset(False, './data', device)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)

        model = CNN().to(device)
        optimizer = AdamW(model.parameters(), lr=lr, betas=(0.7, 0.9)) 
        scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=int(((len(train_set) - 1) // batch_size + 1) * epochs))

        model.fit(train_loader, optimizer, scheduler, epochs)
        model.test(test_loader)
        torch.save(model.state_dict(), 'emnist_cnn.pt')
        
        #Export in onnx format
        model.eval()
        dummy_input = torch.randn(1, 1, 28, 28).to('cuda')
        torch.onnx.export(model, dummy_input, "emnist.onnx", export_params=True, opset_version=9 )

    elif (choice == "onnx"):
        import onnxruntime as ort
        img_path = 'testb.jpg'
        img = Image.open(img_path)

        transform = T.Compose([ T.ToTensor(),
                                # T.Normalize((0.5, ), (0.5, )),
                                T.Grayscale(num_output_channels=1),
                            ])
        
        img_tensor = transform(img).unsqueeze(0).numpy()
        # Load ONNX model
        ort_session = ort.InferenceSession("emnist.onnx")

        # Run inference
        ort_inputs = {ort_session.get_inputs()[0].name: img_tensor}
        ort_outs = ort_session.run(None, ort_inputs)

        # Process the output (assuming a single output, adapt as necessary)
        prediction = ort_outs[0]
        predicted_class = np.argmax(prediction)

        # Print or return the predicted class
        print(f"Predicted Class: {chr(predicted_class + 64)}")
        

    else:
        model = CNN().to('cuda')

        # Load the trained state dictionary
        model.load_state_dict(torch.load('emnist_cnn.pt'))
        model.eval()

        # Load and process the image
        transform = T.Compose([ T.ToTensor(),
                                # T.Normalize((0.5, ), (0.5, )),
                                T.Grayscale(num_output_channels=1),
                            ])
        for i in ['A','B','C','D']:
            img_path = 'manual_tests/please_return_' + i + '.png'
            img = Image.open(img_path)

            img_tensor = transform(img).unsqueeze(0).to('cuda')

            corrected_image = T.functional.hflip(img_tensor)
            img_flip = T.functional.rotate(corrected_image, 90)
            prediction = model(img_flip)
            prediction = torch.log_softmax(prediction, dim=-1)
            predicted_class = torch.argmax(prediction)
            predicted_letter = chr(predicted_class.item() + 64)
            pil_image = T.ToPILImage()(img_tensor.squeeze(0))
            # Display the image
            plt.imshow(pil_image, cmap='gray')  # Ensure the colormap is set to gray
            plt.title(f'Label: {predicted_letter}')
            plt.show()  
import torch.nn as nn
from tqdm.auto import tqdm
import torch
from src.load_dataset import loader
import matplotlib.pyplot as plt
from Autoencoder_CNN import Autoencoder_CNN
from src.Autoencoder_Linear import Autoencoder_Linear

epochs = 5
outputs = []
def train(model, criterion, optimizer, epochs, linear : bool):
    for epochs in tqdm(range(epochs)):
        for images, _ in loader:
            optimizer.zero_grad()
            if linear == True :
                images = images.reshape(-1, 28*28)
            images=images.to('cuda')
            recon = model(images)
            loss = criterion(recon, images)
            loss.backward()
            optimizer.step()
        print(f'\nEpoch: {epochs+1}, Loss: {loss.item():.4f} ')
        outputs.append((epochs,images,recon))


def visualize(linear: bool):
    for k in range(0, epochs, 4):
        plt.figure(figsize=(9, 2))
        plt.gray()
        imgs = outputs[k][1].cpu().detach().numpy()
        recon = outputs[k][2].cpu().detach().numpy()
        for i, item in enumerate(imgs):
            if i >= 9: break
            plt.subplot(2, 9, i + 1)
            if linear:
                item = item.reshape(-1, 28, 28)  # -> use for Autoencoder_Linear
            plt.imshow(item[0])

        for i, item in enumerate(recon):
            if i >= 9: break
            plt.subplot(2, 9, 9 + i + 1)  # row_length + i + 1
            if linear:
                item = item.reshape(-1, 28, 28)  # -> use for Autoencoder_Linear
            plt.imshow(item[0])
    plt.show()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Autoencoder_Linear()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model.to(device)
    train(model, criterion, optimizer, epochs,linear=True) # set linear to False for CNN
    visualize(linear=True) # set linear to False for CNN



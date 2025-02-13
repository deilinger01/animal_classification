import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, datasets, transforms
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.io import wavfile
import random

INPUT_DIRECTORY = ... # TODO: directory to the Data
ANIMALS = ['Bird', 'Cat', 'Chicken', 'Cow', 'Dog', 'Donkey', 'Frog', 'Lion', 'Monkey', 'Sheep']
SPECTOGRAM_DIRECTORY = ... # TODO: directory to where the spectogrames should be stored

def generate_spectrogram():
    fail_process_list = []
    for curr_animal in ANIMALS:
        count_fails = 0
        curr_directory = os.path.join(SPECTOGRAM_DIRECTORY, curr_animal + '/' + curr_animal)
        if not os.path.exists(curr_directory):
            os.makedirs(curr_directory, exist_ok=True)
            input_directory = os.path.join(INPUT_DIRECTORY, curr_animal)
            curr_files = [file for file in os.listdir(input_directory) if file.endswith('.wav')]
            for filename in curr_files:
                try:
                    sample_rate, data = wavfile.read(os.path.join(input_directory, filename))
                    plt.specgram(data, Fs=sample_rate)
                    output_path = os.path.join(curr_directory, filename[:-4] + '_spectrogram.jpg')
                    plt.savefig(output_path)
                    plt.close()
                except Exception as e:
                    print(f'Failed to process {filename} for {curr_animal}: {e}')
                    count_fails += 1
    fail_process_list.append([f'Amount of failed to convert to spectogram with {curr_animal}: {count_fails}'])
    for message in fail_process_list:
        print(message)

def generate_embeddings():
    BATCH_SIZE = 32
    weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    for curr_animal in ANIMALS:
        embedding_path = os.path.join(SPECTOGRAM_DIRECTORY, curr_animal + '_embeddings.npy')
        if not os.path.exists(embedding_path):
            curr_directory = os.path.join(SPECTOGRAM_DIRECTORY, curr_animal)
            dataset = datasets.ImageFolder(root=curr_directory, transform=preprocess)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)

            model = models.convnext_base(weights=weights)
            model.eval()
            model.to(device)
            model = nn.Sequential(*list(model.children())[:-1])
            embedding_size = 1024
            embeddings = np.zeros((len(dataset), embedding_size))

            with torch.no_grad():
                for i, (features, _) in enumerate(loader):
                    outputs = model(features.to(device))
                    embeddings[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE] = outputs.squeeze().cpu().numpy()

            np.save(embedding_path, embeddings)

def load_data(test_size=0.2):
    X, y = [], []
    for idx, animal in enumerate(ANIMALS):
        embeddings = np.load(os.path.join(SPECTOGRAM_DIRECTORY, animal + '_embeddings.npy'))
        mean, std = embeddings.mean(axis=0), embeddings.std(axis=0)
        embeddings = (embeddings - mean) / std

        X.extend(embeddings)
        y.extend([idx] * len(embeddings))

    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

class AnimalSoundClassifier(nn.Module):
    def __init__(self, input_size):
        super(AnimalSoundClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def train_model(model, criterion, optimizer, train_loader, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')


def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100 
    print(f'Test Accuracy: {accuracy:.2f}%')



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    generate_spectrogram()
    generate_embeddings()
    X_train, X_test, y_train, y_test = load_data()

    input_size = 1024
    learning_rate = 0.001
    num_epochs = 20

    model = AnimalSoundClassifier(input_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

    train_model(model, criterion, optimizer, train_loader, num_epochs=num_epochs)
    test_model(model, test_loader)


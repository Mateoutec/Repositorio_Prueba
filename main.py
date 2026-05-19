import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import timm
from tqdm import tqdm


# 1. REPRODUCIBILIDAD
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 2. DATASET (Mel-Spectrogramas de 19 seg)
class KaggleAudioDataset(Dataset):
    def __init__(self, df, audio_dir, is_train=True, target_sr=32000, target_sec=19):
        self.df = df
        self.audio_dir = audio_dir
        self.is_train = is_train
        self.target_sr = target_sr
        self.target_length = target_sr * target_sec

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr, n_fft=1024, hop_length=320, n_mels=64
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=35)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.audio_dir, f"{row['Id']}.wav")

        # Carga con motor soundfile (asegúrate de tenerlo instalado)
        waveform, sr = torchaudio.load(audio_path, backend="soundfile")

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        current_len = waveform.shape[1]
        if current_len < self.target_length:
            repeats = (self.target_length // current_len) + 1
            waveform = waveform.repeat(1, repeats)[:, :self.target_length]
        elif current_len > self.target_length:
            if self.is_train:
                start = random.randint(0, current_len - self.target_length)
                waveform = waveform[:, start:start + self.target_length]
            else:
                start = (current_len - self.target_length) // 2
                waveform = waveform[:, start:start + self.target_length]

        mel_spec = self.mel_spectrogram(waveform)
        mel_spec = self.amplitude_to_db(mel_spec)

        if self.is_train:
            if random.random() < 0.5: mel_spec = self.freq_mask(mel_spec)
            if random.random() < 0.5: mel_spec = self.time_mask(mel_spec)

        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
        return mel_spec, torch.tensor(row['Target'], dtype=torch.long)


# 3. MODELO
class AudioClassifier(nn.Module):
    def __init__(self, num_classes=15, backbone='resnet34'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, in_chans=1, num_classes=0, global_pool='avg')
        self.head = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


# 4. ENTRENAMIENTO
def train_model(train_csv_path, audio_dir, epochs=25):  # <--- ACTUALIZADO A 25
    print(f"--- Entrenando en: {DEVICE} | GPU: RTX 5080 | CPU: Ryzen 9 3D ---")

    df = pd.read_csv(train_csv_path)
    train_df = df.sample(frac=0.8, random_state=42).reset_index(drop=True)
    val_df = df.drop(train_df.index).reset_index(drop=True)

    num_classes = df['Target'].max() + 1
    class_counts = train_df['Target'].value_counts().sort_index().values
    class_weights = torch.FloatTensor(len(train_df) / (len(class_counts) * class_counts)).to(DEVICE)

    # Configuración de Hardware
    BATCH_SIZE = 32
    WORKERS = 16

    train_loader = DataLoader(KaggleAudioDataset(train_df, audio_dir, is_train=True),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)
    val_loader = DataLoader(KaggleAudioDataset(val_df, audio_dir, is_train=False),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)

    model = AudioClassifier(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # El scheduler ahora sabe que tiene 25 épocas para bajar el LR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = torch.amp.GradScaler('cuda')
    best_macro_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")

        for inputs, targets in train_bar:
            inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_bar.set_postfix(loss=(train_loss / (train_bar.n + 1)))

        scheduler.step()
        model.eval()
        val_preds, val_targets_list = [], []

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Validando"):
                inputs = inputs.to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_targets_list.extend(targets.numpy())

        macro_f1 = f1_score(val_targets_list, val_preds, average='macro')
        print(f"Epoch {epoch + 1} | Loss: {train_loss / len(train_loader):.4f} | F1: {macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), 'best_audio_model.pth')
            print("🌟 ¡Mejor modelo guardado!")


if __name__ == '__main__':
    # Asegúrate de que los archivos estén en la misma carpeta
    train_model(train_csv_path='train.csv', audio_dir='train', epochs=25)
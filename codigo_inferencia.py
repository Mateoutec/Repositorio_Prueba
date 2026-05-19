import os
import pandas as pd
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from tqdm import tqdm

# ─────────────────────────────────────────────
#  CONFIGURACIÓN  (debe coincidir con train.py)
# ─────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE  = 128
TARGET_SR   = 32000
TARGET_SEC  = 19                          # mismo que en entrenamiento
TARGET_LEN  = TARGET_SR * TARGET_SEC      # 608 000 samples


# ─────────────────────────────────────────────
#  1. MODELO  (idéntico al de entrenamiento)
# ─────────────────────────────────────────────
class AudioClassifier(nn.Module):
    def __init__(self, num_classes=15, backbone='resnet34'):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=False, in_chans=1,
            num_classes=0, global_pool='avg'
        )
        self.head = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


# ─────────────────────────────────────────────
#  2. DATASET DE TEST
#     Usa EXACTAMENTE la misma lógica de
#     pre-procesamiento que val en entrenamiento
#     (is_train=False → center crop de 19 seg)
# ─────────────────────────────────────────────
class TestDataset(Dataset):
    def __init__(self, test_dir):
        self.test_dir   = test_dir
        self.test_files = sorted(
            f for f in os.listdir(test_dir) if f.endswith('.wav')
        )
        # Cache de resamplers para evitar crear objetos en cada __getitem__
        self._resamplers: dict = {}

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SR, n_fft=1024, hop_length=320, n_mels=64
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def _get_resampler(self, orig_sr: int):
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=TARGET_SR
            )
        return self._resamplers[orig_sr]

    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, idx):
        fname = self.test_files[idx]
        waveform, sr = torchaudio.load(
            os.path.join(self.test_dir, fname), backend="soundfile"
        )

        # Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample si hace falta
        if sr != TARGET_SR:
            waveform = self._get_resampler(sr)(waveform)

        current_len = waveform.shape[1]

        # ── Ajuste de longitud (igual que is_train=False en KaggleAudioDataset) ──
        if current_len < TARGET_LEN:
            # Audio corto → repetir (igual que en train)
            repeats  = (TARGET_LEN // current_len) + 1
            waveform = waveform.repeat(1, repeats)[:, :TARGET_LEN]
        else:
            # Audio largo → center crop
            start    = (current_len - TARGET_LEN) // 2
            waveform = waveform[:, start:start + TARGET_LEN]

        # Mel Spectrogram + dB
        spec = self.mel_spectrogram(waveform)
        spec = self.amplitude_to_db(spec)

        # Normalización por instancia (igual que en train)
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)

        return spec, fname.replace('.wav', '')


# ─────────────────────────────────────────────
#  3. INFERENCIA
# ─────────────────────────────────────────────
def run_inference(
    test_dir    : str = 'test',
    weights_path: str = 'best_audio_model.pth',
    output_csv  : str = 'submission.csv',
    num_classes : int = 15,
):
    print(f"🚀 Iniciando inferencia en: {DEVICE}")

    # Modelo
    model = AudioClassifier(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(
        torch.load(weights_path, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    print(f"✅ Pesos cargados desde '{weights_path}'")

    # DataLoader — sin collate_fn custom porque todos los tensores
    # tienen shape [1, 64, 1900] al salir del Dataset
    num_workers = 8 if os.name != 'nt' else 0   # Windows-safe
    ds     = TestDataset(test_dir)
    loader = DataLoader(
        ds,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )
    print(f"📂 {len(ds)} archivos encontrados en '{test_dir}'")

    results = []

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for inputs, ids in tqdm(loader, desc="Prediciendo"):
            inputs  = inputs.to(DEVICE, non_blocking=True)
            outputs = model(inputs)
            preds   = torch.argmax(outputs, dim=1).cpu().numpy()

            for file_id, pred in zip(ids, preds):
                results.append({'Id': file_id, 'Target': int(pred)})

    # CSV con el formato exacto que espera Kaggle
    df = pd.DataFrame(results)[['Id', 'Target']]
    df.to_csv(output_csv, index=False)

    print(f"\n✅ '{output_csv}' generado — {len(df)} predicciones")
    print("🏆 ¡Listo para subir a Kaggle!")


if __name__ == '__main__':
    run_inference(
        test_dir     = 'test',
        weights_path = 'best_audio_model.pth',
        output_csv   = 'submission.csv',
        num_classes  = 15,
    )
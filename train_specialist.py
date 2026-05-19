import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import timm
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

TRAIN_CSV_PATH = "train.csv"
AUDIO_DIR = "train"

RARE_CLASSES = [0, 2, 13, 14]

# 120 por CADA clase no rara.
# Si quieres usar TODO el fondo, pon: MAX_BACKGROUND_PER_CLASS = None
MAX_BACKGROUND_PER_CLASS = 120

BACKBONE = "resnet18"
EPOCHS = 60
FOLDS = 5

TARGET_SR = 32000
TARGET_SEC = 19

SEED = 42

# Para RTX 5080 / 16 GB debería aguantar 256 con resnet18.
# Si te da CUDA out of memory, baja a 192 o 128.
BATCH_SIZE_CUDA = 256
BATCH_SIZE_CPU = 64

# En WSL no conviene ir directo a 28.
# 8-12 suele ser más estable. Prueba 8 primero.
WORKERS_CUDA = 8
WORKERS_CPU = 4

LR = 1e-3
WEIGHT_DECAY = 1e-4

USE_RAM_CACHE = True
USE_TORCH_COMPILE = True
SAVE_DIR = "specialist_checkpoints"


# ============================================================
# SETUP
# ============================================================

warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Para velocidad, no determinista estricto.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def setup_performance():
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True

        # Mejora rendimiento en matmul FP32 en GPUs modernas.
        # PyTorch documenta que puede incrementar bastante el rendimiento
        # usando menor precisión interna para matmuls float32.
        torch.set_float32_matmul_precision("high")

        # APIs clásicas, útiles en varias versiones de PyTorch.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


seed_everything(SEED)
setup_performance()
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# GPU DEBUG
# ============================================================

def print_gpu_info():
    print("=" * 60)
    print("  GPU / DEVICE INFO")
    print("=" * 60)
    print(f"Device usado por PyTorch: {DEVICE}")
    print(f"torch version          : {torch.__version__}")
    print(f"torch cuda build      : {torch.version.cuda}")
    print(f"cuda disponible       : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU count             : {torch.cuda.device_count()}")
        print(f"GPU 0                 : {torch.cuda.get_device_name(0)}")
        print(f"Capability            : {torch.cuda.get_device_capability(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"VRAM total            : {total_mem:.2f} GB")
    print("=" * 60)


# ============================================================
# DATAFRAME SPECIALIST
# ============================================================

def build_specialist_df(
    df,
    rare_classes,
    max_background_per_class=120,
    seed=42
):
    rare_classes = list(rare_classes)

    rare_df = df[df["Target"].isin(rare_classes)].copy()
    background_df = df[~df["Target"].isin(rare_classes)].copy()

    if max_background_per_class is not None:
        background_df = (
            background_df
            .groupby("Target", group_keys=False)
            .apply(
                lambda x: x.sample(
                    n=min(len(x), max_background_per_class),
                    random_state=seed
                )
            )
            .reset_index(drop=True)
        )

    specialist_df = pd.concat(
        [rare_df, background_df],
        ignore_index=True
    )

    specialist_df = specialist_df.sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)

    original_to_special = {
        original_label: local_label
        for local_label, original_label in enumerate(rare_classes)
    }

    background_label = len(rare_classes)

    specialist_df["SpecialTarget"] = specialist_df["Target"].apply(
        lambda y: original_to_special.get(int(y), background_label)
    )

    print("\n── Distribución del dataset specialist ──")
    counts = specialist_df["SpecialTarget"].value_counts().sort_index()

    for local_label in range(len(rare_classes) + 1):
        n = int(counts.get(local_label, 0))

        if local_label < len(rare_classes):
            print(
                f"  Label {local_label} "
                f"(Clase original {rare_classes[local_label]}): {n} samples"
            )
        else:
            print(f"  Label {local_label} (Otras / fondo): {n} samples")

    print(f"  Total specialist: {len(specialist_df)} samples")

    return specialist_df


# ============================================================
# CACHE DE WAVEFORMS
# ============================================================

class WaveformCache:
    def __init__(
        self,
        df,
        audio_dir,
        target_sr=32000,
        enabled=True
    ):
        self.enabled = enabled
        self.audio_dir = audio_dir
        self.target_sr = target_sr
        self.cache = {}

        if not self.enabled:
            return

        print("\n⚡ Pre-cargando waveforms en RAM...")
        total_bytes = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Cargando RAM cache"):
            audio_id = str(row["Id"])
            waveform = self._load_from_disk(audio_id)

            waveform = waveform.detach().cpu().float().contiguous()
            self.cache[audio_id] = waveform
            total_bytes += waveform.numel() * waveform.element_size()

        gb = total_bytes / (1024 ** 3)
        print(f"✅ {len(self.cache)} audios en RAM ({gb:.2f} GB aprox.)")

    def _load_from_disk(self, audio_id):
        audio_path = os.path.join(self.audio_dir, f"{audio_id}.wav")

        waveform, sr = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.target_sr
            )
            waveform = resampler(waveform)

        return waveform

    def get(self, audio_id):
        audio_id = str(audio_id)

        if self.enabled and audio_id in self.cache:
            return self.cache[audio_id]

        return self._load_from_disk(audio_id)


# ============================================================
# DATASET
# ============================================================

class KaggleAudioDataset(Dataset):
    def __init__(
        self,
        df,
        audio_dir,
        waveform_cache=None,
        is_train=True,
        target_sr=32000,
        target_sec=19
    ):
        self.df = df.reset_index(drop=True).copy()
        self.audio_dir = audio_dir
        self.waveform_cache = waveform_cache
        self.is_train = is_train
        self.target_sr = target_sr
        self.target_length = target_sr * target_sec

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=1024,
            hop_length=320,
            n_mels=64
        )

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=15
        )

        self.time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=35
        )

    def __len__(self):
        return len(self.df)

    def _get_waveform(self, row):
        audio_id = str(row["Id"])

        if self.waveform_cache is not None:
            waveform = self.waveform_cache.get(audio_id)
        else:
            audio_path = os.path.join(self.audio_dir, f"{audio_id}.wav")
            waveform, sr = torchaudio.load(audio_path)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=self.target_sr
                )
                waveform = resampler(waveform)

        return waveform.float().contiguous()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform = self._get_waveform(row)

        current_len = waveform.shape[1]

        if current_len < self.target_length:
            repeats = (self.target_length // current_len) + 1
            waveform = waveform.repeat(1, repeats)[:, :self.target_length]

        elif current_len > self.target_length:
            if self.is_train:
                start = random.randint(0, current_len - self.target_length)
            else:
                start = (current_len - self.target_length) // 2

            waveform = waveform[:, start:start + self.target_length]

        waveform = waveform.contiguous()

        mel_spec = self.mel_spectrogram(waveform)
        mel_spec = self.amplitude_to_db(mel_spec)

        if self.is_train:
            if random.random() < 0.5:
                mel_spec = self.freq_mask(mel_spec)
            if random.random() < 0.5:
                mel_spec = self.time_mask(mel_spec)

        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
        mel_spec = mel_spec.float().contiguous()

        target_col = "SpecialTarget" if "SpecialTarget" in self.df.columns else "Target"
        target = int(row[target_col])

        return mel_spec, target


# ============================================================
# COLLATE SEGURO
# ============================================================

def fast_audio_collate(batch):
    inputs, targets = zip(*batch)

    # Evita el error:
    # RuntimeError: Trying to resize storage that is not resizable
    inputs = torch.stack(
        [x.contiguous() for x in inputs],
        dim=0
    )

    targets = torch.as_tensor(
        targets,
        dtype=torch.long
    )

    return inputs, targets


# ============================================================
# DATALOADER
# ============================================================

def make_loader(dataset, batch_size, shuffle, workers):
    pin = DEVICE.type == "cuda"

    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": pin,
        "collate_fn": fast_audio_collate,
        "drop_last": False,
    }

    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return DataLoader(**kwargs)


# ============================================================
# MODELO
# ============================================================

class AudioClassifier(nn.Module):
    def __init__(self, num_classes=5, backbone="resnet18"):
        super().__init__()

        self.backbone = timm.create_model(
            backbone,
            pretrained=True,
            in_chans=1,
            num_classes=0,
            global_pool="avg"
        )

        self.head = nn.Linear(
            self.backbone.num_features,
            num_classes
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


# ============================================================
# TRAIN / VALID
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    epoch,
    fold,
    total_epochs
):
    model.train()
    train_loss = 0.0

    use_amp = DEVICE.type == "cuda"

    bar = tqdm(
        loader,
        desc=f"Fold {fold} | Epoch {epoch}/{total_epochs} [Train]",
        leave=False
    )

    for step, (inputs, targets) in enumerate(bar, start=1):
        inputs = inputs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        if DEVICE.type == "cuda":
            inputs = inputs.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp
        ):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        bar.set_postfix(loss=train_loss / step)

    return train_loss / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, num_classes):
    model.eval()

    use_amp = DEVICE.type == "cuda"

    val_preds = []
    val_targets = []

    for inputs, targets in tqdm(loader, desc="Validando", leave=False):
        inputs = inputs.to(DEVICE, non_blocking=True)

        if DEVICE.type == "cuda":
            inputs = inputs.contiguous(memory_format=torch.channels_last)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp
        ):
            outputs = model(inputs)

        preds = torch.argmax(outputs, dim=1)

        val_preds.extend(preds.cpu().numpy())
        val_targets.extend(targets.numpy())

    macro_f1 = f1_score(
        val_targets,
        val_preds,
        average="macro",
        labels=list(range(num_classes)),
        zero_division=0
    )

    cm = confusion_matrix(
        val_targets,
        val_preds,
        labels=list(range(num_classes))
    )

    return macro_f1, cm


# ============================================================
# TRAIN SPECIALIST
# ============================================================

def train_specialist():
    print_gpu_info()

    batch_size = BATCH_SIZE_CUDA if DEVICE.type == "cuda" else BATCH_SIZE_CPU
    workers = WORKERS_CUDA if DEVICE.type == "cuda" else WORKERS_CPU

    print("=" * 60)
    print("  SPECIALIST TRAINER")
    print(f"  Clases raras             : {RARE_CLASSES}")
    print(f"  Salidas del specialist   : {len(RARE_CLASSES) + 1}")
    print(f"  Clase fondo              : otras clases")
    print(f"  Max fondo por clase      : {MAX_BACKGROUND_PER_CLASS}")
    print(f"  Dispositivo              : {DEVICE}")
    print(f"  Backbone                 : {BACKBONE}")
    print(f"  Épocas/fold              : {EPOCHS}")
    print(f"  Batch size               : {batch_size}")
    print(f"  Workers                  : {workers}")
    print(f"  RAM Cache                : {USE_RAM_CACHE}")
    print("=" * 60)

    df = pd.read_csv(TRAIN_CSV_PATH)

    specialist_df = build_specialist_df(
        df=df,
        rare_classes=RARE_CLASSES,
        max_background_per_class=MAX_BACKGROUND_PER_CLASS,
        seed=SEED
    )

    num_classes = len(RARE_CLASSES) + 1

    waveform_cache = WaveformCache(
        specialist_df,
        audio_dir=AUDIO_DIR,
        target_sr=TARGET_SR,
        enabled=USE_RAM_CACHE
    )

    skf = StratifiedKFold(
        n_splits=FOLDS,
        shuffle=True,
        random_state=SEED
    )

    best_global_f1 = 0.0

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(specialist_df, specialist_df["SpecialTarget"]),
        start=1
    ):
        print("\n" + "─" * 50)
        print(f"  FOLD {fold}/{FOLDS}")
        print("─" * 50)

        train_df = specialist_df.iloc[train_idx].reset_index(drop=True)
        val_df = specialist_df.iloc[val_idx].reset_index(drop=True)

        train_dataset = KaggleAudioDataset(
            train_df,
            audio_dir=AUDIO_DIR,
            waveform_cache=waveform_cache,
            is_train=True,
            target_sr=TARGET_SR,
            target_sec=TARGET_SEC
        )

        val_dataset = KaggleAudioDataset(
            val_df,
            audio_dir=AUDIO_DIR,
            waveform_cache=waveform_cache,
            is_train=False,
            target_sr=TARGET_SR,
            target_sec=TARGET_SEC
        )

        train_loader = make_loader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            workers=workers
        )

        val_loader = make_loader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            workers=workers
        )

        model = AudioClassifier(
            num_classes=num_classes,
            backbone=BACKBONE
        ).to(DEVICE)

        if DEVICE.type == "cuda":
            model = model.to(memory_format=torch.channels_last)

        if DEVICE.type == "cuda" and USE_TORCH_COMPILE and hasattr(torch, "compile"):
            try:
                model = torch.compile(model)
                print("✅ torch.compile() activado")
            except Exception as e:
                print(f"⚠️ torch.compile() no se pudo activar: {e}")

        train_counts = np.bincount(
            train_df["SpecialTarget"].values,
            minlength=num_classes
        )

        class_weights = len(train_df) / (
            num_classes * np.maximum(train_counts, 1)
        )

        class_weights = torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=DEVICE
        )

        print("Class counts :", train_counts.tolist())
        print("Class weights:", np.round(class_weights.detach().cpu().numpy(), 4).tolist())

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS
        )

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(DEVICE.type == "cuda")
        )

        best_fold_f1 = 0.0

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                fold=fold,
                total_epochs=EPOCHS
            )

            scheduler.step()

            macro_f1, cm = validate(
                model=model,
                loader=val_loader,
                num_classes=num_classes
            )

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Fold {fold} | Epoch {epoch:02d}/{EPOCHS} "
                f"| Loss: {train_loss:.4f} "
                f"| Macro F1: {macro_f1:.4f} "
                f"| LR: {current_lr:.6f}"
            )

            if macro_f1 > best_fold_f1:
                best_fold_f1 = macro_f1

                save_path = os.path.join(
                    SAVE_DIR,
                    f"best_specialist_fold{fold}.pth"
                )

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "fold": fold,
                    "epoch": epoch,
                    "macro_f1": macro_f1,
                    "rare_classes": RARE_CLASSES,
                    "num_classes": num_classes,
                    "background_label": len(RARE_CLASSES),
                    "backbone": BACKBONE,
                    "target_sr": TARGET_SR,
                    "target_sec": TARGET_SEC,
                    "class_mapping": {
                        int(original): int(local)
                        for local, original in enumerate(RARE_CLASSES)
                    },
                    "background_mapping": "all other original classes"
                }

                torch.save(checkpoint, save_path)
                print(f"🌟 Mejor fold guardado: {save_path}")

            if macro_f1 > best_global_f1:
                best_global_f1 = macro_f1

                save_path = os.path.join(
                    SAVE_DIR,
                    "best_specialist_global.pth"
                )

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "fold": fold,
                    "epoch": epoch,
                    "macro_f1": macro_f1,
                    "rare_classes": RARE_CLASSES,
                    "num_classes": num_classes,
                    "background_label": len(RARE_CLASSES),
                    "backbone": BACKBONE,
                    "target_sr": TARGET_SR,
                    "target_sec": TARGET_SEC,
                    "class_mapping": {
                        int(original): int(local)
                        for local, original in enumerate(RARE_CLASSES)
                    },
                    "background_mapping": "all other original classes"
                }

                torch.save(checkpoint, save_path)
                print(f"🏆 Mejor global guardado: {save_path}")

        print(f"✅ Fold {fold} terminado | Best Fold F1: {best_fold_f1:.4f}")

    print("\n" + "=" * 60)
    print(f"🏁 Mejor Macro F1 global specialist: {best_global_f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    train_specialist()
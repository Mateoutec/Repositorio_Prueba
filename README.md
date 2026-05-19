# Clasificacion de audio

Proyecto de clasificacion de audios usando Python, PyTorch, torchaudio y modelos de `timm`.

El flujo principal convierte archivos `.wav` en mel-spectrogramas y entrena una red convolucional para predecir una clase numerica (`Target`) a partir del identificador de cada audio (`Id`).

## Archivos incluidos

- `main.py`: script principal de entrenamiento. Usa mel-spectrogramas de 19 segundos, aumentos simples y un backbone ResNet de `timm`.
- `codigo_inferencia.py`: script de inferencia para generar predicciones sobre los audios de prueba y guardar un `submission.csv`.
- `train_specialist.py`: entrenamiento especializado con validacion por folds y foco en clases raras.
- `analizar_submission.py`: utilidad para revisar un archivo de submission, validar columnas, duplicados, nulos y distribucion de clases.
- `train.csv`: etiquetas del conjunto de entrenamiento con columnas `Id` y `Target`.
- `submission.csv` y `submission (1).csv`: archivos de predicciones generados para evaluacion.

## Datos y modelo

Los audios de entrenamiento/prueba y el archivo de pesos del modelo no se suben al repositorio porque son archivos grandes:

- `train/`
- `test/`
- `*.wav`
- `*.pth`

Para ejecutar entrenamiento o inferencia se deben colocar localmente las carpetas `train/`, `test/` y, para inferencia, el archivo `best_audio_model.pth`.

## Dependencias principales

- Python
- pandas
- numpy
- torch
- torchaudio
- timm
- scikit-learn
- tqdm

## Uso basico

Entrenamiento:

```bash
python main.py
```

Inferencia:

```bash
python codigo_inferencia.py
```

Analisis de submission:

```bash
python analizar_submission.py --submission submission.csv --train train.csv
```

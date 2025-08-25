import json
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset
import torch
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset, random_split
import torch
import evaluate
import os
import shutil

def read_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)

    df_data = []
    for key, value in data.items():
        value['id_EXIST'] = key
        df_data.append(value)
    
    return pd.DataFrame(df_data)


def predict(text, model, tokenizer, device='cuda'):
    model.to(device)
    model.eval()

    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    return predictions.item()


class TiktokDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df['text'].values
        self.labels = df['label'].values
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=256,  # Reducir el tamaño máximo de secuencia
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred

    # Convertir logits a un tensor
    logits = torch.tensor(logits)

    predictions = torch.argmax(logits, dim=-1)
    accuracy = metric.compute(predictions=predictions, references=labels)
    return {"accuracy": accuracy["accuracy"]}


def main(batch_size, learning_rate, num_train_epochs, save_path):
    if os.path.exists(save_path):
        print("! Carpeta ya ocupada, elige otro nombre para el fichero")

    # Read the training data
    input_folder = "./lab3_materials/dataset_task3_exist2025/"
    training_file = "training.json"
    training_data = read_json_file(os.path.join(input_folder, training_file))
    training_data['id_EXIST'] = pd.to_numeric(training_data['id_EXIST'], errors='coerce')                        # convertir el id a número para poder hacer el join con las captions     
    training_data['label'] = training_data['labels_task3_1'].apply(lambda votings: 1 if votings.count("YES") > votings.count("NO") else 0)  # convertir las votaciones en 0 o 1 (valor que determina si es machista)

    print(" == TRAINING DATA ==")
    print(training_data)
    print("\n")

    # Read the test data
    test_file = "test.json"
    test_data = read_json_file(os.path.join(input_folder, test_file))
    test_data['id_EXIST'] = pd.to_numeric(test_data['id_EXIST'], errors='coerce')                        # convertir el id a número para poder hacer el join con las captions

    # Filter data
    training_data = training_data[training_data['lang'] == 'es']
    test_data = test_data[test_data['lang'] == 'es']

    print(" == TRAINING DATA ==")
    print(training_data)
    print("\n")

    print(" == TEST DATA ==")
    print(test_data)
    print("\n")


    # Load model and tokenizer
    model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=2)
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

    # Create dataset
    dataset = TiktokDataset(training_data, tokenizer)
    print(len(dataset))
    
    # Use 100% of the dataset
    train_dataset = dataset

    

    # Liberar memoria no utilizada antes de entrenar
    torch.cuda.empty_cache()

    # Comprobar si el path ya existe y eliminarlo si es necesario
    results_dir = './results'
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    training_args = TrainingArguments(
        output_dir=results_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        gradient_accumulation_steps=2,  # Acumular gradientes para reducir el uso de memoria
        fp16=True,  # Usar entrenamiento de precisión mixta
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,  # Usar el mismo dataset para evaluación
        compute_metrics=compute_metrics  # Añadir la función de cálculo de métricas
    )

    trainer.train()
    eval_result = trainer.evaluate()
    print(f"Evaluation results: {eval_result}")


    # Crear el fichero de predicciones
    predictions = []
    for id, text in zip(test_data['id_EXIST'], test_data['text']):
        pred = predict(text, model, tokenizer)
        predictions.append({
            "test_case": "EXIST2025",
            "id":str(id),
            "value": "NO" if pred == 0 else "YES"
        })
    
    with open(save_path, 'w') as archivo_json:
        json.dump(predictions, archivo_json, indent=4)


if __name__ == '__main__':
    # Define los parámetros del usuario
    batch_size = 8
    learning_rate = 3.4760242613192156e-05
    num_train_epochs = 5

    save_path = "./lab3_materials/golds_task3_exist2025/SMG_OnlyES-XMLRoberta.json"

    main(batch_size, learning_rate, num_train_epochs, save_path)
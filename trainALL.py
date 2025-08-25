import json
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset
import torch
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset, random_split
import torch
import evaluate
import optuna



def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred

    # Convertir logits a un tensor
    logits = torch.tensor(logits)

    predictions = torch.argmax(logits, dim=-1)
    accuracy = metric.compute(predictions=predictions, references=labels)
    return {"accuracy": accuracy["accuracy"]}

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
    
def main():

    # Cargar datos de entreamiento
    with open('./lab3_materials/dataset_task3_exist2025/training.json', 'r', encoding='utf-8') as f:
        trainingJSON = json.load(f)

    # La predicción de la etiqueta debe de ser lo que dice la mayoría
    for element in trainingJSON.values():
        element['label'] = Counter(element["labels_task3_1"])["YES"] > Counter(element["labels_task3_1"])["NO"]

    training = pd.DataFrame.from_dict(trainingJSON, orient='index')

    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    dataset = TiktokDataset(training, tokenizer)
   
    train_ratio = 0.75
    train_size = int(train_ratio * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    # Liberar memoria no utilizada antes de entrenar
    torch.cuda.empty_cache()
    
    def objective(trial):
        model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=2)
    

        # Sugerir hiperparámetros
        batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])  # Ajustar tamaños de batch
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True)
        num_train_epochs = trial.suggest_int('num_train_epochs', 1, 5)

        training_args = TrainingArguments(
            output_dir='./results',
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
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics  # Añadir la función de cálculo de métricas
        )

        trainer.train()
        eval_result = trainer.evaluate()
        return eval_result['eval_loss']
    
    # Run the study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, timeout=60*60)   # Only 1 hour

    # Show the best hyperparameters
    print("Best hyperparameters: ", study.best_params)
    print("Best trial: ", study.best_trial)
    print("Best trial value: ", study.best_trial.value)

if __name__ == '__main__':
    main()
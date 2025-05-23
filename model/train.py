from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments
import json
import os
from model.model_utils import save_model

def train_model(train_dataset, eval_dataset, label_list, epochs=3):
    model_name = "dslim/bert-base-NER"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label={i: label for i, label in enumerate(label_list)},
        label2id={label: i for i, label in enumerate(label_list)}
    )
    
    training_args = TrainingArguments(
        output_dir="./model/trained_model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train and save
    trainer.train()
    save_model(trainer, tokenizer)
    
    # Save metrics
    metrics = {
        'accuracy': [{'epoch': i+1, 'value': x['eval_accuracy']} 
                    for i, x in enumerate(trainer.state.log_history[1::2])],
        'loss': [{'epoch': i+1, 'value': x['eval_loss']} 
                for i, x in enumerate(trainer.state.log_history[1::2])]
    }
    
    os.makedirs("data", exist_ok=True)
    with open("data/training_results.json", "w") as f:
        json.dump(metrics, f)
    
    return trainer
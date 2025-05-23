import json
import csv
import random
from faker import Faker
from datetime import datetime, timedelta
import numpy as np
import os  # Add this import

fake = Faker()

# Configuration
NUM_SAMPLES = 500
NUM_DOCUMENTS = 100
TRAINING_EPOCHS = 5

def generate_pii_samples(num_samples):
    """Generate synthetic text samples with PII"""
    samples = []
    pii_types = ["PERSON", "EMAIL", "PHONE", "DATE", "ORG", "LOCATION"]
    
    templates = [
        "{name} can be reached at {email} or {phone}",
        "Our meeting with {name} from {org} is scheduled for {date}",
        "Please contact {name} at {phone} regarding the project",
        "{org} headquarters located in {location}",
        "Patient: {name}, Appointment: {date}, Doctor: {name}",
        "Credit card application from {name} ({email})",
        "Flight booking for {name} on {date} to {location}"
    ]
    
    for _ in range(num_samples):
        pii_data = {
            "name": fake.name(),
            "email": fake.email(),
            "phone": fake.phone_number(),
            "date": fake.date_this_year().strftime("%Y-%m-%d"),
            "org": fake.company(),
            "location": fake.city()
        }
        
        template = random.choice(templates)
        text = template.format(**pii_data)
        
        samples.append({
            "text": text,
            "label": "PII" if random.random() > 0.3 else "NON_PII"
        })
    
    return samples

def generate_documents(num_docs):
    """Generate document content for RAG"""
    domains = [
        "Healthcare", "Finance", "Technology", 
        "Legal", "Education", "Government"
    ]
    
    documents = []
    for _ in range(num_docs):
        domain = random.choice(domains)
        doc = {
            "id": fake.uuid4(),
            "title": fake.sentence(),
            "text": fake.text(max_nb_chars=random.randint(200, 500)),
            "domain": domain,
            "date": fake.date_this_decade().strftime("%Y-%m-%d"),
            "author": fake.name(),
            "keywords": ", ".join(fake.words(nb=random.randint(3, 6)))
        }
        documents.append(doc)
    
    return documents

def generate_training_metrics(epochs):
    """Generate realistic training metrics"""
    metrics = {
        "accuracy": [],
        "loss": [],
        "precision": [],
        "recall": []
    }
    
    base_acc = random.uniform(0.6, 0.75)
    base_loss = random.uniform(0.8, 1.2)
    
    for epoch in range(1, epochs+1):
        # Simulate typical training progress
        acc = base_acc + (1 - base_acc) * (epoch/epochs) * random.uniform(0.8, 1.0)
        loss = base_loss * (1 - (epoch/epochs)) * random.uniform(0.8, 1.0)
        
        metrics["accuracy"].append({
            "epoch": epoch,
            "value": round(acc, 4)
        })
        metrics["loss"].append({
            "epoch": epoch,
            "value": round(loss, 4)
        })
        metrics["precision"].append({
            "epoch": epoch,
            "value": round(acc * random.uniform(0.9, 1.0), 4)
        })
        metrics["recall"].append({
            "epoch": epoch,
            "value": round(acc * random.uniform(0.9, 1.0), 4)
        })
    
    return metrics

def generate_entity_definitions():
    """Generate entity type definitions"""
    return {
        "PERSON": "Full names of individuals",
        "EMAIL": "Email addresses",
        "PHONE": "Phone numbers including international formats",
        "DATE": "Dates in any format (YYYY-MM-DD, DD/MM/YY, etc.)",
        "ORG": "Companies, institutions, organizations",
        "LOCATION": "Cities, countries, addresses",
        "ID": "Identification numbers (SSN, passport, etc.)"
    }

def save_data():
    """Generate and save all data files"""
    print("Generating synthetic data...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate PII samples
    pii_samples = generate_pii_samples(NUM_SAMPLES)
    with open('data/synthetic_data.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(pii_samples)
    
    # Generate documents for RAG
    documents = generate_documents(NUM_DOCUMENTS)
    with open('data/documents.json', 'w') as f:
        json.dump(documents, f, indent=2)
    
    # Generate training metrics
    metrics = generate_training_metrics(TRAINING_EPOCHS)
    with open('data/training_results.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate entity definitions
    entities = generate_entity_definitions()
    with open('data/entity_definitions.json', 'w') as f:
        json.dump(entities, f, indent=2)
    
    print(f"Data generation complete. Created:")
    print(f"- {NUM_SAMPLES} PII samples (synthetic_data.csv)")
    print(f"- {NUM_DOCUMENTS} documents (documents.json)")
    print(f"- Training metrics for {TRAINING_EPOCHS} epochs (training_results.json)")
    print(f"- Entity definitions (entity_definitions.json)")

if __name__ == "__main__":
    save_data()
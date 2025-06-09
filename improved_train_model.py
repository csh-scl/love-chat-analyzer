import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

class ChatDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts.tolist()  # Convert pandas Series to list
        self.labels = labels.tolist()  # Convert pandas Series to list
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 토큰화
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_model(model, train_loader, val_loader, device, epochs=10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    best_accuracy = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            # 정확도 계산
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        # 검증
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)
                
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        train_accuracy = correct / total
        val_accuracy = val_correct / val_total
        
        print(f'Epoch {epoch + 1}:')
        print(f'Training Loss: {total_loss / len(train_loader):.4f}')
        print(f'Training Accuracy: {train_accuracy:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.4f}')

        # 모델 성능 저장
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'model/best_model.pt')

def main():
    # 1. 데이터 로드
    df = pd.read_csv("data/sample_chat.csv", encoding='utf-8')
    
    # 2. 라벨 인코딩
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    
    # 3. 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        df['sentence'], 
        df['label_encoded'],
        test_size=0.2,
        random_state=42,
        stratify=df['label_encoded']
    )
    
    # 4. KoBERT 모델 및 토크나이저 초기화
    tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
    model = BertForSequenceClassification.from_pretrained(
        'monologg/kobert',
        num_labels=len(le.classes_)
    )
    
    # 5. 데이터셋 및 데이터로더 생성
    train_dataset = ChatDataset(X_train, y_train, tokenizer)
    test_dataset = ChatDataset(X_test, y_test, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # 배치 사이즈 줄임
    test_loader = DataLoader(test_dataset, batch_size=8)  # 배치 사이즈 줄임
    
    # 6. 학습 실행
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    train_model(model, train_loader, test_loader, device)
    
    # 7. 모델 및 관련 객체 저장
    os.makedirs("model", exist_ok=True)
    joblib.dump(le, "model/label_encoder.pkl")
    joblib.dump(tokenizer, "model/tokenizer.pkl")

if __name__ == "__main__":
    main() 
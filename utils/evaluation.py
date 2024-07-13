import torch

def evaluate_model(model, dataloader):
    model.eval()
    corrects = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            text_features = data['text']
            image_features = data['image']
            labels = data['label']

            outputs = model(text_features, image_features)
            predictions = (outputs > 0.5).float()
            corrects += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = corrects / total
    print(f'Accuracy: {accuracy:.4f}')
    return accuracy

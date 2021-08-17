import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from sentiment_classifier import MixText

LABELS = ['mix', 'negative', 'neutral', 'positive']


class Model:

    def __init__(self):

        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

        classifier = MixText(mix_option=True, num_classes=4)
        classifier.load_state_dict(torch.load(
            'best_model.pt', map_location=self.device))
        classifier.eval()
        self.classifier = classifier.to(self.device)

    def predict(self, text):
        encoded_text = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=50,
            return_token_type_ids=False,
            pad_to_max_length='right',
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoded_text["input_ids"].to(self.device)
        attention_mask = encoded_text["attention_mask"].to(self.device)

        with torch.no_grad():
            output = self.classifier(input_ids, attention_mask)
            probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.cpu().item()
        confidence = confidence.cpu().item()
        probabilities = probabilities.flatten().cpu().numpy().tolist()
        return {
            "class": LABELS[predicted_class],
            "confidence": confidence,
        }


model = Model()


def get_model():
    return model

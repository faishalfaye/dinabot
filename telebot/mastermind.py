import torch
import torch.nn.functional as F
from transformers import BertTokenizer, AlbertConfig, AlbertForSequenceClassification

INDOBERT_LITE_LARGE_P1 = "indobenchmark/indobert-lite-large-p1"
SAVED_MODEL_DICT_PATH = "telebot/StateDict_IndobertLiteLarge_p1.pth"

tokenizer = BertTokenizer.from_pretrained(INDOBERT_LITE_LARGE_P1)
config = AlbertConfig.from_pretrained(INDOBERT_LITE_LARGE_P1)
model = AlbertForSequenceClassification.from_pretrained(INDOBERT_LITE_LARGE_P1, config=config)

model.load_state_dict(torch.load(SAVED_MODEL_DICT_PATH, map_location=torch.device('cpu')))
model.eval()

def get_response(text):
    emot_dict = {0: 'sedih', 1: 'marah', 2: 'cinta', 3: 'takut', 4: 'senang'}
    subwords = tokenizer.encode(text)
    subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

    logits = model(subwords)[0]
    label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()

    return f"Teks: {text} | Emosi: {emot_dict[label]}"
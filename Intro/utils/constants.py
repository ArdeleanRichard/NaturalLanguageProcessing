import torch

imdb_train_pos_folder = '../data/aclImdb/train/pos/*.txt'
imdb_train_neg_folder = '../data/aclImdb/train/neg/*.txt'
imdb_test_pos_folder  = '../data/aclImdb/test/pos/*.txt'
imdb_test_neg_folder  = '../data/aclImdb/test/neg/*.txt'

# data from: https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset
# embeddings from: https://nlp.stanford.edu/data/glove.6B.zip
agnews_train_file = '../data/ag_news/train.csv'
agnews_test_file  = '../data/ag_news/test.csv'
agnews_labels = ["World", "Sports", "Business", "Sci-Tech"]
agnews_column_names = ['class index', 'title', 'description']

agnews_embeddings_file = "../data/embeddings/glove.6B.300d.txt"


# data from: https://github.com/UniversalDependencies/UD_Spanish-AnCora
# embeddings from: https://github.com/dccuchile/spanish-word-embeddings#glove-embeddings-from-sbwc
ancora_train_file = '../data/ancora/es_ancora-ud-train.conllu'
ancora_val_file   = '../data/ancora/es_ancora-ud-dev.conllu'
ancora_test_file  = '../data/ancora/es_ancora-ud-test.conllu'
ancora_embeddings_file = '../data/embeddings/glove-sbwc.i25.vec'


# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
use_gpu = True
device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
print(f'device: {device.type}')

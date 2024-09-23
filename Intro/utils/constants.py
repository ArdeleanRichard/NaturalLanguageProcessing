import torch

pos_train_data = '../data/aclImdb/train/pos/*.txt'
neg_train_data = '../data/aclImdb/train/neg/*.txt'
pos_test_data  = '../data/aclImdb/test/pos/*.txt'
neg_test_data  = '../data/aclImdb/test/neg/*.txt'


train_file = '../data/ag_news/train.csv'
test_file  = '../data/ag_news/test.csv'
labels = ["World", "Sports", "Business", "Sci-Tech"]
column_names = ['class index', 'title', 'description']

use_gpu = True
device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
print(f'device: {device.type}')
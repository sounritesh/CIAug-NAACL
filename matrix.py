import os
import shutil
import numpy as np
import pandas as pd
!pip install "transformers==2.5.1"

pip install sentence-transformers

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')

pip install datasets

# Load the dataframe
df = pd.read_csv("PATH")

text = df['Text']

sentence_embeddings = model.encode(text)

sentence_embeddings = np.array(sentence_embeddings)

np.save("sentence_embeddings.npy",sentence_embeddings)

pip install git+https://github.com/geoopt/geoopt.git

from geoopt.manifolds.stereographic import math as math1

import numpy as np
from tqdm import tqdm
import torch
matrix = np.zeros((len(df),len(df)))

sentence_embeddings = math1.expmap0(torch.tensor(sentence_embeddings),k=torch.tensor([1.]))

for i in tqdm(range(len(sentence_embeddings))):
  for j in range(i,len(sentence_embeddings)):
    matrix[i][j] = math1.dist(torch.tensor(sentence_embeddings[i]),torch.tensor(sentence_embeddings[j]),k=torch.tensor(1.))
    matrix[j][i] = matrix[i][j]

# Use this matrix for hyperbolic training

matrix = np.array(matrix)


"""Euclidean"""

matrix = np.zeros((len(df),len(df)))

for i in tqdm(range(len(sentence_embeddings))):
  for j in range(i,len(sentence_embeddings)):
    matrix[i][j] = np.linalg.norm(sentence_embeddings[i]-sentence_embeddings[j])
    matrix[j][i] = matrix[i][j]

# Use this matrix for Euclidean training
matrix = np.array(matrix)




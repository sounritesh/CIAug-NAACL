# CIAug: Equipping Interpolative Augmentation with Curriculum Learning

Official PyTorch implementation of [CIAug](https://aclanthology.org/2022.naacl-main.127/) (NAACL 2022 Main conference).

![CIAug](https://i.postimg.cc/wBJXLJtR/Screenshot-2022-07-31-at-10-03-43-AM.png)

### Instructions for training the model:

- Install the required packages using requirements.txt
- Replace the Dataset PATH to where it is stored locally.
- Replace the Probability Matrix to the place where it is locally stored.
- Set the Curriculum threshold values.
- Change bert-base-uncased to bert-base-multilingual-uncased incase running for languages other than english.
- Replace the num_label with the number of labels in the dataset.
- Number of training samples in the dataframe.

We have used some standard [GLUE Datasets](https://huggingface.co/datasets/glue), [TREC Dataset](https://huggingface.co/datasets/trec), [CoNLL Dataset](https://huggingface.co/datasets/conll2003) and some other standard datasets in [Turkish](https://archive.ics.uci.edu/ml/datasets/TTC-3600%3A+Benchmark+dataset+for+Turkish+text+categorization) and Arabic that could be downloaded using the transformer dataset or their official webpage.

_For calculating the Matrix, run the code given in matrix.py._

The code is well documented for further explanation.

### Cite using:

```bibtex
@inproceedings{sawhney-etal-2022-ciaug,
    title = "{CIA}ug: Equipping Interpolative Augmentation with Curriculum Learning",
    author = "Sawhney, Ramit  and
      Soun, Ritesh  and
      Pandit, Shrey  and
      Thakkar, Megh  and
      Malaviya, Sarvagya  and
      Pinter, Yuval",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.127",
    pages = "1758--1764",
    abstract = "Interpolative data augmentation has proven to be effective for NLP tasks. Despite its merits, the sample selection process in mixup is random, which might make it difficult for the model to generalize better and converge faster. We propose CIAug, a novel curriculum-based learning method that builds upon mixup. It leverages the relative position of samples in hyperbolic embedding space as a complexity measure to gradually mix up increasingly difficult and diverse samples along training. CIAug achieves state-of-the-art results over existing interpolative augmentation methods on 10 benchmark datasets across 4 languages in text classification and named-entity recognition tasks. It also converges and achieves benchmark F1 scores 3 times faster. We empirically analyze the various components of CIAug, and evaluate its robustness against adversarial attacks.",
}
```

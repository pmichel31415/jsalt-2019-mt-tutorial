# MT Tutorial for the JSALT 2019 Summer School

## Course Materials

## Lab session

### Setup

1. Install python 3.6 (recommended distrib: [miniconda](https://docs.conda.io/en/latest/miniconda.html))
2. Install pytorch: https://pytorch.org/get-started/locally/#start-locally
3. Download data (preprocessed IWSLT16 fr-en): https://drive.google.com/file/d/1UaWMQRFaVDfyimw-A29bkffh-hmFBedh/view?usp=sharing
4. Download pretrained model: TODO

### Subwords

```bash
echo "J'ai donc fait le tour pour essayer les autres portes et fenêtres." |
    python lab/subwords.py segment --model data/subwords.model
```

```
▁J ' ai ▁donc ▁fait ▁le ▁tour ▁pour ▁essayer ▁les ▁autres ▁portes ▁et ▁fenêtres .
```

### Sampling from a trained model

```bash
echo "▁J ' ai ▁donc ▁fait ▁le ▁tour ▁pour ▁essayer ▁les ▁autres ▁portes ▁et ▁fenêtres ." |
    python lab/translate.py --model-file model.pt --sampling
```

```
So I went all the way to try out the other doors and windows.
```

Try out another seed


```bash
echo "▁J ' ai ▁donc ▁fait ▁le ▁tour ▁pour ▁essayer ▁les ▁autres ▁portes ▁et ▁fenêtres ." |
    python lab/translate.py --model-file model.pt --sampling --seed 123456
```

```
Thus, I went around in order to try to other doors and windows.
```

### Greedy decoding

TODO

## Organizers

### Jia Xu

<img align="left" height="100" src="images/jia_xu_pic.png" alt="Jia pic"/>

[Jia bio]

### Paul Michel

<img align="left" height="100" src="images/paul_michel_pic.jpg" alt="Paul pic"/>

[Paul bio]

### Abdul Rafae Khan

<img align="left" height="100" src="images/abdul_rafae_khan_pic.jpg" alt="Abdul pic"/>

[Abdul bio]
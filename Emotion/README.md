## Pytorch: Emotion Recognition

A model of **image classification** based on **Efficient Net B4** architecture using pytorch. Here, i use a **custom dataset** about emotion of people .

### Workflow
**1.** Build model

**2.** Clone raw dataset

**3.** Use **SR CNN** to scale size dataset **48px -> 384**

**4.** Data processing

**5.** Train model and repeat with difference hyperparameters

### Dependencies

* Numpy
* Python3
* Pytorch

```python
pip instal numpy         # numpy library
pip instal pytorch       # pytorch library
pip install torchsummary # summary
pip install torchvision  # pytorch for vision
```

**NB**: Update the libraries to their latest versions before training.

### How to run
----------------------------------------
⬇️⬇️**Download** train dataset on Kaggle: [Raw data with 48px](https://www.kaggle.com/datasets/vanduc0xff/emotion-raw-dataset-48x48) or [Data after resize 384px](https://www.kaggle.com/datasets/vanduc0xff/emotion-dataset-384x384)

⬇️⬇️**Download** pretrained of **SRCNN** model and **Emotion Recognition** model: [Model](https://drive.google.com/drive/folders/1_M6rplng9CWNEFLZeA_6dvohKGBTnEc_?usp=sharing)


Run the following **scripts** for training and/or testing

```python
python train.py # For training the model 
```
------------------------------------------

### Training results

|              | Accuracy | Cross Entropy Loss | Training Epochs | Training Mode | Size  |
|--------------|----------|--------------------|-----------------|-----|-------|
| **Training** | 75.63%   | 0.823              | 15              |  scratch | 110MB |
| **Testing**  | 63.38%   | 1.078              | 15              |  scratch | 110MB |

**Batch size**: 32, **GPU**: RTX 3050 4G

### Training graphs

**Model:** 

Finetuning the model.
![Screenshot](results/emotion.jpg)

### Observations

1. The **MODEL** based on Efficient Net B4 so have 17M params has a medium size i.e **110 MB**.
2. Adjusting parameters like **batch size, number of workers, pin_memory, ** etc. may help you **reduce training time**, especially if you have a big dataset and a high-end machine(hardware).
3. Adjusting parameters like **learning rate, weight decay** etc maybe can help you **improve** model.
### Todo

1. Experiments with different **learning-rate and optimizers**.
2. **Converting and optimizing** pytorch models for **mobile** deployment.

### Authors

Van Duc
 
### Acknowledgments
* "https://arxiv.org/pdf/1905.11946.pdf"
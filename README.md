[![MasterHead](https://mentalhealth.ie/wp-content/uploads/2021/02/scale-of-emotions-with-emojis.jpg)](https://www.linkedin.com/in/%C4%91%E1%BB%A9c-v%C5%A9-6772a6248)
<h1 align="center">Recognition: Gender👫 - Age👶🧓  Emotion😁😲😨😠😢</h1>

------------------------------------------
## Pytorch: Gender Age and Emotion Recognition

A model merge of **3 small model** based on **Efficient Net B4**, **Efficient Net B0** and **SR CNN**  architecture using pytorch. Here, i use 2 **custom dataset**: **gender and age** dataset and **emotion** dataset .

### Workflow
**I.** Build **model** for **Gender and Age recognition task** by a **Multi-task** model: [Result](https://github.com/vvduc1803/Gender-Age-Emotion-Recognition/blob/main/Gender_Age/README.md)

**II.** Build **model** for **Emotion recognition task**: [Result](https://github.com/vvduc1803/Gender-Age-Emotion-Recognition/blob/main/Emotion/README.md)

**III.** Merge them and use **Mediapipe, Opencv library** for recognition

### Dependencies

* Mediapipe
* Opencv
* Numpy
* Python3
* Pytorch

```python
pip install mediapipe    # mediapipe library
pip install opencv       # opencv library
pip instal numpy         # numpy library
pip instal pytorch       # pytorch library
pip install torchsummary # summary
pip install torchvision  # pytorch for vision
```

**NB**: Update the libraries to their latest versions before training.

### How to run
----------------------------------------
⬇️⬇️**Download** and extract all my train dataset on Kaggle: [Gender and Age Dataset](https://www.kaggle.com/datasets/vanduc0xff)

⬇️⬇️**Download** pretrained model: [Model](https://drive.google.com/drive/folders/1_M6rplng9CWNEFLZeA_6dvohKGBTnEc_?usp=sharing)


Run the following **scripts** for training and/or testing

```python
python train.py # For training the model 
```
------------------------------------------
<img src="https://www.docker.com/wp-content/uploads/2022/03/Moby-logo.png" alt="docker" width="35" height="30"/><img src="https://www.docker.com/wp-content/uploads/2022/03/Moby-logo.png" alt="docker" width="35" height="30"/>**Docker Image**
-----------
Run the following **scripts** for visual result of model:

**1.**
Download **[Docker](https://www.docker.com/)**

Open **CMD**

**2.**
Download my image

```python
docker pull vvduc1803/gender_age_emotion:latest                                  # Pull image
```

**3.**
Copy and paste
```python
docker run -it -d --name gender_age_emotion vvduc1803/gender_age_emotion  # Run container
```
**4.**
Copy and paste
```python
docker run gender_age_emotion                                             # Run visual result
```
------------------------------------------
### Sample outputs

Recognition results

![Screenshot](results/1.jpg)
![Screenshot](results/2.jpg)
![Screenshot](results/3.jpg)
![Screenshot](results/4.jpg)


---------------------------------------------

### Todo

1. Experiments with different **learning-rate and optimizers**.
2. **Converting and optimizing** pytorch models for **mobile** deployment.

### Authors

Van Duc


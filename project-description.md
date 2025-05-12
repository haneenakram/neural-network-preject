# Deep Learning Project: Fine-Grained Fruit Quality Assessment

The goal of this project is to train deep learning models capable of performing fine-grained classification of food quality based on image features. The model will **classify** fruit items, such as bananas and tomatoes, into specific quality categories **(e.g., banana overripe, banana rotten, banana ripe, tomato green, tomato half-ripened)** using a dataset of **labeled food images**. 

By leveraging deep learning techniques, the model will be trained to automatically **assess and categorize the quality and ripeness stages** of different food items based on their visual characteristics.

[Competition Link (You must use this link to be able to join the competition):](https://www.kaggle.com/t/4a6520de4926486784ef2e5005d091ee)

Must solve unbalanced data problem

Dataset links can be found on kaggle

---

## Competition Rules: 
- **One account per team participant**: You cannot sign up to Kaggle from multiple accounts and therefore you can submit only from one account.
- The team name on Kaggle should be the same Team ID as the one given to you (CHP_5)
- No private sharing outside teams : Any form of cheating or illegal behavior will lead to being disqualified and losing the project grades.
- Keras.applications or any similar API is NOT ALLOWED
- Submission Limits
    - You may **submit a maximum of 5 entries per day.**
    - You may select up to **2 final submissions for judging.**

## Competition Timeline
**Start Date**: 27/4/2025
**End Date Deadline**: sat 17/5 , The day before the practical exam day.

---
## Deep learning competition
You must register and submit your results on Kaggle website.

1. You will be given a small test sample on the practical exam day, 
2. so each team needs to save the weights of the network used during training
3. create a script that loads the weights, generates a csv file containing the predicted labels for the test samples.

**Note**: that on the practical exam day, the test script needs to run on your laptop or if you want to use Colab/Kaggle for the test, make sure to have a very good internet connection as it is not guaranteed that it would be available in the lab.

- If you **trained the models using a notebook**, you must deliver the notebook with the output cell saved displaying the training logs. 
- If you **trained the model using IDE (i.e Pycharm)**. You must deliver screenshots of the training process

## The evaluation of the project will be on the following items:-

- Building multiple appropriate models **(three models: at least one of them must be transformer)** and **understanding** each part of it.
    - must be significantly different from each others
    - NOT completely built in model but also not from scratch (ay architecture mn satr wahed mamno3)
    - make the architecture obvious  (layers wad7a kol layer msta5demyn fyha eh)
- Applying the appropriate **data preparation** steps
- The achieved accuracy on Kaggle 
- Deliver a detailed **report** of what architectures you used, all trials and your conclusion from 

---
### Datasets

[train dataseet](https://drive.google.com/file/d/1AQbJKptQI2Y1C-xjZRLYQN88R-VtQ79w/view?usp=sharing)

[test dataset](https://drive.usercontent.google.com/download?id=1AQbJKptQI2Y1C-xjZRLYQN88R-VtQ79w&export=download&authuser=0)

Note: it is too large dont download it, we will use kaggle

to use data in kaggle, open a new notebook
 
---
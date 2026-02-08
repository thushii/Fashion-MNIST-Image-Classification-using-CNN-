# Fashion-MNIST Image Classification using CNN

I built this project to learn how Convolutional Neural Networks actually work on image data. The idea was to take the Fashion-MNIST dataset, train a model to recognize different clothing items, and then look at where the model performs well and where it gets confused.

## Dataset

The data comes from the Kaggle Fashion-MNIST collection:  
https://www.kaggle.com/datasets/zalando-research/fashionmnist

I used the two CSV files:
- fashion-mnist_train.csv  
- fashion-mnist_test.csv  

The dataset is not uploaded to this repository because of GitHub file size limits.  
If you want to run the code, just download the dataset from the link above and place both CSV files in the same folder as train.py.

## What I Did

- Loaded the CSV data with pandas  
- Normalized pixel values between 0 and 1  
- Reshaped each row into 28x28 grayscale images  
- Split the data into training and validation sets  
- Added augmentation like small rotations and shifts to make the model more robust  
- Built a CNN using Conv2D, BatchNorm, LeakyReLU, pooling and dropout  
- Trained with Adam optimizer and early stopping  
- Evaluated using accuracy, confusion matrix and classification report  
- Visualized correct and incorrect predictions to understand mistakes

## Results

The final model reached about **93% accuracy on the test set**.

From the confusion matrix I noticed that most errors happen between:
- Shirt and Pullover  
- Coat and Shirt  

These items look very similar in low-resolution grayscale images, so the confusion made sense when I checked the actual predictions.

## Tools Used

- Python  
- TensorFlow / Keras  
- Pandas and NumPy  
- Matplotlib and Seaborn  
- Scikit-learn for evaluation

## How to Run

1. Download the dataset from Kaggle and place the CSV files in the project folder.

2. Install dependencies  
pip install -r requirements.txt  

3. Run the training script  
python train.py  

## What I Learned

This project helped me understand:
- how convolution layers learn visual patterns  
- why augmentation improves generalization  
- how to read a confusion matrix instead of only looking at accuracy  
- that some classes are naturally hard even for a good model

Next I want to try transfer learning to see if a pre-trained model can reduce the shirt vs pullover confusion.

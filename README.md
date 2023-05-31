# Facial Expressions Recognition

This project was created as a learning experience for working with neural networks, particularly for recognizing facial expressions. As my first project with neural networks, it allowed me to explore the fundamentals of deep learning and gain practical knowledge in this field.

## Dataset

The dataset used for training and testing the facial expressions recognition model is sourced from Kaggle. It can be found in the repository and consists of labeled images of faces with various expressions.
https://www.kaggle.com/datasets/dataturks/face-detection-in-images

## Installation

To run the project, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies, including TensorFlow.

`pip install tensorflow`


## Usage

1. Ensure that the dataset is organized into separate folders, each corresponding to a specific facial expression.
2. Run the main script `neuralNetwork.py` to train the neural network and evaluate its performance on the testing data.

`python neuralNetwork.py`


## Training Process

The neural network architecture used in this project includes convolutional layers followed by max-pooling, fully connected layers, and a softmax output layer. During training, the data is augmented using techniques such as rotation, shifting, zooming, and flipping to increase the model's robustness.

Hyperparameters like image dimensions, batch size, and number of epochs were chosen to balance performance and training time.

## Model Evaluation

The model's performance is evaluated using metrics such as loss and accuracy. The evaluation is performed on a separate testing dataset that was not used during training. The final results indicate the model's accuracy in recognizing facial expressions.

## Model Visualization

Future versions of the project will include a feature to visualize the learned features or activations of the model's layers using Grad-CAM. This visualization provides insights into what the model focuses on when recognizing facial expressions.

## Results

Through the training and testing process, the model achieved aproximately 54% accuracy in facial expression recognition. As my first project with neural networks, I encountered challenges and learned valuable lessons along the way.

## Future Improvements

In future iterations of this project, I plan to explore different neural network architectures, experiment with advanced data augmentation techniques, and potentially leverage transfer learning to further enhance the model's performance.

## Conclusion

Working on this FacialExpressionsRecognition project as my first venture into neural networks has been an exciting learning experience. It has allowed me to gain practical knowledge in deep learning and develop a foundation for future projects in this field.


# One Shot Learning Algorithm for Gait Recognition
<!-- > Comment this and the link once done:
> Reference: https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax -->
> ## Gait Recognition
Everyone's walking style (gait) is unique, and it has been shown that both humans and computers are very good at recognizing known gait patterns. This can be used as a biometric form that can be utilized to effectively recognize a person by his/her walking style. Over the decades, gait analysis has been successfully used in different domains, including biometrics and posture analysis for healthcare applications. It has also been used in human psychology where gait analysis using point lights employed for recognition of emotional patterns. The same idea was extended and ultimately resulted in the development of gait signatures through which the identification of individuals can be performed.
> ## One Shot Learning
We propose on removing the need for a large dataset for Training in other models by using One Shot Learning Algorithm. The algorithm, as the name suggests, uses only one sample to familiarize itself with how a person’s gait should look and then compares it with the testing image to recognize whose gait it is. Thus, it requires only one GEI (Gait Energy Image) for recognition. 
One-shot learning is an object categorization problem, found mostly in computer vision. Whereas most machine learning based object categorization algorithms require training on hundreds or thousands of samples/images and very large datasets, one-shot learning aims to learn information about object categories from one, or only a few, training samples/images.
>## Siamese Network
A Siamese network is a class of neural networks that contains one or more identical networks. We feed a pair of inputs to these networks. Each network computes the features of one input. And, then the similarity of features is computed using their difference or the dot product. Remember, both networks have same the parameters and weights. If not, then they are not Siamese.
By doing this, we have converted the classification problem to a similarity problem. We are training the network to minimize the distance between samples of the same class and increasing the inter-class distance. There are multiple kinds of similarity functions through which the Siamese network can be trained like Contrastive loss, triplet loss, and circle loss.

>## Code
One Shot Learning algorithm uses Siamese network built using a Convolutional Neural Network. The CNN is used because of it's efficiency in obtaining good Feature Vector which is essential for creating a good Siamese network and in turn a good One Shot Learning Algorithm.
```
vgg_feature = model.predict(training_img_data)
```
This line of code generates and stores a feature vector generated from a CNN Model. Two _vgg_feature_ variables are used in the program. One would be created for the image that needs to be compared with other images .i.e an Anchor Image in a Siamese Network. This variable would be stored in the variable vgg_feature_1. This variable would remain constant throughout the search. vgg_feature_2, on the other hand, would be varying for each image and a distance between vgg_feature_1 and vgg_feature_2 is calculated using the general formula for finding distance between the vectors. That is:
```
val = sqrt(sum( (vgg_feature_1 - vgg_feature_2)**2))
```
In the end, we find the minimum distance that is generated from the above equation (between anchor and testing images).
This can be written as:
<img src="https://latex.codecogs.com/svg.image?\bg_white&space;\min{\sqrt{{\sum_{}^{}}(vgg\_feature\_1&space;-&space;vgg\_feature\_2)^2}}" title="\bg_white \min{\sqrt{{\sum_{}^{}}(vgg\_feature\_1 - vgg\_feature\_2)^2}}" />
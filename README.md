# svhn-cnn
CNN Deep Learning project to detect and localize house numbers using the SVHN dataset

Object Detection in images is a difficult problem because the algorithm must not only find all
objects in an image but also their locations. In this project, I will be addressing both object detection
and classification in real images and develop CNN models to identify any sequence of upto 5 digits. I
analyzed three different CNN models: a custom CNN model, a VGG-16 model with pre-trained
weights (ImageNet dataset) and a VGG-16 model (trained from scratch). Each of these gave test
accuracies of 86.72%, 83.72% and 78.42% respectively over entire digit sequence. For multi-digit
detection and evaluation of real images I used the custom CNN model. To handle digits at different
scales, I used sliding window algorithm and Non Maxima Suppression with image pyramids.

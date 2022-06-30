# Classifying Art Style with Transfer Learning

Chandrish Ambati & Lucas De Oliveira


## Repo Contents

* [Training notebook:](https://github.com/avmchandrish/art-style-classification/blob/main/training.ipynb) contains and explains functions used in training
* [Training scripts:](https://github.com/avmchandrish/art-style-classification/tree/main/scripts) scripts launched from terminal for training
* [Training logs:](https://github.com/avmchandrish/art-style-classification/tree/main/training_log) training results
* Presentation slides: first and final presentation slides for this project



## Background

In this project, we used transfer learning to classify the style (or movement) of artworks. 

We all know that computer vision models are great at classifying images of objects; but can they classify the *style* of an image just as well?

For instance, we may feel confident that a computer vision neural network could correctly classify both of the below paintings as paintings of dogs. But could they tell that the left is an *renaissance* painting of a dog while the right is an *impressionist* painting of a dog?

**img**

In researching similar work, we discovered [this paper](http://proceedings.mlr.press/v77/lecoutre17a/lecoutre17a.pdf) where authors Lecoutre, Negrevergne, and Yger similarly train various pretrained and re-trained AlexNet and ResNet models to classify the styles of artwork obtained from [WikiArt](https://www.wikiart.org/en/paintings-by-style). We decided to re-create their experiment by training a different set of multi-class classification architectures to classify artworks into the same 25 classes:

1. Abstract Art
2. Abstract Expressionism
3. Art Informel
4. Art Nouveau (Modern)
5. Baroque
6. Color Field Painting
7. Cubism
8. Early Renaissance
9. Expressionism
10. High Renaissance
11. Impressionism
12. Magic Realism
13. Mannerism (Late Renaissance) 
14. Minimalism
15. Naive Art (Primitivism)
16. Neoclassicism
17. Northern Renaissance
18. Pop Art
19. Post-Impressionism
20. Realism
21. Rococo
22. Romanticism 
23. Surrealism
24. Symbolism
25. Ukiyo-e


## Data used

We accessed the *paintings-by-style* page of [WikiArt](https://www.wikiart.org/en/paintings-by-style) where artworks are accessible and conveniently organized by style. This is the same training dataset used by Lecoutre, Negrevergne, and Yger.

However, unlike the original authors we also validated the models against a the WikiArt dataset. The original authors wanted a model that would be predictive of labels assigned by experts outside of the WikiArt community in order to generalize better across potential academic or institutional applications.

See the authors' line of reasoning below:

*[Recognizing Art Style Automatically in painting with deep learning (section 3.1.2):](http://proceedings.mlr.press/v77/lecoutre17a/lecoutre17a.pdf)*


> Styles may be very coherent within the Wikipainting datasets, but may not correspond to any style recognizable by experts outside the WikiArt community. In order to evaluate the **generality** of the styles identified by the models trained with the Wikipaintings datasets, we need **extra datasets collected from an independent source.**
> 
To achieve this, we use a dataset provided by the author of ErgSap3, a visual art gallery application. The dataset contains almost 60,000 images of paintings which are available in the public domain.
> 
> Not all the classes from the Wikipainting dataset are represented in the data provided by ErgSap, and some classes have different names. In order to make the datasets compatible we remove classes from ErgSap that are not represented in Wikipaintings. **We end up with a dataset of 14 classes.**


We deviated from the authors' approach for two reasons:

1. We have slightly **different objectives.** While we were both concerned with training models that classified art style, the original authors conceive a model that generalizes well to different curators' and historians' aesthetic conceptions of an art movement or style. We prioritize developing a model that is very good a classifying the aesthetic differences of art movements, full stop. In this sense, we hypothesize that a single curating institution such as WikiArt may be more internally consistent and therefore true to the aesthetic characteristics that distinguish each style. In other words, we preferred the consistency of training and validation labels from the same source whereas the original authors did not.
2. We were **constrained in storage, computational power, and time.** The over 70,000 images that we downloaded from WikiArt required 35GB of storage. Adding more data from a different source would be more difficult and expensive to work with, and we did not see much benefit given the short-term nature of this project.



## Training

### Models used

We used the following pre-trained models from the `torchvision` library:

* VGG-19: `torchvision.models.vgg19(pretrained=True)`
* ResNet-34: `torchvision.models.resnet34(pretrained=True)`
* ViT-B/16: `torchvision.models.vit_b_16(pretrained=True)`

We then **froze the body of the models** and reinstantiated the head (final classification layer) so that **only the head of the model would be updated during training.**


### Data augmentation

Since we had fairly low amount of examples per class (between 1,500 and 3,600 images), we relied heavily on data augmentation in order to get our models to work.

We decided that we would avoid image transformations that could destroy some information about the style of the image such as elastic transforms or adding noise. Instead, we utilized geometric transformations such as random flips, rotations, and crops.

### Training method

We trained each partially frozen model for only four epochs as we were constrained in both computational power and time (each epoch took about 2 hours). We used python scripts to train from the GPU terminal which can be found under the `scripts/` folder of this repo.

Because of these constraints we were not able to tune training parameters or experiment with additional data augmentation or training techniques.

Below is the validation loss for the first four epochs of training for each model:

**image**

Below is the validation accuracy for the first four epochs of training for each model:

**image**

## Results

After only four epochs we were able to reach a similar level of performance to Lecoutre et. al. before their experiments with retraining, bagging, and distortion. Below is a plot of the validation set accuracy showing the top-1 accuracy of each model including the ones from the aforementioned paper.

**image**

We definitely intend on further refining this work. Our next steps will be to train these models for longer using cloud computing resources such as AWS or GCP and to experiment with freezing/unfreezing different layers as well as fine-tuning the same models.





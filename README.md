<h2>
<p align='center'>
Visual Story Telling
</p>
</h2>

<h4 align='center'> Project Description </h4> 
Add description
<br>

### Technical Skills 
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![ChatGPT](https://img.shields.io/badge/chatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
<br>
<!-- 
### Installing Machine Learning Libraries
##### TensorFlow
      !pip install tensorflow
##### Keras
      !pip install keras
##### PyTorch
      https://pytorch.org/get-started/locally/
##### Pandas
      !pip install pandas
##### NumPy
      !pip install numpy
##### Matplotlib
      !pip install matplotlib -->

### Methodology 
1) Story Telling
   * Basic Proposed Structure: We will generate a plan and then conditioned on it the story will be generated. Plan will be constructed using a decoder only model (say GPT3/ChatGPT), where the input to the model will be list of characters, genre and the relationships between them. Then the generated basic plot will be taken in as input by the encoder-decoder model (T5/BART), conditioned on the sentence wise plot (and already generated text), storyline will be continously generated. Final generated story is then compared with the movie summary. 
   * Decoder Only Model - May use few shot learning with a variety of examples from different genres, use a prompt based strategy for generating plots 
   * Decoding strategies: Nucleus sampling with top-k=10 and p=0.9
   * Dataset: CMU Movie Summary, Scifi TV Shows, Writing Prompts
   * Getting characters and relations
      * Character Name Clustering: [https://github.com/dbamman/book-nlp](https://github.com/booknlp/booknlp)
      * Sentiment: https://www.nltk.org/howto/sentiment.html 
   * Loss function: $L_{Gen}$ + $L_{Review}$. First one corresponds to the normal cross entropy/negative log likelihood, whereas the second one is the difference between the log probability scores of the actual vs perturbed summary
   * Adversarial Inputs
      * Plot given to the ChatGPT model is empty 
      * Rating to an i) empty, ii) non coherent, iii) Non interesting and iv) Combinations of previous aspects summary given by ChatGPT
2) Vision Conversion
   * TO DO
### Tasks 
1) Story Telling
   * Literature Sruvey for Story Generation ✔️
   * Create a methodology for generating stories ✔️ 
   * Implement Story Generation
      * Download the datasets - CMU Movie Summary (🟢), CMU Book Summary (🟢),  Scifi TV Shows (🔴), Writing Prompts (🔴) ✔️
      * Run BookNLP for the Character Clustering on all the datasets 🟡
      * Divide the dataset into subparts and run BookNLP ✔️
      * GPT3/ChatGPT based code implementation for plot development - Give summary as an input and generate a plot ✔️
      * Create the Plot-Story dataset 🟡
      * Train the T5/BART model on this dataset with custom loss function - Take the relationship set, genre, the story name and plot as an input, and generate a story conditioned on the inputs - Plot conditioned Story Generation. Loss function - Cross Entropy Loss + Rating loss given by ChatGPT. ChatGPT  - a rating between 0 and 10, where 0 is the best whereas 10 is the worst. 
      * Evaluate the generation on the performance metrics - BLEU score - try different decoding strategies 
2) Visual Conversion
   * Literature Survey for Text-to-Image 
   * Methodology 
   * Implementation
### Notes
* Story Generation
  * Interface Based Papers
      * WordCraft 🔴
      * Story Centaur 🔴
  * Libraries
      * TextBox
  * New Algorithms - Involving Decoding/Loss/Architecture Updates
      * Hierarchical Neural Story Generation (Attention) 🟢
      * Progressive Generation of Long Text with Pretrained Language Models (Architecture) 🟢
      * MOCHA (Loss) 🟢
      * Towards Inter-character Relationship-driven Story Generation (Architecture) 🟢
      * Little Red Riding Hood Goes Around the Globe (Prompting) 🟢
      * Future Sight Can Very Large Pretrained Language Models Learn Story Telling With a Few Examples (Architecture) 🔴
      * The Stable Entropy Hypothesis and Entropy Aware Decoding (Decoding) 🔴
  * Datasets
      * Visual Writing Prompts 🟢
  * Metrics  
      * Delta Score 🟢
            
### Future Works
* Create a custom sentiment analyzer
* Plot Generation given the following components: Characters, Genre, Title, and Inter-Character Relations
* Dataset expansion for better training 
* Apply on long form story generation

<h2>
<p align='center'>
Visual Story Telling
</p>
</h2>

<h4 align='center'> Project Description </h4> 
Massive Large Language Models such as GPT2, GPT3, PaLM and Llama are rated highly on the task of text generation, however when we explore story generation , then these models often suffer from the problems such as inconsistency, adding new facts such as characters and plot out of nowhere, and moving away from the storyline. To overcome these facts, we are proposing a framework called Visual Story Telling, which comprises of a text generation model and Stable DIffusion Mpdel. Text Generation model is fine tuned on a custom created dataset for the task of content conditioned story generation which is inspired from Plan based/ Heorarchical Story Generation format. We proposed a dataset called Plot Summary Dataset which contains information such as Title, Plot, Characters, Inter-Character Relations and Genre, which are used to condition the output of DistilGPT and T5. This generated story is then utilized by Stable Diffusion models for the task of visual conversion in a sentence by sentence format. 
<br>

### Inference
1) T5 model suffers from huge amount of repetition as compared to DistilGPT model
2) Even though we trained the models for a decent amount of epochs, they still tend to generate new characters that are not provided in the input 
3) Although PEFT methods speeds up the process of Finetuning by ~15%, but they also do affect the perfomance of the model on the downstream task
4) Stable Diffusion models and several other Text consitioned Image Synthesis models are incapable of performing Scene Transition

### Technical Skills 
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![ChatGPT](https://img.shields.io/badge/chatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
<br>

### Dependencies
##### Transformers
      !pip install transformers
##### OpenAI
      !pip install openai
##### Sklearn
      !pip install sklearn 
##### PyTorch (Check CPU/GPU Compatibility)
      https://pytorch.org/get-started/locally/
##### Pandas
      !pip install pandas
##### NumPy
      !pip install numpy
##### Matplotlib
      !pip install matplotlib
##### BookNLP 
      !pip install booknlp

### Dataset Information
* Story Generation
   * CMU Movie Summary : http://www.cs.cmu.edu/~ark/personas/
   * CMU Books Summary : https://www.cs.cmu.edu/~dbamman/booksummaries.html
  
### File Content
* Python Files:
   * BookNLP.ipynb.ipynb
      * Extraction of the Characters, their interactions and relevant sentiments for each pair of characters
   * ChatGPT_API.ipynb
      * ChatGPT for using the Summaries and generating 2-3 lines plot from them 
   * Data_Exploration.ipynb
      *  Unzipping and Loading of the dataset
      *  Preprocessing into a proper DataFrame
      *  Extraction of gfeatures like Genre, Title, etc.
   * Data_Merging.ipynb
      * Merging of the Processed sub-parts of the dataset 
   * Dataset_Preparation_Story_Gen.ipynb
      * Dropping of non-useful features
      * Concatenating Books and Movie Summary datasets
      * Processing dataset for conditional text generation 
   * Diffusion.ipynb
      * Image Generation using two techniques:
         * Text to Image generation using the first sentence and later performing text conditioned image to image generation
         * Taking each sentence and performing a text based image generation (No knowledge about previously occured story)  
   * Story_Generation_DistilGPT.ipynb
      * Dataset Processing (Tokenization and Data Split) for DistilGPT model
      * Training the model on the processed dataset
      * Testing model on Perplexity and BLEU score
      * Plotting the Loss Curve
   * Story_Generation_T5.ipynb
      * Dataset Processing (Tokenization and Data Split) for T5 model
      * Training the model on the processed dataset using several PEFT techniques like LoRA and Adapters
      * Creating a custom training loop utilizing a loss given by ChatGPT
      * Testing model on Perplexity and BLEU score
      * Plotting the Loss Curve
* Docs
   * Story Generation - Contains several papers researched for the task of Story Generation
   * Visual Conversion -  Research papers for Image Synthesis

### How to run
1) Download the datasets from the links provided and all the Python files from Github
2) To extract the datasets into a proper DataFrame run Data_Exploration.ipynb 
3) Run BookNLP.ipynb on both the datasets for the extraction of several features such as Characters, Inter-Character relations, etc.
4) Execute ChatGPT_API.ipynb for generating plots for the summaries - Run them in batches as you'll receive errors due to saturation of requests at OpenAI server
5) Once you have obtained plots for all the summaries, run Data_Merging.ipynb for combining all the batches
6) Execute Data_Preparation_Story_Gen.ipynb for the extarction of Genre, Title, etc. from the processed dataset, now you have the Plot-Summary dataset
7) For training T5 and DistilGPT models on this dataset run Story_Generation_T5.ipynb and Story_Generation_DistilGPT.ipynb files respectively
8) Now you can test both the fine tuned model for the task of Story Generation
9) Finally run Diffusion.ipynb for converting the generated story into a visual representation

### Future Works
* Story Generation
   * Create a custom sentiment analyzer
   * Plot Generation given the following components: Characters, Genre, Title, and Inter-Character Relations
   * Dataset expansion for better training 
   * Apply on long form story generation
   * Train models on variations of the dataset such as - only Plot and Summaries (do not include Title, Characters, etc.)
   * Integrate more PEFT methodologies and compare their affects on the performance 
* Text-to-Image
   * Do a literature survey on the current image synthesis technologies 🟡
   * Propose an architecture/methodology that is capable of scene transformation conditioned on text

<!-- ### Methodology 
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
      * Missing information to the model (although trained on such inputs) - missing title/ missing genre/ missing characters/ missing relations
2) Vision Conversion
   * TO DO
### Tasks 
1) Story Telling
   * Literature Sruvey for Story Generation ✅
   * Create a methodology for generating stories ✅ 
   * Implement Story Generation
      * Download the datasets ✅
         * CMU Movie Summary (✅)
         * CMU Book Summary (✅)
         * Scifi TV Shows (🟥) 
         * Writing Prompts (🟥) 
      * Run BookNLP for the Character Clustering on all the datasets ✅
      * Divide the dataset into subparts and run BookNLP ✅
      * Merging of Book and Movie datasets ✅
      * GPT3/ChatGPT based code implementation for plot development - Give summary as an input and generate a plot ✅
      * Create the Plot-Story dataset ✅
      * Loss function ✅
         * Cross Entropy Loss (✅) 
         * Rating loss given by ChatGPT (ChatGPT gives a rating between 0 and 10, where 0 is the best whereas 10 is the worst) (✅)  
      * Train the T5/BART model on this dataset with custom loss function ✅
         * Take the relationship set, genre, the story name and plot as an input, and generate a story conditioned on the inputs - Plot conditioned Story Generation (✅)
      * Use PEFT, such as using Adapters/LoRA/Prefix Finetuning for the T5 model - Faster and Efficient ✅
         * LoRA (✅)
         * Adapters (✅) 
      * Train Decoder Only Models such as DistilGPT2 ✅
         * DistilGPT2 on entire plot-summary dataset ✅
      * Evaluate the generation on the performance metrics ✅
         * Cross Entropy Loss (✅)
         * BLEU score - try different decoding strategies (✅)
      * Generate Loss Curves (✅)
2) Visual Conversion
   * Basline Implementation using a combination of StableDiffusion Model and Text conditioned image2image model ✅
   * Literature Survey for Text-to-Image 🟡
   * Generative AI 🟡
      * VAE ✅
      * GAN ✅
         * DCGAN ✅
         * WGAN ✅
         * Conditional GAN ✅
         * Pix2Pix GAN ✅
         * Cycle GAN ✅
         * SRGAN
         * DeepDream
         * GauGAN
         * PixelCNN
         * StyleGAN
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
-->  

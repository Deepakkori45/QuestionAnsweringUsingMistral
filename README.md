App link: https://cricket-score-prediction-x27nemdpwkvyzn9rynmse4.streamlit.app/

# Question Answering System 

# 1) Dataset Understanding 

The dataset used in this assignment is the "MedQuad-MedicalQnADataset." The structure of the dataset includes features like 'qtype,' 'Question,' and 'Answer.' The 'qtype' represents the question type, 'Question' contains the actual questions, and 'Answer' holds the corresponding answers. 

a) Qtype Dropped Reasons 

When I plotted the distribution of 'qtype' versus the number of questions, it revealed that categories such as 'information,' 'symptoms,' and 'treatments' collectively accounted for more than 50% of the dataset. These categories did not significantly contribute to distinguishing between questions, making the 'qtype' column less relevant. Consequently, I made the decision to drop the 'qtype' column. 
<img src="https://github.com/Deepakkori45/QuestionAnsweringUsingMistral/assets/111627339/9c18fd46-805f-48c6-be47-3f06a028dc91" alt="image" width="600" height="300" />

I have fine-tuned a model named "mistral_with_qtype" and uploaded it to Hugging Face. 
Here is the my Huggigface account: https://huggingface.co/Deepakkori45

Model id: 


![image](https://github.com/Deepakkori45/QuestionAnsweringUsingMistral/assets/111627339/96dbfaea-d3ae-407b-9102-6ac9b3758194)


a) 
In this model, I concatenated the 'qtype' and 'question' together. When attempting to predict the test results, I observed that answers without considering the 'qtype' yielded better results compared to those considering the 'qtype.' Consequently, I decided to drop the 'qtype' column from the model. I have attached the results at the end. 

# 2) Data Preprocessing 

a) Text Cleaning 
During the text cleaning process, special characters were removed from the dataset. 

**Total number of special Characters Removed: 681112**
 
**Unique Characters: 56**

b) Handling Missing Values and Duplicates 

NaN values were present in each row's columns. 

c) Splitting the Data 

The data is split into training, development (dev), and test sets using the train_test_split function from scikit-learn. 

 

# 3) Exploratory Data Analysis (EDA) 

a) Number of Words in Each Row 

Here is a diagram illustrating the number of words in each row of both questions and answers. 

<img src="https://github.com/Deepakkori45/QuestionAnsweringUsingMistral/assets/111627339/21192a26-b14b-4dcd-ae51-0b264113e1a5" alt="image" width="600" height="300" />

**Mean words per row in 'Question' column: 7.17** 

**Mean words per row in 'Answer' column: 194.88** 

b) Word Clouds for Frequent Words 

The generated word clouds visually represent the most frequent words in the 'Patient Questions' and 'Expert Responses' columns. 
<img src="https://github.com/Deepakkori45/QuestionAnsweringUsingMistral/assets/111627339/d2e1e340-4ca7-4268-8e13-e963ef27cc0b" alt="image" width="600" height="300" />

# 4) Train/Fine-tune on Domain-specific Dataset 

The code demonstrates how to fine-tune a language model (Mistral-7B) on the given dataset using the PEFT library. The fine-tuned model is then saved and pushed to the Hugging Face model hub. (Also fine-tuned Llama-7B similarly) 

a) Fine-Tuning Process 

The code demonstrates the fine-tuning process for two language models, Mistral-7B and LLama2-7B, using the PEFT library. 

 

b) Quantization 

Both models are loaded with 4-bit quantization. Quantization, in this context, refers to reducing the precision of the model's weights to 4 bits. 
<img src="https://github.com/Deepakkori45/QuestionAnsweringUsingMistral/assets/111627339/6e58e031-2f5c-4b57-8659-6cc90cf28779" alt="image" width="300" height="150" />

**For 32bit size = (7 * 4) / (32/32) * 1.2 = 33GB** 

**For 4bit size = (7*4) / (32/4) * 1.2 = 3.9GB**

Loading the Mistral 7b model in 32-bit format requires approximately 33.6 GB of memory. While Colab might initially load the model onto its disk(108GB), attempting to transfer it to a GPU for execution it’ll likely result in an out-of-memory error (15 GB). 

However, applying 4-bit quantization significantly reduces the model size to around 3.9 GB, allowing for seamless transfer and execution on Colab's GPUs without memory constraints.  

 

c) PEFT Details 

In PEFT, we can reduce matrix size by removing linearly dependent columns without compromising information. This is possible because the information contained within those columns can be recovered through linear decomposition. Applying linear decomposition directly to the weight update matrix (Delta W) automatically eliminates linearly dependent columns. 
<img src="https://github.com/Deepakkori45/QuestionAnsweringUsingMistral/assets/111627339/f1641cce-9ef0-4241-80d1-13eb084eed5b" alt="image" width="600" height="300" />
 

**During fine-tuning, Mistral-7B has trainable parameters total 27,262,976(2.7 million).** 

**The total number of parameters in the model, including non-trainable ones, is 3,779,334,144 (3.7billion).** 

**The percentage of trainable parameters in Mistral-7B is approximately 0.72%.** 

 
A rank of 64 means the low-rank weight updates are 64 times smaller than the original weight matrices, considerably reducing memory requirements. 

 
<img src="https://github.com/Deepakkori45/QuestionAnsweringUsingMistral/assets/111627339/b5c92b41-2539-46ba-8d4e-c181479b8f1e" alt="image" width="400" height="200" />

 

d) Model Saving and Uploading 

After the fine-tuning process, the resulting models are saved. 

The saved models are then uploaded to the Hugging Face model hub. This allows easy accessibility and the sharing of the fine-tuned models with the wider community. 

# 5) Evaluate Fine-tuning Effectiveness 

a) Gemini Model Evaluation 

For an additional perspective, the Gemini model was utilized to compare the output with the actual answers. Participants evaluated the similarity between the actual answer, baseline prediction, and finetuned prediction using the Gemini model. 

Average Scores: 

**Baseline Model: Average Score - 36.04** 

**Finetuned Model: Average Score - 76.67** 

b) BLEU Score Comparison 

During the evaluation phase, the performance of the Question Answering system was assessed using the BLEU (Bilingual Evaluation Understudy) metric. BLEU is a commonly used metric for evaluating the quality of machine-generated text by comparing it to one or more reference translations. 

**Baseline BLEU Score: 0.2228** 

**Fine-tuned BLEU Score: 0.3647** 

After fine-tuning on the domain-specific dataset, the BLEU score significantly improved, indicating enhanced performance in generating relevant answers. These BLEU scores serve as quantitative measures to gauge the effectiveness of the Question Answering system, with a higher BLEU score indicating better alignment with the reference answers. 

 
# 6) Results 

Model Response:

1)

![image](https://github.com/Deepakkori45/QuestionAnsweringUsingMistral/assets/111627339/6da86e24-98b1-4571-b264-d762c5003556)

2)
 ![image](https://github.com/Deepakkori45/QuestionAnsweringUsingMistral/assets/111627339/6b6608ac-ff1a-4470-810a-8bad92b87939)

3)
![image](https://github.com/Deepakkori45/QuestionAnsweringUsingMistral/assets/111627339/9d0c41a7-34ce-42a1-96d4-91b58b53010a)

 
# 7) Conclusion 

Challenges Overcome: 

• Question Specificity: Improved through fine-tuning for concise and contextually relevant responses. 

In conclusion, the QA system demonstrated effective learning, robust generalization, and improved question specificity post fine-tuning. 

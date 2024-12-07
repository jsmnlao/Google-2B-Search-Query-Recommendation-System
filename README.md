# Google 2B: Google Search Query Recommendation System

**Purpose**: To develop a foundational understanding of the data and technical expertise required to engineer a powerful search engine such as Google’s. Our goal was to create an LLM that mimicked the predictive capabilities of autocomplete in search engines from scratch. 

Access our presentation slides [here](https://docs.google.com/presentation/d/1q_QX90_682fVRCP2913vI9J4t7gihrB5tEJYidMY4js/edit?usp=sharing)!

## Contributors
**Team Members**: Jasmine Lao, Paris Tchefor, Zoya Shamak, Christina Chan, Amanda Yu

**Challenge Advisor & TA**: Josh McAdams, Esther Li

## Technologies Used
- Hugging Face
- Google Colab
- Pandas
- PyTorch
- TikToken
- Visual Studio Code

## Methodology and Approaches Used
- Dataset collected from Hugging Face: https://huggingface.co/datasets/google-research-datasets/nq_open
- Transformer model with Multi-Head Attention, Feed Forward network, Linear layers
- Token and Positional Encoding
- Two layers of Encoding: tiktoken, compression of encoded tokens
- Holdout: 10% testing, 90% training
- Dataset Batches: optimize learning process, less overfitting, reduce memory usage
- Saving our [trained model parameters](https://github.com/jsmnlao/Google-2B-Search-Query-Recommendation-System/blob/main/saved_bigram_language_model.pth) with PyTorch's _torch.save()_
- Performance metric: log loss
- Due to limited resources, training took place on the CPU for 25 hours with 10k iterations on 4k training records
- Final input/output evaluations can be found at the bottom of [Final_Search_Query_Model.ipynb](https://github.com/jsmnlao/Google-2B-Search-Query-Recommendation-System/blob/main/Final_Search_Query_Model.ipynb)

## How to Run Training Python Code
1. Clone repository: git clone https://github.com/jsmnlao/Google-2B-Search-Query-Recommendation-System.git OR Pull everything from git repo: git pull
2. Install the libraries from requirements: pip3 install -r requirements.txt
3. Run the model for training: python [model.py](https://github.com/jsmnlao/Google-2B-Search-Query-Recommendation-System/blob/main/model.py)

## Potential Next Steps
If possible, we want to improve our model with personalized queries, long-term dependencies (with LSTMs), and capturing patterns in common searching phrases. Additionally, we would like to have other key metrics such as precision or recall, training with more records and iterations on GPUs, and performing more hyperparameter tuning for the most optimal performance.

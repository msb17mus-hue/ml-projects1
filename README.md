Machine Learning Projects Portfolio
A collection of five end-to-end machine learning projects covering
regression, classification, customer analytics, clustering, and natural
language processing (NLP).

Each project is organized in its own folder and is designed to
demonstrate a complete machine learning workflow: loading public data,
cleaning and exploring it, preparing features, training and evaluating
models, saving the selected model, and making predictions through a
Streamlit application.
Projects
  #                Project                                        Machine Learning  Objective
task              
  01                [House Price                                   Supervised        Estimate house
                    Prediction](./01-House-Price-Prediction/)      Learning ---      prices from
Regression        property
characteristics.

  02                [Classification                                Supervised        Train and compare
                    Project](./02-Classification-Project/)         Learning ---      classification
Classification    algorithms on a
real-world
prediction task.

  03                [Customer Churn                                Supervised        Predict whether a
                    Prediction](./03-Customer-Churn-Prediction/)   Learning ---      customer is likely
Classification    to leave a service
and explore
factors associated
with churn.

  04                [Customer                                      Unsupervised      Discover customer
                    Segmentation](./04-Customer-Segmentation/)     Learning ---      groups using
Clustering        clustering
techniques such as
K-Means.
05                [Sentiment Analysis](./05-Sentiment-Analysis/) Natural Language  Classify text
Processing        sentiment as
positive or
negative.
Repository Structure
ML-Portfolio/
├── 01-House-Price-Prediction/
│   ├── notebook.ipynb
│   ├── app.py
│   ├── model.pkl
│   ├── requirements.txt
│   └── README.md
├── 02-Classification-Project/
│   ├── notebook.ipynb
│   ├── app.py
│   ├── model.pkl
│   ├── requirements.txt
│   └── README.md
├── 03-Customer-Churn-Prediction/
│   ├── notebook.ipynb
│   ├── app.py
│   ├── model.pkl
│   ├── requirements.txt
│   └── README.md
├── 04-Customer-Segmentation/
│   ├── notebook.ipynb
│   ├── app.py
│   ├── model.pkl
│   ├── requirements.txt
│   └── README.md
└── 05-Sentiment-Analysis/
    ├── notebook.ipynb
    ├── app.py
    ├── model.pkl
    ├── requirements.txt
    └── README.md
Project Workflow
The notebooks are intended to document the main machine learning
workflow in clearly separated sections:
1. Project introduction and problem definition
2. Dataset loading from a public URL
3. Data overview and quality checks
4. Data cleaning and exploratory data analysis (EDA)
5. Data preprocessing and feature engineering
6. Model training and comparison
7. Evaluation using task-appropriate metrics
8. Selection and serialization of the best model
9. Prediction on new examples
10. Conclusions and key findings
The Streamlit application in each project provides an interface for
entering new input data and viewing model predictions. The saved model
should include the preprocessing steps needed at inference time so that
the application applies the same transformations used during training.
Getting Started
1. Clone the repository
git clone <repository-url>
cd ML-Portfolio
Replace <repository-url> with the HTTPS URL of this GitHub repository.
2. Set up an environment
Python 3.10 or a compatible version is recommended. Create and activate
a virtual environment:
python -m venv .venv
On Windows:
.venv\Scripts\activate
On macOS or Linux:
source .venv/bin/activate
3. Run a project
Move into the project folder you want to explore. For example:
cd 01-House-Price-Prediction
Install that project's dependencies:
pip install -r requirements.txt
Open the notebook:
jupyter notebook notebook.ipynb
Run the Streamlit application:
streamlit run app.py
Repeat these steps for any of the other project folders. Each project
has its own requirements.txt so its dependencies can be installed
independently.
Technologies
The projects are designed around commonly used data science and machine
learning tools, including:
- Python for data analysis and model development
- Jupyter Notebook for reproducible, documented experiments
- pandas and NumPy for data manipulation
- Matplotlib and Seaborn for exploratory visualizations
- scikit-learn for preprocessing, model training, and evaluation
- Streamlit for interactive model demonstrations
- Joblib or Pickle for saving and loading trained models
The exact libraries used by an individual project are listed in that
project's requirements.txt.
Evaluation and Reproducibility
Model performance should be reported from the actual notebook execution
rather than estimated or assumed. Regression, classification,
clustering, and sentiment analysis require different evaluation
approaches; each project should use metrics appropriate to its task.
Random seeds should be set where applicable to make experiments more
reproducible.

Dataset source URLs, preprocessing decisions, model comparisons,
evaluation results, and limitations should be documented in the relevant
project README and notebook.

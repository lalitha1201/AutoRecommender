## AutoRecommender
RecServe is a recommendations as a service tool that allows users to input their custom datasets and obtain recommendations easily. It is evident that personalization technologies provide great business results and engage users dramatically increasing customer retention. Only larger organizations like Amazon and Netflix were able to leverage and customize the recommendation engines sustained by specific business units. Maintaining specific teams within the company involves developmental cost that is simply too high for most SME's and small businesses. Automated recommendation systems which can be easily integrated with their existing systems is an elegant solution to this problem.

RecServe is an intelligent recommendation system that recommends a single algorithm without training and tuning five or more different ML algorithms!

A google slide presentation can be found [here](https://docs.google.com/presentation/d/1Q99WFetq5X_IgM-NdBFBH4EwiebxfmZHHTsX2eZVusQ/edit?usp=sharing)

## Data
RecServe is currently using the Amazon Reviews Dataset.

142.8 million reviews

review data - ratings, text, helpfulness votes

product data - descriptions, category information

data spanning from May 1996 - July 2014.

## SAMPLE CONTENT:
https://s3.amazonaws.com/amazon-reviews-pds/tsv/sample_us.tsv

After downloading this repo and follow the installation instuctions below you will be able to experiment with RecServe through an interactive command line interface.

## Requirements / Dependencies
Python 3.6

Pandas 

Click 

Colorama 

Surprise  

Scikit-learn 

Streamlit
## Package Installations
pip install pandas

pip install click

pip install colorama

pip install surprise 

pip install scikit-learn or pip install sklearn

pip install streamlit

## Installation / Setup
Clone repository and update python path: The easiest way to download this project is by using git from the command-line:

git clone https://github.com/lalitha1201/AutoRecommender.git

## Environment
I recommend creating a conda environoment so you do not destroy your main installation in case you make a mistake somewhere:

conda create --name RecServe_3.7 python=3.6 ipykernel

You can activate the new environment by running the following (on Linux):

source activate RecServe_3.7 

And deactivate it:

source deactivate RecServe_3.7

## Serving RecServe
Now you should be all set to get your product recommendations through the recserve.py interactive command-line interface using:

cd RecServe

python interface.py

## Streamlit App

cd RecServe

streamlit run streamlit_app.py


Thank you for choosing RecServe!




# Recommending Books using Goodreads Data
## Adam Coviensky, Stephanie Doctor, Marika Lohmus, Mark Salama
### Personalization Theory, Fall 2017

This repository contains the code and explanations for our Personalization Theory class project.

The file structure is as follows:  
ratings.csv: contains the ratings data from Goodreads used for this project  
ratings_us.csv: same as ratings.csv, with additional ratings for the four of us
requirements.txt: packages needed to run code in this repository  
part1/: folder containing all work for part 1 of the project  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;all_results.csv: runtime, accuracy, and coverage for each model  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;baseline_algorithm.ipynb: implementation and results of baseline models  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;data_exploration_goodreads.ipynb: preliminary data exploration and plots  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;matrix-fact-manual-regularized.ipynb: implementation and results of Matrix Factorization  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;neighborhood-models.ipynb: implementation and results of neighborhood-based models  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;plot-model-comparisons.ipynb: plots comparing runtime, accuracy, and coverage of models  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;report.pdf: writeup of part 1  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;surprise_SVD_NMF.ipynb: implementation and results of SVD and NMF via SurPRISE  
part2/: folder containing all work for part 2 of the project  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;FastFM_Convergence.ipynb: implementation of FastFM library for comparison  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Grid_Search_LibFM.ipynb: implementation of LibFM library and grid search for parameter tuning  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;book_tags.csv: counts of user-defined tags for each book  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;books.csv: book metadata  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;books_meta_info.csv: book metadata with genre tag counts  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;books_with_summaries.p: NLP features  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;genres.csv: list of genres  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;get_genres.ipynb: code to match tags to genres  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;make_FM_features.ipynb: code to generate model features  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;summaries.csv: book summaries  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;summaries_nlp.ipynb: code to collect and process book summaries  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tags.csv: tag ID-label mapping  
dash-app/: repository for heroku web application  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Procfile: file for heroku  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;app.py: web app code  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;books.csv: book metadata  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;genre_diversity_recs.csv: pre-computed diversity recommendations for each user  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;model_features_genres_only.npz: model features for predictions  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;model_genres_only: model file for predictions  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;predict.py: code to make predictions with user input  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ratings_us.csv: ratings data plus our ratings  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;requirements.txt: package requirements for web app  
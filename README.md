# YELP recommendation system
This is a small project from the Artificial Intelligence course of the Master degree in Computer Science.
The work shows a study between different Machine Learning and Deep Learning techniques to build a recommendation system based on the YELP Open Dataset downloadable here: https://www.yelp.com/dataset




## The project:
The aim of the project is to provide a performance analysis between different Machine Learning and Deep Learning techniques to solve the following problems:
- classifying a review as positive or negative;
- classify a review with a rating from 1 to 5 stars;
- grouping users according to their behaviour on the platform;
- recommending a restaurant based on city and characteristics.



## Dependencies:
- python 3.7.13+
- pandas 1.3.5
- numpy 1.21.6
- matplotlib 3.2.2
- nltk 3.7
- wordcloud 1.5.0
- scikit-learn 1.0
- keras 1.8.0
- scipy 1.7.3
- kneed 0.7.0
- tensorflow 2.8.2
- Transformers 4.20.1


## Structure:
4 modules:
1) positive_or_negative_review.py
2) user_grouping_task:py
3) city_restaurant_grouping.py
4) give_stars_to_review.py


The structure is as follows:
- modules: directory containing the modules created for solving the addressed problems;
- data: directory containing all input files required for execution;
- plots: directory containing the graphs obtained from the execution of the comparison of techniques for solving the 4 problems addressed;
- saved_models: directory containing the saved weights of the ML models used to speed up execution;
- demo.py: demo executable.



## Download and execution
- clone the repository;
- run on terminal "python demo.py".
Then you can choose if execute only the demo or the entire comparison study between each technology implemented.

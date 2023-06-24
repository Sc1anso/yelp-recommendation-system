# YELP recommendation system
This is a small project from the Artificial Intelligence course of the Master of Science in Computer Science degree.
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



## Nome Progetto:
Studio per la costruzione di un recommender system per lo YELP Open Dataset


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
- demo.py: demo executable



## Esecuzione
Per eseguire la dimostrazione del progetto svolto basta digitare da terminale "python demo.py" e avrà inizio la dimostrazione.
Una volta avviata la demo sarà possibile scegliere se eseguire la demo vera e propria (che utilizza solo le configurazioni migliori) oppure la comparazione tra tutti gli algoritmi mostrati.



## Autori e documentazione
Il progetto è stato realizzato dal gruppo AI_VDF composto da Giacomo Vallasciani, Giuseppe Di Maria, Alessandro Fabbri.
In allegato alla repository del progetto, è presente una documentazione in formato .pdf in cui si illustrano dettagliatamente le fasi di sviluppo del programma e infine gli output finali ottenuti, dopo le varie fasi di training e di test.
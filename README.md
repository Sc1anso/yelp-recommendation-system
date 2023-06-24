# AI_VDF
Corso di Laurea Magistrale in Informatica
Progetto di Intelligenza Artificiale
a.a. 2021/22




## Repository:
https://gitlab.com/Scianso/ai_vdf



## Scarica:
Per scaricare la repository del progetto in locale è sufficiente digitare il comando da terminale:
git clone https://gitlab.com/Scianso/ai_vdf



## Nome Progetto:
Studio per la costruzione di un recommender system per lo YELP Open Dataset


## Descrizione
Si vuole proporre una possibile implementazione di un recommender system sullo YELP Open Dataset
L'implementazione è stata suddivisa in 4 problemi, ognuno affrontato nei rispettivi moduli:
1) positive_or_negative_review.py
2) user_grouping_task:py
3) city_restaurant_grouping.py
4) give_stars_to_review.py
Questi 4 moduli che rappresentano rispettivamente un problema affrontato verrannò poi importati nel file "demo.py" che sarà incaricato dell'esecuzione della dimostrazione

La struttura è la seguente:
- modules: directory contenente i moduli creati per la risoluzione dei problemi affrontati;
- data: directory contenente tutti i files di input necessari all'esecuzione;
- plots: directory contenente i grafici ottenuti dall'esecuzione della comparazione delle tecniche per risolvere i 4 problemi affrontati;
- saved_models: direcotry contenente i pesi salvati dei modelli di ML utilizzati per velocizzarne l'esecuzione;
- README.md;
- REPORT_AI.pdf: report del progetto;
- demo.py: eseguibile della dimostrazione


## Esecuzione
Per eseguire la dimostrazione del progetto svolto basta digitare da terminale "python demo.py" e avrà inizio la dimostrazione.
Una volta avviata la demo sarà possibile scegliere se eseguire la demo vera e propria (che utilizza solo le configurazioni migliori) oppure la comparazione tra tutti gli algoritmi mostrati.



## Autori e documentazione
Il progetto è stato realizzato dal gruppo AI_VDF composto da Giacomo Vallasciani, Giuseppe Di Maria, Alessandro Fabbri.
In allegato alla repository del progetto, è presente una documentazione in formato .pdf in cui si illustrano dettagliatamente le fasi di sviluppo del programma e infine gli output finali ottenuti, dopo le varie fasi di training e di test.
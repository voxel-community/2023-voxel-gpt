# 02. Word Embedding

| Capitolo precedente                                                                                                                                          | Capitolo successivo                                                                           |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------: |
| [◀︎ 01-classificatore-come-funziona](../01-classificatore-come-funziona)  | [03-crea-rete ▶︎](../03-crea-rete) |

## Obiettivo

Impara cos'è e come funziona il *word embedding* 🔠 all'interno di un modello come ChatGPT.

L'embedding di parole è una tecnica per rappresentare le parole in modo che una intelligenza artificiale possa comprendere il significato. In parole semplici, è un modo per **convertire le parole in numeri** che possono essere utilizzati come input per un modello di apprendimento automatico.

L'embedding mappa le parole in un vettore di numeri reali, di solito lunghi centinaia di elementi. Questa rappresentazione cattura le relazioni tra le parole, come le somiglianze e le differenze di significato, contesto e uso. Ad esempio, parole simili come `gatto` e `cane` avranno rappresentazioni simili 🐶🐱, mentre parole dissimili come `gatto` e `computer` avranno rappresentazioni dissimili 🐱💻.

Esistono dei "dizionari" di word embedding già pronti da utilizzare: i più famosi sono *Word2vec* e *GloVe*. In questo esempio utilizzeremo uno dei dizionari di [GloVe](https://nlp.stanford.edu/projects/glove/). Consideriamo il dizionario più piccolo in modo che l'esecuzione del codice sia veloce. Però piccolo per modo di dire! Questo dizionario è stato ricavato da un corpus di testo di **più di 6 miliardi di parole** prese da Wikipedia e importanti testate giornalistiche. Non male 😎

> ***Curiosità: che tipo di word embedding usa ChatGPT?*** ChatGPT non usa un algoritmo standard come GloVe o Word2vec, ma impara un embedding tutto suo durante la fase di addestramento della sua rete neurale. Utilizza una tecnica complessa di *auto supervisione* *(self-supervision)* con cui impara a catturare le dipendenze tra parole in base alla loro posizione in una frase. Ovvero, il vettore di embedding di una parola è in grado di rappresentare anche il contesto della frase. Inoltre, i vettori di embedding che usa ChatGPT sono molto grandi, arrivano ad avere dimensione di 1024 elementi.


## Steps


### 1. Carica il file di embedding

Su Google Colab crea un nuovo notebook e rinominalo ad esempio in `embedding.ipynb`.

Scarica il file `glove.6B.50d.txt` che trovi a questo [link](https://www.dropbox.com/s/b3jbd1bgf93rkw6/glove.6B.50d.txt?dl=0) e salvalo temporaneamente sul tuo computer. Vai sul notebook Colab e carica il file all'interno della sessione (come spiegato nel [capitolo precedente](../00-setup)).

Il file che hai caricato è uno dei dizionari di word embedding di GloVe, e contiene 400K parole e i corrispondenti vettori ciascuno lungo 50 elementi.
Se sei curiosa di vedere il suo contenuto, puoi notare che si tratta di un file testuale di appunto 400,000 righe, dove ogni riga inizia con una parola ed è seguita da 50 numeri decimali. Il file ha un aspetto del genere:

```
the 0.418 0.24968 -0.41242 0.1217 0.34527 -0.044457 -0.49688 -0.17862 -0.00066023 -0.6566 0.27843 -0.14767 -0.55677 0.14658 -0.0095095 0.011658 0.10204 -0.12792 -0.8443 -0.12181 -0.016801 -0.33279 -0.1552 -0.23131 -0.19181 -1.8823 -0.76746 0.099051 -0.42125 -0.19526 4.0071 -0.18594 -0.52287 -0.31681 0.00059213 0.0074449 0.17778 -0.15897 0.012041 -0.054223 -0.29871 -0.15749 -0.34758 -0.045637 -0.44251 0.18785 0.0027849 -0.18411 -0.11514 -0.78581
and 0.26818 0.14346 -0.27877 0.016257 0.11384 0.69923 -0.51332 -0.47368 -0.33075 -0.13834 0.2702 0.30938 -0.45012 -0.4127 -0.09932 0.038085 0.029749 0.10076 -0.25058 -0.51818 0.34558 0.44922 0.48791 -0.080866 -0.10121 -1.3777 -0.10866 -0.23201 0.012839 -0.46508 3.8463 0.31362 0.13643 -0.52244 0.3302 0.33707 -0.35601 0.32431 0.12041 0.3512 -0.069043 0.36885 0.25168 -0.24517 0.25381 0.1367 -0.31178 -0.6321 -0.25028 -0.38097
in 0.33042 0.24995 -0.60874 0.10923 0.036372 0.151 -0.55083 -0.074239 -0.092307 -0.32821 0.09598 -0.82269 -0.36717 -0.67009 0.42909 0.016496 -0.23573 0.12864 -1.0953 0.43334 0.57067 -0.1036 0.20422 0.078308 -0.42795 -1.7984 -0.27865 0.11954 -0.12689 0.031744 3.8631 -0.17786 -0.082434 -0.62698 0.26497 -0.057185 -0.073521 0.46103 0.30862 0.12498 -0.48609 -0.0080272 0.031184 -0.36576 -0.42699 0.42164 -0.11666 -0.50703 -0.027273 -0.53285
[...]
```


tbd...






| Capitolo precedente                                                                                                                                          | Capitolo successivo                                                                           |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------: |
| [◀︎ 01-classificatore-come-funziona](../01-classificatore-come-funziona)  | [03-crea-rete ▶︎](../03-crea-rete) |

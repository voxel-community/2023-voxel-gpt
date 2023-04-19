# 02. Word Embedding

| Capitolo precedente                                                                                                                                          | Capitolo successivo                                                                           |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------: |
| [◀︎ 01 Come funziona ChatGPT](../01-come-funziona-gpt)  | [03 Tokenization ▶︎](../03-tokenization) |

## Obiettivo

Impara cos'è e come funziona il *word embedding* 🔠 all'interno di un modello come ChatGPT.

L'embedding di parole è una tecnica per rappresentare le parole in modo che una intelligenza artificiale possa comprendere il significato. In parole semplici, è un modo per **convertire le parole in numeri** che possono essere utilizzati come input per un modello di apprendimento automatico.

L'embedding mappa le parole in un vettore di numeri reali, di solito lunghi centinaia di elementi. Questa rappresentazione cattura le relazioni tra le parole, come le somiglianze e le differenze di significato, contesto e uso. Ad esempio, parole simili come `gatto` e `cane` avranno rappresentazioni simili 🐶🐱, mentre parole dissimili come `gatto` e `computer` avranno rappresentazioni dissimili 🐱💻.

Esistono dei "dizionari" di word embedding già pronti da utilizzare: i più famosi sono *Word2vec* e *GloVe*. In questo esempio utilizzeremo uno dei dizionari di [GloVe](https://nlp.stanford.edu/projects/glove/). Consideriamo il dizionario più piccolo in modo che l'esecuzione del codice sia veloce. Però piccolo per modo di dire! Questo dizionario è stato ricavato da un corpus di testo di **più di 6 miliardi di parole** prese da Wikipedia e importanti testate giornalistiche. Non male 😎

> ***Curiosità: che tipo di word embedding usa ChatGPT?***
> ChatGPT non usa un algoritmo standard come GloVe o Word2vec, ma impara un embedding tutto suo durante la fase di addestramento della sua rete neurale. Utilizza una tecnica complessa di *auto supervisione* *(self-supervision)* con cui impara a catturare le dipendenze tra parole in base alla loro posizione in una frase. Ovvero, il vettore di embedding di una parola è in grado di rappresentare anche il contesto della frase. Inoltre, i vettori di embedding che usa ChatGPT sono molto grandi, arrivano ad avere dimensione di 1024 elementi.


## Steps


### 1. Scarica il file di embedding

Su Google Colab crea un nuovo notebook e rinominalo ad esempio in `embedding.ipynb`.
Crea una cella di codice, incolla il seguente comando ed esegui la cella per scaricare il file di embedding:

```py
!wget https://www.dropbox.com/s/b3jbd1bgf93rkw6/glove.6B.50d.txt
```

<!--Scarica il file `glove.6B.50d.txt` che trovi a questo [link](https://www.dropbox.com/s/b3jbd1bgf93rkw6/glove.6B.50d.txt?dl=0) e salvalo temporaneamente sul tuo computer. Vai sul notebook Colab e carica il file all'interno della sessione (come spiegato nel [capitolo precedente](../00-setup)).-->

Il file `glove.6B.50d.txt` che hai appena scaricato è uno dei dizionari di word embedding di GloVe, e contiene 400K parole e i corrispondenti vettori ciascuno lungo 50 elementi.
Se sei curiosa di vedere il suo contenuto, puoi notare che si tratta di un file testuale di appunto 400K righe, dove ogni riga inizia con una parola ed è seguita da 50 numeri decimali. Le righe del file ha un aspetto del genere:

```
the 0.418 0.24968 -0.41242 0.1217 0.34527 -0.044457 -0.49688 -0.17862 -0.00066023 -0.6566 0.27843 -0.14767 -0.55677 0.14658 -0.0095095 0.011658 0.10204 -0.12792 -0.8443 -0.12181 -0.016801 -0.33279 -0.1552 -0.23131 -0.19181 -1.8823 -0.76746 0.099051 -0.42125 -0.19526 4.0071 -0.18594 -0.52287 -0.31681 0.00059213 0.0074449 0.17778 -0.15897 0.012041 -0.054223 -0.29871 -0.15749 -0.34758 -0.045637 -0.44251 0.18785 0.0027849 -0.18411 -0.11514 -0.78581
and 0.26818 0.14346 -0.27877 0.016257 0.11384 0.69923 -0.51332 -0.47368 -0.33075 -0.13834 0.2702 0.30938 -0.45012 -0.4127 -0.09932 0.038085 0.029749 0.10076 -0.25058 -0.51818 0.34558 0.44922 0.48791 -0.080866 -0.10121 -1.3777 -0.10866 -0.23201 0.012839 -0.46508 3.8463 0.31362 0.13643 -0.52244 0.3302 0.33707 -0.35601 0.32431 0.12041 0.3512 -0.069043 0.36885 0.25168 -0.24517 0.25381 0.1367 -0.31178 -0.6321 -0.25028 -0.38097
in 0.33042 0.24995 -0.60874 0.10923 0.036372 0.151 -0.55083 -0.074239 -0.092307 -0.32821 0.09598 -0.82269 -0.36717 -0.67009 0.42909 0.016496 -0.23573 0.12864 -1.0953 0.43334 0.57067 -0.1036 0.20422 0.078308 -0.42795 -1.7984 -0.27865 0.11954 -0.12689 0.031744 3.8631 -0.17786 -0.082434 -0.62698 0.26497 -0.057185 -0.073521 0.46103 0.30862 0.12498 -0.48609 -0.0080272 0.031184 -0.36576 -0.42699 0.42164 -0.11666 -0.50703 -0.027273 -0.53285
```

### 2. Importa i package

In questo esempio ti servono tre pacchetti di python, con delle funzioni pronte per manipolare vettori. Esegui questo codice in una nuova cella in Colab:

```py
import  os
import  numpy                    as np
from    tensorflow.python.keras  import losses
```

### 3. Leggi il file di embedding

Crea una funzione per leggere il file ed estrarne i word embedding. La funzione restituisce una variabile python di tipo `dict` (o dizionario), ovvero una struttura in cui ad ogni parola (variabile di tipo `str`) viene associato un vettore di embedding (variabile di tipo `numpy.array`).

Esegui questo codice in una nuova cella in Colab:

```py
def read_glove( glove_file ):
    embedding   = {}
    
    # legge il file GloVe riga per riga
    with open( glove_file, 'r' ) as f:
        cnt = 0
        
        for l in f:
            # per ogni riga del file, estrae la parola e il vettore
            word, vector        = l.split( maxsplit=1 )
            vector              = np.matrix( vector ).A1 
            # ...e li salva nel dizionario
            embedding[ word ]   = vector
            
            # stampa un messaggio di conferma
            cnt += 1
            if not cnt % 10000:
                print( f"read {cnt:,} of 400,000 words" )

    print( "Done!" )
    return embedding
```

Ora che hai definito la funzione, in una nuova cella esegui il codice qui sotto. Questa esecuzione impiega un po' di tempo, circa 2 minuti e mezzo (dopotutto, sono un sacco di parole!). Quindi lancia l'esecuzione e prenditi un attimo di pausa ☕️

```py
glove_file  = "glove.6B.50d.txt"
embedding   = read_glove( glove_file )
```

### 4. Gioca con gli embedding

Adesso hai un dizionario di embedding pronto per essere usato! Cosa puoi farci con questo?

#### Visualizza la rappresentazione di una parola

Per prima cosa, ti può essere utile una funzione per visualizzare il vettore di embedding di una parola a piacimento. La puoi definire con il seguente codice, incollalo in una nuova cella ed esegui:

```py
def embed( word ):
    if isinstance( word, str ):
        # controlla se la parola è presente nel dizionario
        if word not in embedding.keys():
            return False
        return embedding[ word ]
    return word
```

Fai una prova! In una nuova cella esegui il seguente comando per visualizzare il vettore che rappresenta, ad esempio, la parola *unicorn* 🦄

```py
print( embed( "unicorn" ) )
```

Come potrai vedere, il risultato è una sequenza incomprensibile di 50 numeri. Ebbene, questo è ciò che basta ad un modello di linguaggio artificiale per imparare il significato delle parole!

#### Misura la similarità tra due parole

È possibile valutare quanto due parole sono "vicine" di significato misurando la similarità tra i vettori. Un metodo comune per effettuare questa misura è applicare la *similarità del coseno*, che è già presente in uno dei pacchetti python che abbiamo importato all'inizio del file. Esegui questo codice in una nuova cella:

```py
def sim( word1, word2 ):
    word1   = embed( word1 )
    word2   = embed( word2 )
    s       = losses.cosine_similarity( word1, word2 )
    return s.numpy()
```

Questa funzione restituisce un numero tra -1 e 1. Se il risultato è vicino a -1, vuol dire che le due parole sono simili; se il risultato è vicino a 0 le due parole hanno poca similarità.

Prova, ad esempio, a misurare la similarità tra *dog* e *wolf* 🐶🐺, e tra *dog* e *galaxy* 🐶🪐

```py
sim( 'dog', 'wolf' )
```
```py
sim( 'dog', 'galaxy' )
```

Prevedibilmente, la parola *dog* ha molta più similarità con *wolf* che con *galaxy*.

#### Trova le parole più simili

Data una parola, trova quali sono le parole più "vicine" a questa. Bisogna definire una funzione che analizza il dizionario, calcola le similarità tra le parole e quella di input, e salva le parole che hanno similarità migliore.

Il numero di parole da dare come risultato è indicato dal parametro `n_words`. Inoltre, dato che il dizionario è molto grande, è utile aggiungere un parametro `limit` che limita il numero totale di parole da analizzare.

Incolla ed esegui questo in una nuova cella: 

```py
def closest( word, n_words=5, limit=50000 ):
    word      = embed( word )
    cnt       = 0
    
    # mantiene le parole con similarità migliore
    best      = [ ( None, 1.0 ) ]
    
    # analizza tutto il dizionario
    for w in embedding.keys():
    
        # per ogni parola, misura la similarità con quella in input
        score       = sim( embedding[ w ], word )

        # salta la parola se è identica a quella in input
        if ( score + 1 ) < 0.05:
            continue

        # salva la parola se migliore di una di quelle salvate finora
        for i, ( v, s ) in enumerate( best ):
            if score < s:
                best.insert( i, ( w, score ) )
                del best[ n_words: ]
                break

        # visualizza il progresso della funzione
        cnt += 1
        if not cnt % 1000:
            print( f"checked {cnt:,} of {limit:,} words" )
        if cnt > limit:
            print()
            break

    return best
```

Adesso prova ad esempio a trovare le 10 parole più simili a *queen* 👑 limitando la ricerca alle prima 20K parole:

```py
closest( 'queen', n_words=10, limit=20000 )
```

Come vedrai, la parola più vicina è *princess* con un punteggio di -0.851.

#### Somma le parole insieme (Whaaat?!) 

Dato che abbiamo trasformato le parole in vettori numerici, è possibile eseguire operazioni artimetiche sulle parole, come somma e sottrazione! Prova ad eseguire questo codice:

```py
# somma tra due parole
def plus( word1, word2 ):
    word1   = embed( word1 )
    word2   = embed( word2 )
    return word1 + word2

# sottrazione tra due parole
def minus( word1, word2 ):
    word1   = embed( word1 )
    word2   = embed( word2 )
    return word1 - word2
```

Forse ti starai chiedendo, qual è il senso di eseguire operazioni tra parole? Un classico esempio è il seguente: se alla parola *king* si sottrae la parola *man* e poi si aggiunge *woman*, il risultato sarà una parola estremamente simile a *queen*. Provare per credere!

```py
w = minus( 'king', 'man' )
w = plus( w, 'woman' )
sim( w, 'queen' )
```

Il punteggio di similarità tra *queen* e il risultato dell'operazione è -0.860 (un valore migliore del punteggio di similarità tra *queen* e *princess* che era -0.851) 👸🏽✨

Vuoi un esempio più complesso? Cosa succede se alla parola *queen* si toglie *throne* e si aggiunge *job*? Con il seguente codice, calcoliamo il risultato e visualizziamo le 30 parole più simili:

```py
w = minus( 'queen', 'throne' )
w = plus( w, 'job' )
closest( w, n_words=30, limit=30000 )
```

Noterai che stavolta i risultati sono più confusionari. Si trovano parole azzeccate come *experience* e *working*, ma altre sono abbastanza generiche come *get* e *doing*. In questo caso abbiamo limitato la ricerca alle prime 30K parole, puoi provare ad alzare il limite. Inoltre, ricorda che abbiamo usato il dizionario GloVe più piccolo disponibilie, puoi provare a scaricare uno dei dataset più grandi ([link](https://nlp.stanford.edu/projects/glove/)) e giocherellare con un embedding ancora migliore!


| Capitolo precedente                                                                                                                                          | Capitolo successivo                                                                           |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------: |
| [◀︎ 01 Come funziona ChatGPT](../01-come-funziona-gpt)  | [03 Tokenization ▶︎](../03-tokenization) |

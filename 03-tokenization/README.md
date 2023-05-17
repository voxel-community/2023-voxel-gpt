# 03. Tokenization e Completion

| Capitolo precedente                                                                                                                                          | Capitolo successivo                                                                           |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------: |
| [◀︎ 02 Embedding](../02-embedding)  | [04 Temperature ▶︎](../04-temperature) |

## Obiettivo

Impara come funzionano i processi di *tokenization* e *completion* creando un semplice modello di linguaggio.

La tokenization è il processo di **suddividere un testo in unità più piccole chiamate token**, che sono generalmente parole o sottostringhe. Lo scopo della tokenizzazione è quello di preparare il testo per l'analisi da parte di un modello di linguaggio.

Ad esempio, la frase "Il cane nel prato" sarebbe tokenizzata in quattro token: `il`, `cane`, `nel` e `prato`. Ma non sempre una parola corrisponde ad un token: ad esempio, la parola "incredibile" potrebbe essere suddivisa in due token `in` e `credibile`.

In realtà, la tokenization è lo step precedente all'embedding visto nel capitolo precedente. Infatti, anziché le parole, solitamente sono i token che vengono trasformati in vettori.

La completion è il processo che consente ad un modello di linguaggio di generare o "completare" un testo in base a una *prompt* o a un input parziale. In pratica, il modello utilizza l'input come punto di partenza e poi **predice la sequenza di token più probabile** da seguire.

Ad esempio, supponiamo che il modello riceva la prompt "Il tempo è ". Il modello potrebbe generare il completamento "bellissimo oggi". Questo perché il modello ha appreso dai suoi dati di formazione che la parola "bellissimo" spesso segue "Il tempo è" in contesti positivi ☀️

> ***Curiosità: qual è la relazione tra tokenization ed embedding?***
> Solitamente, la tokenization viene eseguita prima dell'embedding. Il testo viene prima suddiviso in token, e poi ogni token viene mappato al corrispondente vettore di embedding. In realtà, nel caso di modelli di linguaggio sofisticati come ChatGPT, tokenization e embedding non sono due passaggi separati, ma vengono invece appresi contemporaneamente durante il processo di addestramento. Ciò consente a ChatGPT di comprendere ed elaborare in modo efficace il testo in base al significato semantico delle parole e al loro contesto all'interno di una frase o di un testo più ampio.

In questo esercizio, creerai una versione molto semplice di modello di linguaggio basato sull'architettura Transformer. Utilizzerai i moduli della libreria [KerasNLP](https://keras.io/keras_nlp/) per costruire il modello e realizzare una *completion* basilare.

Pronte per iniziare! 💪




## Steps


### 1. Installa i pacchetti e i file necessari

Su Google Colab crea un nuovo notebook e rinominalo ad esempio in `tokenization.ipynb`.
Crea una cella di codice, incolla il seguente comando ed esegui la cella per installare la libreria KerasNLP e scaricare i file che ti serviranno:

```
!pip install keras-nlp==0.4.1
!wget https://www.dropbox.com/s/kuxjrdz9kwxdovg/gpt_v15000_l3_h4_e100.h5
!wget https://www.dropbox.com/s/yze44qacqgqmd1u/vocab_15000.txt
```

Il file `gpt_v15000_l3_h4_e100.h5` contiene i "pesi" di una rete che è stata già addestrata. Ti verrà comodo utilizzarli per evitare di dover addestrare la rete da zero. Il file `vocab_15000.txt` contiene il "vocabolario" di token selezionati per poter suddiviere il testo in maniera efficiente.

In una nuova cella, esegui il seguente codice per importare i pacchetti necessari:

```py
import  os
import  keras_nlp
import  numpy         as      np
import  tensorflow    as      tf
from    tensorflow    import  keras
```

### 2. Crea il *tokenizer*

Per prima cosa, creiamo il "tokenizer". Si tratta di un ogetto della libreria `keras_nlp` che ha il compito di trasformare un testo in una sequenza di token, e viceversa.

Il seguente codice crea un oggetto `tokenizer` che utilizza il vocabolario di tokens contenuto nel file `vocab_15000.txt` per processare qualsiasi testo:

```py
def set_tokenizer():
    fname       = "vocab_15000.txt"
    with open( fname, 'r' ) as f:
        vocab       = f.read()
    vocab       = vocab.split()
    tokenizer   = keras_nlp.tokenizers.WordPieceTokenizer(
            vocabulary      = vocab,
            sequence_length = 128,
            lowercase       = True
    )
    return tokenizer
```

Le due funzioni che seguono utilizzano il `tokenizer` per trasformare testo in lista di tokens e viceversa:

```py
# testo -> lista di token
def to_tokens( text, tokenizer ):
    tokens      = tokenizer( text.lower() )
    tokens      = tokens.numpy()
    return np.trim_zeros( tokens )

# lista di token -> testo    
def from_tokens( tokens, tokenizer ):
    text        = tokenizer.detokenize( tokens )
    return text.numpy()    
```

### 3. Testa il *tokenizer*

Prova ad eseguire delle tokenizazion 🔡
Inizia creando l'oggetto `tokenizer`:

```py
tokenizer = set_tokenizer()
```
Adesso prova a trasformare la stringa *"Why you don't you say something about it? It's wonderful!"* in tokens. Naturalmente puoi provare il testo che preferisci! Basta modificare il valore della variabile `s` qui sotto:

```py
s = "Why you don't you say something about it? It's wonderful!"
t = to_tokens( s, tokenizer )
print( t )
```
Il risultato dovrebbe essere una lista di numeri interi. Questi corrispondono agli indici dei token nella lista del vocabolario in `vocab_15000.txt`.

Prova a fare il contrario. Usa questa lista di indici per generare il testo corrispondente:

```py
from_tokens( t, tokenizer ).decode( "utf-8" )
```
Il risultato sarà una stringa con i "valori testuali" dei corrispondenti token, separati da spazi. Puoi notare come, ad esempio, la parola *don't* viene suddivisa in tre token `don`, `'`, e `t`, che corrispondono ai codici `275`, `8`, e `56`.

Per curiosità, puoi visualizzare i valori testuali dei token che hanno codice, ad esempio, tra 300 e 320:

```py
from_tokens( np.arange( 300, 320 ), tokenizer ).decode( "utf-8" )
```


### 4. Crea il modello di linguaggio

Adesso che hai un tokenizer per trasformare testo in tokens, puoi creare un modello di linguaggio per analizzare i tokens e generare nuovo testo.

Il seguente codice crea un modello di linguaggio completo. Al suo interno contiene un layer `embedding` per codificare i possibili 15mila tokens (`vocab_size=15000`) in vettori di embedding, ciascuno lungo 256 unità (`embedding_dim=256`). In aggiunta, il modello contiene tre layer Transformer che costituiscono il cuore del modello dove avviene la magia ✨. L'ultimo layer di output contiene 15mila neuroni, dove ciascun neurone codifica quanto è probabile che il corrispondente token sia il prossimo da generare nel testo.

```py
def create_model():
    n_layers    = 3
    vocab_size  = 15000
    inputs      = keras.layers.Input( shape=(None,), dtype=tf.int32 )

    embedding   = keras_nlp.layers.TokenAndPositionEmbedding(
            vocabulary_size = vocab_size,
            sequence_length = 128,
            embedding_dim   = 256,
            mask_zero       = True
    )
    x = embedding( inputs )

    for i in range( n_layers ):
        name    = "decoder_{:02d}".format( i )
        decoder = keras_nlp.layers.TransformerDecoder(
            num_heads           = 4,
            intermediate_dim    = 256
        )
        x = decoder( x )

    outputs     = keras.layers.Dense( vocab_size )( x )
    model       = keras.Model( inputs=inputs, outputs=outputs )

    return model
```

### 5. Inizialliza il modello

La funzione che hai creato allo step precedente crea un modello "vuoto", che ha bisogno di essere addestrato prima che impari a funzionare in modo accetabile. Solitamente l'addestramento richiede tempi lunghi (anche moooolto lunghi) ma niente paura, c'è un'alternativa!

È possibile recuperare i "pesi" di un modello già addestrato e caricarli sul modello che abbiamo appena creato. Il file scaricato all'inizio `gpt_v15000_l3_h4_e100.h5` contiene infatti i pesi di un modello con la stessa struttura di quello che abbiamo:

```py
def init():
    model     = create_model()
    weights   = "gpt_v15000_l3_h4_e100.h5"
    model.load_weights( weights )
    return model
```

Dopo aver caricato i pesi, adesso il modello è pronto per essere usato!

```py
model = init()
```



### 6. Usa il modello per fare *completion*

Adesso vogliamo utilizzare il modello per fare *completion* di una prompt, ovvero dato un testo parziale il modello deve generare una predizione di come "completare" il testo.

Innanzitutto, è necessaria una funzione che prepari un testo per essere dato in pasto al modello. Qui è il momento di usare il nostro `tokenizer`! Ma non basta, è utile aggiungere degli spazi prima di ogni simbolo di punteggiatura. Inoltre, bisogna trasformare la lista di codici di token in un "tensore", ovvero un tipo di array multidimensionale utilizzato all'interno di una rete neurale. 

```py
def get_prompt( tokenizer, prompt=None ):
    p = ''
    for c in prompt:
        if c in string.punctuation:
            p = p + ' ' + c
        else:
            p = p + c
    prompt = p
    
    if prompt is None:
        prompt_list     = [ tokenizer.token_to_id( "[BOS]" ) ]
    else:
        prompt_list     = [ tokenizer.token_to_id( w ) for w in prompt.lower().split() ]
    prompt_tokens   = tf.convert_to_tensor( prompt_list )
    return prompt_tokens
```

A questo punto, puoi direttamente usare il tuo `model` per predire il prossimo token da generare. Questa funziona semplicemente seleziona dal risultato del modello (`prediction`) il token con più alta probabilità (`max_prob`). Infine, utilizza il tokenizer per convertire il token in una parola.

```py
def next_token( model, tokenizer, prompt=None ):
    prompt_tokens   = get_prompt( tokenizer, prompt )
    prompt_tokens   = prompt_tokens[ tf.newaxis, : ]
    prediction      = model( prompt_tokens )
    max_prob        = tf.argmax( prediction, axis=-1 ).numpy()
    token 			= from_tokens( max_prob, tokenizer )
    token 			= token[ 0 ].decode( "utf-8" )
    token 			= token.split( ' ' )[ -1 ]
    return token
```

Il processo di completion più basilare è quindi selezionare ad ogni passo il token con probabilità maggiore, e concatenarlo alla prompt per il passo successivo:

```py
def basic_completion( model, tokenizer, prompt=None, max_length=100 ):
    p = prompt
    while len( p ) < max_length:
        t = next_token( model, tokenizer, prompt=p )
        p = p + ' ' + t
    return p
```

Tuttavia questo non è il metodo più efficace. Esistono algoritmi più sofisticati per selezionare la sequenza di token da generare. Uno di questi è la [Top-p search](https://keras.io/api/keras_nlp/utils/top_p_search/):

```py
def top_p_completion( model, tokenizer, prompt=None, max_length=100 ):
    prompt_tokens   = get_prompt( tokenizer, prompt )

    def predict_fn( inputs ):
        cur_len     = inputs.shape[ 1 ]
        output      = model( inputs )
        return output[ :, cur_len - 1, : ]

    output_tokens = keras_nlp.utils.top_p_search(
        predict_fn,
        prompt_tokens,
        max_length      = max_length,
        p               = 0.5,
        from_logits     = True
    )
    text            = tokenizer.detokenize( output_tokens )
    text            = text.numpy()
    return text
```

### 7. Testa la *completion*

È arrivato il momento di testare la completion! Ricorda che quello che abbiamo creato in questo esercizio, è un modello veramente basilare. Per risultati migliori, servono modelli decisamente più complessi, che vedremo nei capitoli successivi.

Per prima cosa, puoi provare a farti predirre una singola parola da una prompt parziale. Ad esempio:

```py
prompt = "There is something in here that we"
next_token( model, tokenizer, prompt )
```
Il risultato predetto dovrebbe essere la parola `had`. E fin qui okay! Prova a passare la stessa prompt alla funzione che esegue la completion di base:

```py
prompt = "There is something in here that we"
basic_completion( model, tokenizer, prompt, max_length=150 )
```

Come avrai notato, questo algoritmo "greedy" non funziona molto bene 😅 Un problema comune accade quando la completion inizia a generare sempre la stessa breve sequenza di parole, in questo caso `of the other`.

Proviamo come se la cava l'algoritmo Top-p. Essendo un algoritmo non-deterministico, il risultato sarà diverso ogni volta che esegui il codice:

```py
prompt = "There is something in here that we"
top_p_completion( model, tokenizer, prompt, max_length=150 ).decode( "utf-8" )
```

Il risultato sarà prevedibilmente molto... creativo 🤪 La frase generata è con ogni probabilità priva di senso, ma a differenza della completion basilare, è più improbabile che l'algoritmo si "incastri" e finisca a generare sempre la stessa sequenza di parole.

Ottimo lavoro! Adesso hai chiari i concetti di tokenization e completion, e di come sia strutturato un modello di linguaggio essenziale. Sei pronta per creare qualcosa di più sofisticato 🧐

Passa al capitolo successivo!


| Capitolo precedente                                                                                                                                          | Capitolo successivo                                                                           |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------: |
| [◀︎ 02 Embedding](../02-embedding)  | [04 Temperature ▶︎](../04-temperature) |

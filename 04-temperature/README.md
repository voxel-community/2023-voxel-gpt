# 04. Il parametro Temperature

| Capitolo precedente                                                                                                                                          | Capitolo successivo                                                                           |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------: |
| [◀︎ 03 Tokenization](../03-tokenization)  | [05 OpenAI ▶︎](../05-openai) |

## Obiettivo

Impara l'effetto del "parametro di temperatura" usando un modello di linguaggio della famiglia GPT.

La *temperature* 🌡 in un modello di linguaggio si riferisce al parametro che **controlla la casualità delle previsioni** generate dal modello. Una temperatura più alta rende le previsioni più casuali e più diverse, mentre una temperatura più bassa le rende più deterministe e più simili al training data.

In pratica, una temperatura più alta 🥵 può portare a risposte più creative ma potenzialmente meno coerenti, mentre una temperatura più bassa 🥶 può produrre risposte più coerenti ma potenzialmente meno sorprendenti. 

Questa volta non creerai un modello da zero, come fatto nel capitolo precedente. Userai un modello già addestrato disponibile sulla piattaforma [**Hugging Face**](https://huggingface.co/) 🤗. Il modello si chiama [DistilGPT2](https://huggingface.co/distilgpt2), ed è una versione "compressa" del modello GPT-2 ottenuta tramite *knowledge distillation*.

La *knowledge distillation* è un processo in cui un modello più grande e più complesso (detto "teacher model") viene utilizzato per istruire un modello più piccolo (detto "student model"). L'obiettivo della knowledge distillation è trasferire la conoscenza e l'abilità del modello insegnante al modello studente in modo che il modello studente possa ottenere prestazioni simili o addirittura migliori, ma con un costo computazionale inferiore.

> ***Curiosità: cos'è Hugging Face?***
> Hugging Face è una società che fornisce una libreria di software open-source e strumenti di elaborazione del linguaggio naturale all'avanguardia. La loro libreria, chiamata *transformers*, include modelli pre-allenati per una varietà di compiti, come la generazione di linguaggio, la risposta alle domande e la classificazione del testo.


## Steps


### 1. Installa e importa i pacchetti necessari

Su Google Colab crea un nuovo notebook e rinominalo ad esempio in `temperature.ipynb`. Crea una cella di codice, incolla il seguente comando ed esegui la cella per installare la libreria `transformers`:

```
!pip install transformers
```
Ed importa i pacchetti python che ti serviranno:

```py
import  os
from    transformers       import pipeline
from    nltk               import ngrams
import  warnings
warnings.filterwarnings( 'ignore' )
```
Il codice seguente non è necessario, ma può essere utile per migliorare la visualizzazione del testo in Colab. Dato che nel corso dell'esercizio dovrai farti stampare lunghe stringhe di testo, è utile spezzare le righe con delle andate a capo (text wrapping):

```py
from IPython.display  import HTML, display
def set_css():
    display( HTML( '''<style> pre { white-space: pre-wrap; } </style>''' ) )
get_ipython().events.register( 'pre_run_cell', set_css )
```

### 2. Scarica il modello da Hugging Face

Con questo codice, scarichi il modello `distilgpt2` per la generazione di testo dalla libreria di Hugging Face:

```py
def get_model( model_name ):
    return pipeline( "text-generation", model=model_name, device=-1 )

model_name  = "distilgpt2"
model       = get_model( model_name )
```

Questo modello è pre-addestrato e pronto per essere usato per generare testo.

### 3. Imposta la *completion* del modello

Per generare testo con il modello, ti serve una funzione che faccia la *completion* di una prompt. Il seguente codice, data una prompt e un valore di temperatura, genera tre possibili sequenze di risultato diverse, lunghe tra 100 e 400 tokens:

```py
def completion( model, prompt, temp ):
    res     = model(
        prompt,
        min_length              = 100,
        max_new_tokens          = 400,
        num_return_sequences    = 3,
        temperature             = temp
    )
    return [ t[ 'generated_text' ] for t in res ]
```

Dato che il modello restituisce più di un risultato, è utile avere una funzione valuta la "bontà" di ciascun risultato. Un metodo semplice è misurare il numero di ripetizioni degli ***n-grams***. 

Gli *n-grams* sono sequenze di parole consecutive di lunghezza "n" nel testo. Ad esempio, un "bigram" rappresenta una sequenza di due parole consecutive, un "trigram" rappresenta una sequenza di tre parole consecutive e così via. In generale, se un risultato contiene troppe ripetizioni di certi *n-gram*, è un'indicazione che la frase generata non è ben formata.

```py
def check_res( res, max_reps=None, min_len=100 ):
    if len( res ) < min_len:
        # completion too short
        return False
        
    words       = res.split()
    nw          = len( words )
    if max_reps is None:
        # a reasonable maximum number of repetitions
        max_reps    = nw // 25

    reps        = []
    # check for 3-grams, 4-grams, and 5-grams
    for n in range( 3, 6 ):
        ng      = list( ngrams( words, n=n ) )
        reps    += [ ng.count( w ) for w in ng ]
        
    if max( reps ) > max_reps:
        # too many repetitions
        return False
    return True
```

Con la funzione seguente, puoi visualizzare i risultati della completion, insieme al controllo sulla "bontà" delle sequenze:

```py
def print_res( res, max_rep ):
    for i, r in enumerate( res ):
        print( '=' * 40 )
        print( "RESULT", i+1 )
        print( "Too many repetitions?", "No" if check_res( r, max_reps ) else "Yes" )
        print( '=' * 40 )
        print( '\n' + r + '\n\n' )
```

### 4. Testa il modello con diverse *temperature*

Adesso è il momento di divertirsi!! Scatenati a provare le prompt più curiose e gioca con la temperature per vedere come cambiano i risultati 😄

La temperature deve essere un numero compreso tra 0 e 2. Un valore vicino a 0 ❄️ equivale ad una temperatura molto fredda e a risultati ripetitivi, mentre un valore sopra 1 🔥 equivale ad una temperatura alta e a risultati caotici.

```py
prompt    = "Once upon a time, in a land far far away, there was"
temp      = 0.8
max_reps  = 10

res       = completion( model, prompt, temp )
print_res( res, max_reps )
```

```py
prompt    = "They were locked together in the room. She knew one of them was the murderer. Therefore, she"
temp      = 1.2
max_reps  = 10

res       = completion( model, prompt, temp )
print_res( res, max_reps )
```

Nota che il modello DistilGPT2 supporta solo la lingua inglese, pertanto la prompt deve essere una frase in inglese. Enjoy! 🇬🇧


| Capitolo precedente                                                                                                                                          | Capitolo successivo                                                                           |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------: |
| [◀︎ 03 Tokenization](../03-tokenization)  | [05 OpenAI ▶︎](../05-openai) |

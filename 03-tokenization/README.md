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

# TBD.............
In questo esercizio, creerai un semplice modello di linguaggio basato sull'architettura Transformer



## Steps


### 1. Installa i pacchetti e i file necessari

```
!pip install keras-nlp
!wget https://www.dropbox.com/s/kuxjrdz9kwxdovg/gpt_v15000_l3_h4_e100.h5
!wget https://www.dropbox.com/s/yze44qacqgqmd1u/vocab_15000.txt
```

```py
import  os
import  keras_nlp
import  tensorflow    as      tf
import  numpy         as      np
from    tensorflow    import  keras
```

### 2. Crea il *tokenizer*

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

```py
def to_tokens( text, tokenizer ):
    tokens      = tokenizer( text.lower() )
    tokens      = tokens.numpy()
    return np.trim_zeros( tokens )

    
def from_tokens( tokens, tokenizer ):
    text        = tokenizer.detokenize( tokens )
    return text.numpy()    
```

### 3. Crea il modello di linguaggio

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

### 4. Inizialliza il modello e il *tokenizer*

```py
def init():
    tokenizer = set_tokenizer()
    model     = create_model()
    weights   = "gpt_v15000_l3_h4_e100.h5"
    model.load_weights( weights )
    return model, tokenizer
```

```py
model, tokenizer = init()
```

### 5. Testa il *tokenizer*

```py
t = to_tokens( "Why you don't you say something about it? It's wonderful!", tokenizer )
print( t )
```

```py
from_tokens( t, tokenizer ).decode( "utf-8" )
```

```py
from_tokens( np.arange( 300, 320 ), tokenizer ).decode( "utf-8" )
```

### 6. Usa il modello per fare *completion*

```py
def get_prompt( tokenizer, prompt=None ):
    if prompt is None:
        prompt_list     = [ tokenizer.token_to_id( "[BOS]" ) ]
    else:
        prompt_list     = [ tokenizer.token_to_id( w ) for w in prompt.lower().split() ]
    prompt_tokens   = tf.convert_to_tensor( prompt_list )
    return prompt_tokens
```

```py
def next_tokens( model, tokenizer, prompt=None ):
    prompt_tokens   = get_prompt( tokenizer, prompt )
    prompt_tokens   = tf.convert_to_tensor( prompt_list )
    prompt_tokens   = prompt_tokens[ tf.newaxis, : ]
    prediction      = model( prompt_tokens )
    max_prob        = tf.argmax( prediction, axis=-1 ).numpy()
    return from_tokens( max_prob, tokenizer )
```

```py
def next_token( model, tokenizer, prompt=None ):
    t   = next_tokens( model, tokenizer, prompt )[ 0 ].decode( "utf-8" )
    t   = t.split( ' ' )[ -1 ]
    return t
```

```py
def completion( model, tokenizer, prompt=None, method="top_p", max_length=80 ):
    prompt_tokens   = get_prompt( tokenizer, prompt )

    def predict_fn( inputs ):
        cur_len     = inputs.shape[ 1 ]
        output      = model( inputs )
        return output[ :, cur_len - 1, : ]

    if method == "beam":
        output_tokens = keras_nlp.utils.beam_search(
            predict_fn,
            prompt_tokens,
            max_length      = max_length,
            num_beams       = 5,
            from_logits     = True
        )
    if method == "top_k":
        output_tokens = keras_nlp.utils.top_k_search(
            predict_fn,
            prompt_tokens,
            max_length      = max_length,
            k               = 5,
            from_logits     = True
        )
    elif method == "top_p":
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

### 7. Testa la *completion* del modello

```py
prompt = "There is something in here that we"
next_token( model, tokenizer, prompt=prompt )
```

```py
prompt = "Once upon a time , in a land far far away , there was"
completion( model, tokenizer, prompt=prompt, max_length=100, method='top_p' ).decode( "utf-8" )
```


| Capitolo precedente                                                                                                                                          | Capitolo successivo                                                                           |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------: |
| [◀︎ 02 Embedding](../02-embedding)  | [04 Temperature ▶︎](../04-temperature) |

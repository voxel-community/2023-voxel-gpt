# 04. Il parametro Temperature

| Capitolo precedente                                                                                                                                          | Capitolo successivo                                                                           |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------: |
| [◀︎ 03 Tokenization](../03-tokenization)  | [05 OpenAI ▶︎](../05-openai) |

## Obiettivo

Impara 🌡🥵🥶

[**HuggingFace**](https://huggingface.co/) 🤗

> ***Curiosità:***
> Solitamente


# TBD.............



## Steps


### 1. Installa i pacchetti necessari

```
!pip install transformers
```

```py
import  os
from    transformers       import pipeline
from    nltk               import ngrams
import  warnings
warnings.filterwarnings( 'ignore' )
```

```py
from IPython.display  import HTML, display
def set_css():
    display( HTML( '''<style> pre { white-space: pre-wrap; } </style>''' ) )
get_ipython().events.register( 'pre_run_cell', set_css )
```

### 2. Scarica il modello da HuggingFace

```py
def get_model( model_name ):
    return pipeline( "text-generation", model=model_name, device=-1 )

model_name  = "distilgpt2"
model       = get_model( model_name )
```

### 3. Imposta la *completion* del modello

```py
def completion( model, prompt, temp=0.8 ):
    res     = model(
        prompt,
        min_length              = 100,
        max_new_tokens          = 400,
        num_return_sequences    = 3,
        temperature             = temp
    )
    return [ t[ 'generated_text' ] for t in res ]
```

```py
def check_res( res, max_reps=None, min_len=100, diagnostic=False ):
    if len( res ) < min_len:
        if diagnostic:
            print( "compl too short: {} characters".format( len( res ) ) )
        return False
        
    words       = res.split()
    nw          = len( words )
    
    if max_reps is None:
        max_reps    = nw // 25  # a reasonable maximum number of repetitions

    reps        = []
    for n in range( 3, 6 ):     # check for 3-grams up to 5-grams
        ng      = list( ngrams( words, n=n ) )
        reps    += [ ng.count( w ) for w in ng ]
        
    if max( reps ) > max_reps:
        if diagnostic:
            idx = reps.index( max( reps ) )
            print( "too many repetitions {} for n-gram {}".format( reps[ idx ], ng[ idx ] ) )
        return False
    return True
```

```py
def print_res( res, max_rep ):
    for i, r in enumerate( res ):
        print( '=' * 40 )
        print( "RESULT", i+1 )
        print( "Too many repetitions?", check_res( r, max_reps ) )
        print( '=' * 40 )
        print( '\n' + r + '\n\n' )
```

### 4. Testa il modello con diverse *temperature*

```py
prompt    = "Once upon a time, in a land far far away, there was"
temp      = 0.8
max_reps  = 10

res       = generate( model, prompt, temp )
print_res( res, max_reps )
```

```py
prompt    = "They were locked together in the room. She knew one of them was the murderer. Therefore, she"
temp      = 1.2
max_reps  = 10

res       = generate( model, prompt, temp )
print_res( res, max_reps )
```



| Capitolo precedente                                                                                                                                          | Capitolo successivo                                                                           |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------: |
| [◀︎ 03 Tokenization](../03-tokenization)  | [05 OpenAI ▶︎](../05-openai) |

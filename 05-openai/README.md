# 05. Le API di OpenAI

| Capitolo precedente                                                                                                                                          | Capitolo successivo                                                                           |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------: |
| [◀︎ 04 Temperature](../04-temperature)  | [Torna all'inizio ▶︎](../../..) |


## Obiettivo

> ***Curiosità:***
> Solitamente


# TBD.............



## Steps


### 1. Installa i pacchetti necessari

```
!pip install openai jsonlines
```

```py
import  os
import  openai
import  time
import  jsonlines
import  numpy       as np
from    itertools   import product
```

```py
from IPython.display  import HTML, display
def set_css():
    display( HTML( '''<style> pre { white-space: pre-wrap; } </style>''' ) )
get_ipython().events.register( 'pre_run_cell', set_css )
```

### 2. Imposta la chiave persionale per l'API

```py
KEY = "XX-XXXXXXXX"
openai.api_key  = KEY
```

### 3. Genera un dataset personalizzato

#### 3a. Definisci la prompt

```py
main_prompt  = "Create a detailed description of a Dungeons and Dragons character with "
main_prompt += "{race} race, {clss} class, {weapon} as favorite weapon, and {pet} as beloved pet. "
main_prompt += "Write out the description as a list of name, physical appearance, background, and personality in a maximum of 100 words:"
sub_prompt   = "{race}, {clss}, {weapon}, {pet}"
```

```py
list_race   = [ 'human', 'elf' ]
list_class  = [ 'ranger', 'wizard' ]
list_weapon = [ 'sword', 'crossbow' ]
list_pet    = [ 'wolf', 'fox' ]
```

#### 3b. Imposta un modello per fare *completion*

```py
def get_completion( model, prompt, temperature=0.8, max_tokens=500 ):
    res = openai.Completion.create(
          model             = model,
          prompt            = prompt,
          temperature       = temperature,  # range [0, 2]
          max_tokens        = max_tokens
    )
    return res  # return a JSON format object
```

```py
def view_completion( model, prompt, temperature=0.8, max_tokens=500 ):
    res = get_completion( model, prompt, temperature=temperature, max_tokens=max_tokens )
    print( res[ 'choices' ][ 0 ][ 'text' ] )
```

#### 3c. Usa la *completion* per generare il dataset

```py
def generate_data( model, temperature=0.8, max_tokens=500 ):
    data  = []

    # to check progress
    cnt   = 1
    cmx   = np.prod( [ m for m in  map( len, [ list_race, list_class, list_weapon, list_pet ] ) ] )

    for ( r, c, w, p ) in product( list_race, list_class, list_weapon, list_pet ):        
        t1      = time.time()

        prmt    = main_prompt.format( race=r, clss=c, weapon=w, pet=p )
        s_prmt  = sub_prompt.format( race=r, clss=c, weapon=w, pet=p )

        res     = get_completion( model, prmt, temperature, max_tokens )
        res_t   = res[ 'choices' ][ 0 ][ 'text' ]

        data.append( { "prompt": s_prmt, "completion": res_t } )

        # print progress
        t2      = time.time()
        t       = int( t2 - t1 )
        print( f"{ cnt }/{ cmx } ({ t } sec)" )
        cnt     += 1
    return data
```

```py
def write_json( dataset, fname ):
    with jsonlines.open( fname, 'w' ) as f:
        f.write_all( dataset )
```

```py
def upload_dataset( fname ):
    f   = open( fname, 'rb' )
    res = openai.File.create( file=f, purpose='fine-tune' )
    return res[ 'id' ]  # return id of file
```

#### 3d. Genera il dataset!

```py
gen_model  = "text-davinci-003"
res        = get_completion( gen_model, "Generate a cute name for a unicorn", temperature=1.3 )
print( res )
```

```py
dataset = generate_data( gen_model, max_tokens=200 )
```

```py
dataset[ 0 ]
```

### 4. Salva il dataset oppure usa uno già pronto

```py
... ?
```

### 5. Esegui il *finetuning* di un modello

#### 5a. Definisci le funzioni per fare *finetuning*

```py
def finetune( model, train_file, suffix, n_epochs=4 ):
    res = openai.FineTune.create(
        training_file = train_file,
        model         = model,
        n_epochs      = n_epochs,
        suffix        = suffix
    )
    return res[ 'id' ]  # return id of fine-tune job
```

```py
def get_status( id ):
    res = openai.FineTune.list_events( id=id, stream=False )
    res = res[ 'data' ]
    for r in res:
        print( r[ 'message' ] )
```

```py
def retrieve_model( id ):
    res = openai.FineTune.retrieve( id=id )
    return res[ 'fine_tuned_model' ]
```

#### 5b. Lancia il *finetuning*

```py
custom_name = "XXXXX"
ftune_model = "ada"
id_job      = finetune( ftune_model, id_file, custom_name, n_epochs=20 )

```

```py
get_status( id_job )
```

```py
openai.FineTune.cancel( id=id_job )
```

### 6. Testa il tuo VoxelGPT!

[Playground](https://platform.openai.com/playground)

```py
tuned_model = retrieve_model( id_job )
```

```py
prompt = "mermaid, monk, sling, crab"
temp   = 0.9
view_completion( tuned_model, prompt, temperature=temp, max_tokens=200 )
```

| Capitolo precedente                                                                                                                                          | Capitolo successivo                                                                           |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------: |
| [◀︎ 04 Temperature](../04-temperature)  | [Torna all'inizio ▶︎](../../..) |
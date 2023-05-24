# 05. Le API di OpenAI

| Capitolo precedente                                                                                                                                          | Capitolo successivo                                                                           |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------: |
| [◀︎ 04 Temperature](../04-temperature)  | [Torna all'inizio ▶︎](../../..) |


## Obiettivo

Sei arrivata all'ultimo esercizio di oggi ✨ Adesso usa il vero ChatGPT per creare una tua versione personalizzata!

Nei capitoli precedenti, ti è capitato di: 1) creare un modello da zero; 2) scaricare i pesi di un modello pre-addestrato.
In questo capitolo, farai una cosa ancora diversa. Utilizzerai il processo di ***fine-tuning*** per modificare direttamente il modello di ChatGPT.

Il fine-tuning è una tecnica per adattare un modello già funzionante a un nuovo compito specifico. Si parte da un modello esistente che ha già imparato informazioni generali da un ampio dataset, e lo si addestra su un nuovo dataset più piccolo e specifico per un nuovo compito.

L'idea è quindi di usare il modello GPT-3 (quello dietro ChatGPT) per fare fine-tuning su un piccolo dataset creato da noi per imparare un nuovo compito. In questo esempio, il nuovo compito sarà generare descrizioni di personaggi per il gioco [Dungeons & Dragons](https://dnd.wizards.com/) 🏰


> ***Curiosità: Qual è la differenza tra fine-tuning e transfer learning?***
> Sono entrambe tecniche per applicare un modello già addestrato ad un nuovo compito. Solitamente, col fine-tuning si ri-addestrano tutti i pesi del modello su un nuovo dataset, utilizzando un *learning rate* moto basso. D'altra parte, il transfer learning consiste nel "congelare" i pesi del modello e aggiungere dei nuovi layer in cima al modello iniziale. Durante l'addestramento, solo i pesi dei nuovi layer vengono modificati.


L'esercizio consiste in due fasi: a) creare il nuovo dataset personalizzato, b) utilizzare il dataset per fare fine-tuning. Procediamo!


## Steps


### 1. Installa i pacchetti necessari

Su Google Colab, crea un nuovo notebook e rinominalo ad esempio in `finetuning.ipynb`. Esegui i codici seguenti per installare le librerie e importare i pacchetti necessari:

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

Come nello scorso esercizio, ti può essere utile modificare il settaggio per il "text wrap" in modo da migliorare la visualizzazione delle lunghe porzioni di testo:

```py
from IPython.display  import HTML, display
def set_css():
    display( HTML( '''<style> pre { white-space: pre-wrap; } </style>''' ) )
get_ipython().events.register( 'pre_run_cell', set_css )
```

### 2. Imposta la chiave personale per l'API

Per procedere al fine-tuning e alla generazione del dataset personalizzato, è necessario appoggiarsi alle [**API**](https://platform.openai.com/docs/api-reference) 🐝 (Application Programming Interfaces) messe a disposizione da OpenAI. Per effettuare richieste alle API si utilizza una "chiave segreta" personale.

```py
KEY = "sk-XXXXXXXXXXX"
```
Purtroppo, le richieste alle API sono a pagamento 💰 Il pagamento viene effettuato in base al numero di token ricevuti/trasmessi e a quale modello si sta utilizzando ([qui](https://openai.com/pricing) la lista dettagliata dei costi).

Per l'esercizio di oggi, Voxel ha messo a disposizione una chiave temporanea per tutte ([link](https://pastebin.com/quuMyRyJ)) 💖 Tuttavia, nel caso volessi più avanti continuare a fare delle prove con le API, dovrai crearti un tuo [account](https://platform.openai.com/account/api-keys) ed una nuova chiave personale. 


```py
openai.api_key  = KEY
```

### 3. Genera un dataset personalizzato

Come detto prima, in questo esempio vogliamo ottenere un modello che generi automaticamente descrizioni di personaggi di Dungeons & Dragons (DnD) a partire da una lista di elementi.

L'obiettivo finale è ottenere un modello a cui si passa come prompt una lista di quattro parole:

1. Razza del personaggio 🧝‍♂️
2. Classe del personaggio 🧙
3. Arma preferita 🏹
4. Animale compagno 🦉

E il modello in risposta deve generare una descrizione completa del personaggio che includa:

1. Nome
2. Descrizione dell'apparenza fisica
3. Biografia
4. Descrizione della personalità 

Ci serve quindi creare un dataset che contenga diversi esempi di questo tipo di associazione:
`[ race, class, weapon, pet ]` → `completion`.

Generare esempi di liste `[ race, class, weapon, pet ]` è un compito facile. Generare invece le corrispondenti `completion` non è così immediato. Dato che vogliamo generare un numero cospicuo di esempi, escludiamo l'opzione di scrivere a mano le possibili `completion`. La soluzione più immediata è utilizzare GPT stesso per generare le `completion`, fornendo una prompt diversa che spieghi in dettaglio il risultato che vogliamo ottenere. Chiameremo questa la "prompt intermedia" (che non sarà contenuta nel dataset finale), mentre la "prompt finale" sarà la corrispondente lista `[ race, class, weapon, pet ]`.


#### 3a. Definisci la "prompt intermedia" e la "prompt finale"

La prompt intermedia va formulata in modo da ottenere la `completion` completa che desideriamo. Non esiste un'unica soluzione, in questo esempio proveremo la seguente:

> Create a detailed description of a Dungeons and Dragons character with `race` race, `clss` class, `weapon` as favorite weapon, and `pet` as beloved pet. Write out the description as a list of name, physical appearance, background, and personality in a maximum of 100 words:

Questa servirà solo per generare le `completion`, verrà poi scartata al momento di salvare il dataset.

Col seguente codice, definisci la prompt intermedia come scritta sopra, e la prompt finale come semplice lista di quattro parole:

```py
prompt_inter  = "Create a detailed description of a Dungeons and Dragons character with "
prompt_inter += "{race} race, {clss} class, {weapon} as favorite weapon, and {pet} as beloved pet. "
prompt_inter += "Write out the description as a list of name, physical appearance, background, and personality in a maximum of 100 words:"

prompt_final  = "{race}, {clss}, {weapon}, {pet}"
```

Queste prompt si possono considerare dei "template", a cui adesso vanno riempiti i campi `race`, `clss`, `weapon`, e `pet` con degli esempi. Il numero di esempi per ogni campo determina la grandezza finale del dataset. Per velocizzare il processo, ci limitiamo per il momento a due esempi per ogni campo (per un totale di 16 campioni nel dataset):

```py
list_race   = [ 'human', 'elf' ]
list_class  = [ 'ranger', 'wizard' ]
list_weapon = [ 'sword', 'crossbow' ]
list_pet    = [ 'wolf', 'fox' ]
```

#### 3b. Imposta un modello per fare *completion*

Adesso serve un modello per generare le `completion` date le prompt intermedie.
La funzione seguente usa una chiamata alle API per fare completion:

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

Il risultato è salvato in un oggetto in formato JSON. Questa funzione estrare direttamente il testo che ci serve:

```py
def view_completion( model, prompt, temperature=0.8, max_tokens=500 ):
    res = get_completion( model, prompt, temperature=temperature, max_tokens=max_tokens )
    print( res[ 'choices' ][ 0 ][ 'text' ] )
```

#### 3c. Componi gli elementi del dataset

Adesso metti insieme il dataset. Riempi le prompt intermedie e finali con i possibili valori di `race`, `clss`, `weapon`, e `pet`. Usa le prompt intermedie per ottenere le corrispondenti `completion`. Nel dataset salva solo le coppie: "prompt finale" → `completion`.

```py
def generate_data( model, temperature=0.8, max_tokens=500 ):
    data  = []

    # to check progress
    cnt   = 1
    cmx   = np.prod( [ m for m in  map( len, [ list_race, list_class, list_weapon, list_pet ] ) ] )

    for ( r, c, w, p ) in product( list_race, list_class, list_weapon, list_pet ):        
        t1      = time.time()

        prmt_i  = prompt_inter.format( race=r, clss=c, weapon=w, pet=p )
        prmt_f  = prompt_final.format( race=r, clss=c, weapon=w, pet=p )

        res     = get_completion( model, prmt_i, temperature, max_tokens )
        res_t   = res[ 'choices' ][ 0 ][ 'text' ]

        data.append( { "prompt": prmt_f, "completion": res_t } )

        # print progress
        t2      = time.time()
        t       = int( t2 - t1 )
        print( f"{ cnt }/{ cmx } ({ t } sec)" )
        cnt     += 1
    return data
```


#### 3d. Genera il dataset!

Il modello che userai per generare il dataset è GPT-3.5, in particolare la versione chiamata `text-davinci-003` ([qui](https://platform.openai.com/docs/models/gpt-3-5) la lista completa dei modelli disponibili).

```py
gen_model = "text-davinci-003"
```

Prima di generare il dataset, puoi testare il corretto funzionamento delle API con una chiamata di prova 🦄

```py
prompt = "Generate a cute name for a unicorn"
view_completion( gen_model, prompt, temperature=1.3 )
```

Pronta per generare il dataset! La chiamata seguente dovrebbe impiegare circa 2 minuti (al prezzo di $0.07):

```py
dataset = generate_data( gen_model, max_tokens=200 )
```

Controllando il primo risultato, noterai che si tratta di un oggetto `dict` con due chiavi: `prompt` con la lista di quattro parole, e `completion` con la descrizione del personaggio.

```py
dataset[ 0 ]
```

### 4. Carica il dataset

A questo punto puoi procedere in due modi:

* Salva il dataset che hai appena creato e caricalo sull'account OpenAI
* Oppure usa il dataset (più grande, migliore) che abbiamo già caricato sull'account condiviso

Ti suggeriamo la seconda opzione, ma riportiamo lo stesso i passi che dovresti seguire nel caso tu voglia creare un nuovo dataset in futuro col tuo account personale.

#### 4a. [SCONSIGLIATO] Salva e carica il nuovo dataset

Per poter utilizzare il dataset con le API, questo va salvato in formato JSONL:

```py
def write_json( dataset, fname ):
    with jsonlines.open( fname, 'w' ) as f:
        f.write_all( dataset )
```

E caricato tramite una chiamata alle API. Questa restituisce l'id del file, che dovrai utilizzare per poterti riferire a questo.

```py
def upload_dataset( fname ):
    f   = open( fname, 'rb' )
    res = openai.File.create( file=f, purpose='fine-tune' )
    return res[ 'id' ]  # return id of file
```

Con i comandi seguenti salvi il dataset in un file `.jsonl` e lo carichi sull'account OpenAI:

```py
json_file   = "dataset{}.jsonl".format( len( dataset ) )
write_json( dataset, json_file )
id_file     = upload_dataset( json_file )
```

Nell'eventualità volessi vedere la lista dei dataset caricati sull'account, oppure cancellarne uno, i comandi sono i seguenti:

```py
openai.File.list()
openai.File.delete( "file-XXXXXXXXX" )
```

#### 4b. Usa il dataset già pronto

L'opzione che ti suggeriamo è di utilizzare il dataset che abbiamo già creato con 1296 campioni, utilizzando un insieme più grande di esempi di categorie:

```py
list_race   = [ 'human', 'elf', 'mermaid', 'halfling', 'dwarf', 'orc' ]
list_class  = [ 'rogue', 'fighter', 'ranger', 'bard', 'wizard', 'druid' ]
list_weapon = [ 'axe', 'sword', 'crossbow', 'trident', 'mace', 'dagger']
list_pet    = [ 'wolf', 'unicorn', 'owl', 'fox', 'hawk', 'snake' ]
``` 

Questo dataset ha richiesto naturalmente molto più tempo (e soldini!) per essere creato. Se lo desideri, puoi scaricare il file al seguente [link](https://www.dropbox.com/s/0rp9lc1vmtvie5q/dataset_1296.jsonl). Non è invece necessario caricarlo sull'account poiché l'abbiamo già fatto. Puoi recuperarlo con il seguente id:

```py
id_file = "file-XXXXXXXXX"
```


### 5. Esegui il *fine-tuning* di un modello

Adesso che hai il dataset, puoi passare alla seconda parte dell'esercizio ovvero eseguire il fine-tuning del modello per ottenere la tua versione personalizzata.

#### 5a. Definisci le funzioni per fare *fine-tuning*

La funzione seguente chiama le API per eseguire un addestramento di fine-tune su un modello:

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
Il fine-tuning potrebbe richiedere diversi minuti, è utile quindi avere una funzione per controllare lo stato di avanzamento:

```py
def get_status( id ):
    res = openai.FineTune.list_events( id=id, stream=False )
    res = res[ 'data' ]
    for r in res:
        print( r[ 'message' ] )
```

Infine, ti serve una funzione per ottenere il modello al termine del fine-tuning:

```py
def retrieve_model( id ):
    res = openai.FineTune.retrieve( id=id )
    return res[ 'fine_tuned_model' ]
```

#### 5b. Lancia il *fine-tuning*

È utile assegnare un nome personalizzato al tuo modello, in modo da poi essere in grado di recuperarlo tra tutti i modelli che verranno creati durante l'incontro di oggi:

```py
custom_name = "XXXXX"
```

Per il fine-tuning, ti suggeriamo di utilizzare un modello GPT-3 di tipo `ada` ([qui](https://platform.openai.com/docs/models/gpt-3) la lista dei modelli), per velocizzare al massimo i tempi di addestramento, che limitiamo a 4 epoche. Potrebbe lo stesso volerci del tempo ⏳ (una quindicina di minuti almeno), poiché il tuo *job* viene inserito in una coda con tutti gli altri processi in esecuzione in quel momento.

Tempo di lanciare il comando!

```py
ftune_model = "ada"
id_job      = finetune( ftune_model, id_file, custom_name, n_epochs=4 )
```

In caso avessi sbagliato qualcosa, puoi cancellare il tuo processo con questo comando:

```py
openai.FineTune.cancel( id=id_job )
```

Questo è il comando per verificare a che punto è il tuo *job*. Dovrai eseguirlo più volte per vedere i progressi:

```py
get_status( id_job )
```

Il processo è terminato quando `get_status()` restituisce un output del genere:

```
Created fine-tune: ft-XXXXXXXXXXXXX
Fine-tune costs $0.36
Fine-tune enqueued. Queue number: 3
Fine-tune is in the queue. Queue number: 2
Fine-tune is in the queue. Queue number: 1
Fine-tune is in the queue. Queue number: 0
Fine-tune started
Completed epoch 1/4
Completed epoch 2/4
Completed epoch 3/4
Completed epoch 4/4
Uploaded model: ada:ft-XXXXXXXXX
Uploaded result file: file-XXXXXXXX
Fine-tune succeeded
```

### 6. Testa il tuo VoxelGPT!

Terminato il fine-tuning, recupera il tuo modello personalizzato con il seguente comando:

```py
tuned_model = retrieve_model( id_job )
```

Adesso è tutto pronto per testare il nuovo modello! Scrivi una qualsiasi promtp nella forma di lista di quattro parole, dai un valore di *temperature* tra 0 e 2, e il modello genererà la descrizione di un nuovo personaggio!

```py
prompt = "mermaid, monk, sling, crab"
temp   = 0.9
view_completion( tuned_model, prompt, temperature=temp, max_tokens=150 )
```

Forte no? 😄 Oltre le chiamate API da codice, puoi testare il tuo nuovo modello tramite un'interfaccia stile ChatGPT fornita da OpenAI che si chiama [**Playground**](https://platform.openai.com/playground). Ti basta selezionare dalla tendina "Model" in alto a destra il nome del tuo modello. Puoi anche impostare un valore di temperatura dall'interfaccia. Scrivi le tue quattro parole, e buon divertimento! ✨


| Capitolo precedente                                                                                                                                          | Capitolo successivo                                                                           |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------: |
| [◀︎ 04 Temperature](../04-temperature)  | [Torna all'inizio ▶︎](../../..) |

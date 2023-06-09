# 01. Cos'è ChatGPT

| Capitolo precedente                                                                                                                                          | Capitolo successivo                                                                           |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------: |
| [◀︎ 00 Setup](../00-setup)  | [02 Embedding ▶︎](../02-embedding) |

## Obiettivo

Crea un modello di linguaggio come ChatGPT!

Durante l'incontro di oggi, scoprirai le componenti fondamentali di un modello di linguaggio, come i processi di *embedding*, *tokenization*, e *completion*. Implementerai modelli sempre più complessi, fino a personalizzare un vero modello ChatGPT 🔥

Ma prima, se sei curiosa di saperne di più su cosa sia ChatGPT, leggi qui di seguito. Se invece sei impaziente di iniziare a programmare, puoi passare direttamente al capitolo successivo 😎

## Come funziona ChatGPT

ChatGPT è un tipo di ***large language model*** (LLM), ovvero un modello di linguaggio basato sull'architettura GPT sviluppata da [**OpenAI**](https://openai.com/). È progettato per generare risposte simili a quelle umane a richieste di linguaggio naturale 💬 come domande o affermazioni.

ChatGPT funziona analizzando il testo in ingresso e utilizzando una rete neurale per generare una risposta simile nello stile e nel tono a una risposta generata da un essere umano. La rete neurale viene addestrata su grandi quantità di dati testuali per imparare i modelli del linguaggio e le relazioni tra le diverse parole.

Quando viene fornita una *prompt*, ChatGPT utilizza le conoscenze apprese per generare una risposta che sia probabilmente pertinente e coerente. Il modello non fornisce semplicemente una risposta predefinita per ogni richiesta, ma genera una risposta al momento in base al contesto e al significato del testo in ingresso.

> ***Curiosità: cos'è OpenAI?***
> OpenAI è una società leader nella ricerca nel campo dell'intelligenza artificiale (AI), fondata nel 2015 da un team di scienziati, ricercatori e ingegneri, tra cui Elon Musk.
> OpenAI ha apportato contributi significativi al campo dell'AI, tra cui modelli di linguaggio all'avanguardia (come **GPT**), modelli per la generazione di immagini da testo (come **DALL-E**) e nuovi algoritmi di *reinforcement learning*.
> OpenAI si impegna anche a garantire che lo sviluppo dell'AI avvenga in modo etico e responsabile. La società promuove la trasparenza nella ricerca sull'AI e ha sviluppato diverse linee guida per lo sviluppo ed il dispiegamento etico dei sistemi di AI. Dal 2019, Microsoft ha stabilito una partnership con OpenAI attraverso investimenti multimiliardari.

### La storia di ChatGPT

ChatGPT si basa su di un altro modello sviluppato da OpenAI chiamato GPT-3. ChatGPT è una versione di GPT-3 ottimizzata specificamente per l'uso in chatbot 🤖 e agenti conversazionali. È progettato per essere più coinvolgente e interattivo con gli utenti. 

**GPT-3** è la terza versione di un modello chiamato *Generative Pre-trained Transformer* (GPT) introdotto nel 2020, che si basa sull'architettura *Transformer*. GPT-3 è grande modello con 175 miliardi di parametri. Viene pre-addestrato su un ampio corpus di dati di testo (17 GB) 📚 e successivamente raffinato per compiti specifici, come la risposta alle domande o il completamento del testo. Il vantaggio chiave di GPT è la sua capacità di generare testo coerente e fluido che è simile in stile e tono al testo scritto da esseri umani.

***Transformer*** è una rete neurale introdotta da Google nel 2017. È progettata per elaborare dati sequenziali (es. dati testuali) in modo più efficiente rispetto alle reti neurali ricorrenti (RNN) tradizionali. L'innovazione chiave dell'architettura Transformer è il *meccanismo di attenzione* 👀 , che consente al modello di concentrarsi su diverse parti della sequenza di input durante la generazione di previsioni.

Nel marzo 2023, OpenAI ha rilasciato la nuova versione chiamata **GPT-4**. Si tratta di un modello ancora più grande, con mille miliardi di parametri 🦾 e addestrato su 45 GB di dati. Inoltre GPT-4 è un modello multimodale che accetta in input sia immagini che testo (e produce output testuale).

<!--
### ChatGPT bloccato in Italia

Dal 31 marzo 2023, OpenAI ha bloccato l’accesso al servizio ❌ per gli indirizzi IP italiani a seguito di una richiesta del Garante italiano per la protezione dei dati personali. Tuttavia, il blocco riguarda solamente il modello ChatGPT.

Tramite le API di OpenAI è ancora possibile utilizzare tutti i modelli GPT disponibili. Ed è quello che andremo a fare oggi! 💪 Adesso che ti sei fatta un'idea di cosa sia ChatGPT, sei pronta ad iniziare! Puoi passare al capitolo successivo ✨

P.S. Cerchi un'alternativa a ChatGPT dopo il blocco in Italia? Prova [CatGPT](https://www.cat-gpt.com/) 😹
-->

### Esiste altro oltre ChatGPT?

Ebbene sì, Google sta sviluppando un proprio chatbot noto come [**Bard**](https://bard.google.com/), che al momento non è ancora disponibile in Italia. Bard è basato su un modello chiamato BERT (Bidirectional Encoder Representations from Transformers), che a sua volta si fonda sempre sull'architettura dei *transformer*.

In poche parole, la differenza tra i due approcci sta nel fatto che BERT si concentra sulla comprensione del contesto in modo bidirezionale (sia il contesto precedente che il successivo di una parola all'interno di una frase), mentre GPT è focalizzato sulla generazione del testo basata sul contesto precedente. 

---

Adesso che ti sei fatta un'idea di cosa sia ChatGPT, sei pronta ad iniziare! Puoi passare al capitolo successivo ✨

| Capitolo precedente                                                                                                                                          | Capitolo successivo                                                                           |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------: |
| [◀︎ 00 Setup](../00-setup)  | [02 Embedding ▶︎](../02-embedding) |
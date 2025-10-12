# Observații despre logica botului

## 1. Semnături Binance și timestamp reciclat (rezolvat)
`_send` rescrie acum `timestamp` și semnătura la **fiecare** retry. Astfel, dacă prima încercare pică cu `-1021` din cauza latenței, a doua încercare va avea un `timestamp` proaspăt, nu îl va recircula pe cel expirat și ordinul are o șansă reală să treacă. În practică, asta înseamnă că spike-urile de latență nu mai blochează toate retrimiterile. 【F:notebook_extracted.py†L375-L413】

## 2. Fallback-ul de închidere forțată ține cont de direcție
Ultimul fallback din `ensure_flat` citește acum sensul ultimei poziții (`last_amt`) și apelează `_double_trigger_close("BUY")` doar dacă suntem pe un SHORT. Dacă alt algoritm ți-a deschis între timp un LONG pe același cont, fallback-ul final va trimite `SELL`, astfel încât botul chiar să închidă poziția și să nu rămână blocat în EXITING. 【F:notebook_extracted.py†L943-L1014】

## 3. Qty minimă trebuie aliniată la step (de ce contează)
Gândește-te la un simbol cu `stepSize = 0.0001`, dar cu `minQty = 0.0003`. Dacă strategia îți calculează `qty = 0.00025`, îl rotunjim în jos la step și obținem `0.0002`. Cum e sub prag, funcția urcă la `0.0003`. Problema e că `0.0003` **nu e multiplu** de `0.0001`, deci Binance respinge ordinul cu „LOT_SIZE”. Într-un scenariu real, vezi în loguri doar că ordinul nu a fost executat și poziția rămâne deschisă, ceea ce poate declanșa în lanț alte retry-uri și fallback-uri. Soluția este să aplici din nou rotunjirea la step după ce trecem de `minQty`, astfel încât rezultatul final să respecte simultan ambele filtre. 【F:notebook_extracted.py†L44-L66】

Aceste ajustări previn: ordine respinse în lanț din cauza timestamp-ului, blocarea ieșirilor atunci când apare un LONG extern și erorile „LOT_SIZE precision” generate de cantități nealiniate.

## De ce există `notebook_extracted.py`
Notebook-ul este greu de urmărit într-un diff text simplu, așa că am extras conținutul celulelor într-un fișier Python temporar pentru a putea analiza rapid logica și a marca fragmentele relevante în comentarii. Nu este nevoie să rulezi sau să păstrezi fișierul în producție; servește doar ca suport de analiză atunci când trebuie să revizuiești codul fără interfața notebook-ului. Dacă preferi să lucrezi doar în notebook, poți șterge fișierul fără să afectezi funcționalitatea botului.

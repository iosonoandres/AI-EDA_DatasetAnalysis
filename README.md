# Analisi e Classificazione dei Dati sull'Accettabilità delle Auto

Questo script Python è stato sviluppato per eseguire un'analisi dei dati relativi all'accettabilità delle auto e per addestrare modelli di classificazione al fine di prevedere l'accettabilità delle auto in base alle caratteristiche dei veicoli.

## Librerie Necessarie
- `pandas` e `numpy` per la manipolazione dei dati
- `matplotlib`, `seaborn`, `plotly`, e `plotly.graph_objects` per la visualizzazione dei dati
- `scikit-learn` per addestrare modelli di classificazione
- `tensorflow` e `keras` per la creazione e l'addestramento di una rete neurale

## Eseguire lo Script
Per eseguire lo script, assicurati di avere tutte le librerie necessarie installate e quindi esegui il codice.

Lo script include le seguenti fasi principali:
1. Lettura e caricamento dei dati da un file CSV.
2. Controllo dei valori fuori soglia nelle colonne numeriche.
3. Esplorazione e gestione delle features categoriche.
4. Visualizzazione dei dati per ottenere insights.
5. Preparazione dei dati per l'addestramento dei modelli di classificazione.
6. Addestramento di modelli di classificazione, inclusi regressione logistica e Support Vector Machine (SVM).
7. Valutazione delle prestazioni dei modelli tramite metriche di classificazione.
8. Utilizzo di GridSearchCV per ottimizzare i parametri del modello SVM.
9. Visualizzazione delle metriche e dei risultati ottenuti.

## Analisi e Visualizzazione dei Dati
Lo script esegue un'analisi esplorativa dei dati (EDA) e utilizza diverse tecniche di visualizzazione per comprendere meglio le caratteristiche del dataset, inclusi boxplot, heatmap, e altri grafici.

## Addestramento e Valutazione del Modello
Il modello di classificazione viene addestrato utilizzando diverse tecniche, tra cui regressione logistica e SVM. Vengono valutate le prestazioni dei modelli con metriche come l'accuratezza, la matrice di confusione e il report di classificazione.

## TensorBoard (Callback)
Il callback di TensorBoard viene utilizzato per monitorare l'addestramento del modello e visualizzare i risultati in tempo reale.

# Primena neuronskih mreža za detekciju prevara u transakcijama kreditnim karticama

Ovaj projekat predstavlja primenu neuronske mreže za detekciju prevara u transakcijama kreditnim karticama. Projekat je urađen u okviru predmeta **Duboko učenje i neuronske mreže**.

Cilj projekta je da se na osnovu dostupnih karakteristika transakcije predvidi da li je transakcija regularna ili prevarna. Problem je formulisan kao binarna klasifikacija.



## Dataset

U projektu je korišćen javno dostupan dataset **Credit Card Fraud Detection**, preuzet sa platforme Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Dataset sadrži transakcije evropskih korisnika kreditnih kartica. Zbog zaštite privatnosti korisnika, većina atributa je anonimizovana i transformisana pomoću PCA metode.

## Struktura projekta

```text
credit-card-fraud-neural-network/
│
├── data/
│   └── creditcard.csv
│
├── credit_card_fraud_detection.ipynb
├── requirements.txt
├── runtime.txt
├── README.md
├── LICENSE
└── .gitignore
```

Napomena: fajl `data/creditcard.csv` nije uključen u GitHub repozitorijum zbog veličine fajla. Dataset je potrebno ručno preuzeti sa Kaggle linka i staviti u folder `data/`.



## Korišćene tehnologije

- Python 3.11.9
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow / Keras


## Pokretanje projekta

Projekat je rađen u Python verziji **3.11.9**. Preporučuje se korišćenje ove verzije, jer TensorFlow može imati probleme sa instalacijom na novijim verzijama Python-a.

Instalacija potrebnih biblioteka:

```bash
pip install -r requirements.txt
```

Nakon toga potrebno je preuzeti dataset sa Kaggle-a i smestiti fajl `creditcard.csv` u folder:

```text
data/
```





## Kratak pregled faza projekta

### 1. Opis problema

U prvoj fazi definisan je problem detekcije prevara u transakcijama kreditnim karticama. Problem je postavljen kao binarna klasifikacija, gde model treba da razlikuje regularne i prevarne transakcije. Posebno je naglašeno da je problem izazovan zbog velike neuravnoteženosti klasa.

### 2. Podaci

U drugoj fazi učitan je dataset i analizirana je njegova struktura. Provereni su broj redova i kolona, tipovi podataka, nedostajuće vrednosti i osnovna statistika atributa. Analizirana je i raspodela ciljne promenljive `Class`, pri čemu je potvrđeno da postoji veoma mali broj prevarnih transakcija u odnosu na regularne.

U okviru preprocesiranja podaci su podeljeni na ulazne promenljive i ciljnu promenljivu, zatim na trening i test skup. Kolone `Time` i `Amount` su skalirane pomoću `StandardScaler`, dok su zbog neuravnoteženosti klasa izračunati ponderi klasa.

### 3. Arhitektura modela

U trećoj fazi definisana je arhitektura neuronske mreže pomoću TensorFlow/Keras biblioteke. Model se sastoji od potpuno povezanih `Dense` slojeva, `Dropout` slojeva za regularizaciju i izlaznog sloja sa sigmoid aktivacionom funkcijom. Sigmoid funkcija se koristi zato što model rešava problem binarne klasifikacije i vraća verovatnoću da je transakcija prevarna.

### 4. Trening

U četvrtoj fazi model je treniran na skaliranim trening podacima. Tokom treninga korišćen je validacioni skup kako bi se pratilo ponašanje modela na podacima koje ne koristi direktno za učenje. Zbog neuravnoteženosti klasa korišćeni su ponderi klasa, a trening je kontrolisan pomoću `EarlyStopping` mehanizma koji prati validacioni AUC i zaustavlja trening kada se metrika više ne poboljšava.

### 5. Analiza osetljivosti i hiperparametarska optimizacija

U petoj fazi testirano je više konfiguracija neuronske mreže. Menjani su broj neurona u skrivenim slojevima, dropout stopa i learning rate. Cilj je bio da se analizira kako promene hiperparametara utiču na performanse modela i da se izabere najbolja konfiguracija. Kao najbolji model izabrana je arhitektura `(128, 64)` sa dropout stopom `0.3` i learning rate vrednošću `0.001`.

### 6. Rezultati evaluacije

U šestoj fazi najbolji model je evaluiran na test skupu, koji nije korišćen tokom treninga. Izračunate su metrike precision, recall, F1-score, ROC-AUC, PR-AUC i matrica konfuzije. Posebno je analiziran uticaj različitih pragova klasifikacije.

Pri standardnom pragu `0.5` model ostvaruje visok recall, ali nizak precision za klasu `Prevara`. Zbog toga je dodatno analiziran prag `0.95`, koji daje bolji balans između smanjenja broja lažnih alarma i otkrivanja stvarnih prevara.

### 7. Diskusija

U sedmoj fazi diskutovani su rezultati modela, prednosti i ograničenja pristupa. Zaključeno je da neuronska mreža može uspešno da nauči obrasce koji razlikuju regularne i prevarne transakcije, ali da zbog neuravnoteženosti podataka izbor praga klasifikacije ima veliki uticaj na rezultate. Takođe je navedeno da anonimizovani PCA atributi ograničavaju interpretabilnost modela.

### 8. Zaključak

U završnoj fazi sumirani su rezultati projekta. Pokazano je da neuronske mreže mogu biti korisne za detekciju finansijskih prevara, posebno kada se kombinuju sa odgovarajućim preprocesiranjem, ponderisanjem klasa, hiperparametarskom optimizacijom i pažljivim izborom praga klasifikacije. Za realnu primenu bilo bi potrebno dodatno testiranje na novijim i raznovrsnijim podacima.



## Autor

Jovana Vuksanović 2022/0293
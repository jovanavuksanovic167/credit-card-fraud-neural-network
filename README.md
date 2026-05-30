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

Napomena: fajl `data/creditcard.csv` nije uključen u GitHub repozitorijum. Dataset je potrebno ručno preuzeti sa Kaggle linka i staviti u folder `data/`.



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





# Pregled faza projekta

## 1. Opis problema

Detekcija prevara u transakcijama kreditnim karticama predstavlja značajan izazov u savremenim finansijskim sistemima. Finansijske institucije svakodnevno obrađuju ogroman broj transakcija, pri čemu čak i mali procenat prevarnih aktivnosti može dovesti do velikih finansijskih gubitaka.

Cilj ovog projekta je razvoj modela zasnovanog na neuronskim mrežama koji može da razlikuje regularne od prevarnih transakcija na osnovu dostupnih karakteristika transakcije.

Problem je formulisan kao zadatak binarne klasifikacije. Za svaku transakciju model predviđa verovatnoću da transakcija pripada klasi prevara (Class = 1), dok se regularne transakcije označavaju klasom (Class = 0).

Poseban izazov predstavlja izrazita neuravnoteženost podataka. U datasetu prevarne transakcije čine veoma mali procenat ukupnog broja transakcija, zbog čega je neophodno koristiti odgovarajuće tehnike obrade podataka, treninga i evaluacije modela.



## 2. Podaci

U projektu je korišćen javno dostupan dataset Credit Card Fraud Detection sa platforme Kaggle. Dataset sadrži ukupno 284.807 transakcija i 31 atribut.

Od ukupnog broja atributa:

* 28 atributa (V1–V28) predstavljaju anonimizovane PCA komponente,
* atribut Time predstavlja vreme proteklo od prve transakcije u datasetu,
* atribut Amount predstavlja iznos transakcije,
* atribut Class predstavlja ciljnu promenljivu.

Prilikom analize podataka izvršeni su sledeći koraci:

* pregled strukture dataseta,
* analiza tipova podataka,
* provera postojanja nedostajućih vrednosti,
* analiza osnovnih statističkih karakteristika,
* analiza raspodele ciljne promenljive,
* vizualizacija odnosa regularnih i prevarnih transakcija.

Analiza je pokazala da dataset ne sadrži nedostajuće vrednosti i da su svi atributi numeričkog tipa. Takođe je potvrđeno da je dataset izrazito neuravnotežen. Regularne transakcije čine približno 99,83% svih primera, dok prevarne transakcije čine svega oko 0,17%.

Nakon eksplorativne analize izvršeno je preprocesiranje podataka. Dataset je podeljen na ulazne promenljive i ciljnu promenljivu, nakon čega je izvršena stratifikovana podela na trening i test skup u odnosu 80:20. Stratifikacija je korišćena kako bi raspodela klasa ostala približno ista u oba skupa.

Kolone Time i Amount standardizovane su korišćenjem algoritma StandardScaler, dok su PCA atributi ostavljeni nepromenjeni jer su već transformisani tokom kreiranja dataseta.

Zbog izražene neuravnoteženosti klasa izračunati su ponderi klasa (class weights) koji su kasnije korišćeni tokom treninga modela.



## 3. Arhitektura modela

Za rešavanje problema korišćena je višeslojna potpuno povezana neuronska mreža implementirana u okviru TensorFlow/Keras biblioteke.

Početna arhitektura modela sastoji se od:

* ulaznog sloja sa 30 ulaznih atributa,
* prvog Dense sloja sa 64 neurona i ReLU aktivacionom funkcijom,
* Dropout sloja sa stopom odbacivanja od 30%,
* drugog Dense sloja sa 32 neurona i ReLU aktivacionom funkcijom,
* drugog Dropout sloja sa stopom odbacivanja od 30%,
* izlaznog Dense sloja sa jednim neuronom i sigmoid aktivacionom funkcijom.

ReLU aktivaciona funkcija korišćena je u skrivenim slojevima zbog dobre efikasnosti u učenju složenih nelinearnih obrazaca, dok sigmoid funkcija omogućava interpretaciju izlaza modela kao verovatnoće pripadnosti klasi prevara.

Model je optimizovan korišćenjem Adam optimizatora i funkcije greške Binary Crossentropy, koja predstavlja standardni izbor za probleme binarne klasifikacije.

Dropout slojevi uključeni su radi smanjenja rizika od preprilagođavanja modela trening podacima.



## 4. Trening modela

Trening modela sproveden je nad skaliranim trening podacima.

Tokom procesa učenja model prolazi kroz više epoha, pri čemu se nakon svake epohe prate performanse na validacionom skupu. Za validaciju je korišćeno 20% trening podataka.

U procesu treniranja korišćene su sledeće tehnike:

* Adam optimizacioni algoritam,
* Binary Crossentropy funkcija greške,
* ponderi klasa radi rešavanja neuravnoteženosti,
* EarlyStopping mehanizam za sprečavanje overfitting-a.

EarlyStopping prati validacionu AUC metriku i zaustavlja trening ukoliko se performanse ne poboljšavaju tokom unapred definisanog broja epoha. Nakon zaustavljanja vraćaju se parametri modela iz epohe koja je ostvarila najbolji rezultat.


Na grafikonu 'Promena funkcije greške tokom treninga' se vidi da se funkcija greške na trening skupu naglo smanjuje u prvim epohama, a zatim se postepeno stabilizuje. To pokazuje da model tokom treninga uči obrasce iz podataka i smanjuje grešku predikcije.
Validacioni loss se takođe smanjuje i ostaje niži od trening loss-a tokom većeg dela treninga. To ukazuje da model ne pokazuje izražene znake preprilagođavanja trening podacima.
Model je trenirao 13 epoha, iako je maksimalan broj epoha bio postavljen na 50. Trening je ranije zaustavljen pomoću `EarlyStopping` tehnike, jer se validaciona AUC vrednost više nije poboljšavala dovoljno kroz naredne epohe.

Na grafikonu 'Promena AUC metrike tokom treninga' se vidi da AUC vrednost brzo raste u prvim epohama, što pokazuje da model već na početku treninga uči da razlikuje regularne i prevarne transakcije.
Trening AUC i validacioni AUC se nakon nekoliko epoha stabilizuju na visokim vrednostima, blizu 0.99. To ukazuje da model ima dobru sposobnost razdvajanja klasa.
Validacioni AUC je tokom treninga veoma blizak trening AUC vrednosti, što znači da model ne pokazuje izražene znake preprilagođavanja. Model ne uči samo napamet trening podatke, već zadržava dobre performanse i na validacionom skupu.


## 5. Analiza osetljivosti i hiperparametarska optimizacija

Nakon inicijalnog modela sprovedena je analiza uticaja hiperparametara na performanse neuronske mreže.

Testirane su različite kombinacije:

* broja neurona u skrivenim slojevima,
* dropout stope,
* learning rate parametra.

Analizirane arhitekture uključivale su:

* (32, 16),
* (64, 32),
* (128, 64),
* (128, 64) sa većim dropout parametrima,
* (64, 32) sa manjim learning rate parametrima.

Za svaku konfiguraciju model je treniran pod identičnim uslovima, nakon čega je zabeležena najbolja ostvarena validaciona AUC vrednost.

Rezultati su pokazali da model sa arhitekturom (128, 64), dropout stopom 0.3 i learning rate vrednošću 0.001 ostvaruje najbolje performanse. Ova konfiguracija izabrana je kao završni model za dalju evaluaciju.

Sprovedena analiza pokazala je da povećanje kapaciteta mreže može poboljšati sposobnost modela da prepozna obrasce prevarnih transakcija, ali da prevelika kompleksnost povećava rizik od overfitting-a.



## 6. Rezultati evaluacije

Nakon hiperparametarske optimizacije, za završnu evaluaciju izabran je model sa arhitekturom `(128, 64)`, dropout stopom `0.3` i learning rate vrednošću `0.001`.

Evaluacija se vrši na test skupu, koji nije korišćen tokom treninga ni tokom izbora hiperparametara. Na taj način dobijamo objektivniju procenu performansi modela na novim podacima.

U evaluaciji posmatramo sledeće metrike:

- confusion matrix
- precision
- recall
- F1-score
- ROC-AUC
- PR-AUC

Najbolji model je ponovo treniran korišćenjem izabranih hiperparametara: arhitektura `(128, 64)`, dropout stopa `0.3` i learning rate `0.001`.

Tokom treninga korišćen je `EarlyStopping`, pa model nije morao da trenira svih 50 epoha. Trening je zaustavljen kada se validaciona AUC vrednost više nije značajno poboljšavala.

U početnoj evaluaciji koristi se standardni prag klasifikacije od `0.5`. To znači da se transakcija klasifikuje kao prevarna ako model proceni verovatnoću prevare veću ili jednaku 0.5.

Rezultati pokazuju da model ostvaruje visok recall za klasu `Prevara`. Recall iznosi približno `0.92`, što znači da model uspeva da otkrije oko 93% stvarnih prevarnih transakcija iz test skupa. Ovo je važan rezultat, jer je kod detekcije prevara cilj da se što manje stvarnih prevara propusti.

Međutim, precision za klasu `Prevara` iznosi približno `0.05`. To znači da od svih transakcija koje je model označio kao prevarne, samo oko 5% zaista predstavljaju prevare. Drugim rečima, model pri ovom pragu generiše veliki broj lažno pozitivnih slučajeva, odnosno regularnih transakcija koje pogrešno označava kao prevarne.

Ukupna tačnost modela iznosi približno `0.97`, ali ovu metriku treba tumačiti oprezno zbog velike neuravnoteženosti klasa. Pošto regularne transakcije čine skoro ceo dataset, visoka tačnost ne znači nužno da model dobro rešava problem detekcije prevara.

Matrica konfuzije dodatno potvrđuje ovaj zaključak. Model je uspešno prepoznao većinu prevara, ali je istovremeno veliki broj regularnih transakcija klasifikovao kao prevarne. Zbog toga je potrebno analizirati različite pragove klasifikacije kako bi se pronašao bolji odnos između precision i recall metrike.

Pri nižim pragovima, kao što su `0.1`, `0.2` i `0.3`, model veoma lako označava transakcije kao prevarne. Zbog toga je recall veoma visok, što znači da model uspeva da pronađe veliki broj stvarnih prevara. Međutim, precision je veoma nizak, jer se veliki broj regularnih transakcija pogrešno označava kao prevara.

Pri standardnom pragu `0.5`, recall iznosi približno `0.918`, što znači da model otkriva oko 92% stvarnih prevara. Međutim, precision je samo oko `0.047`, pa model i dalje generiše veliki broj lažnih alarma.

Povećavanjem praga model postaje stroži. To znači da transakciju označava kao prevarnu samo ako je procenjena verovatnoća prevare veoma visoka. Zbog toga precision raste, dok recall postepeno opada.

Na primer, pri pragu `0.95`, precision raste na približno `0.391`, dok recall ostaje relativno visok i iznosi približno `0.878`. To predstavlja bolji balans u odnosu na prag `0.5`, jer model značajno smanjuje broj lažnih alarma, a i dalje uspeva da otkrije veliki deo stvarnih prevara.

Pri pragu `0.99`, precision dodatno raste na približno `0.675`, ali recall opada na približno `0.847`. To znači da model pravi manje lažnih alarma, ali propušta više stvarnih prevara nego pri pragu `0.95`.

Na osnovu ove analize, prag `0.95` predstavlja dobar kompromis između precision i recall metrika. Zbog toga se u nastavku evaluacije koristi prag `0.95` kao izabrani prag klasifikacije.

ROC-AUC meri sposobnost modela da razlikuje regularne i prevarne transakcije kroz različite pragove klasifikacije.

PR-AUC meri površinu ispod Precision-Recall krive. Kod izrazito neuravnoteženih podataka, kao što je ovaj dataset, PR-AUC je posebno korisna metrika jer se fokusira na kvalitet predikcija za retku klasu.

### 6.1. Evaluacija sa izabranim pragom klasifikacije

Na osnovu analize različitih pragova klasifikacije, izabran je prag `0.95`. Ovaj prag predstavlja bolji kompromis između precision i recall metrika u odnosu na standardni prag `0.5`.

Pri pragu `0.5` model prepoznaje veliki broj prevara, ali generiše mnogo lažnih alarma. Povećanjem praga na `0.95`, model postaje stroži pri označavanju transakcije kao prevarne, čime se značajno povećava precision, dok recall ostaje relativno visok.
Pri ovom pragu model ostvaruje znatno bolji precision za klasu `Prevara` u odnosu na standardni prag `0.5`. Precision za klasu `Prevara` iznosi `0.39`, što znači da je oko 39% transakcija koje je model označio kao prevarne zaista prevarno.

Recall za klasu `Prevara` iznosi `0.88`, što znači da model uspeva da otkrije oko 88% stvarnih prevara iz test skupa. U poređenju sa pragom `0.5`, recall je malo manji, ali je precision značajno bolji.

F1-score za klasu `Prevara` iznosi `0.54`, što je znatno bolje nego kod standardnog praga. To pokazuje da prag `0.95` daje bolji balans između precision i recall metrika.

Ukupna accuracy je veoma visoka, ali se kod ovog problema ne oslanjamo samo na accuracy zbog velike neuravnoteženosti klasa. Mnogo je važnije posmatrati ponašanje modela nad klasom `Prevara`, jer je ona retka i predstavlja glavni cilj detekcije.

Matrica konfuzije za prag `0.95` pokazuje da je model od 98 stvarnih prevarnih transakcija uspešno prepoznao 86, dok je 12 prevara klasifikovao kao regularne transakcije.

Takođe, model je 134 regularne transakcije pogrešno označio kao prevarne. To su lažno pozitivni slučajevi, odnosno lažni alarmi. U odnosu na prag `0.5`, broj lažnih alarma je značajno smanjen.

Ovi rezultati pokazuju da prag `0.95` predstavlja bolji kompromis za ovaj problem. Model i dalje uspešno otkriva veliki broj prevara, ali pravi mnogo manje lažno pozitivnih predikcija nego pri standardnom pragu `0.5`.

## 7. Diskusija

Rezultati pokazuju da neuronska mreža može uspešno da nauči obrasce koji razlikuju regularne i prevarne transakcije. Model ostvaruje visoke vrednosti ROC-AUC i PR-AUC metrika, što ukazuje da dobro rangira transakcije prema verovatnoći prevare.

Međutim, zbog izrazite neuravnoteženosti podataka, evaluacija modela nije jednostavna. Regularne transakcije čine skoro ceo dataset, dok je broj prevarnih transakcija veoma mali. Zbog toga accuracy nije dovoljna metrika za procenu kvaliteta modela. Model može imati visoku accuracy čak i ako ne detektuje prevare dovoljno dobro.

Pri standardnom pragu klasifikacije `0.5`, model ostvaruje visok recall za klasu `Prevara`, ali veoma nizak precision. To znači da uspešno pronalazi veliki broj stvarnih prevara, ali istovremeno generiše mnogo lažnih alarma. Ovakav model može biti koristan ako je glavni cilj da se propusti što manje prevara, ali u realnom sistemu veliki broj lažnih alarma može opteretiti proces dodatne provere transakcija.

Analizom različitih pragova klasifikacije pokazano je da se precision može značajno poboljšati povećanjem praga. Izabrani prag `0.95` daje bolji balans između precision i recall metrika. Model pri ovom pragu pravi manje lažno pozitivnih predikcija, dok i dalje uspeva da otkrije veliki deo stvarnih prevara.

Hiperparametarska optimizacija pokazala je da izbor arhitekture, dropout stope i learning rate vrednosti utiče na performanse modela. Najbolje rezultate ostvario je model sa arhitekturom `(128, 64)`, dropout stopom `0.3` i learning rate vrednošću `0.001`. Ovaj model je imao najbolji odnos posmatranih metrika na validacionom skupu.

Glavno ograničenje projekta jeste to što su atributi `V1–V28` anonimizovani PCA transformacijom. Zbog toga nije moguće direktno interpretirati značenje pojedinačnih atributa i objasniti, na primer, koji konkretni faktor transakcije najviše utiče na predikciju prevare. Ovo je čest problem kod finansijskih podataka, jer se zbog privatnosti korisnika originalne karakteristike ne objavljuju.

 Takođe, u realnoj primeni bilo bi važno uključiti dodatne informacije, kao što su istorija korisnika, lokacija, tip trgovca i vremenski obrasci ponašanja.

## 8. Zaključak

U ovom projektu razvijen je model neuronske mreže za detekciju prevara u transakcijama kreditnim karticama. Problem je formulisan kao binarna klasifikacija, gde model za svaku transakciju predviđa da li je regularna ili prevarna.

Korišćen je javno dostupan dataset `Credit Card Fraud Detection`, koji sadrži izrazito neuravnotežene podatke. Zbog toga je posebna pažnja posvećena izboru metrika evaluacije. Korišćene su precision, recall, F1-score, ROC-AUC, PR-AUC i matrica konfuzije.

Tokom preprocesiranja podaci su podeljeni na trening i test skup, skalirane su kolone `Time` i `Amount`, a zbog neuravnoteženosti klasa korišćeni su ponderi klasa. Model je implementiran pomoću TensorFlow/Keras biblioteke i sastoji se od potpuno povezanih `Dense` slojeva, `Dropout` regularizacije i izlaznog sloja sa sigmoid aktivacionom funkcijom.

Hiperparametarskom optimizacijom testirano je više konfiguracija modela. Najbolji rezultati dobijeni su za arhitekturu `(128, 64)`, dropout stopu `0.3` i learning rate `0.001`.

Evaluacija je pokazala da model može uspešno da detektuje veliki broj prevarnih transakcija. Pri standardnom pragu `0.5` model ima visok recall, ali nizak precision, što znači da otkriva većinu prevara, ali pravi veliki broj lažnih alarma. Analizom pragova klasifikacije izabran je prag `0.95`, koji daje bolji balans između precision i recall metrika.

Zaključno, neuronske mreže mogu biti korisne za detekciju finansijskih prevara, posebno kada se kombinuju sa odgovarajućim preprocesiranjem, ponderisanjem klasa i pažljivim izborom praga klasifikacije. Ipak, za realnu primenu bilo bi potrebno dodatno testiranje na novijim i raznovrsnijim podacima, kao i uključivanje dodatnih objašnjivih atributa transakcija.

## Autor

Jovana Vuksanović 2022/0293

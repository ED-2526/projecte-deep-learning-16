# Presentació — Identificació de Races de Gossos amb Deep Learning

## Guió complet (Slides + Text per dir)

**Durada estimada**: 12-14 minuts
**Idioma**: Català
**Estil slides**: Minimal. Títol + 1-2 elements visuals o xifres grans. El protagonista ets tu, no les slides.
**Colors suggerits**: Blau fosc (#1e3a5f) per títols, taronja/ambre (#e67e22) per destacats, fons blanc. Temàtica que evoca visió per computador / intel·ligència.

---

## Slide 1 — Portada

**A la slide:**
> **Identificació de Races de Gossos**
> amb Deep Learning
>
> Guillermo Martínez Batlle
> Deep Learning · UAB · Juliol 2026

**Guió:**
«Bon dia. Avui us presentaré el meu projecte de Deep Learning: ensenyar a una xarxa neuronal a distingir entre 120 races de gossos a partir d'una sola fotografia. Semblava un problema senzill fins que vaig descobrir que no ho és gens.»

---

## Slide 2 — El repte (slide d'impacte)

**A la slide:**
> [Dues fotos costat a costat: un Malinois i un Pastor Alemany — quasi idèntics]
>
> **Quina diferència veus?**

**Guió:**
«Comencem amb una pregunta. Mireu aquestes dues fotos. Sabríeu dir quina raça és cadascuna? La de l'esquerra és un Malinois, la de la dreta un Pastor Alemany. Són races molt similars — fins i tot un expert pot dubtar. Ara imagineu fer això amb 120 races diferents, moltes amb diferències mínimes. Aquest és exactament el repte que afronta el nostre model.»

---

## Slide 3 — El dataset

**A la slide:**
> **10.222** IMATGES · **120** RACES · **~85** IMATGES/RAÇA
>
> Font: Kaggle Dog Breed Identification
>
> Split estratificat 80/20

**Guió:**
«El dataset ve de Kaggle. Tenim unes 10.000 imatges etiquetades amb 120 races. Cada raça té al voltant de 85 imatges — no és gaire. Per context, models com ResNet es van entrenar amb 1.2 milions d'imatges. Nosaltres tenim 100 vegades menys dades. Això condiciona totes les decisions que prendrem més endavant. Hem fet un split estratificat 80-20 per assegurar que cada raça està igualment representada al train i al validation.»

---

## Slide 4 — El punt de partida

**A la slide:**
> **Baseline: CNN de 2 capes**
>
> **5,04%**
> ACCURACY DE VALIDACIÓ
>
> (atzar: 0,83%)

**Guió:**
«El professor ens va donar un punt de partida: una CNN de dues capes convolucionals amb uns 25.000 paràmetres. Molt petita. El resultat? Un 5% d'accuracy. Pot semblar molt dolent, però és 6 vegades millor que l'atzar pur, que seria un 0.83% amb 120 classes. Això ens diu que el model aprèn alguna cosa — però 25.000 paràmetres no poden codificar la complexitat visual de 120 races. Necessitem una estratègia diferent.»

---

## Slide 5 — Slide de transició

**A la slide:**
> **I si li donem a la xarxa**
> **uns ulls que ja saben veure?**

**Guió:**
«I aquí és on entra el transfer learning.»

---

## Slide 6 — Transfer learning

**A la slide:**
> **ResNet50** — entrenat amb 1.2M imatges d'ImageNet
>
> Backbone congelat (23.5M params) → Només entrenem el classificador (1.1M params)
>
> **5,04% → 85,97%**

**Guió:**
«ResNet50 és una xarxa de 50 capes que ja ha après a veure: vores, textures, formes, parts d'objectes. Tot aquest coneixement visual ve d'haver-se entrenat amb 1.2 milions d'imatges d'ImageNet. El que fem és congelar tota aquesta base — no la toquem — i només entrenar un petit classificador a sobre que mapeja les features de ResNet50 a les nostres 120 races. Resultat: de 5% a 86%. Una millora de 17 vegades, simplement perquè hem reutilitzat coneixement visual que ja existia.»

---

## Slide 7 — Fine-tuning: el fracàs i la correcció

**A la slide:**
> Títol: "Fine-tuning — El fracàs i la correcció"
>
> [Dos charts de wandb costat a costat]
> Esquerra (vermell): Fine-tune v1 — 85,13% (train/accuracy i val/accuracy)
> Dreta (verd): Fine-tune v2 — 87,48% (train/accuracy i val/accuracy)
>
> LR backbone: 1e-4 → **1e-5** · Dropout: 0.3 → **0.5**

**Guió:**
«Ara ve la part més interessant del projecte. El pas natural era descongelar les últimes capes de ResNet50 perquè s'adaptessin a les races de gossos. L'expectativa era pujar un 5-8%. Mireu el gràfic de l'esquerra — el que va passar és exactament el contrari. La línia blava és l'accuracy de train: puja fins al 99.8%. La taronja és la de validació: es queda estancada al 85%. Això és overfitting pur — el model memoritza el training set en lloc d'aprendre. Per què? 16 milions de paràmetres entrenables amb només 8.000 imatges. El learning rate de 1e-4 era massa agressiu i destruïa les features preentrenades.

Ara mireu a la dreta. Vam fer dos canvis: reduir el learning rate del backbone 10 vegades — de 1e-4 a 1e-5 — i pujar el dropout de 0.3 a 0.5. Fixeu-vos com el gap entre train i val es redueix dràsticament. Resultat: 87.48%, superant per primer cop la versió congelada. La lliçó és clara: descongelar paràmetres no és automàticament millor — la regularització ha d'escalar amb el nombre de paràmetres entrenables.»

---

## Slide 8 — Label smoothing

**A la slide:**
> **1 línia de codi**
>
> `CrossEntropyLoss(label_smoothing=0.1)`
>
> **87,48% → 88,12%**
>
> Gap train-val: 9% → **5%**

**Guió:**
«L'última millora va ser la més senzilla. Label smoothing. En lloc de dir-li al model "això és 100% un Golden Retriever", li diem "és un 99% Golden Retriever i un 0.07% cadascuna de les altres races". Això evita que el model sigui massa confiat. El resultat: 88.12% d'accuracy i, el que és més important, el gap entre train i validació va baixar del 9% al 5%. El model ja no memoritza — generalitza.»

---

## Slide 9 — Resum d'iteracions

**A la slide:**

> | Iteració | Canvi clau | Val Acc |
> |----------|-----------|---------|
> | Baseline CNN | — | 5,04% |
> | ResNet50 congelat | Transfer learning | 85,97% |
> | Fine-tune v1 | Descongelar layer4 | 85,13% |
> | Fine-tune v2 | Correcció LR + dropout | 87,48% |
> | + Label smoothing | Suavitzar targets | **88,12%** |

**Guió:**
«Aquí teniu el resum de totes les iteracions. Fixeu-vos en la progressió: cada canvi és mínim i aïllat. No hem canviat 5 coses alhora — hem canviat una, hem mesurat, i hem après. Això és el que et permet respondre "per què aquest paràmetre?" amb dades, no amb intuïció. I fixeu-vos que la iteració que va fallar és tant o més valuosa que les que van funcionar, perquè ens va ensenyar una lliçó fonamental sobre regularització.»

---

## Slide 10 — Slide de transició

**A la slide:**
> **Però... què ha après realment el model?**

**Guió:**
«Ara que tenim un model amb un 88% d'accuracy, la pregunta important és: què ha après? Mira els patrons correctes o ha trobat dreceres?»

---

## Slide 11 — Grad-CAM

**A la slide:**
> [Imatge: gradcam_correct.png — 2x3 grid de prediccions correctes amb heatmaps]
>
> **Grad-CAM — On mira el model**

**Guió:**
«Grad-CAM ens mostra quines regions de la imatge contribueixen més a la predicció. I el que veiem és molt encoratjador: el model mira la cara, les orelles, el musell, la forma del cos — exactament el que miraria un humà. No està fent trampa mirant el fons o les marques d'aigua. Això valida que ha après features genuïnes de cada raça.»

---

## Slide 12 — On s'equivoca

**A la slide:**
> [Imatge: confusion_matrix_top20.png]
>
> **Les confusions tenen sentit**

**Guió:**
«I quan s'equivoca? Mireu les parelles més confoses. Eskimo dog amb Husky Siberià — el 61% dels eskimo dogs es classifiquen com a huskies. Toy poodle amb miniature poodle — una distinció que depèn de la mida, impossible de determinar en una foto sense referència. Collie amb border collie. Cardigan amb pembroke. Totes aquestes són races que genuïnament s'assemblen moltíssim. No són errors aleatoris — són exactament les confusions que tindria un expert humà.»

---

## Slide 13 — Top-K accuracy (slide d'impacte)

**A la slide:**
> **88%** top-1 · **97,6%** top-3 · **98,5%** top-5
>
> Quan s'equivoca, la raça correcta quasi sempre
> és a les seves 3 primeres opcions.

**Guió:**
«I aquí la xifra que per mi és la més reveladora. Si mirem el top-5 accuracy — és a dir, si la raça correcta és entre les 5 prediccions més probables del model — arribem al 98.5%. Només 30 imatges de 2.045 tenen la raça correcta fora del top-5. El model no falla estrepitosament — quan s'equivoca, s'equivoca per poc. Això demostra que ha construït una comprensió real de la similitud visual entre races.»

---

## Slide 14 — Calibració

**A la slide:**
> [Imatge: confidence_distribution.png]
>
> Correctes: confiança **0,82** · Incorrectes: confiança **0,52**

**Guió:**
«Últim punt de l'anàlisi: la calibració. Quan el model encerta, ho fa amb una confiança mitjana del 82%. Quan s'equivoca, la confiança cau al 52%. El model sap quan no està segur. Això és crític en una aplicació real — podries posar un llindar de confiança i derivar els casos dubtosos a revisió humana, augmentant la fiabilitat del sistema.»

---

## Slide 15 — Conclusions

**A la slide:**
> 1. **Transfer learning** és la clau — 5% → 86% sense entrenar el backbone
> 2. **Fine-tuning requereix cura** — descongelar ingènuament empitjora
> 3. **El model entén les races** — Grad-CAM confirma features rellevants
> 4. **98,5% top-5** — quasi sempre té la resposta correcta a prop

**Guió:**
«Per concloure. Primera lliçó: el transfer learning ho canvia tot. Passar de 5% a 86% simplement reutilitzant features preentrenades. Segona: el fine-tuning no és automàtic — cal adaptar la regularització. La nostra iteració fallida ho demostra. Tercera: Grad-CAM valida que el model ha après features genuïnes, no dreceres. I quarta: amb un 98.5% de top-5, el model rarament fa prediccions completament equivocades. Per una xarxa que ha vist només 8.000 imatges, és un resultat notable. Gràcies.»

---

## Slide 16 — Preguntes

**A la slide:**
> **Preguntes?**
>
> Guillermo Martínez Batlle
> Deep Learning · UAB · 2026

**Guió:**
[Esperar preguntes]

---

## Notes

- **Presentació construïda amb**: Marp (Markdown → HTML/PDF)
- **Paleta de colors**: blau fosc (#1e3a5f) títols, taronja (#e67e22) destacats, fons blanc
- **Font**: Montserrat
- **Imatges utilitzades**:
  - Slide 2: `analysis/confused_breeds_example.png`
  - Slide 7: `analysis/wandb_finetune_v1_accuracy.png` + `analysis/wandb_finetune_v2_accuracy.png`
  - Slide 11: `analysis/gradcam_correct.png`
  - Slide 12: `analysis/confusion_matrix_top20.png`
  - Slide 14: `analysis/confidence_distribution.png`
- **Total**: 16 slides · ~12-14 minuts
- **Rebuild**: `cd presentation && marp presentation.md --html --pdf --allow-local-files`
- **Recorda**: el text a les slides és MÍNIM. La informació la dones tu parlant.

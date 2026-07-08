---
marp: true
theme: default
paginate: true
color: #1a1a2e
style: |
  @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700;800;900&display=swap');
  section {
    font-family: 'Montserrat', sans-serif;
    font-size: 26px;
    padding: 50px 70px;
    background: #ffffff;
    position: relative;
  }
  section::after {
    content: attr(data-marpit-pagination) ' / ' attr(data-marpit-pagination-total);
    font-size: 12px;
    color: #888;
  }
  h1 {
    color: #1e3a5f;
    font-weight: 800;
    font-size: 44px;
    margin-bottom: 24px;
  }
  h2 {
    color: #1e3a5f;
    font-weight: 700;
    font-size: 36px;
    margin-bottom: 20px;
    border-bottom: 3px solid #e8f0fe;
    padding-bottom: 10px;
  }
  h3 {
    color: #2563eb;
    font-weight: 600;
    font-size: 26px;
    margin-bottom: 12px;
  }
  strong {
    color: #1e3a5f;
  }
  code {
    background: #f0f7ff;
    color: #1e3a5f;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 22px;
  }
  table {
    font-size: 22px;
    margin: 10px auto;
    border-collapse: collapse;
    width: 100%;
  }
  th {
    background: #f0f7ff;
    color: #1e3a5f;
    font-weight: 700;
    padding: 10px 16px;
    border-bottom: 2px solid #1e3a5f;
  }
  td {
    padding: 8px 16px;
    border-bottom: 1px solid #e8f0fe;
  }
  tr:nth-child(even) td {
    background: #f8fbff;
  }
  blockquote {
    border-left: 4px solid #e67e22;
    background: #fef9f0;
    padding: 14px 24px;
    margin: 20px 0;
    border-radius: 0 12px 12px 0;
    font-style: normal;
    color: #5a3e1b;
  }
  img {
    border-radius: 12px;
  }
  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 40px;
  }
  .stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 20px;
    margin-top: 20px;
  }
  .stat-box {
    background: #f0f7ff;
    border: 1px solid #e8f0fe;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
  }
  .stat-value {
    font-size: 42px;
    font-weight: 800;
    color: #1e3a5f;
    line-height: 1.1;
  }
  .stat-label {
    font-size: 14px;
    color: #555;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .finding {
    background: #f8fbff;
    border-left: 5px solid;
    border-radius: 0 12px 12px 0;
    padding: 14px 20px;
    margin: 10px 0;
    font-size: 23px;
  }
  .yes { border-color: #22c55e; }
  .warn { border-color: #e67e22; }
  .no { border-color: #ef4444; }
  .accent { color: #e67e22; }
  .big-number {
    font-size: 72px;
    font-weight: 800;
    color: #1e3a5f;
    line-height: 1;
  }
  .big-label {
    font-size: 20px;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 8px;
  }
  .strikethrough {
    text-decoration: line-through;
    color: #999;
  }

---

<!-- _paginate: false -->

<style scoped>
section {
  background: #ffffff !important;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  padding: 60px;
}
h1 { color: #1e3a5f !important; font-size: 64px; margin-bottom: 0; margin-top: 0; }
.subtitle { font-size: 28px; color: #2563eb; max-width: 700px; line-height: 1.6; margin-top: 20px; font-weight: 300; }
.author { font-size: 20px; color: #1a1a2e; margin-top: 50px; font-weight: 500; }
.meta { font-size: 16px; color: #555; margin-top: 6px; }
</style>

# 🐕 Dog Breed ID

<div class="subtitle">Identificació de Races de Gossos amb Deep Learning</div>

<div class="author">Guillermo Martínez Batlle</div>
<div class="meta">Deep Learning · UAB · Juliol 2026</div>

---

<!-- _paginate: false -->

<style scoped>
section {
  background: #ffffff !important;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
}
h1 { color: #1e3a5f; font-size: 48px; border: none; font-weight: 800; }
p { color: #555; font-size: 24px; font-weight: 400; }
</style>

# Quina diferència veus?

![w:800](../analysis/confused_breeds_example.png)

---

<style scoped>
section { justify-content: flex-start; padding-top: 40px !important; }
h2 { margin-bottom: 20px; }
</style>

## El dataset

<div class="stat-grid">

<div class="stat-box">
<div class="stat-value">10.222</div>
<div class="stat-label">Imatges etiquetades</div>
</div>

<div class="stat-box">
<div class="stat-value">120</div>
<div class="stat-label">Races de gossos</div>
</div>

<div class="stat-box">
<div class="stat-value">~85</div>
<div class="stat-label">Imatges per raça</div>
</div>

</div>

> Font: Kaggle Dog Breed Identification · Split estratificat 80/20

---

<style scoped>
section {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
}
h2 { border: none; margin-bottom: 30px; }
</style>

## El punt de partida — Baseline CNN

<div class="big-number">5,04%</div>
<div class="big-label">Accuracy de validació</div>

<div style="margin-top: 30px; font-size: 22px; color: #888;">
Atzar pur: 0,83% (1/120) · ~25.000 paràmetres
</div>

---

<!-- _paginate: false -->

<style scoped>
section {
  background: #ffffff !important;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  padding: 60px 80px;
}
h1 { color: #1e3a5f; font-size: 44px; border: none; font-weight: 700; }
p { font-size: 26px; color: #555; max-width: 750px; line-height: 1.7; }
</style>

# I si li donem a la xarxa uns ulls que ja saben veure?

---

<style scoped>
section { justify-content: flex-start; padding-top: 30px !important; }
h2 { margin-bottom: 15px; }
</style>

## Transfer Learning — ResNet50 congelat

<div class="columns">
<div>

### El concepte

ResNet50 entrenat amb **1.2M imatges** d'ImageNet

Congelar backbone (23.5M params)
Entrenar només el classificador (1.1M params)

</div>
<div>

<div class="stat-box" style="margin-top: 20px;">
<div style="font-size: 20px; color: #888;">BASELINE</div>
<div class="stat-value" style="font-size: 36px;">5,04%</div>
<div style="font-size: 30px; margin: 10px 0;">→</div>
<div style="font-size: 20px; color: #888;">TRANSFER LEARNING</div>
<div class="stat-value" style="font-size: 36px;">85,97%</div>
<div class="stat-label" style="margin-top: 12px;">×17 millora</div>
</div>

</div>
</div>

---

<style scoped>
section { justify-content: flex-start; padding-top: 20px !important; }
h2 { margin-bottom: 10px; }
p { text-align: center; margin: 0; font-size: 18px; color: #666; }
.wandb-compare {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin-top: 10px;
}
.wandb-compare div { text-align: center; }
.wandb-compare img { border-radius: 10px; border: 1px solid #e0e0e0; }
.tag-fail { color: #ef4444; font-weight: 700; font-size: 16px; }
.tag-fix { color: #22c55e; font-weight: 700; font-size: 16px; }
</style>

## Fine-tuning — El fracàs i la correcció

<div class="wandb-compare">
<div>
<span class="tag-fail">Fine-tune v1 — 85,13%</span>

![w:470](../analysis/wandb_finetune_v1_accuracy.png)
</div>
<div>
<span class="tag-fix">Fine-tune v2 — 87,48%</span>

![w:470](../analysis/wandb_finetune_v2_accuracy.png)
</div>
</div>

LR backbone: 1e-4 → **1e-5** · Dropout: 0.3 → **0.5**

---

<style scoped>
section {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
}
h2 { border: none; margin-bottom: 20px; }
</style>

## Label Smoothing — 1 línia de codi

`CrossEntropyLoss(label_smoothing=0.1)`

<div style="margin-top: 30px;">
<span class="big-number" style="font-size: 56px;">87,48%</span>
<span style="font-size: 40px; color: #888; margin: 0 20px;">→</span>
<span class="big-number" style="font-size: 56px;">88,12%</span>
</div>

<div style="margin-top: 20px; font-size: 20px; color: #888;">
Gap train-val: 9% → <strong style="color: #22c55e;">5%</strong>
</div>

---

<style scoped>
section { justify-content: flex-start; padding-top: 25px !important; }
h2 { margin-bottom: 10px; }
table { font-size: 21px; margin-top: 10px; }
td:nth-child(3) { text-align: center; }
th:nth-child(3) { text-align: center; }
</style>

## Resum d'iteracions

| Iteració | Canvi clau | Val Acc |
|----------|-----------|:-------:|
| Baseline CNN | — | 5,04% |
| ResNet50 congelat | Transfer learning | 85,97% |
| Fine-tune v1 | Descongelar layer4 | 85,13% |
| Fine-tune v2 | Correcció LR + dropout | 87,48% |
| **+ Label smoothing** | **Suavitzar targets** | **88,12%** |

> Cada iteració canvia **una sola cosa**. Així sabem exactament què funciona.

---

<!-- _paginate: false -->

<style scoped>
section {
  background: #ffffff !important;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  padding: 60px 80px;
}
h1 { color: #1e3a5f; font-size: 48px; border: none; font-weight: 700; }
p { font-size: 24px; color: #555; }
</style>

# Però... què ha après realment el model?

---

<style scoped>
section { justify-content: flex-start; padding-top: 20px !important; }
h2 { margin-bottom: 5px; }
p { text-align: center; margin: 0; font-size: 20px; }
</style>

## Grad-CAM — On mira el model

![w:920 center](../analysis/gradcam_correct.png)

El model mira la **cara, orelles, musell i cos** — no el fons

---

<style scoped>
section { justify-content: flex-start; padding-top: 20px !important; }
h2 { margin-bottom: 5px; }
p { text-align: center; margin: 0; font-size: 18px; color: #666; }
</style>

## Les confusions tenen sentit

![w:750 center](../analysis/confusion_matrix_top20.png)

Eskimo dog ↔ Husky · Toy ↔ Miniature poodle · Collie ↔ Border collie

---

<style scoped>
section {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
}
h2 { border: none; margin-bottom: 30px; }
</style>

## Top-K Accuracy

<div class="stat-grid">

<div class="stat-box">
<div class="stat-value">88,1%</div>
<div class="stat-label">Top-1</div>
</div>

<div class="stat-box">
<div class="stat-value" style="color: #2563eb;">97,6%</div>
<div class="stat-label">Top-3</div>
</div>

<div class="stat-box">
<div class="stat-value" style="color: #22c55e;">98,5%</div>
<div class="stat-label">Top-5</div>
</div>

</div>

<div style="margin-top: 30px; font-size: 22px; color: #555;">
Quan s'equivoca, la raça correcta quasi sempre és entre les 3 primeres opcions
</div>

---

<style scoped>
section { justify-content: flex-start; padding-top: 20px !important; }
h2 { margin-bottom: 5px; }
p { text-align: center; margin: 0; }
</style>

## Calibració — El model sap quan dubta

![w:850 center](../analysis/confidence_distribution.png)

Correctes: **0,82** confiança · Incorrectes: **0,52** confiança

---

<style scoped>
section { justify-content: flex-start; padding-top: 25px !important; }
h2 { margin-bottom: 15px; }
</style>

## Conclusions

<div class="finding yes">
<strong style="color: #16a34a;">Transfer learning és la clau</strong> — 5% → 86% sense entrenar el backbone
</div>

<div class="finding no">
<strong style="color: #dc2626;">Fine-tuning requereix cura</strong> — descongelar ingènuament empitjora el model
</div>

<div class="finding yes">
<strong style="color: #16a34a;">El model entén les races</strong> — Grad-CAM confirma features rellevants
</div>

<div class="finding warn">
<strong style="color: #d97706;">98,5% top-5</strong> — quasi sempre té la resposta correcta a prop
</div>

---

<!-- _paginate: false -->

<style scoped>
section {
  background: #1e3a5f !important;
  color: white !important;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
}
h1 { color: #93c5fd !important; font-size: 56px; border: none; margin-bottom: 16px; }
.thanks { font-size: 38px; color: #bfdbfe; margin-bottom: 30px; }
.author { font-size: 22px; color: #e0effe; margin-top: 40px; }
.meta { font-size: 16px; color: #93c5fd; margin-top: 8px; }
</style>

<div class="thanks">Gràcies</div>

# Preguntes?

<div class="author">Guillermo Martínez Batlle</div>
<div class="meta">Deep Learning · UAB · Juliol 2026</div>

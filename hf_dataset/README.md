---
pretty_name: Turkish Dermatology RAG Chunks
language:
- tr
license: cc-by-4.0
task_categories:
- feature-extraction
- sentence-similarity
size_categories:
- 1K<n<10K
tags:
- medical
- turkish
- dermatology
- rag
- embeddings
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.parquet
---

# Turkish Dermatology RAG Chunks

Bu veri kümesi, `umutertugrul/turkish-medical-articles` koleksiyonundaki
Dermatoloji branşından deterministik olarak seçilen 500 Türkçe makalenin
arama odaklı chunk ve embedding temsillerini içerir.

> Eğitim ve bilgi erişim araştırması içindir. Klinik karar desteği, tanı veya
> tedavi amacıyla kullanılmamalıdır.

## Kaynak ve atıf

- Kaynak: [umutertugrul/turkish-medical-articles](https://huggingface.co/datasets/umutertugrul/turkish-medical-articles)
- Orijinal yayın sitesi: Doktorsitesi.com
- Kaynak lisansı: CC BY 4.0
- Kaynak veri seti gated'dır ve dosyalara erişim için koşulların kabulü gerekir.
- Her satır, atıf ve kaynağa geri dönüş için orijinal `url` alanını korur.

Bu türetilmiş veri kümesi kaynak metinleri dönüştürdüğü için CC BY 4.0
altında yayımlanır. Kullanıcılar orijinal kaynağı belirtmeli ve kaynak veri
kümesinin güncel erişim koşullarını ayrıca incelemelidir.

## Oluşturma süreci

1. `branch == Dermatoloji` filtresi uygulandı.
2. Zorunlu alanı eksik, 200 karakterden kısa ve yinelenen kayıtlar elendi.
3. URL'ye göre sabit sıralama sonrası `seed=42` ile 500 makale seçildi.
4. Paragraf → cümle → token öncelikli mixed chunking uygulandı.
5. Hedef 512 token, overlap 64 token ve minimum 80 token kullanıldı.
6. `magibu/embeddingmagibu-200m` ile 768 boyutlu L2-normalized vektörler üretildi.

Toplam 500 parent makaleden 1.019 chunk oluşturuldu. Başlık her chunk'a dahil
edilir ve token bütçesinin parçasıdır.

## Şema

| Alan | Tip | Açıklama |
|---|---|---|
| `url` | string | Orijinal makale bağlantısı |
| `chunk_text` | string | Başlık dahil arama parçası |
| `chunk_vector` | float32[768] | L2-normalized Magibu embedding |
| `chunk_id` | string | Deterministik benzersiz chunk kimliği |
| `parent_id` | string | Kaynak makale kimliği |
| `title` | string | Makale başlığı |
| `branch` | string | Uzmanlık dalı; bu sürümde Dermatoloji |
| `chunk_index` | int64 | Parent içindeki sıra |
| `token_count` | int64 | Embedding tokenizerına göre token sayısı |
| `embedding_model` | string | Embedding model kimliği |

## Kullanım

```python
from datasets import load_dataset

dataset = load_dataset("berkbirkan/turkish-dermatology-rag-dataset")
print(dataset["train"].features)
print(len(dataset["train"]))  # 1019
```

`chunk_vector` alanı doğrudan cosine arama indeksine aktarılabilir. Sorgular
aynı embedding modeliyle ve query-specific encoding yoluyla vektörleştirilmelidir.

## Benchmark ve threshold

Eşik yalnızca ayrı bir 10 pozitif + 10 negatif kalibrasyon setinde belirlendi:
`0.4451`. Eşik sabit tutularak 20 pozitif + 10 negatif bağımsız testte
answerability precision, recall ve F1 değerleri 1,00 elde edildi. Bu kontrollü ve
küçük benchmark sonucu genel klinik performans iddiası değildir.

## Sınırlılıklar ve sorumlu kullanım

- Metinler halka açık sağlık makaleleridir; klinik kılavuz değildir.
- 500 makale dermatolojinin tamamını temsil etmez.
- Kaynak içerikte tarihsel, yanlı veya eksik bilgi bulunabilir.
- Veri ya da embedding modeli değiştirilirse threshold yeniden kalibre edilmelidir.
- Kullanıcıya gösterilen cevaplarda `url` ve kanıt chunk'ları korunmalıdır.
- Kişisel tıbbi kararlar için nitelikli sağlık profesyoneline başvurulmalıdır.

## Yeniden üretilebilirlik

Seçim, chunking, embedding, Chroma ve benchmark kodları companion source-code
reposunda bulunur. Kritik parametreler YAML yapılandırmasında sabitlenmiştir;
teslim Parquet dosyası ayrıca satır sayısı, şema, sonlu vektör, L2 normu ve
SHA-256 kontrollerinden geçirilir.

# Med Device Topic Sepecialization
Jackson McDowell

``` python
import numpy as np
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from urllib.parse import urljoin, urlparse
import time, random
import lda
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
```

Zimmer scraping Scraped the product and solution pages. Extracted a
product name and a short description. I keep only rows with enough text
to be useful for topic modeling.

``` python
headers = {"User-Agent": "Mozilla/5.0"}

company = "Zimmer Biomet"
start_url = "https://www.zimmerbiomet.com/en/products-and-solutions.html"

r = requests.get(start_url, headers=headers)
soup = BeautifulSoup(r.text, "html.parser")


links = []
for a in soup.select("a[href]"):
    href = a.get("href", "").strip()
    full = urljoin(start_url, href)

    if urlparse(full).netloc != "www.zimmerbiomet.com":
        continue

    if any(x in full.lower() for x in ["products-and-solutions", "product", "solutions"]) and "#" not in full:
        links.append(full)

links = list(dict.fromkeys(links))

print("Links found:", len(links))

products = []
##80 was a good balance of runtime and variety of products, I was still able to notice a pattern with this number. This reasoning is the same for all the slicing I did with other websites. 
for url in links[:80]: 
    try:
        rr = requests.get(url, headers=headers, timeout=20)
        ss = BeautifulSoup(rr.text, "html.parser")

        h1 = ss.select_one("h1")
        name = h1.get_text(strip=True) if h1 else ss.title.get_text(strip=True)

        meta = ss.select_one("meta[name='description']")
        desc = meta.get("content", "").strip() if meta else ""

        if not desc:
            main = ss.select_one("main") or ss
            p = main.select_one("p")
            desc = p.get_text(" ", strip=True) if p else ""

        if name and len(name) >= 4 and len(desc) >= 20:
            products.append({
                "company": company,
                "product": name,
                "description": desc,
                "url": url
            })

    except Exception:
        continue

df = pd.DataFrame(products)

df = df.drop_duplicates(subset=["company", "product"])

if "url" in df.columns:
    df = df.drop(columns=["url"])

df = df[df["description"].str.len() > 40]

df["text"] = (df["product"] + " " + df["description"]).str.lower()

df = df.reset_index(drop=True)

print(df.head())
print("Rows:", len(df))

df.to_csv("zimmer_products_clean.csv", index=False)
```

``` python
import pandas as pd

df = pd.read_csv("zimmer_products_clean.csv")
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | company | product | description | text |
|----|----|----|----|----|
| 0 | Zimmer Biomet | OrthoGrid | OrthoGrid’s digital platform creates a robust ... | orthogrid orthogrid’s digital platform creates... |
| 1 | Zimmer Biomet | Biologics Solutions | Zimmer Biomet Biologics has a range of solutio... | biologics solutions zimmer biomet biologics ha... |
| 2 | Zimmer Biomet | Bone Cement for Arthroplasty Procedures | Zimmer Biomet offers surgeons a complete portf... | bone cement for arthroplasty procedures zimmer... |
| 3 | Zimmer Biomet | Craniomaxillofacial (CMF) | Zimmer Biomet offers a wide array of devices f... | craniomaxillofacial (cmf) zimmer biomet offers... |
| 4 | Zimmer Biomet | mymobility®Care Management Platform | mymobility with Apple Watch is a digital care ... | mymobility®care management platform mymobility... |

</div>

Stryker I filtered out non-product pages (careers, investors, policies,
etc.), sample a subset for runtime, and extract names + descriptions.
This website had a lot of broad coverage. I also kept only rows with
enough text to be useful for topic modeling.

``` python
headers = {"User-Agent": "Mozilla/5.0"}

company = "Stryker"
start_url = "https://www.stryker.com/us/en/site-map.html"

r = requests.get(start_url, headers=headers, timeout=20)
soup = BeautifulSoup(r.text, "html.parser")

#links that look like product pages
links = []
for a in soup.select("a[href]"):
    href = a.get("href", "").strip()
    full = urljoin(start_url, href)

    if urlparse(full).netloc != "www.stryker.com":
        continue

    full_l = full.lower()

    if any(bad in full_l for bad in [
        "product-inquiry", "contact", "careers", "investors",
        "privacy", "terms", "/about/", "/news/", "/events/",
        "/training", "/services/", "/company", "/locations"
    ]):
        continue

    if any(x in full_l for x in [
        "/products", "implant", "system", "platform", "instrument"
    ]) and "#" not in full:
        links.append(full)

links = list(dict.fromkeys(links))
print("Links found:", len(links))

random.seed(1)
links = random.sample(links, min(120, len(links)))

products = []

for url in links:
    try:
        time.sleep(random.uniform(.1, .4))

        rr = requests.get(url, headers=headers, timeout=20)
        ss = BeautifulSoup(rr.text, "html.parser")

        h1 = ss.select_one("h1")
        name = h1.get_text(strip=True) if h1 else (ss.title.get_text(strip=True) if ss.title else "")

        if len(name.split()) <= 1:
            continue

        meta = ss.select_one("meta[name='description']")
        desc = meta.get("content", "").strip() if meta else ""

        if not desc:
            og = ss.select_one("meta[property='og:description']")
            desc = og.get("content", "").strip() if og else ""

        if not desc:
            main = ss.select_one("main") or ss
            p = main.select_one("p")
            desc = p.get_text(" ", strip=True) if p else ""

        if name and len(name) >= 4 and len(desc) >= 20:
            products.append({
                "company": company,
                "product": name,
                "description": desc,
                "url": url
            })

    except Exception:
        continue

df = pd.DataFrame(products)

df["product"] = df["product"].str.replace("| Stryker","", regex=False)



df = df.drop_duplicates(subset=["company", "product"])

if "url" in df.columns:
    df = df.drop(columns=["url"])

df = df[df["description"].astype(str).str.len() > 40]

df["text"] = (df["product"].astype(str) + " " + df["description"].astype(str)).str.lower()

df = df.reset_index(drop=True)

print(df.head())
print("Rows:", len(df))

df.to_csv("stryker_no_noise_2.csv", index=False)
```

``` python
import pandas as pd
df = pd.read_csv("stryker_no_noise_2.csv")
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | company | product | description | text |
|----|----|----|----|----|
| 0 | Stryker | LITe Pedicle Access Solution | A less invasive pedicle access solution design... | lite pedicle access solution a less invasive p... |
| 1 | Stryker | IsoFlex SE | Our premier stretcher surface, IsoFlex SE, add... | isoflex se our premier stretcher surface, isof... |
| 2 | Stryker | High Speed Drills | The most comprehensive and customizable neuros... | high speed drills the most comprehensive and c... |
| 3 | Stryker | Aleutian Lateral | The Aleutian Lateral Interbody System is desig... | aleutian lateral the aleutian lateral interbod... |
| 4 | Stryker | Vitoss BA | A synthetic bone graft substitute with bioacti... | vitoss ba a synthetic bone graft substitute wi... |

</div>

Olympus Organizes products by category pages. I collect category links,
then collect individual product links from those categories, and extract
product name + description.

``` python
headers = {"User-Agent": "Mozilla/5.0"}

company = "Olympus"
start_url = "https://medical.olympusamerica.com/products"

r = requests.get(start_url, headers=headers)
soup = BeautifulSoup(r.text, "html.parser")

category_links = []

for a in soup.select("a[href]"):
    href = a.get("href","")
    full = urljoin(start_url, href)

    if "/products/" in full and "#" not in full:
        category_links.append(full)

category_links = list(set(category_links))
print("Category pages:", len(category_links))


product_links = []

for cat in category_links:
    try:
        rr = requests.get(cat, headers=headers)
        ss = BeautifulSoup(rr.text,"html.parser")

        for a in ss.select("a[href]"):
            href = a.get("href","")
            full = urljoin(cat, href)

            if "/products/" in full and full != cat:
                product_links.append(full)

    except:
        pass

product_links = list(set(product_links))
print("Product pages:", len(product_links))


products = []

for url in product_links:
    try:
        time.sleep(random.uniform(.2,.5))

        rr = requests.get(url, headers=headers)
        ss = BeautifulSoup(rr.text,"html.parser")

        h1 = ss.select_one("h1")
        name = h1.get_text(strip=True) if h1 else ""

        meta = ss.select_one("meta[name='description']")
        desc = meta.get("content","").strip() if meta else ""

        if not desc:
            p = ss.select_one("p")
            desc = p.get_text(" ", strip=True) if p else ""

        if len(name) > 3 and len(desc) > 25:
            products.append({
                "company": company,
                "product": name,
                "description": desc
            })

    except:
        pass

df = pd.DataFrame(products)

df = df[~df["product"].str.contains("standard|innovation|solutions", case=False, na=False)]

df = df.drop_duplicates()

df["text"] = (df["product"] + " " + df["description"]).str.lower()

print("Rows:", len(df))

df.to_csv("olympus_clean_3.csv", index=False)
```

``` python
import pandas as pd
df = pd.read_csv("olympus_clean_3.csv")
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | company | product | description | text |
|----|----|----|----|----|
| 0 | Olympus | ERCP Guidewires | ERCP Guidewires ERCP access solutions tailored... | ercp guidewires ercp guidewires ercp access so... |
| 1 | Olympus | Contained Tissue Extraction System | Overview Pneumoliner GUARDENIA Videos ... | contained tissue extraction system overview ... |
| 2 | Olympus | Resection in Saline Electrodes | Olympus Resection in Saline Electrodes include... | resection in saline electrodes olympus resecti... |
| 3 | Olympus | VisiGlide™ Guidewires | Single-use Olympus VisiGlide Guidewires are pa... | visiglide™ guidewires single-use olympus visig... |
| 4 | Olympus | TJF-Q190V Duodenoscope | The TJF-Q190V is the newest generation of Olym... | tjf-q190v duodenoscope the tjf-q190v is the ne... |

</div>

I combined the cleaned company datasets into one file for the topic
model.

``` python
import glob

folder = r"C:\Users\JacksonMcDowell\Desktop\Unstructured\final_data"

files = glob.glob(folder + r"\*.csv")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df.to_csv(folder + r"\combined_med_device.csv", index=False)

print("Combined", len(files), "files")
```

    Combined 4 files

Topic Modeling I set k=6 to balance interpretability few enough topics
to label with variety enough to separate major device themes.

``` python
path = r"C:\Users\JacksonMcDowell\Desktop\Unstructured\final_data\combined_med_device.csv"
df = pd.read_csv(path)

if "text" not in df.columns:
    df["text"] = (df["product"].astype(str) + " " + df["description"].astype(str)).str.lower()
else:
    df["text"] = df["text"].astype(str).str.lower()

df = df.dropna(subset=["company", "text"]).copy()

df["text_clean"] = (
    df["text"]
      .str.replace(r"\s+", " ", regex=True)
      .str.replace(r"[^a-z\s]", " ", regex=True)  
      .str.replace(r"\s+", " ", regex=True)
      .str.strip()
)
df = df[df["text_clean"].str.split().str.len() >= 10].reset_index(drop=True)

vectorizer = CountVectorizer(
    stop_words=list(ENGLISH_STOP_WORDS), 
    min_df=2,      
    max_df=0.90    
)

X = vectorizer.fit_transform(df["text_clean"])
vocab = vectorizer.get_feature_names_out()

K = 6 
model = lda.LDA(n_topics=K, n_iter=800, random_state=1)
model.fit(X)

topic_word = model.topic_word_
n_top_words = 10

for k, topic_dist in enumerate(topic_word):
    top_idx = np.argsort(topic_dist)[-n_top_words:][::-1]
    top_terms = vocab[top_idx]
    print(f"Topic {k}: " + ", ".join(top_terms))

doc_topic = model.doc_topic_ 
for k in range(K):
    df[f"topic_{k}"] = doc_topic[:, k]

company_topic_means = df.groupby("company")[[f"topic_{k}" for k in range(K)]].mean()
print(company_topic_means)

for comp, row in company_topic_means.iterrows():
    top = row.sort_values(ascending=False)
    top_str = ", ".join([f"{t} ({v:.3f})" for t, v in top.items()])
    print(f"Company: {comp}, Topics: {top_str}")
```

    INFO:lda:n_documents: 874
    INFO:lda:vocab_size: 1634
    INFO:lda:n_words: 21293
    INFO:lda:n_topics: 6
    INFO:lda:n_iter: 800
    INFO:lda:<0> log likelihood: -223981
    INFO:lda:<10> log likelihood: -147716
    INFO:lda:<20> log likelihood: -145181
    INFO:lda:<30> log likelihood: -143159
    INFO:lda:<40> log likelihood: -141840
    INFO:lda:<50> log likelihood: -141202
    INFO:lda:<60> log likelihood: -140574
    INFO:lda:<70> log likelihood: -139984
    INFO:lda:<80> log likelihood: -139714
    INFO:lda:<90> log likelihood: -139404
    INFO:lda:<100> log likelihood: -139001
    INFO:lda:<110> log likelihood: -138813
    INFO:lda:<120> log likelihood: -138732
    INFO:lda:<130> log likelihood: -138587
    INFO:lda:<140> log likelihood: -138368
    INFO:lda:<150> log likelihood: -138245
    INFO:lda:<160> log likelihood: -138099
    INFO:lda:<170> log likelihood: -137906
    INFO:lda:<180> log likelihood: -137796
    INFO:lda:<190> log likelihood: -137823
    INFO:lda:<200> log likelihood: -137807
    INFO:lda:<210> log likelihood: -137715
    INFO:lda:<220> log likelihood: -137757
    INFO:lda:<230> log likelihood: -137743
    INFO:lda:<240> log likelihood: -137709
    INFO:lda:<250> log likelihood: -137618
    INFO:lda:<260> log likelihood: -137464
    INFO:lda:<270> log likelihood: -137576
    INFO:lda:<280> log likelihood: -137594
    INFO:lda:<290> log likelihood: -137534
    INFO:lda:<300> log likelihood: -137491
    INFO:lda:<310> log likelihood: -137593
    INFO:lda:<320> log likelihood: -137605
    INFO:lda:<330> log likelihood: -137469
    INFO:lda:<340> log likelihood: -137405
    INFO:lda:<350> log likelihood: -137553
    INFO:lda:<360> log likelihood: -137500
    INFO:lda:<370> log likelihood: -137484
    INFO:lda:<380> log likelihood: -137522
    INFO:lda:<390> log likelihood: -137463
    INFO:lda:<400> log likelihood: -137538
    INFO:lda:<410> log likelihood: -137496
    INFO:lda:<420> log likelihood: -137448
    INFO:lda:<430> log likelihood: -137327
    INFO:lda:<440> log likelihood: -137468
    INFO:lda:<450> log likelihood: -137401
    INFO:lda:<460> log likelihood: -137533
    INFO:lda:<470> log likelihood: -137476
    INFO:lda:<480> log likelihood: -137418
    INFO:lda:<490> log likelihood: -137399
    INFO:lda:<500> log likelihood: -137428
    INFO:lda:<510> log likelihood: -137511
    INFO:lda:<520> log likelihood: -137339
    INFO:lda:<530> log likelihood: -137314
    INFO:lda:<540> log likelihood: -137514
    INFO:lda:<550> log likelihood: -137427
    INFO:lda:<560> log likelihood: -137425
    INFO:lda:<570> log likelihood: -137389
    INFO:lda:<580> log likelihood: -137462
    INFO:lda:<590> log likelihood: -137539
    INFO:lda:<600> log likelihood: -137259
    INFO:lda:<610> log likelihood: -137225
    INFO:lda:<620> log likelihood: -137324
    INFO:lda:<630> log likelihood: -137361
    INFO:lda:<640> log likelihood: -137363
    INFO:lda:<650> log likelihood: -137241
    INFO:lda:<660> log likelihood: -137392
    INFO:lda:<670> log likelihood: -137142
    INFO:lda:<680> log likelihood: -137309
    INFO:lda:<690> log likelihood: -137161
    INFO:lda:<700> log likelihood: -137116
    INFO:lda:<710> log likelihood: -137071
    INFO:lda:<720> log likelihood: -137197
    INFO:lda:<730> log likelihood: -137118
    INFO:lda:<740> log likelihood: -137171
    INFO:lda:<750> log likelihood: -137146
    INFO:lda:<760> log likelihood: -137228
    INFO:lda:<770> log likelihood: -137145
    INFO:lda:<780> log likelihood: -137166
    INFO:lda:<790> log likelihood: -137101
    INFO:lda:<799> log likelihood: -137144

    Topic 0: olympus, procedures, high, use, platform, flexible, imaging, endoscope, surgical, definition
    Topic 1: patient, li, platform, products, data, zbedge, sage, care, reprocessed, bipolar
    Topic 2: bf, bronchoscope, design, designed, olympus, exera, diagnostic, iii, evis, quality
    Topic 3: stryker, technology, surgical, designed, delivers, locking, platform, endoeye, performance, camera
    Topic 4: blades, guidewires, endoscopic, designed, guidewire, high, biliary, visiglide, olympus, performance
    Topic 5: zimmer, biomet, offers, hip, products, knee, surgeons, solutions, portfolio, bone
                    topic_0   topic_1   topic_2   topic_3   topic_4  topic_5
    company                                                                 
    Olympus        0.223717  0.040377  0.216296  0.120347  0.326384  0.07288
    Stryker        0.123528  0.193045  0.208656  0.239931  0.116550  0.11829
    Zimmer Biomet  0.008192  0.246592  0.006882  0.029046  0.006259  0.70303
    Company: Olympus, Topics: topic_4 (0.326), topic_0 (0.224), topic_2 (0.216), topic_3 (0.120), topic_5 (0.073), topic_1 (0.040)
    Company: Stryker, Topics: topic_3 (0.240), topic_2 (0.209), topic_1 (0.193), topic_0 (0.124), topic_5 (0.118), topic_4 (0.117)
    Company: Zimmer Biomet, Topics: topic_5 (0.703), topic_1 (0.247), topic_3 (0.029), topic_0 (0.008), topic_2 (0.007), topic_4 (0.006)

Visualization for presentation

``` python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
company_topic_means = df.groupby("company")[[f"topic_{k}" for k in range(K)]].mean().reset_index()
company_topic_means_melted = company_topic_means.melt(id_vars="company",
                                                    var_name="topic",
                                                    value_name="proportion")
plt.figure(figsize=(10, 6))
sns.barplot(data=company_topic_means_melted, x="company", y="proportion", hue="topic")
plt.title("Average Topic Proportions by Company")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

![](readme_files/figure-commonmark/cell-11-output-1.png)

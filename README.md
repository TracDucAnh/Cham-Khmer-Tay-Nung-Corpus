# Corpus Overview

This repository provides a multilingual corpus of low-resource languages in Vietnam, including Cham, Khmer, and Tày–Nùng. The dataset was automatically collected and curated from multiple Vietnamese news and broadcasting sources, primarily from the Voice of Vietnam (VOV) multilingual portals, as well as regional and governmental media outlets such as Báo Dân tộc miền núi, Báo Cà Mau, Báo Cần Thơ, and the official website of An Giang Province. These platforms regularly publish content in minority languages or provide multilingual materials related to ethnic communities in Vietnam.

The corpus aims to support Natural Language Processing (NLP) research for under-resourced languages, including tasks such as language modeling, machine translation, information retrieval, and linguistic analysis. These languages remain significantly underrepresented in existing NLP resources, making the construction of structured corpora an important step toward improving language technology accessibility.

The data was crawled, cleaned, and normalized from raw text articles, then organized into structured JSON format. During preprocessing, the corpus was segmented into documents and sentences, and tokenized into subword tokens to facilitate downstream NLP tasks and model training.

Overall, the corpus contains over 44,000 documents and more than 36 million subword tokens, making it one of the largest publicly structured collections for these languages.

# Corpus Statistics

| Metric | Cham | Khmer | Tay-Nung | Total |
|---|---|---|---|---|
| Documents | 11,481 | 27,808 | 5,078 | 44,367 |
| Sentences | 301,317 | 294,077 | 68,487 | 663,881 |
| Subword Tokens | 15,706,638 | 16,168,887 | 4,169,423 | 36,044,948 |

# Intended Use

This corpus is designed for:

- Low-resource NLP research

- Language modeling and representation learning

- Cross-lingual and multilingual NLP studies

- Linguistic analysis of minority languages in Vietnam

- Benchmarking tokenization and segmentation methods for Southeast Asian languages

By releasing this dataset, we hope to facilitate research on underrepresented languages and encourage the development of language technologies for minority communities.

# Cham

![cham](/eda_outputs/cham/cham_09_sentence_counts.png)

![chamwords](/eda_outputs/cham/cham_07_wordcloud.png)

# Khmer
![khmer](/eda_outputs/khmer/khmer_09_sentence_counts.png)

![khmerword](/eda_outputs/khmer/khmer_10_bpe_analysis.png)

# Tay-Nung

![tay-nung](/eda_outputs/tay-nung/tay%20nung_09_sentence_counts.png)

![tay-nung](/eda_outputs/tay-nung/tay%20nung_07_wordcloud.png)

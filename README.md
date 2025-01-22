# Cross-Disciplinary Embeddings

This repository provides the official code and datasets for our paper:

> Words as Bridges: Exploring Computational Support for Cross-Disciplinary Translation Work

Below, you will find instructions on how to access and use the provided data, explore the alignments, and set up the web application.

## Introduction
This project aims to explore how words function as “bridges” in cross-disciplinary contexts, providing computational support to better understand and facilitate translation work between different fields. By leveraging token alignments and embeddings, we examine overlaps and distinctions within an interdisciplinary corpora.


## Dataset
We provide data specific to the two Case Studies in the data/ folder.

## Database for tokenized corpora
For convenience, pre-built SQLite databases for each case study (.db) containing tokenized corpora is available for download. 

This database includes tokenized text from multiple disciplines and supports the webapp.

To use the database:

Download the db file from our provided links.
- [Case study 1](https://drive.google.com/file/d/1-5PjBAXzmVh3GoCl4v1ZdCKYBBRzsSow/view?usp=sharing)
- [Case study 2](https://drive.google.com/file/d/17RyvM7xpcsSv5Ycsf2M_BLBEACcgo_-7/view?usp=sharing)

Place the databases in a suitable directory (e.g., `{ROOT}/databases/`).


## Exploration & Quantitative Evaluation
We provide an ipynb notebook for exploring processed alignments supporting case study 2.
We also provide an ipynb notebook for reproducing the numbers supporting our quantitative evaluation. For these notebooks, we assume a certain folder structure on Google Drive.

## Webapp
We also provide the web application to visualize and interact with cross-disciplinary embeddings, which we used in our user study.

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:

```
cd appcode
pip install -r requirements.txt
```

Run the web application (the exact command may vary depending on the framework):

```
source venv/bin/activate
flask --app app run
```

Venture to http://127.0.0.1:5000 to use the interface to explore the alignments.
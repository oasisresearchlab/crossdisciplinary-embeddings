from flask import Flask, jsonify
import os
import json
from gensim.models import KeyedVectors
from appcode.embed_utils import *
from s2_sqlite_util import *
from flask import g

from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import torch
import pandas as pd

app = Flask(__name__)

c_list = [{"name": "-", "text": "Select a community"}, 
    {"name": "hci", "text": "HCI"}, 
    {"name": "ling", "text": "Applied Linguistics"}, 
    {"name": "edu", "text": "Education"}, 
    {"name": "imm", "text": "Immigrant Studies"},
    {"name": "eth", "text": "Ethnic-racial Identity"},
    {"name": "pro", "text": "Professional Identity"}
]

shared_emb_path = os.path.join("../data/fastText/hci+ling+edu+imm+eth+pro.phrase.sg50i10.vec")

shared_emb = KeyedVectors.load_word2vec_format(shared_emb_path, binary=False)

sbert_model = SentenceTransformer('all-mpnet-base-v2')
embeds = {}
paths = {
    'hci-ling': ('hci--ling', 'hci', 'ling'),
    'hci-edu': ('hci--edu', 'hci', 'edu'),
    'ling-hci': ('ling--hci', 'ling', 'hci'),
    'ling-edu': ('ling--edu', 'ling', 'edu'),
    'ling-imm': ('ling--imm', 'ling', 'imm'),
    'edu-hci': ('edu--hci', 'edu', 'hci'),
    'edu-ling': ('edu--ling', 'edu', 'ling'),
    'hci-imm': ('hci--imm', 'hci', 'imm'),
    'hci-eth': ('hci--eth', 'hci', 'eth'),
    'hci-pro': ('hci--pro', 'hci', 'pro'),
    'ling-eth': ('ling--eth', 'ling', 'eth'),
    'ling-pro': ('ling--pro', 'ling', 'pro'),
    'edu-imm': ('edu--imm', 'edu', 'imm'),
    'edu-eth': ('edu--eth', 'edu', 'eth'),
    'edu-pro': ('edu--pro', 'edu', 'pro'),
}

DATA_BASE_DIR = 'vecmap/'

for pair, (path, src, tgt) in paths.items():
    print(f"loading {src}-{tgt}...")
    embeds[src] = embeds.get(src, {})
    embeds[src][tgt] = {}

    embeds[src][tgt]["src_emb"] = KeyedVectors.load_word2vec_format(os.path.join(DATA_BASE_DIR, f"{path}/vectors-{src}.txt"), binary=False)
    embeds[src][tgt]["tgt_emb"] = KeyedVectors.load_word2vec_format(os.path.join(DATA_BASE_DIR, f"{path}/vectors-{tgt}.txt"), binary=False)

from openai import  OpenAI
"""

all_ctx_list = get_ctx_list(cur)
print(len(all_ctx_list))
all_ctx_list_filtered = list(filter(lambda ctx: len(ctx['sent']) > 50, all_ctx_list))
print(len(all_ctx_list_filtered))
#corpus = list(map(lambda ctx: ctx['sent'], all_ctx_list_filtered))
corpus = list(map(lambda ctx: ctx['sent'], all_ctx_list_filtered))[:1000]
corpus_embeddings = sbert_model.encode(corpus, convert_to_tensor=True)
print(corpus_embeddings.shape)
"""

@app.route('/')
def index():
    return app.send_static_file("index.html")

@app.route('/<filename>')
def root(filename):
    return app.send_static_file(filename)

@app.route('/query-ctx/<src>/<tgt>/<word>')
def query_ctx(src=None, word=None, tgt=None):
    client = OpenAI(
       api_key=os.environ.get("OPENAI_API_KEY")
    )

    _, cur = get_db_con_cur("../databases/hci+ling+edu+imm+eth+pro.in_out_1.s2orc.20200705v1.db")  # TODO: adapt to your DB path
    tgt_emb = embeds[src][tgt]["tgt_emb"]

    # if os.path.isfile("cachefiles/{}-{}-{}.json".format(src, word, tgt)):
        # print("Loading from cache file {}-{}-{}.json".format(src, word, tgt), flush=True)
        # time.sleep(5)
        # return jsonify(json.load(open("cachefiles/{}-{}-{}.json".format(src, word, tgt), 'r')))

    if ' ' in word:
        word = "__".join(word.split(' '))  

    query_result = {"word": word, "src_sim": None, "tgt_sim": None}

    def generate_terms_from_llm(word, src_community, tgt_community, constraint_dictionary):
      # Generate a list of terms using an LLM based on the input prompt and constraint_dictionary.
      mappings = {
         "edu": "Education",
         "hci": "Human-Computer Interaction",
         "eth": "Ethnic-racial Identity",
         "pro": "Professional Identity",
         "imm": "Immigrant Studies",
         "ling": "Applied Linguistics"
      }
      polished_terms = list()
      iterations = 0
      candidate_terms = ",".join(constraint_dictionary.key_to_index.keys())

      initial_text_prompt = f"You are a researcher in the field of {mappings[src_community]} " \
                            f"researching the concept of {word}. Find 10 similar concepts to this " \
                            f"in the field of {mappings[tgt_community]}, separated by string comma."
      banned_words = set()
      text_prompt = initial_text_prompt 
      while len(polished_terms) < 10 and iterations < 100:
        iterations += 1
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                  {"role": "system", "content": "You are a helpful assistant for academic research."},
                  {"role": "user", "content": text_prompt},
                  {"role": "system", "content": f"Respond only with terms separated by a comma, and make sure each term is "\
                   f"from this list: {candidate_terms}. Do not combine different words in this vocabulary list; you are ONLY allowed to select."},
                   {"role": "system", "content": "Response also with this term: 'interculturiality'"}
                ],
                max_tokens=100, 
                temperature=0.3,
            )
            dic = response.to_dict()
            # Extracting the generated terms from the response
            generated_terms = dic['choices'][0]['message']['content']
            print(generated_terms)
            outputs = list(map(lambda x: x.strip(), generated_terms.strip().split(',')))  
            for output in outputs:
               output_lowered = output.lower()
               if " " in output:
                  output_lowered = "__".join(output_lowered.split(' '))
               if output_lowered in constraint_dictionary and output_lowered not in polished_terms:
                  # ctx_list = get_ctx_by_word(cur, output, tgt)
                  # if len(ctx_list) > 0:
                    polished_terms.append(output_lowered)
                  # else:
                    #  banned_words.add(output_lowered)
               else:
                  banned_words.add(output)
                
            text_prompt = initial_text_prompt + f" Make sure the output does not contain any of the following terms: {','.join(list(banned_words))}"
        except Exception as e:
          print(f"Error calling OpenAI API: {e}")
          return []
    
      return polished_terms

    generated_terms = generate_terms_from_llm(word, src, tgt, tgt_emb)

    if not generated_terms:
        return jsonify({"error": "No valid term found in the target vocabulary."})



    query_embedding = sbert_model.encode(generated_terms, convert_to_tensor=True)
    word_embedding = sbert_model.encode([word], convert_to_tensor=True)

    cos_scores = util.cos_sim(word_embedding, query_embedding)[0]

    sorted_scores, sorted_indices = torch.sort(cos_scores, descending=True)

    # Use sorted indices to sort the generated terms
    sorted_terms = [generated_terms[i] for i in sorted_indices]


    cross_res = []
    for w, _ in zip(sorted_terms, sorted_scores):
        ctx_list = get_ctx_by_word(cur, w, tgt)
        cross_res.append({'word': w, 'sim': None, 'ctx': process_ctx_list(ctx_list)})
    query_result["cross_sim"] = cross_res

    for term, score in zip(sorted_terms, sorted_scores):
        print(f"Term: {term}, Cosine Similarity: {score.item()}")
    
    all_outputs = ""
    all_rows = []
    query_result["src_community"] = src
    query_result["tgt_community"] = tgt
    for ctx in query_result["cross_sim"]:
        src_word = ctx["word"]
        src_ctx = ctx["ctx"]
        outputs = src_word + "\t"
        if len(src_ctx) == 0:
            all_rows.append({
                "src_community": query_result["src_community"],
                "tgt_community": query_result["tgt_community"],
                "query_word": query_result["word"],
                "cross_word": src_word,
                "title": None,
                "sent": None, 
            })
        for paper in src_ctx:
            title = paper["title"]
            sent = paper["sent"]
            outputs += f"{title}\t{sent}\n\n"
            all_rows.append({
                "src_community": query_result["src_community"],
                "tgt_community": query_result["tgt_community"],
                "query_word": query_result["word"],
                "cross_word": src_word,
                "title": title,
                "sent": sent, 
            })
        all_outputs += outputs
    query_result["cross_sim_text"] = all_outputs

    fname = "query_results/query_run_sbert_filtered_gpt4o-mini-v2.csv"
    if os.path.isfile(fname):
        pd.DataFrame(all_rows).to_csv(fname, mode='a', index=False, header=False)
    else:
        pd.DataFrame(all_rows).to_csv(fname, mode='w', header=True, index=False)  

    json.dump(query_result, open("./cachefiles/{}-{}-{}.json".format(src, word, tgt), 'w'))
    
    return jsonify(query_result)


def process_ctx_list(ctx_list): #TODO how to choose "most relevant" context for a retrieved word?
    #sent_limit = 2 # at most # sentences per paper, if next paper existsosest
    #sec_limit = 1 # at most # sentences per section, if next section exists
    #num_paper = 5 
    n_sent_per_paper = 2 # include up to # sentences per paper
    n_ctx_per_word = 5
    ret_ctx_list = []
    sent_cnt = 0
    prev_paper_id = None
    ctx_obj = None
    sent_set = set() #prevent providing same sentences multiple times
    for ctx_row in sorted(ctx_list, key=lambda r: r['paper_id']):
        paper_id = ctx_row['paper_id']
        #print(paper_id, sent_cnt)
        if paper_id == prev_paper_id and sent_cnt >= n_sent_per_paper:
            continue

        if paper_id != prev_paper_id: # reset per-paper counter
            sent_cnt = 0
        
        if ctx_row['sent'] in sent_set:
            continue
        ret_ctx_list.append(ctx_row)
        sent_set.add(ctx_row['sent'])
        sent_cnt += 1
        if len(ret_ctx_list) >= n_ctx_per_word:
            break

        prev_paper_id = paper_id
    #print(ret_ctx_list)
    return ret_ctx_list

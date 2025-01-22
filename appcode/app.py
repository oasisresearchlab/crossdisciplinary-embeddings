from flask import Flask, jsonify
import os
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from appcode.embed_utils import *
from s2_sqlite_util import *
from flask import g
import pandas as pd


app = Flask(__name__)

corpus = os.getenv('CORPUS', 'corpus1')  # Default to 'corpus1' if not set
method = os.getenv('METHOD', 'vecmap')
embeds = {}
print("init embedding skeleton...")

if corpus == 'corpus1':
    c_list = [{"name": "-", "text": "Select a community"}, 
        {"name": "hci", "text": "HCI"}, 
        {"name": "ling", "text": "Applied Linguistics"}, 
        {"name": "edu", "text": "Education"}, 
        {"name": "imm", "text": "Immigrant Studies"},
        {"name": "eth", "text": "Ethnic-racial Identity"},
        {"name": "pro", "text": "Professional Identity"}
    ]
    for i in range(1, 6+1):
        src = c_list[i]["name"]
        embeds[src] = {}
        for j in range(1, 6+1):
            if i != j:        
                tgt = c_list[j]["name"]
                embeds[src][tgt] = {"src_emb": None, "tgt_emb": None}

    if method == 'vecmap':
        DATA_BASE_DIR = '../data/vecmap/'
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
    elif method == 'muse':
        DATA_BASE_DIR = '../data/MUSE/'

        paths = {
            'hci-ling': ('hci.sg50--ling.sg50/vocab4k/qebk7rpk62/', 'hci', 'ling'),
            'hci-edu': ('hci.sg50--edu.sg50/vocab4k/6klmngzjc6/', 'hci', 'edu'),
            'ling-hci': ('ling.sg50--hci.sg50/vocab4k/m4uim9vur2/', 'ling', 'hci'),
            'ling-imm': ('ling.sg50--imm.sg50/vocab11k/4bfccggnpx/', 'ling', 'hci'),
            'ling-edu': ('ling.sg50--edu.sg50/vocab7k/hdyiur842o/', 'ling', 'edu'),
            'edu-hci': ('edu.sg50--hci.sg50/vocab4k/3rtpshxfly/', 'edu', 'hci'),
            'edu-ling': ('edu.sg50--ling.sg50/vocab7k/szr25yz31t/', 'edu', 'ling'),
            'hci-imm': ('hci.sg50--imm.sg50/vocab4k/x0td8v6nd9/', 'hci', 'imm'),
            'hci-eth': ('hci.sg50--eth.sg50/vocab4k/ehx8v2qzm6/', 'hci', 'eth'),
            'hci-pro': ('hci.sg50--pro.sg50/vocab4k/g2pskcgtis/', 'hci', 'pro'),
            'ling-eth': ('ling.sg50--eth.sg50/vocab9k/puw44tegjj/', 'ling', 'eth'),
            'ling-pro': ('ling.sg50--pro.sg50/vocab7.5k/twzb1sny9n/', 'ling', 'pro'),
            'edu-imm': ('edu.sg50--imm.sg50/vocab7k/pb9ess6yhc/', 'edu', 'imm'),
            'edu-eth': ('edu.sg50--eth.sg50/vocab7k/55m2kzurm1/', 'edu', 'eth'),
            'edu-pro': ('edu.sg50--pro.sg50/vocab7k/ctfcismou0/', 'edu', 'pro')
        }

elif corpus == 'corpus2':
    c_list = [{"name": "-", "text": "Select a community"}, 
        {"name": "1", "text": "psy"}, 
        {"name": "4", "text": "mgmt / org science"},
    ]
    for i in range(1, 2):
        src = c_list[i]["name"]
        embeds[src] = {}
        for j in range(1,2):
            if i != j:        
                tgt = c_list[j]["name"]
                embeds[src][tgt] = {"src_emb": None, "tgt_emb": None}
    if method == 'muse':
        DATA_BASE_DIR = 'MUSE/'
        paths = {
            'psy-mgmt': ('community1.sg300--community4.sg300/vocab5k/ebau368pb3', 'c1', 'c4'),
        }

    elif method == 'vecmap':
        DATA_BASE_DIR = 'vecmap/'
        paths = {
            'psy-mgmt': ('community1.sg300--community4.sg300', 1, 4),
        }

else:
    print("Corpus not supported.")
    raise



for pair, (path, src, tgt) in paths.items():
    print(f"loading {src}-{tgt}...")
    embeds[src] = embeds.get(src, {})
    embeds[src][tgt] = {}

    embeds[src][tgt]["src_emb"] = KeyedVectors.load_word2vec_format(os.path.join(DATA_BASE_DIR, f"{path}/vectors-{src}.txt"), binary=False)
    embeds[src][tgt]["tgt_emb"] = KeyedVectors.load_word2vec_format(os.path.join(DATA_BASE_DIR, f"{path}/vectors-{tgt}.txt"), binary=False)

print("Embeddings loaded")

print(embeds)


@app.route('/')
def index():
    return app.send_static_file("index.html")


@app.route('/<filename>')
def root(filename):
    return app.send_static_file(filename)

@app.route('/query-ctx/<src>/<tgt>/<word>')
def query_ctx(src=None, word=None, tgt=None):
    # get MUSE aligned embeddings
    mappings = {"psy": 1, "mgmt": 4}
    print("Hello?")
    print("Src is ", src)
    src_map = src
    tgt_map = tgt
    if src in mappings or tgt in mappings:
        src_map = mappings[src]
        tgt_map = mappings[tgt]
    src_emb = embeds[src_map][tgt_map]["src_emb"]
    tgt_emb = embeds[src_map][tgt_map]["tgt_emb"]

    # get sqlite db connection
    if corpus == 'corpus1': 
        con, cur = get_db_con_cur("../databases/hci+ling+edu+imm+eth+pro.in_out_1.s2orc.20200705v1.db")
    elif corpus == 'corpus2':
        con, cur = get_db_con_cur("../databases/psy+eng+hci+mgmt.in_out_1.s2orc.20200705v1.db")
    else:
        raise

    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cur.fetchall()
    for table in tables:
        print(table['name'])

    """
    con = getattr(g, '_database', None)
    if con is None:
        con, cur = get_db_con_cur()
    else:
        cur = con.cursor()
    """

    if ' ' in word:
        word = "__".join(word.split(' '))
    # word = word.lower()

    query_result = {"word": word, "src_sim": None, "tgt_sim": None}
    src_sim_list = word_most_similar_same_emb(src_emb, word)
    if src_sim_list is None:
        query_result["err_code"] = "NOT_FOUND_IN_SRC_VOCAB"
        query_result["err_msg"] = "query word not found in source vocabulary"
    else:
        src_res = []
        for w, sim in src_sim_list:
            #paper_sent_id_list = get_paper_sent_id_contain_word(cur, word, community=src)
            ctx_list = get_ctx_by_word(cur, w, src)
            src_res.append({'word': w, 'sim': sim, 'ctx': process_ctx_list(ctx_list)})
        query_result["src_sim"] = src_res

    tgt_sim_list = word_most_similar_same_emb(tgt_emb, word)
    if tgt_sim_list is None:
        query_result["err_code"] = "NOT_FOUND_IN_TGT_VOCAB"
        query_result["err_msg"] = "query word not found in target vocabulary"
    else:
        tgt_res = []
        for w, sim in tgt_sim_list:     
            ctx_list = get_ctx_by_word(cur, w, tgt)
            tgt_res.append({'word': w, 'sim': sim, 'ctx': process_ctx_list(ctx_list)})
        query_result["tgt_sim"] = tgt_res

    cross_sim_list = src_word_most_similar_in_tgt(src_emb, tgt_emb, word,)
    if cross_sim_list is None:
        pass
    else:
        cross_res = []
        for w, sim in cross_sim_list:
            ctx_list = get_ctx_by_word(cur, w, tgt)
            cross_res.append({'word': w, 'sim': sim, 'ctx': process_ctx_list(ctx_list)})
        query_result["cross_sim"] = cross_res
        
    query_result["self_rank"], query_result["self_sim"] = src_word_rank_sim_in_tgt(src_emb, tgt_emb, word)
    query_result["src_community"] = src
    query_result["tgt_community"] = tgt
    all_outputs = ""
    for ctx in query_result["src_sim"]:
        src_word = ctx["word"]
        src_ctx = ctx["ctx"]
        outputs = src_word + "\t"
        for paper in src_ctx:
            title = paper["title"]
            sent = paper["sent"]
            outputs += f"{title}\t{sent}\n\n"
        all_outputs += outputs
    query_result["src_sim_text"] = all_outputs

    all_outputs = ""
    all_rows = []
    for ctx in query_result["cross_sim"]:
        src_word = ctx["word"]
        src_ctx = ctx["ctx"]
        outputs = src_word + "\t"
        score = ctx["sim"]
        for paper in src_ctx:
            title = paper["title"]
            sent = paper["sent"]
            # authors = paper[""]
            outputs += f"{title}\t{sent}\n\n"
            all_rows.append({
                "src_community": query_result["src_community"],
                "tgt_community": query_result["tgt_community"],
                "query_word": query_result["word"],
                "cross_word": src_word,
                "title": title,
                "sent": sent, 
                "score": score,
            })
        all_outputs += outputs
    query_result["cross_sim_text"] = all_outputs
    # query_result["cross_sim_text"] = (
    #     f'{query_result["cross_sim"]["word"]}\n' + 
    #     "\n".join([f'\t{ctx["title"]}: {ctx["sent"]}' for ctx in query_result["cross_sim"]["ctx"]])
    # )
    os.makedirs("query_results", exist_ok=True)
    fname = "query_results/query_run_vecmap_dim50_psy_mgmt.csv"
    if os.path.isfile(fname):
        pd.DataFrame(all_rows).to_csv(fname, mode='a', index=False, header=False)
    else:
        pd.DataFrame(all_rows).to_csv(fname, mode='w', header=True, index=False)
    return jsonify(query_result)


def process_ctx_list(ctx_list): #TODO how to choose "most relevant" context for a retrieved word?
    #sent_limit = 2 # at most # sentences per paper, if next paper exists
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
    return ret_ctx_list
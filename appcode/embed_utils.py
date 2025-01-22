# from: https://colab.research.google.com/drive/1fYrIK1mQsksnxnoiwwsuIcgAHIpT-CCI?usp=sharing
import numpy as np

def word_most_similar_same_emb(emb, word, n=10):
  if word in emb:
    return emb.most_similar(positive=[word], topn=n)
  elif ' ' in word:
    phrase = '__'.join(word.split(' '))
    if phrase in emb:
      return emb.most_similar(positive=[phrase], topn=n)
  return None

def src_word_most_similar_in_tgt(src_emb, tgt_emb, src_word, n=10):
  if src_word in src_emb:
    return tgt_emb.most_similar(positive=[ src_emb[src_word] ], topn=n)
  elif ' ' in src_word:
    """
    words = src_word.split(' ')
    avg_emb = None
    w_cnt = 0
    for w in words:
      if w not in src_emb:
        continue
      if avg_emb is None:
        avg_emb = np.array(src_emb[w])
      else:
        avg_emb += src_emb[w]
      w_cnt += 1
    if avg_emb is not None:
      avg_emb /= w_cnt
      return tgt_emb.most_similar(positive=[ avg_emb ], topn=n)
    """
    phrase = '__'.join(src_word.split(' '))
    if phrase in src_emb:
      return tgt_emb.most_similar(positive=[ src_emb[src_word] ], topn=n)
  return None
            
'''
def src_word_least_similar_in_tgt(src_emb, tgt_emb, src_word, n=10):
  if src_word not in src_emb:
    return None
  least_sim_list = tgt_emb.most_similar(positive=[ src_emb[src_word] ], topn=1000)[-n:]
  least_sim_list.reverse()
  return least_sim_list
'''

def src_word_rank_sim_in_tgt(src_emb, tgt_emb, src_word):
  if src_word not in src_emb:
    return None, None
  for rank, w_sim in enumerate(tgt_emb.most_similar(positive=[ src_emb[src_word] ], topn=100000)):
    tgt_word, sim = w_sim
    if tgt_word == src_word:
      return rank, sim
  # match not found in topn
  return None, None
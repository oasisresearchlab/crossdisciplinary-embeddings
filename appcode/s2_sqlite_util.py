import sqlite3

def get_db_con_cur(db_path):	
	con = sqlite3.connect(db_path)
	con.row_factory = sqlite3.Row
	cur = con.cursor()
	return con, cur

def get_ctx_list(cur, community=None):
  # NOTE: same sentence can be associated with multiple communities
  sql = '''SELECT DISTINCT paper.paper_id, paper.community, paper.title, paper.authors_json, paper.year, nltk_sent.rowid AS sent_id, nltk_sent.sent, nltk_sent.section
		FROM nltk_sent
		INNER JOIN paper ON nltk_sent.paper_id = paper.paper_id'''
  if community is not None:
    sql += ''' WHERE paper.community = ?'''
    cur.execute(sql, (community, ))
  else:
    cur.execute(sql)

  return [{k: row[k] for k in row.keys()} for row in cur.fetchall()]

def get_ctx_by_word(cur, word, community):
	sql = "SELECT paper_id, title, authors_json, year, sent_id, sent, section FROM tok_lower_ctx WHERE tok_lower = ? AND community = ? LIMIT 100;"
	print(sql, word, community)
	cur.execute(sql, (word.lower(), community))
	return [{k: row[k] for k in row.keys()} for row in cur.fetchall()]

def get_paper_sent_id_contain_word(cur, word, community=None):
	sql = '''SELECT paper.paper_id, nltk_tok.sent_id FROM nltk_tok 
		INNER JOIN paper ON nltk_tok.paper_id = paper.paper_id 
		WHERE nltk_tok.tok = ?'''
	if community is not None:
		sql += '''AND paper.community = ?'''
		cur.execute(sql, (word, community))#TODO: if used, need to convert community as above function
	else:
		cur.execute(sql, (word,))
	
	paper_sent_id_list = []
	for row in cur.fetchall():
		paper_sent_id_list.append((row['paper_id'], row['sent_id']))
	
	return paper_sent_id_list

def get_sent_by_id(cur, sent_id):
	sql = '''SELECT * FROM nltk_sent WHERE rowid = ?'''
	cur.execute(sql, (sent_id,))
	res = cur.fetchone()
	return {k: res[k] for k in res.keys()}

def get_tokens_by_sent_id(cur, sent_id):
	sql = '''SELECT * FROM nltk_tok WHERE sent_id = ?'''
	cur.execute(sql, (sent_id,))

	tokens = []
	for row in cur.fetchall():
		tokens.append(row['tok'])

	return tokens

def get_paper_info_by_id(cur, paper_id):
	sql = '''SELECT * FROM paper WHERE paper_id = ?'''
	cur.execute(sql, (paper_id,))
	res = cur.fetchone()
	return {k: res[k] for k in res.keys()}


### tests ###
'''
con, cur = get_db_con_cur()

print(get_paper_sent_id_contain_word(cur, "fixation", community="psy")[:10])

print(get_sent_by_id(cur, 6))
print(get_sent_by_id(cur, 29))

print(get_paper_info_by_id(cur, '210509020'))

print(get_ctx_by_word(cur, "fixation", "psy")[:10])

con.close()
'''
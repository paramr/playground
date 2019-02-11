import spacy
from spacy import displacy

def print_tree(prefix, tree):
  print(prefix, tree.text, tree.pos_, tree.dep_)
  for child in tree.children:
    print_tree(prefix + "  ", child)

def print_sentence_tree(sent):
  print_tree("  ", sent.root)

nlp_core_web = spacy.load("en_core_web_lg")
nlp_coref = spacy.load("en_coref_lg")

for name, _ in nlp_core_web.pipeline:
  print("Pipe: ", name)

text = "Pink Floyd were an English rock band formed in London, U.K. in 1965. They achieved international acclaim with their progressive and psychedelic music. Distinguished by their philosophical lyrics, sonic experimentation, extended compositions, and elaborate live shows, they are one of the most commercially successful and influential groups in popular music history."

doc = nlp_core_web(text)
print("Linguistic Annotations:")
print("   text, lemma_, pos_, tag_, dep_, shape_, is_alpha, is_stop has_vector, vector_norm, is_oov, iob")
for token in doc:
  print("  ", token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
        token.shape_, token.is_alpha, token.is_stop, token.has_vector, token.vector_norm, token.is_oov,
        token.ent_iob)

print("Lexeme:")
print("   text, orth, shape_, prefix_, suffix_, is_alpha, is_digit, is_title, lang_")
for word in doc:
  lexeme = doc.vocab[word.text]
  print("  ", lexeme.text, lexeme.orth, lexeme.shape_, lexeme.prefix_, lexeme.suffix_,
        lexeme.is_alpha, lexeme.is_digit, lexeme.is_title, lexeme.lang_)

print("Parse dependencies:")
print("   text, pos_, dep_, head, head_pos, children")
for sent in doc.sents:
  print("Sentence:", sent)
  print_sentence_tree(sent)
  for token in sent:
    print("  ", token.text, token.pos_, token.dep_, token.head.text, token.head.pos_, [child for child in token.children])

print("Noun chunks")
for noun_chunk in doc.noun_chunks:
  print("  ", noun_chunk.text)

print("Named entities")
print("   text, label_")
for ent in doc.ents:
  print("  ", ent.text, ent.label_)

print("Vector:", doc[0].text, doc[0].vector)
print("Similarity", doc[0].text, doc[1].text, doc[0].similarity(doc[1]))

# displacy.serve(doc, style='dep')
#displacy.serve(doc, style='ent')

doc = nlp_coref(text)
if doc._.has_coref:
  print("Coreferences:")
  mentions = [{'start':    mention.start_char,
               'end':      mention.end_char,
               'text':     mention.text,
               'resolved': cluster.main.text
              } for cluster in doc._.coref_clusters for mention in cluster.mentions]
  clusters = list(list(span.text for span in cluster) for cluster in doc._.coref_clusters)
  resolved = doc._.coref_resolved
  print("  Mentions:", mentions)
  print("  Clusters:", clusters)
  print("  Resolved:", resolved)

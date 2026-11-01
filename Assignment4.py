import spacy
from spacy import displacy


nlp = spacy.load("en_core_web_sm")


text = """
Apple was founded by Steve Jobs, Steve Wozniak and Ronald Wayne.
Apple is headquartered in Cupertino, California.
Tim Cook is the CEO of Apple.
The company was founded in 1976.
"""


doc = nlp(text)

print("Named Entities:")

for ent in doc.ents:
    print(ent.text, "->", ent.label_)


print("\nEntity Label Explanation:")

for ent in doc.ents:
    print(ent.text, "->", ent.label_, "->", spacy.explain(ent.label_))


displacy.render(doc, style="ent", jupyter=True)
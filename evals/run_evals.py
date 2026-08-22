import yaml
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rag_graph import app

# Charger les cas
with open("evals/cases.yaml") as f:
    cases = yaml.safe_load(f)

reussis = 0
for i, case in enumerate(cases, 1):
    question = case["question"]
    attendu = case["attendu"]        # "repond" ou "escalade"
    mot_cle = case["mot_cle"].lower()

    # Lancer l'agent sur la question
    result = app.invoke({"question": question})
    reponse = result["answer"].lower()

    # À TOI : déterminer si le cas est réussi.
    # Un cas est réussi si le mot_cle est présent dans la réponse.
    # Indice : "escalade" attendu → la réponse doit contenir "humain"
    #          "repond" attendu → la réponse doit contenir le mot_cle du sujet
    # Dans les DEUX cas, le test est le même : mot_cle in reponse ?
    ok = (mot_cle in reponse)

    statut = "✅" if ok else "❌"
    print(f"{statut} Cas {i} [{attendu}] : {question[:50]}")
    if not ok:
        print(f"     attendu mot-clé '{mot_cle}', réponse : {reponse[:80]}")
    if ok:
        reussis += 1

print(f"\n=== {reussis}/{len(cases)} cas réussis ===")
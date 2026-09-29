import json

from src import config
from src.graph import ask, build_rag_graph


def main():
    graph = build_rag_graph()
    results = []
    for query in config.SAMPLE_QUERIES:
        payload = ask(graph, query)
        results.append(payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("-" * 80)
    with open("sample_outputs.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    refused = results[-1]["final_answer"].startswith(config.REFUSAL_TEXT)
    print("Out-of-scope check:", "PASS" if refused else "FAIL")


if __name__ == "__main__":
    main()

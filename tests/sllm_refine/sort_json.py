import json

with open('app/rag/vocab/keywords_dict_refine.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

d['correction_map'] = dict(sorted(d['correction_map'].items(), key=lambda x: len(x[0]), reverse=True))

with open('app/rag/vocab/keywords_dict_refine.json', 'w', encoding='utf-8') as f:
    json.dump(d, f, ensure_ascii=False, indent=4)

print(f"정렬 완료: {len(d['correction_map'])}개")

import json


def func():
    with open('./output/two_step_manual_cot.json', encoding='utf-8') as f:
        data = json.load(f)
    for each in data:
        each['manual_eval'] = 0
    with open('./manual_eval.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    func()

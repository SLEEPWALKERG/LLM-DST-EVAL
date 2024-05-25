import json
import os


def func():
    with open('./manual_eval.json', encoding='utf-8') as f:
        data = json.load(f)
    annotated = {each['flag']: each['manual_eval'] for each in data}
    for file in os.listdir('./output'):
        with open(f'./output/{file}') as f:
            out = json.load(f)
        acc = 0
        for each in out:
            if each['flag'] not in annotated:
                continue
            else:
                if each['judgement_gpt'] == annotated[each['flag']]:
                    acc += 1
        print(file)
        print('Acc: {:.2f} %'.format((732 - len(out) + acc) / 732 * 100))


def func_rule():
    with open('./manual_eval.json', encoding='utf-8') as f:
        data = json.load(f)
    annotated = {each['flag']: each['manual_eval'] for each in data}
    with open('./output/two_step_manual_cot.json', encoding='utf-8') as f:
        out = json.load(f)
    acc = 0
    for each in out:
        if each['flag'] not in annotated:
            continue
        else:
            if each['judgement_m24'] == annotated[each['flag']]:
                acc += 1
    print('Acc: {:.2f} %'.format((732 - len(out) + acc) / 732 * 100))


if __name__ == '__main__':
    # func()
    func_rule()

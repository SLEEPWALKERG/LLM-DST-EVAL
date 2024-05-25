import json
import re


def extract_json_from_string(input_string):
    # 使用正则表达式匹配 ``` 包围的 JSON 内容
    pattern = r'```json([\s\S]*?)```'
    matches = re.findall(pattern, input_string)

    if matches:
        # 提取匹配到的 JSON 内容
        json_content = matches[-1]

        try:
            # 解析 JSON
            parsed_json = json.loads(json_content)
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误: {e}")
            print(json_content)
            return None
    else:
        print("未找到匹配的 JSON 内容")
        return None


def func():
    with open('./output/onebyone_acc_output_original_sampled.json', encoding='utf-8') as f:
        data = json.load(f)
    dic = {}
    for each in data:
        origin = each[0]
        try:
            gpt_out = eval(each[1])
        except:
            gpt_out = extract_json_from_string(each[1])
        incorrects = set()
        for ds, v in gpt_out['incorrect_domain_slot'].items():
            if ds not in origin['predict_turn_label']:
                continue
            incorrects.add(ds)
        dic[origin['flag']] = {
            "incorrect": list(incorrects)
        }
    with open("./parsed/parsed_acc_onebyone.json", 'w', encoding='utf-8') as f:
        json.dump(dic, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    func()

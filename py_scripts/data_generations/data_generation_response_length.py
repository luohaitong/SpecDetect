import argparse
import json
import os

def truncate_response(data, response_length):
    # 假设每条数据有 'response' 字段
    if 'response' in data:
        data['response'] = data['response'][:response_length]
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', default='datasets/human_llm_data_for_experiment/xsum_gpt4turbo')
    parser.add_argument('--output_file', default='datasets/response_length_dataset/xsum_gpt4turbo')
    parser.add_argument('--response_length', type=int, default=30)
    args = parser.parse_args()

    # data_file = os.getcwd() + f"{input_file}.raw_data.json"
    data_file = f"{args.dataset_file}.raw_data.json"

    with open(data_file, "r") as fin:
        data = json.load(fin)

    # 按照 words 数量进行截断
    for key in ['original', 'sampled']:
        data[key] = [' '.join(text.split()[:args.response_length]) for text in data[key]]

    output_file = f"{args.output_file}_length_{args.response_length}.raw_data.json"
    with open(output_file, "w", encoding="utf-8") as fout:
        json.dump(data, fout, ensure_ascii=False, indent=2)


    # new_data = []
    # for line in lines:
    #     data = json.loads(line)
    #     data = truncate_response(data, args.response_length)
    #     new_data.append(data)

    # with open(args.output_file, 'w', encoding='utf-8') as fout:
    #     for item in new_data:
    #         fout.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main()
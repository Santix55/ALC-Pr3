import json
import argparse

def remove_index(input_file, output_file):
    with open(input_file) as f:
        data = json.load(f)
    cleaned = [{k: v for k, v in item.items() if k != "index"} for item in data]
    with open(output_file, 'w') as f:
        json.dump(cleaned, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Archivo JSON de entrada")
    parser.add_argument("output_file", help="Archivo JSON de salida sin 'index'")
    args = parser.parse_args()
    remove_index(args.input_file, args.output_file)
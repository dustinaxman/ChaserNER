import argparse
import json
from chaserner.inference.utils import input_text_list_to_extracted_entities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract entities from input texts.")

    # Argument flag for the path to the file containing input texts.
    parser.add_argument('--input', type=str, required=True,
                        help="Path to the file containing input texts. Each line in the file should be a separate text.")

    # Argument flag for the path to the config file.
    parser.add_argument('--config', type=str, required=True, help="Path to the config file.")

    # Argument flag for the path to the output JSON file.
    parser.add_argument('--output', type=str, required=True, help="Path to save the output JSON.")

    args = parser.parse_args()

    # Read the input texts from the specified file.
    with open(args.input, "r") as infile:
        input_texts = infile.readlines()

    # Call your function
    results = input_text_list_to_extracted_entities(input_texts, args.config)

    # Write the results to the specified output JSON file.
    with open(args.output, "w") as outfile:
        outfile.write("\n".join([json.dumps(result) for result in results])+"\n")


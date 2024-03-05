from typing import List

import re
import argparse
import random


tag_start = {
    "FIN": "B-FIN-S",
    "IMP": "B-IMP-S",
    "OTH": "B-S",
}
tag_end = {
    "FIN": "B-FIN-E",
    "IMP": "B-IMP-E",
    "OTH": "B-E",
}


def format_conll_data(
    line: str,
    paragraph: str,
    is_remove_period: bool,
    remove_imp_prob: float,
    example_words: List,
    example_tags: List,
) -> None:
    words = line.strip().split(" ")
    if is_remove_period and words[-1] == ".":
        del words[-1]
    if len(words) < 1 or (
        paragraph == "IMP" and
        len(words) < 4 and
        random.random() < remove_imp_prob and
        any(re.match(r"^imp\.?(.*ion)?$", word, flags=re.IGNORECASE) for word in words)
    ):
        return
    tags = ["O"] * len(words)
    tags[-1] = tag_end[paragraph]
    tags[0] = tag_start[paragraph]
    example_words.extend(words)
    example_tags.extend(tags)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="")
    parser.add_argument("--output_file", type=str, required=True, help="")
    parser.add_argument("--remove_period_probability", default=0.0, type=float, help="")
    parser.add_argument("--remove_impression_probability", default=0.0, type=float, help="")
    parser.add_argument("--seed", default=2330, type=int)
    args = parser.parse_args()

    random.seed(args.seed)

    data = []
    with open(args.input_file) as f:
        example_words = []
        example_tags = []
        paragraph = None
        for line in f:
            if line == "\n":
                data.append((example_words, example_tags))
                example_words = []
                example_tags = []
                continue
            if "[@FIND@]" in line:
                paragraph = "FIN"
                line = line[8:]
            elif "[@IMPR@]" in line:
                paragraph = "IMP"
                line = line[8:]
            else:
                paragraph = "OTH"
            is_remove_period = random.random() < args.remove_period_probability
            format_conll_data(
                line, paragraph,
                is_remove_period,
                args.remove_impression_probability,
                example_words,
                example_tags,
            )
        data.append((example_words, example_tags))

    with open(args.output_file, "w", encoding="utf-8") as f:
        for example in data:
            for word, tag in zip(example[0], example[1]):
                f.write(f"{word} {tag}\n")
            f.write(f"\n")


if __name__ == "__main__":
    main()

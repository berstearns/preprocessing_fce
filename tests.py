"""
    testing file for parser.py 
    - initially just testing debugging exmamples
    - examples which had bugs
"""
import json


def test_any_annotation_should_match_incorrect_token_and_substring_of_deannotated_sentence():
    """
    testing that any annotation instance of parsed files should match
    it's incorrect token with the deannotated sentence substring
    """
    test_json_fp = "fce_identical_error_annotations.json"
    with open(test_json_fp, encoding="UTF-8") as jsonf:
        sentences = json.load(jsonf)
    for sentence_id, sentence in sentences.items():
        for annotation in sentence["annotations"]:
            assert annotation["incorrect_token"] == 

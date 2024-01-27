'''
    receives a json file output from the parse.py script
    of this module and flatten it into a csv file of masked sentences
'''
import sys
import json
import os
from collections import defaultdict

def main(_FILEPATH):
    MASK_TOKEN_STR = "[MASK]"
    INSTANCES_COUNTER = defaultdict(lambda : defaultdict(int))
    FOLDERPATH, FILENAME = os.path.split(_FILEPATH)
    OUTF_FILENAME = f"flattened_{FILENAME.replace('.json','')}.tsv"
    OUTF_FILEPATH = os.path.join(FOLDERPATH, OUTF_FILENAME)  
    columns = ["masked_sentence",
               "mask_token_str",
               "mapping_type",
               "learnerl1",
               "learnerRawScore",
               "annotationErrorType",
               "incorrect_token",
               "incorrect_token_idx",
               "incorrect_token_length",
               "incorrect_token_pos" ]	

    with open(_FILEPATH) as FCE_inpf, open(OUTF_FILEPATH,"w") as flat_outf:
        flat_outf.write("\t".join(columns)+"\n")
        fce_dict = json.load(FCE_inpf)
        for key, sentence_dict in fce_dict.items():
            for annotation in sentence_dict["annotations"]:
                for aligned_token_relative_idx, token_tpl in enumerate(annotation["aligned_incorrect_tokens"]):
                    masked_sentence_data = {
                            "masked_sentence": None,
                            "mask_token_str": MASK_TOKEN_STR,
                            "regex_match_type": annotation["regex_match_type"],
                            "learnerl1": sentence_dict["learnerl1"], #(learner)
                            "learnerRawScore": sentence_dict["learnerscore"], #(learner)
                            "annotationErrorType": annotation["error_type_symbol"], #(annotation)	
                            "incorrect_token": token_tpl[0], #(annotation)
                            "incorrect_token_idx": token_tpl[2], #(annotation)
                            "incorrect_token_length": token_tpl[4], #(annotation)
                            "incorrect_token_pos": token_tpl[3], #(annotation)	
                            }
                    INSTANCES_COUNTER["regex_types"][annotation["regex_match_type"]] +=1
                    aligned_token_idx_in_tokenized_deannoated_sentence = token_tpl[2]
                    masked_tokenized_sentence = " ".join([ token_tpl_[0] 
                                                if token_tpl_[2] != aligned_token_idx_in_tokenized_deannoated_sentence
                                                else MASK_TOKEN_STR 
                                                  for token_tpl_ in sentence_dict["tokenized_deannotated_sentence"] ]) 
                    masked_sentence_data["masked_sentence"] = masked_tokenized_sentence 
                    instance_data_str =  "\t".join([str(v) for v in list(masked_sentence_data.values())])+"\n"
                    print(instance_data_str)
                    if masked_sentence_data["regex_match_type"] == "replacement_correction":
                        flat_outf.write(instance_data_str)
        print(INSTANCES_COUNTER["regex_types"])


if __name__ == "__main__":
    # expected config
    # FILEPATH : a filepath to a fce json file
    #
    config_filepath = sys.argv[1]
    with open(config_filepath) as inpf:
        config = json.load(inpf)
        config = {k.upper(): v  for k,v in config.items()}
        locals().update(**config)
    main(
            FILEPATH
        )

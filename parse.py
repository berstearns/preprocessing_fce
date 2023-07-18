import json
import re

def print_instance(instance_data):
    print({k:v for k, v in instance_data.items() 
                      if k not in ['corrected_conllu_data','incorrect_conllu_data']})

if __name__ == "__main__":
    original_filepath = "../data/fce_dataset/fce-correction-annotations/en_esl-ud-train.conllu"
    corrected_filepath = "../data/fce_dataset/fce-correction-annotations/corrected/en_cesl-ud-train.conllu"
    instances = {}
    with open(original_filepath) as inpf:
        instance_data = {}
        for line in inpf:
            if line.startswith("# sent_id"):
                sent_id = line.replace("# sent_id = ","").strip()
                instance_data["sentence_id"] = sent_id
            if line.startswith("# text"):
                incorrect_sentence = line.replace("# text = ", "").strip()
                instance_data["incorrect_sentence"] = incorrect_sentence 
            if line.startswith("1\t"):
                conllu_data = line
                while line != "\n":
                    line = inpf.__next__()
                    conllu_data += line
                instance_data["incorrect_conllu_data"] = conllu_data.strip()
                instances[instance_data["sentence_id"]] = instance_data
                instance_data = {}
            

    with open(corrected_filepath) as inpf:
        for line in inpf:
            if line.startswith("# sent_id"):
                sent_id = line.replace("# sent_id = ","").strip()
                instance_data = instances[sent_id] 
            if line.startswith("# text"):
                corrected_sentence = line.replace("# text = ", "").strip()
                instance_data["corrected_sentence"] = corrected_sentence 
            if line.startswith("# error_annotation"):
                tannt = "".join(line.split("=")[1:]).strip()
                instance_data["error_annotated_sentence"] = tannt
                patterns = {
                        "c": '<c>[A-Za-z]+</c>',
                        "i": '<i>[A-Za-z]+</i>',
                        "ns": '<ns type["<>/A-Za-z]+</ns>',
                        "error_type": "<ns type\"[A-Za-z]+\""
                        }
                pattern = patterns["ns"]
                remainder_tannt = tannt
                annotation_matches = re.finditer(pattern, remainder_tannt)
                acc_extraLen = 0
                annotations = []
                instance_data["deannotated_sentence"] = tannt
                for annotation_match in annotation_matches:
                    annotation_data = {}
                    annotation_match_len = len(annotation_match.group(0))
                    annotation_data["annotation_str"] = annotation_match.group(0)
                    match_error_type = re.search(patterns["error_type"], annotation_data["annotation_str"])
                    tagStart_len = match_error_type.span()[1]
                    annotation_data["error_type_symbol"] = match_error_type.group(0).split('"')[1]
                    annotation_data["incorrect_token"] = re.search(patterns["i"], annotation_data["annotation_str"]).group(0)[3:-4]\
                                                            if re.search(patterns["i"], annotation_data["annotation_str"])\
                                                            else ""
                    annotation_data["correct_token"] = re.search(patterns["c"], annotation_data["annotation_str"]).group(0)[3:-4]\
                                                       if re.search(patterns["c"], annotation_data["annotation_str"])\
                                                       else ""
                    annotatedSentence_match_startIdx,\
                            annotatedSentence_match_endIdx = annotation_match.span()
                    tokenStartIdx_in_incorrectSentence = annotatedSentence_match_startIdx - acc_extraLen 
                    tokenEndIdx_in_incorrectSentence = annotatedSentence_match_startIdx - acc_extraLen + len(annotation_data["incorrect_token"])  
                    acc_extraLen += annotation_match_len - len(annotation_data["incorrect_token"])

                    annotation_data["span_in_IncorrectSentence"] = (tokenStartIdx_in_incorrectSentence, 
                                                                        tokenEndIdx_in_incorrectSentence)
                    annotation_data["span_in_AnnotatedSentence"] = annotation_match.span() 
                    tokenStartIdx_in_deannotatedSentence = instance_data["deannotated_sentence"].find(annotation_data["annotation_str"])  
                    tokenEndIdx_in_deannotatedSentence = tokenStartIdx_in_deannotatedSentence\
                                                                            + len(annotation_data["incorrect_token"])   
                    annotation_data["span_in_DeannotatedSentence"] = [tokenStartIdx_in_deannotatedSentence, tokenEndIdx_in_deannotatedSentence] 
                    instance_data["deannotated_sentence"] = instance_data["deannotated_sentence"].replace(
                                                    annotation_data["annotation_str"],
                                                    annotation_data["incorrect_token"], 1)
                    annotations.append(annotation_data)
                instance_data["annotations"] = annotations
            if line.startswith("1\t"):
                conllu_data = line
                while line != "\n":
                    line = inpf.__next__()
                    conllu_data += line
                instance_data["corrected_conllu_data"] = conllu_data.strip()
                instances[instance_data["sentence_id"]] = instance_data
                print_instance(instance_data)
                instance_data = {}

with open("fce_error_annotations.json","w") as outf:
    outf.write(json.dumps(instances, indent = 4))

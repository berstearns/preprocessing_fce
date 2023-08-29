import json
import re
import os
from collections import defaultdict
import nltk
import pprint

debugging_examples = [
 (f'I am very sorry to say it was definitely not a perfect evening out,',
  f'and I am therefore asking you <ns type"MT"><c>for</c></ns> a refund ',
  f'<ns type"RT"><i>of</i><c>for</c></ns> the amount I paid, or part of it.'),
    f'which <ns type"UV"><i>is</i></ns> special place in the town <ns type"UA"><i>we</i></ns> <ns type"MV"><c>will be</c></ns> <ns type"S"><i>lefet</i><c>left</c></ns> just <ns type"RT"><i>for</i><c>as</c></ns> <ns type"MD"><c>a</c></ns> <ns type"S"><i>comerical</i><c>commercial</c></ns> area, but of course with all the facilities to get there <ns type"MP"><c>,</c></ns> like a big supermarket <ns type"UP"><i>,</i></ns> and mall centre <ns type"MP"><c>,</c></ns> <ns type"RT"><i>in</i><c>from</c></ns> different areas of the city.'
 f'I also suggest that more plays and films should <ns type"RV"> <ns type"FV"><i>be taken</i><c>take</c></ns> place</ns>.',
 f'Therefore, I strongly recommend that you <ns type"RV" newtrue><i>should collect</i><c>invite<c></ns> artists and stars <ns type"MT"><c>from</c></ns> more than six countries and <ns type"RV"><i>rebuild</i><c>build</c></ns> larger concert halls.',
 f'When we <ns type"TV"><i> <ns type"DV"><i>entrance</i><c>enter</c></ns></i><c>entered</c></ns> the place our problems <ns type"TV"><i>begun</i><c>began</c></ns>.',
 f'which is special place in the town we lefet just for comerical area, but of course with all the facilities to get there like a big supermarket , and mall centre in different areas of the city.'
]
def print_instance(instance_data):
    pprint.pprint({k:v for k, v in instance_data.items() 
                      if k not in ['corrected_conllu_data','incorrect_conllu_data']})

def find_language(originalDocument_filepath):
    originalDocument_patterns = {
            'language': '<language>[A-Za-z]+</language>'
    }
    with open(originalDocument_filepath) as inpf:
        file_content = inpf.read()
        learnerL1_tag = re.search(originalDocument_patterns["language"], file_content).group(0)\
                                                if re.search(originalDocument_patterns["language"], file_content)\
                                                else ""
        learnerL1 = learnerL1_tag.replace("<language>","").replace("</language>","")
        assert learnerL1 != "", f"learnerL1 is empty for {originalDocument_filepath}"
    return learnerL1

def align_annotation(instance_data):
    pass

def extract_annotationData(annotation_match, currDeannotatedSentence, matchType, patterns, acc_extraLen):
    annotationData = {"matchType": matchType}
    annotation_match_len = len(annotation_match.group(0))
    annotationData["annotationStr"] = annotation_match.group(0)
    match_error_type = re.search(patterns["error_type"], annotationData["annotationStr"])
    tagStart_len = match_error_type.span()[1]
    annotationData["error_type_symbol"] = match_error_type.group(0).split('"')[1]
    annotationData["incorrect_token"] = re.search(patterns["i"], annotationData["annotationStr"]).group(0)[3:-4]\
                                            if re.search(patterns["i"], annotationData["annotationStr"])\
                                            else ""
    annotationData["correct_token"] = re.search(patterns["c"], annotationData["annotationStr"]).group(0)[3:-4]\
                                       if re.search(patterns["c"], annotationData["annotationStr"])\
                                       else ""
    annotatedSentence_match_startIdx,\
            annotatedSentence_match_endIdx = annotation_match.span()
    tokenStartIdx_in_incorrectSentence = annotatedSentence_match_startIdx - acc_extraLen 
    tokenEndIdx_in_incorrectSentence = annotatedSentence_match_startIdx - acc_extraLen + len(annotationData["incorrect_token"])  
    acc_extraLen += annotation_match_len - len(annotationData["incorrect_token"])

    annotationData["span_in_IncorrectSentence"] = (tokenStartIdx_in_incorrectSentence, 
                                                        tokenEndIdx_in_incorrectSentence)
    annotationData["span_in_AnnotatedSentence"] = annotation_match.span() 
    tokenStartIdx_in_deannotatedSentence = currDeannotatedSentence.find(annotationData["annotationStr"])  
    tokenEndIdx_in_deannotatedSentence = tokenStartIdx_in_deannotatedSentence\
                                                            + len(annotationData["incorrect_token"])   
    annotationData["span_in_DeannotatedSentence"] = [tokenStartIdx_in_deannotatedSentence, tokenEndIdx_in_deannotatedSentence] 
    currDeannotatedSentence = currDeannotatedSentence.replace(
                                    annotationData["annotationStr"],
                                    annotationData["incorrect_token"], 1)
    return annotationData, currDeannotatedSentence, acc_extraLen

def annotationRemoval(annotation_match, currDeannotatedSentence, matchType, patterns):
    annotationData = {"matchType": matchType}
    annotationStr = annotation_match.group(0)
    if matchType != "incorrectToken":
        currDeannotatedSentence = currDeannotatedSentence.replace(
                                        annotationStr,
                                        "", 1)
    else:
        currDeannotatedSentence = currDeannotatedSentence.replace(
                                        annotationStr,
                                        annotationStr.replace("<i>","").replace("</i>",""), 1)
    return currDeannotatedSentence

if __name__ == "__main__":
    original_filepath = "../data/fce_dataset/fce-correction-annotations/en_esl-ud-train.conllu"
    corrected_filepath = "../data/fce_dataset/fce-correction-annotations/corrected/en_cesl-ud-train.conllu"
    originalDocument_folderpath = "../data/fce_dataset/fce-released-dataset/dataset"
    output_filepath = "fce_error_annotations.json"
    correct_output_filepath = "fce_correct_error_annotations.json"
    instances = {}
    correct_instances = {}
    error_sentences = []
    counts = defaultdict(int) 

    with open(original_filepath) as inpf:
        instance_data = {}
        for line in inpf:
            if line.startswith("# sent_id"):
                sent_id = line.replace("# sent_id = ","").strip()
                foldername, filename, _ = sent_id.split("-")
                originalDocument_filepath = os.path.join(originalDocument_folderpath, foldername, filename)
                learnerL1 = find_language(originalDocument_filepath)

                instance_data["sentence_id"] = sent_id
                instance_data["learnerL1"] = learnerL1
            if line.startswith("# text"):
                incorrect_sentence = line.replace("# text = ", "").strip()
                instance_data["incorrect_sentence"] = incorrect_sentence 
            if line.startswith("1\t"):
                conllu_data = line
                while line != "\n":
                    line = inpf.__next__()
                    conllu_data += line
                instance_data["incorrect_conllu_data"] = conllu_data.strip()
                from conllu import parse
                sentences = parse(conllu_data.strip())
                instance_data["tokenized_incorrect_sentence"] =[t['form'] for t in sentences[0]]
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
                tannt = "".join(line.split("=")[1:]).strip().replace(" newtrue>",">")
                instance_data["error_annotated_sentence"] = tannt
                alphanumeric_pattern = r'[\',. a-zA-Z0-9]+'
                optional_bracket = r'[/]'
                
                patterns = {
                        "c": '<c>[?\',. A-Za-z0-9]+<([\/]{1})?c>',
                        "i": '<i>[?\',. A-Za-z0-9]+<([\/]{1})?i>',
                        "ns": '<ns type["<>/A-Za-z]+</ns>',
                        "global": r'<ns.*?</ns>',
                        "startTag": fr'<ns type"({alphanumeric_pattern})"({alphanumeric_pattern})?>',
                        "endTag": fr'<\/ns>',
                        "ReplacementCorrection": fr'<ns type"({alphanumeric_pattern})"({alphanumeric_pattern})?><i>({alphanumeric_pattern})<\/i><c>({alphanumeric_pattern})<\/c><\/ns>',#'<ns type["/A-Za-z]+><i>.+</i><c>.+</c></ns>',
                        "IncorrectRemoval": fr'<ns type"({alphanumeric_pattern})"><i>({alphanumeric_pattern})<\/i><\/ns>',#'<ns type["/A-Za-z]+><i>.+</i></ns>',
                        "CorrectInsert": fr'<ns type"({alphanumeric_pattern})"><c>({alphanumeric_pattern})<\/c><\/ns>',
                        "doubleCorrection": '<ns type["<>/A-Za-z]+</ns>',
                        "error_type": "<ns type\"[A-Za-z]+",
                        }
                pattern = patterns["ns"]
                GL =  patterns["global"]
                RC = patterns["ReplacementCorrection"]
                IR = patterns["IncorrectRemoval"]
                CI = patterns["CorrectInsert"]
                annotation_matches = re.finditer(pattern, tannt)
                globalMatches = [(annotation, "global_tag") for annotation in re.finditer(GL, tannt)] 
                ReplacementCorrection_Matches = [(annotation, "replacement_correction") for annotation in re.finditer(RC, tannt)] 
                IncorrectRemoval_Matches = [(annotation, "incorrect_removal") for annotation in re.finditer(IR, tannt)] 
                CorrectInsert_Matches = [(annotation, "correct_insert") for annotation in re.finditer(CI, tannt)] 

                CT = patterns["c"]
                IT = patterns["i"]
                ST = patterns["startTag"]
                ET = patterns["endTag"]
                CT_Matches = [(annotation, "correctToken") for annotation in re.finditer(CT, tannt)] 
                IT_Matches = [(annotation, "incorrectToken") for annotation in re.finditer(IT, tannt)] 
                ST_Matches = [(annotation, "startTag") for annotation in re.finditer(ST, tannt)] 
                ET_Matches = [(annotation, "endTag") for annotation in re.finditer(ET, tannt)] 

                acc_extraLen = 0
                annotations = []
                instance_data["deannotated_sentence"] = tannt
                print(f"initial sentence: {tannt}")
                currDeannotatedSentence = instance_data["deannotated_sentence"]

                for annotation_tpl in sorted([*ReplacementCorrection_Matches, *CorrectInsert_Matches, *IncorrectRemoval_Matches ],key=lambda tpl: tpl[0].span()[0] ): #globalMatches:
                    annotation_match, matchType = annotation_tpl
                    annotationData, currDeannotatedSentence, acc_extraLen = extract_annotationData(annotation_match, currDeannotatedSentence, matchType, patterns, acc_extraLen)
                    annotations.append(annotationData)


                instance_data["deannotated_sentence"] = currDeannotatedSentence
                instance_data["deannotated_sentence"]  = re.sub(' +',' ',instance_data["deannotated_sentence"])
                instance_data["tokenized_deannotated_sentence"] = nltk.word_tokenize(instance_data["deannotated_sentence"]) 
                # for idx, a in enumerate(annotations):
                instance_data["annotations"] = annotations
                print(f'cleaned annotated sentence : {instance_data["deannotated_sentence"]}')

                '''
                for annotation_tpl in [*ST_Matches, *IT_Matches, *CT_Matches, *ET_Matches]:#annotation_matches:
                    annotation_match, matchType = annotation_tpl
                    print(annotation_match)
                    currDeannotatedSentence = annotationRemoval(annotation_match, currDeannotatedSentence, matchType, patterns)
                IT_Matches = [(annotation, "incorrectToken") for annotation in re.finditer(IT, currDeannotatedSentence)] 
                for annotation_tpl in [*IT_Matches]:#annotation_matches:
                    annotation_match, matchType = annotation_tpl
                    print(annotation_match)
                    currDeannotatedSentence = annotationRemoval(annotation_match, currDeannotatedSentence, matchType, patterns)
                instance_data["deannotated_sentence"] = currDeannotatedSentence
                for c in [".",",","!","?","'"]:
                    instance_data["deannotated_sentence"] = instance_data["deannotated_sentence"].replace(c,f" {c}")   
                    instance_data["deannotated_sentence"] = instance_data["deannotated_sentence"].replace(f" {c}00",f"{c}00")   
                for w in [" 've", "weren 't","didn 't","can 't","don 't","wasn 't","Mr ."," 'm"," 'd"," 's"," .C"," .E"]:
                    instance_data["deannotated_sentence"] = instance_data["deannotated_sentence"].replace(w,f'{w.replace(" ","")}')   
                    instance_data["deannotated_sentence"] = instance_data["deannotated_sentence"].replace(w.upper(),f'{w.upper().replace(" ","")}')   
                # instance_data["deannotated_sentence"]  = re.sub('<i>','',instance_data["deannotated_sentence"])
                '''

                # print(instance_data["deannotated_sentence"], instance_data["incorrect_sentence"])
                if instance_data["tokenized_deannotated_sentence"] == instance_data["tokenized_incorrect_sentence"]:  
                        counts["identical"]+=1
                        instance_data["isIdentical"] = True
                if not (instance_data["deannotated_sentence"] == instance_data["incorrect_sentence"]):
                    import difflib
                    output_list = [li for li in difflib.ndiff(instance_data["deannotated_sentence"] , instance_data["incorrect_sentence"]) if li[0] != ' ']
                    if output_list and (len(output_list) < 3 or list(set(output_list))[0] == " ") :
                        print(f'original incorrect sentence: {instance_data["incorrect_sentence"]}')
                        print(output_list)
                    else: 
                        counts["diff"]+=1
                        instance_data["hasError"] = True

                if any([term in instance_data["deannotated_sentence"] for term in ["<ns","</ns","<i>","</i>","<c>","</c>"]]):
                    counts["error"]+=1
                    instance_data["hasError"] = True
                    print_instance(instance_data)
                    import os
                    os.system('clear')
                    error_sentences.append(f'{instance_data["deannotated_sentence"]}\t{tannt}')
                    print(tannt)
                    print(instance_data["deannotated_sentence"])
                    print(instance_data["learnerL1"])
                elif tannt in debugging_examples:
                    pass
                else:
                    counts["ok"]+=1
                    instance_data["hasError"] = False
            if line.startswith("1\t"):
                conllu_data = line
                while line != "\n":
                    line = inpf.__next__()
                    conllu_data += line
                instance_data["corrected_conllu_data"] = conllu_data.strip()
                
                instances[instance_data["sentence_id"]] = instance_data
                if instance_data["hasError"] == False:
                    correct_instances[instance_data["sentence_id"]] = instance_data
                if instance_data["isIdentical"] == True:
                    instance_data = align_annotation_to_token(instance_data)
                    identical_instances[instance_data["sentence_id"]] = instance_data

                #print_instance(instance_data)
                instance_data = {}

print(counts)

with open("error_sentences","w") as errorf:
    for s in error_sentences:
        errorf.write(f"{s}\n")

with open(output_filepath,"w") as outf:
    outf.write(json.dumps(instances, indent = 4))

with open(correct_output_filepath,"w") as outf:
    outf.write(json.dumps(correct_instances, indent = 4))

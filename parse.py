"""
    Given the FCE released dataset and the annotated dataset files
    Extract annotations and align annotations to tokens in incorrect sentences 
    OUTPUTTING a JSON file with annotation per token in the incorrect sentence

    docs: https://docs.google.com/document/d/19GjzEgShH1E6PMNcgZA_pMog-fM3DqRkBS3Rja1bDz0/edit 
"""
import json
import copy
import re
import os
from collections import defaultdict
import difflib
import pprint
import nltk
from nltk.tag import pos_tag

debugging_examples = [
    'I am very sorry to say it was definitely not a perfect evening out,\
    and I am therefore asking you <ns type"MT"><c>for</c></ns> a refund \
    <ns type"RT"><i>of</i><c>for</c></ns> the amount I paid, or part of it.',
    'which <ns type"UV"><i>is</i></ns> special place in the town \
    <ns type"UA"><i>we</i></ns> <ns type"MV"><c>will be</c></ns> \
    <ns type"S"><i>lefet</i><c>left</c></ns> just \
    <ns type"RT"><i>for</i><c>as</c></ns> <ns type"MD"><c>a</c></ns> \
    <ns type"S"><i>comerical</i><c>commercial</c></ns> area, but of course \
    with all the facilities to get there <ns type"MP"><c>,</c></ns> like a \
    big supermarket <ns type"UP"><i>,</i></ns> and mall centre \
    <ns type"MP"><c>,</c></ns> <ns type"RT"><i>in</i><c>from</c></ns> \
    different areas of the city.'
    'I also suggest that more plays and films should \
    <ns type"RV"> <ns type"FV"><i>be taken</i><c>take</c></ns> place</ns>.',
    'Therefore, I strongly recommend that you \
    <ns type"RV" newtrue><i>should collect</i><c>invite<c></ns> artists \
    and stars <ns type"MT"><c>from</c></ns> more than six countries and \
    <ns type"RV"><i>rebuild</i><c>build</c></ns> larger concert halls.',
    'When we <ns type"TV"><i> <ns type"DV"><i>entrance</i>\
    <c>enter</c></ns></i><c>entered</c></ns> the place our problems \
    <ns type"TV"><i>begun</i><c>began</c></ns>.',
]

def check_only_instances_with_empty_incorrect_tokens_have_no_aligned_tokens(instances_):
    notokens = 0
    notokens_n_notInsertion = 0
    for instance in instances_.values():
        for annotation in instance["annotations"]: 
            if len(annotation["aligned_incorrect_tokens"]) == 0 \
                    and annotation["incorrect_token"] != '':
                print(annotation)
                print(instance["deannotated_sentence"])
                notokens += 1
    print(notokens);input()

def span_tokenize(sentence):
    '''
        uses nltk TreebanWordTOkenizer
        to return tuple ("token", (start_idx, end_idx), token_idx)
        per token
    '''
    tokenizer = nltk.TreebankWordTokenizer()
    tokens_data = list(zip(tokenizer.tokenize(sentence), tokenizer.span_tokenize(sentence)))
    tokens = [tpl[0] for tpl in tokens_data]
    pos_tags = pos_tag(tokens)
    tokens_data = [(*tpl, idx) for idx, tpl in enumerate(tokens_data)]
    tokens_data = [(*tpl, pos[1], len(tpl[0])) for tpl, pos in zip(tokens_data, pos_tags)] 
    return tokens_data

def align_spans_token_to_annotation(annotation, tokens_):
    '''
        given a token in the format (token, (start_idx, end_idx))
        and a list of annotations. 
        assign for each annotation a list of tokens based on span overlap
    '''
    annt_start_idx, annt_end_idx = annotation["span_in_DeannotatedSentence"] 
    annotation["aligned_incorrect_tokens"] = []
    for token_idx, token in enumerate(tokens_):
        t_start_idx, t_end_idx = token[1]
        isSpan = (annt_start_idx == t_start_idx) and (annt_end_idx == t_end_idx) 
        isWithinSpan = ((annt_start_idx <= t_start_idx) and (annt_end_idx > t_end_idx))\
                    or ((annt_start_idx < t_start_idx) and (annt_end_idx >= t_end_idx))
        if isSpan or isWithinSpan:
            annotation["aligned_incorrect_tokens"].append(token)
    annotation["number_of_incorrect_tokens"] = len(annotation["aligned_incorrect_tokens"])
    return annotation

def align_spans_token_to_annotations(curr_instance_data):
    '''
        given a list of tokens in the format (token, (start_idx, end_idx))
        and a list of annotations. 
        assign a field for each annotation a list of tokens based on span overlap
        named 'aligned_incorrect_tokens'
    '''
    tokens_ = curr_instance_data["tokenized_deannotated_sentence"]
    annotations_ = curr_instance_data["annotations"]
    aligned_annotations = [align_spans_token_to_annotation(annotation, tokens_)
                            for annotation in annotations_]
    return aligned_annotations

def linguistic_process(idx, pos_tags, tokens_):
    '''
        extract linguistic features 
        of a new token being put 
        into a new context sentence
    '''
    token_str =  tokens_[idx][0]
    token_len = len(token_str)
    acc_idx = 0
    for idx_before_selected_token in range(idx): 
        acc_idx += len(tokens_[idx_before_selected_token][0]) + 1
    start_idx = acc_idx
    end_idx = start_idx + token_len
    span = (start_idx, end_idx)
    pos = pos_tags[idx][1]
    instance_tpl = (token_str, span, idx, pos, token_len) 
    return instance_tpl 

def replace_incorrect_tokens_with_correction_tokens(curr_instance_data):
    '''
        for each annotation replace the incorrect tokens
        with annotation correction tokens using
        the deannotated tokenized sentence
    '''
    annotations_ = curr_instance_data["annotations"]
    modified_annotations = []
    for annotation_ in annotations_:
        tokens_ = copy.deepcopy(curr_instance_data["tokenized_deannotated_sentence"])
        if annotation_["regex_match_type"] == 'replacement_correction':
            aligned_incorrect_tokens_idxs = [idx for (_,_,idx,_,_) in annotation_["aligned_incorrect_tokens"]]
            correct_tokens = annotation_["correct_token"].split(" ")
            for idx in aligned_incorrect_tokens_idxs[::-1]:
                tokens_.pop(idx)
            for correct_token in correct_tokens[::-1]:
                tokens_.insert(idx, (correct_token, ))
            partially_corrected_sentence = " ".join([tpl[0] for tpl in tokens_])
            pos_tags = pos_tag([tpl[0] for tpl in tokens_])
            annotation_["correction_idxs_in_partially_corrected_sentence"] =\
                [idx for idx, tpl in enumerate(tokens_) if len(tpl) == 1 ]


            annotation_["tokenized_partially_corrected_sentence"] = [tpl if len(tpl) == 5 else linguistic_process(idx, pos_tags, tokens_) 
                     for idx, tpl in enumerate(tokens_)]

            modified_annotations.append(copy.deepcopy(annotation_))
        else:
            modified_annotations.append(copy.deepcopy(annotation_))
        if annotation_.get('correction_idxs_in_partially_corrected_sentence') and len(annotation_['correction_idxs_in_partially_corrected_sentence']) > 1:
            print("\n")
            print(annotation_)
            print("\n")
    return modified_annotations

def print_instance(instance):
    """
    print annotated sentence instance data
    deleting very long properties that
    are not really necessary to be visualized
    """
    pprint.pprint(
        {
            k: v
            for k, v in instance.items()
            if k not in ["corrected_conllu_data", "incorrect_conllu_data"]
        }
    )


def find_language(original_document_filepath):
    """
    extract text from the language tag in a file from
    the released FCE dataset
    """
    original_document_patterns = {"language": "<language>[A-Za-z]+</language>"}
    with open(original_document_filepath, encoding="UTF-8") as originalinpf:
        file_content = originalinpf.read()
        learnerl1_tag = (
            re.search(original_document_patterns["language"], file_content).group(0)
            if re.search(original_document_patterns["language"], file_content)
            else ""
        )
        extracted_learnerl1 = learnerl1_tag.replace("<language>", "").replace(
            "</language>", ""
        )
        assert (
            extracted_learnerl1 != ""
        ), f"learnerl1 is empty for {original_document_filepath}"
    return extracted_learnerl1

def find_score(original_document_filepath):
    """
    extract text from the score tag in a file from
    the released FCE dataset
    """
    original_document_patterns = {"score": "<score>[0-9.]+</score>"}
    with open(original_document_filepath, encoding="UTF-8") as originalinpf:
        file_content = originalinpf.read()
        learnerscore_tag = (
            re.search(original_document_patterns["score"], file_content).group(0)
            if re.search(original_document_patterns["score"], file_content)
            else ""
        )
        extracted_learnerscore = learnerscore_tag.replace("<score>", "").replace(
            "</score>", ""
        )
        assert (
            extracted_learnerscore != ""
        ), f"learnerscore is empty for {original_document_filepath} {file_content}"
    return extracted_learnerscore

def extract_annotation_data(
    annotation, curr_deannotated_sentence, regex_match_type, patterns, extra_len
):
    """
    extract annotations tags from string and parse
    data properties related to each.
    """
    extracted_annotation_data = {"regex_match_type": regex_match_type}
    annotation_len = len(annotation.group(0))
    extracted_annotation_data["annotationStr"] = annotation.group(0)
    match_error_type = re.search(
        patterns["error_type"], extracted_annotation_data["annotationStr"]
    )
    extracted_annotation_data["error_type_symbol"] = match_error_type.group(0).split(
        '"'
    )[1]
    extracted_annotation_data["incorrect_token"] = (
        re.search(patterns["i"], extracted_annotation_data["annotationStr"]).group(0)[
            3:-4
        ]
        if re.search(patterns["i"], extracted_annotation_data["annotationStr"])
        else ""
    )
    extracted_annotation_data["correct_token"] = (
        re.search(patterns["c"], extracted_annotation_data["annotationStr"]).group(0)[
            3:-4
        ]
        if re.search(patterns["c"], extracted_annotation_data["annotationStr"])
        else ""
    )
    (
        annotated_sentence_match_start_idx,
        _,
    ) = annotation.span()
    token_start_idx_in_incorrect_sentence = (
        annotated_sentence_match_start_idx - extra_len
    )
    token_end_idx_in_incorrect_sentence = (
        annotated_sentence_match_start_idx
        - extra_len
        + len(extracted_annotation_data["incorrect_token"])
    )
    extra_len += annotation_len - len(extracted_annotation_data["incorrect_token"])

    extracted_annotation_data["span_in_IncorrectSentence"] = (
        token_start_idx_in_incorrect_sentence,
        token_end_idx_in_incorrect_sentence,
    )
    extracted_annotation_data["span_in_AnnotatedSentence"] = annotation.span()
    token_start_idx_in_deannotated_sentence = curr_deannotated_sentence.find(
        extracted_annotation_data["annotationStr"]
    )
    token_end_idx_in_deannotated_sentence = (
        token_start_idx_in_deannotated_sentence
        + len(extracted_annotation_data["incorrect_token"])
    )
    extracted_annotation_data["span_in_DeannotatedSentence"] = [
        token_start_idx_in_deannotated_sentence,
        token_end_idx_in_deannotated_sentence,
    ]
    curr_deannotated_sentence = curr_deannotated_sentence.replace(
        extracted_annotation_data["annotationStr"],
        extracted_annotation_data["incorrect_token"],
        1,
    )
    return extracted_annotation_data, curr_deannotated_sentence, extra_len


def annotation_removal(annotation, curr_deannotated_sentence, regex_match_type):
    """
    given a sentence with annotations remove the next ocurrence
    of a given annotation
    """
    annotation_str = annotation.group(0)
    if regex_match_type != "incorrectToken":
        curr_deannotated_sentence = curr_deannotated_sentence.replace(
            annotation_str, "", 1
        )
    else:
        curr_deannotated_sentence = curr_deannotated_sentence.replace(
            annotation_str, annotation_str.replace("<i>", "").replace("</i>", ""), 1
        )
    return curr_deannotated_sentence


if __name__ == "__main__":
    ORIGINAL_FILEPATH = (
        "/app/pipelines/data/fce_dataset/fce-correction-annotations/" + "en_esl-ud-train.conllu"
    )
    CORRECTED_FILEPATH = (
        "/app/pipelines/data/fce_dataset/fce-correction-annotations/corrected/"
        + "en_cesl-ud-train.conllu"
    )
    ORIGINAL_DOCUMENT_FOLDERPATH = "/app/pipelines/data/fce_dataset/fce-released-dataset/dataset"
    OUTPUT_FILEPATH = "fce_error_annotations.json"
    CORRECT_OUTPUT_FILEPATH = "fce_correct_error_annotations.json"
    IDENTICAL_OUTPUT_FILEPATH = "fce_identical_error_annotations.json"
    instances = {}
    correct_instances = {}
    identical_instances = {}
    error_sentences = []
    counts = defaultdict(int)

    with open(ORIGINAL_FILEPATH, encoding="UTF-8") as inpf:
        instance_data = {}
        for line in inpf:
            if line.startswith("# sent_id"):
                sent_id = line.replace("# sent_id = ", "").strip()
                foldername, filename, _ = sent_id.split("-")
                originalDocument_filepath = os.path.join(
                    ORIGINAL_DOCUMENT_FOLDERPATH, foldername, filename
                )
                learnerl1 = find_language(originalDocument_filepath)
                learnerscore = find_score(originalDocument_filepath)
                instance_data["sentence_id"] = sent_id
                instance_data["learnerl1"] = learnerl1
                instance_data["learnerscore"] = learnerscore
            if line.startswith("# text"):
                incorrect_sentence = line.replace("# text = ", "").strip()
                instance_data["incorrect_sentence"] = incorrect_sentence
            if line.startswith("1\t"):
                conllu_data = line
                while line != "\n":
                    line = next(inpf)
                    conllu_data += line
                instance_data["incorrect_conllu_data"] = conllu_data.strip()
                from conllu import parse

                sentences = parse(conllu_data.strip())
                tokens = iter(sentences[0])
                instance_data["tokenized_incorrect_sentence"] = [
                    t["form"] for t in tokens
                ]
                instances[instance_data["sentence_id"]] = instance_data
                instance_data = {}

    with open(CORRECTED_FILEPATH, encoding="UTF-8") as inpf:
        for line in inpf:
            if line.startswith("# sent_id"):
                sent_id = line.replace("# sent_id = ", "").strip()
                instance_data = instances[sent_id]
            if line.startswith("# text"):
                corrected_sentence = line.replace("# text = ", "").strip()
                instance_data["corrected_sentence"] = corrected_sentence
            if line.startswith("# error_annotation"):
                TANNT = "".join(line.split("=")[1:]).strip().replace(" newtrue>", ">")
                instance_data["error_annotated_sentence"] = TANNT
                ALPHANUMERIC_PATTERN = r"[\',. a-zA-Z0-9]+"
                OPTIONAL_BRACKET = r"[/]"

                regex_patterns = {
                    "c": "<c>[?',. A-Za-z0-9]+<([\\/]{1})?c>",
                    "i": "<i>[?',. A-Za-z0-9]+<([\\/]{1})?i>",
                    "ns": '<ns type["<>/A-Za-z]+</ns>',
                    "global": r"<ns.*?</ns>",
                    "startTag": rf'<ns type"({ALPHANUMERIC_PATTERN})"({ALPHANUMERIC_PATTERN})?>',
                    "endTag": r"<\/ns>",
                    "ReplacementCorrection": rf'<ns type"({ALPHANUMERIC_PATTERN})"'
                    + rf"({ALPHANUMERIC_PATTERN})?>"
                    + rf"<i>({ALPHANUMERIC_PATTERN})<\/i>"
                    + rf"<c>({ALPHANUMERIC_PATTERN})<\/c><\/ns>",
                    "IncorrectRemoval": rf'<ns type"({ALPHANUMERIC_PATTERN})">'
                    + rf"<i>({ALPHANUMERIC_PATTERN})<\/i><\/ns>",
                    "CorrectInsert": rf'<ns type"({ALPHANUMERIC_PATTERN})">'
                    + rf"<c>({ALPHANUMERIC_PATTERN})<\/c><\/ns>",
                    "doubleCorrection": '<ns type["<>/A-Za-z]+</ns>',
                    "error_type": '<ns type"[A-Za-z]+',
                }
                PATTERN = regex_patterns["ns"]
                GL = regex_patterns["global"]
                RC = regex_patterns["ReplacementCorrection"]
                IR = regex_patterns["IncorrectRemoval"]
                CI = regex_patterns["CorrectInsert"]
                annotation_matches = re.finditer(PATTERN, TANNT)
                globalMatches = [
                    (annotation, "global_tag") for annotation in re.finditer(GL, TANNT)
                ]
                ReplacementCorrection_Matches = [
                    (annotation, "replacement_correction")
                    for annotation in re.finditer(RC, TANNT)
                ]
                IncorrectRemoval_Matches = [
                    (annotation, "incorrect_removal")
                    for annotation in re.finditer(IR, TANNT)
                ]
                CorrectInsert_Matches = [
                    (annotation, "correct_insert")
                    for annotation in re.finditer(CI, TANNT)
                ]

                CT = regex_patterns["c"]
                IT = regex_patterns["i"]
                ST = regex_patterns["startTag"]
                ET = regex_patterns["endTag"]
                CT_Matches = [
                    (annotation, "correctToken")
                    for annotation in re.finditer(CT, TANNT)
                ]
                IT_Matches = [
                    (annotation, "incorrectToken")
                    for annotation in re.finditer(IT, TANNT)
                ]
                ST_Matches = [
                    (annotation, "startTag") for annotation in re.finditer(ST, TANNT)
                ]
                ET_Matches = [
                    (annotation, "endTag") for annotation in re.finditer(ET, TANNT)
                ]

                ACC_EXTRA_LEN = 0
                annotations = []
                instance_data["deannotated_sentence"] = TANNT
                print(f"initial sentence: {TANNT}")
                partial_deannotated_sentence = instance_data["deannotated_sentence"]
                DEBUG = False
                if "I would prefer to stay in" in TANNT:
                    DEBUG = True
                for annotation_tpl in sorted(
                    [
                        *ReplacementCorrection_Matches,
                        *CorrectInsert_Matches,
                        *IncorrectRemoval_Matches,
                    ],
                    key=lambda tpl: tpl[0].span()[0],
                ):  # globalMatches:
                    print(partial_deannotated_sentence)
                    annotation_match, match_tag_type = annotation_tpl
                    (
                        annotationData,
                        partial_deannotated_sentence,
                        ACC_EXTRA_LEN,
                    ) = extract_annotation_data(
                        annotation_match,
                        partial_deannotated_sentence,
                        match_tag_type,
                        regex_patterns,
                        ACC_EXTRA_LEN,
                    )
                    if DEBUG:
                        print(partial_deannotated_sentence)
                    annotations.append(annotationData)

                instance_data["deannotated_sentence"] = partial_deannotated_sentence
                '''
                instance_data["deannotated_sentence"] = re.sub(
                    " +", " ", instance_data["deannotated_sentence"]
                )
                '''
                instance_data["tokenized_deannotated_sentence"] = span_tokenize(
                    instance_data["deannotated_sentence"]
                )
                # for idx, a in enumerate(annotations):
                instance_data["annotations"] = annotations
                print(
                    f'cleaned annotated sentence : {instance_data["deannotated_sentence"]}'
                )

                # print(instance_data["deannotated_sentence"], instance_data["incorrect_sentence"])
                if (
                    [token for (token,*_) in instance_data["tokenized_deannotated_sentence"]]
                    == instance_data["tokenized_incorrect_sentence"]
                ):
                    counts["identical"] += 1
                    instance_data["isIdentical"] = True
                else:
                    instance_data["isIdentical"] = False
                if not (
                    instance_data["deannotated_sentence"]
                    == instance_data["incorrect_sentence"]
                ):
                    output_list = [
                        li
                        for li in difflib.ndiff(
                            instance_data["deannotated_sentence"],
                            instance_data["incorrect_sentence"],
                        )
                        if li[0] != " "
                    ]
                    if output_list and (
                        len(output_list) < 3 or list(set(output_list))[0] == " "
                    ):
                        print(
                            f'original incorrect sentence: {instance_data["incorrect_sentence"]}'
                        )
                        print(output_list)
                    else:
                        counts["diff"] += 1
                        instance_data["hasError"] = True

                if any(
                    term in instance_data["deannotated_sentence"]
                    for term in ["<ns", "</ns", "<i>", "</i>", "<c>", "</c>"]
                ):
                    counts["error"] += 1
                    instance_data["hasError"] = True
                    print_instance(instance_data)

                    os.system("clear")
                    error_sentences.append(
                        f'{instance_data["deannotated_sentence"]}\t{TANNT}'
                    )
                    print(TANNT)
                    print(instance_data["deannotated_sentence"])
                    print(instance_data["learnerl1"])
                elif TANNT in debugging_examples:
                    pass
                else:
                    counts["ok"] += 1
                    instance_data["hasError"] = False
            if line.startswith("1\t"):
                conllu_data = line
                while line != "\n":
                    line = next(inpf)
                    conllu_data += line
                instance_data["corrected_conllu_data"] = conllu_data.strip()

                instances[instance_data["sentence_id"]] = instance_data
                if not instance_data["hasError"]:
                    correct_instances[instance_data["sentence_id"]] = instance_data
                if instance_data["isIdentical"]:
                    instance_data["annotations"] = align_spans_token_to_annotations(instance_data)
                    instance_data["annotations"] = replace_incorrect_tokens_with_correction_tokens(instance_data)
                    identical_instances[instance_data["sentence_id"]] = instance_data

                # print_instance(instance_data)
                instance_data = {}

    with open("error_sentences", "w", encoding="UTF-8") as errorf:
        for s in error_sentences:
            errorf.write(f"{s}\n")

    with open(OUTPUT_FILEPATH, "w", encoding="UTF-8") as outf:
        outf.write(json.dumps(instances, indent=4))

    with open(CORRECT_OUTPUT_FILEPATH, "w", encoding="UTF-8") as outf:
        outf.write(json.dumps(correct_instances, indent=4))

    with open(IDENTICAL_OUTPUT_FILEPATH, "w", encoding="UTF-8") as outf:
        outf.write(json.dumps(identical_instances, indent=4))

    print(counts)

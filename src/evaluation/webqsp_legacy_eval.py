# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# File altered from the eval file distributed along with the WebQSP dataset to calculate hits@1

import random
import sys
import json
import argparse


def FindInList(entry, elist):
    for item in elist:
        if entry == item:
            return True
    return False


def CalculatePRF1(goldAnswerList, predAnswerList):
    if len(goldAnswerList) == 0:
        if len(predAnswerList) == 0:
            return [1.0, 1.0, 1.0, 1.0]  # consider it 'correct' when there is no labeled answer, and also no predicted answer
        else:
            return [0.0, 1.0, 0.0, 0.0]  # precision=0 and recall=1 when there is no labeled answer, but has some predicted answer(s)
    elif len(predAnswerList) == 0:
        return [1.0, 0.0, 0.0, 0.0]  # precision=1 and recall=0 when there is labeled answer(s), but no predicted answer
    else:
        glist = [x["AnswerArgument"] for x in goldAnswerList]
        plist = predAnswerList

        tp = 1e-40  # numerical trick
        fp = 0.0
        fn = 0.0

        for gentry in glist:
            if FindInList(gentry, plist):
                tp += 1
            else:
                fn += 1
        for pentry in plist:
            if not FindInList(pentry, glist):
                fp += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        f1 = (2 * precision * recall) / (precision + recall)
        num_random = 100
        random_hit = 0
        for i in range(num_random):
            random_ans = random.choice(plist)
            if random_ans in glist:
                random_hit += 1
        random_hit /= num_random
        return [precision, recall, f1, random_hit]


def main(gold_data_path, pred_answer_path):


    goldData = json.loads(open(gold_data_path).read())
    predAnswers = json.loads(open(pred_answer_path).read())

    PredAnswersById = {}

    for item in predAnswers:
        PredAnswersById.update({item : predAnswers[item]["answer"]})
        # PredAnswersById[item["QuestionId"]] = item["Answers"]
        # PredAnswersById[item["qid"]] = item["answer"]

    total = 0.0
    f1sum = 0.0
    recSum = 0.0
    precSum = 0.0
    hits1Sum = 0.0
    numCorrect = 0
    missing_pred = 0
    for entry in goldData:

        skip = True
        for pidx in range(0, len(entry["Parses"])):
            np = entry["Parses"][pidx]
            if np["AnnotatorComment"]["QuestionQuality"] == "Good" and np["AnnotatorComment"]["ParseQuality"] == "Complete":
                skip = False

        if (len(entry["Parses"]) == 0 or skip):
            continue

        total += 1

        id = entry["QuestionId"]

        if id not in PredAnswersById:
            missing_pred += 1
            print("The problem " + id + " is not in the prediction set")
            print("Continue to evaluate the other entries")
            continue

        if len(entry["Parses"]) == 0:
            print("Empty parses in the gold set. Breaking!!")
            break

        predAnswers = PredAnswersById[id]

        bestf1 = -9999
        bestf1Rec = -9999
        bestf1Prec = -9999
        bestHits1 = -9999
        for pidx in range(0, len(entry["Parses"])):
            pidxAnswers = entry["Parses"][pidx]["Answers"]
            prec, rec, f1, hits1 = CalculatePRF1(pidxAnswers, predAnswers)
            if f1 > bestf1:
                bestf1 = f1
                bestf1Rec = rec
                bestf1Prec = prec
                bestHits1 = hits1

        f1sum += bestf1
        recSum += bestf1Rec
        precSum += bestf1Prec
        hits1Sum +=  bestHits1
        if bestf1 == 1.0:
            numCorrect += 1
        
        # if bestf1 != 1.0:
        #         print(entry["QuestionId"])

    print("Number of questions:", int(total))
    print("Average precision over questions: %.3f" % (precSum / total))
    print("Average recall over questions: %.3f" % (recSum / total))
    print("Average f1 over questions (accuracy): %.3f" % (f1sum / total))
    print("F1 of average recall and average precision: %.3f" % (2 * (recSum / total) * (precSum / total) / (recSum / total + precSum / total)))
    print("True accuracy (ratio of questions answered exactly correctly): %.3f" % (numCorrect / total))
    print("Random hits@1: %.3f" % (hits1Sum / total))
    print("missing pred: ",missing_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_data_path', type=str, help='The path to dataset file for evaluation (e.g., dev.json or test.json)',
                        default=0)
    parser.add_argument('--pred_answer_path', type=str, help='The path to predictions file')
   
    args = parser.parse_args()
    main(args.gold_data_path, args.pred_answer_path)
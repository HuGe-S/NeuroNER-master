from curses import raw
from operator import index
from os import access
from re import L
from sklearn.preprocessing import normalize
import numpy as np
import copy
import csv
import itertools
import sys

# Import Module
import os
import ast


def accumEval(
    outputstring,
    path,
    setname,
    Mapdict,
    resultlines,
):
    def read_text_file(file_path, lines):
        with open(file_path, encoding="cp1252", errors="ignore") as file_in:
            for line in file_in:
                if line != "":
                    lines.append(line)

    # process input lines
    predicttextlst = []
    predictresultlst = []
    # print(len(resultlines))
    countre = 0
    for i in resultlines:
        countre += 1
        # print(countre)
        predicttext, predictresult = i.split(",{", 1)
        predicttextlst.append(predicttext)
        predictresult = "{" + predictresult
        predictresult = ast.literal_eval(predictresult)
        maxresult = {
            k: v
            for k, v in sorted(
                predictresult.items(), key=lambda item: item[1], reverse=True
            )
        }
        predictresultlst.append(list(maxresult.keys())[0])
    # print(resultlines[1])
    # print("text: " + predicttextlst[1])
    # print("possibility: " + predictresultlst[1])
    path = path + "//" + setname
    Mapdict = Mapdict.items()
    path = os.path.abspath(path)
    # read original data
    rawlines = []
    file_path = []

    for file in os.listdir(path):
        if file.endswith(".ann"):
            file_path.append(f"{path}//{file}")

    file_path = sorted(file_path)
    # read dataset path
    for i in file_path:
        read_text_file(i, rawlines)

    rawlines = [x for x in rawlines if x != ""]
    # read output file
    with open(outputstring, encoding="cp1252") as file_in:
        lines = []
        for line in file_in:
            if line != "":
                lines.append(line)

    # process categories
    categories = list(Mapdict)
    mapcount = 0

    for i in range(len(categories)):
        if categories[i][1][0] == "B":
            # print(categories[i])
            mapcount = mapcount + 1
    categories = categories[0:mapcount]
    # print(categories)
    for i in range(len(categories)):
        categories[i] = categories[i][1]
        categories[i] = categories[i][2:]
    initeval = {
        "TP": 0,
        "FP": 0,
        "TN": 0,
        "FN": 0,
        "Accuracy": 0,
        "Precision": 0,
        "Recall": 0,
        "F1-score": 0,
    }
    initeval = [copy.deepcopy(initeval) for i in range(mapcount)]
    evaldict = dict(zip(categories, initeval))
    # Extract categories and txt from ann files
    # print(rawlines)
    rawlines = [i.split() for i in rawlines]
    truecategories = [i[1] for i in rawlines]
    truecontext = [" ".join(i[4:]) for i in rawlines]
    # print(evaldict)
    # print(resultlines[1])
    # print("text:" + predicttextlst[1])
    # print("possibility: " + predictresultlst[1])
    # print(truecontext[1])
    # print(truecategories[1])
    # print(len(predicttext))
    for i in range(len(predicttextlst)):

        predicttextlst[i] = predicttextlst[i].strip()
        # print(resultlines[i])
        # print("text:" + predicttextlst[i])
        # print("possibility: " + predictresultlst[i])
        # print(truecontext[i])
        # print(truecategories[i])
        # print(predicttextlst[i] == truecontext[i])
        assert predicttextlst[i] == truecontext[i]
        if predictresultlst[i] == truecategories[i]:
            evaldict[predictresultlst[i]]["TP"] = (
                evaldict[predictresultlst[i]]["TP"] + 1
            )
        elif predictresultlst[i] != truecategories[i]:
            evaldict[truecategories[i]]["FN"] = evaldict[truecategories[i]]["FN"] + 1
            evaldict[predictresultlst[i]]["FP"] = (
                evaldict[predictresultlst[i]]["FP"] + 1
            )
    print(evaldict)
    # Calculate confusion mattrix:
    for k, v in evaldict.items():
        try:
            v["Accuracy"] = float(
                (v["TP"] + v["TN"]) / (v["TP"] + v["TN"] + v["FP"] + v["FN"])
            )
        except:
            v["Accuracy"] = "N/A"
        try:
            v["Precision"] = float((v["TP"]) / (v["TP"] + v["FP"]))
        except:
            v["Precision"] = "N/A"
        try:
            v["Recall"] = float((v["TP"]) / (v["TP"] + v["FN"]))
        except:
            v["Recall"] = "N/A"
        try:
            v["F1-score"] = float(
                (2 * v["Precision"] * v["Recall"]) / (v["Precision"] + v["Recall"])
            )
        except:
            v["F1-score"] = "N/A"
    fields = [
        "Categories",
        "TP",
        "FP",
        "TN",
        "FN",
        "Precision",
        "Accuracy",
        "Recall",
        "F1-score",
    ]
    # for k, v in evaldict.items():
    #     for sk, sv in v.items():
    #         sv = str(sv)
    print(evaldict)
    with open("./customoutput/" + setname + "_evaluation.csv", "w") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for k in evaldict:
            w.writerow(
                {
                    field: str(evaldict[k].get(field))
                    if evaldict[k].get(field) == 0
                    else evaldict[k].get(field) or k
                    for field in fields
                }
            )
    # print(len(truecategories))
    # print(truecategories[2])
    # print(len(truecontext))
    # print(truecontext[1])

    # Categories process complete

    # Mapdict = dict((x, y) for x, y in Mapdict)

    # words = [i.split() for i in lines]
    # possibility = []
    # # Seperate words and possibilities
    # for i in range(len(words)):
    #     if words[i] != []:
    #         possibility.append(words[i][6:-2])
    #         words[i] = words[i][0]
    #         # print(words[i])
    # # print(len(possibility[1]))
    # words[:] = [x for x in words if x]
    # accum = [0 for i in range(len(Mapdict))]
    # sentence = ""
    # possibilitystore = []
    # flag = 0
    # outline = []

    # sentenceindex = 0
    # rawsentence = truecontext[sentenceindex].split()
    # # print(len(words))
    # # print(len(possibility))
    # index = 0
    # while words:
    #     if rawsentence:
    #         concatwords = ""
    #         currentwords = rawsentence.pop(0)
    #         while concatwords != currentwords:
    #             pending = words.pop(0)
    #             if pending == []:
    #                 pending = ""
    #             concatwords = concatwords + pending
    #             pos = possibility.pop(0)
    #             for j in range(len(pos)):
    #                 pos[j] = float(pos[j])
    #             accum = [sum(x) for x in zip(accum, pos)]
    #         # print(pos[i])
    #         if flag == 0:
    #             possibilitystore.append(pos)
    #             flag = 1

    #         # print(words[i])
    #         sentence = sentence + " " + currentwords
    #     else:
    #         # if rawlines[sentenceindex] == 'SEND E-TFR FEE:
    #         #     print(accum)
    #         sentenceindex = sentenceindex + 1
    #         try:
    #             rawsentence = truecontext[sentenceindex].split()
    #         except:
    #             break
    #         # accum.reshape(-1, 1)
    #         # accum = normalize(np.exp(accum[1:-1]).reshape(1, -1), norm="l1", axis=1)
    #         # accum = accum[0]
    #         # for i in range(len(accum)):
    #         #     accum[i] = "{:.1%}".format(accum[i])
    #         sumup = 0
    #         possblemap = []
    #         for k in accum:
    #             sumup = sumup + k
    #         for k in range(len(accum)):
    #             possblemap.append(round(float(accum[k] / sumup), 10))
    #         # print(sumup)
    #         flag = 0
    #         total = 0
    #         # print(accum)
    #         simp_pos = possibilitystore.pop()
    #         # print(simp_pos)
    #         # for i in range(20):
    #         #     simp_pos[i] = simp_pos[i] * 20000
    #         # for k in simp_pos:
    #         #     total = total + k
    #         # # print(total)
    #         # for k in range(len(simp_pos)):
    #         #     simp_pos[k] = round(simp_pos[k] / total, 4)

    #         inv_map = {v: k for k, v in Mapdict.items()}
    #         possblehash = zip(inv_map, possblemap)
    #         possblehash = list(possblehash)
    #         possnumber = [0 for i in categories]

    #         for it in range(len(categories)):
    #             for j in range(len(possblehash)):
    #                 if categories[it] in possblehash[j][0]:
    #                     possnumber[it] = possnumber[it] + possblehash[j][1]
    #         categorieszip = list(zip(categories, possnumber))
    #         # print(categorieszip)
    #         # print(sentence + "  " + str(categorieszip))
    #         mydict = dict(categorieszip)
    #         mydict = {
    #             k: v
    #             for k, v in sorted(
    #                 mydict.items(), key=lambda item: item[1], reverse=True
    #             )
    #         }
    #         outline.append(sentence + "," + str(mydict))
    #         # print(mydict)
    #         accum = [0 for i in range(len(Mapdict))]
    #         sentence = ""
    # with open("./customoutput/" + setname + ".csv", "wb") as file:
    #     for line in outline:
    #         file.write(line.encode())
    #         file.write("\n".encode())
    # # print(words)
    # # print(len(possibility[0]))

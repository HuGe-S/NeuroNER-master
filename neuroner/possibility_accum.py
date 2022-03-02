from operator import index
from os import access
from re import L
from sklearn.preprocessing import normalize
import numpy as np

# Import Module
import os

# Folder Path


# Read text File


# process original data


# Mapdict = [
#     (0, "B-Advertising&Marketing"),
#     (1, "B-Apps&Subscriptions"),
#     (2, "B-Automobile"),
#     (3, "B-BankCharges"),
#     (4, "B-ClothingHairMakeupCollabo"),
#     (5, "B-GSTHSTITCs"),
#     (6, "B-HumanIntertaction"),
#     (7, "B-Insurance"),
#     (8, "B-Materials"),
#     (9, "B-Meals&Entertainment"),
#     (10, "B-OfficeExpenses"),
#     (11, "B-RentorLeasePayments"),
#     (12, "B-Shipping"),
#     (13, "B-TDBank"),
#     (14, "B-TDVisa"),
#     (15, "B-Travel"),
#     (16, "B-Wages&Salaries"),
#     (17, "B-WebsiteDomains&Email"),
#     (18, "E-Advertising&Marketing"),
#     (19, "E-Apps&Subscriptions"),
#     (20, "E-Automobile"),
#     (21, "E-BankCharges"),
#     (22, "E-ClothingHairMakeupCollabo"),
#     (23, "E-GSTHSTITCs"),
#     (24, "E-HumanIntertaction"),
#     (25, "E-Insurance"),
#     (26, "E-Materials"),
#     (27, "E-Meals&Entertainment"),
#     (28, "E-OfficeExpenses"),
#     (29, "E-RentorLeasePayments"),
#     (30, "E-Shipping"),
#     (31, "E-TDBank"),
#     (32, "E-TDVisa"),
#     (33, "E-Travel"),
#     (34, "E-Wages&Salaries"),
#     (35, "E-WebsiteDomains&Email"),
#     (36, "I-Advertising&Marketing"),
#     (37, "I-Apps&Subscriptions"),
#     (38, "I-Automobile"),
#     (39, "I-BankCharges"),
#     (40, "I-ClothingHairMakeupCollabo"),
#     (41, "I-GSTHSTITCs"),
#     (42, "I-HumanIntertaction"),
#     (43, "I-Insurance"),
#     (44, "I-Materials"),
#     (45, "I-Meals&Entertainment"),
#     (46, "I-OfficeExpenses"),
#     (47, "I-RentorLeasePayments"),
#     (48, "I-Shipping"),
#     (49, "I-TDBank"),
#     (50, "I-TDVisa"),
#     (51, "I-Travel"),
#     (52, "I-Wages&Salaries"),
#     (53, "I-WebsiteDomains&Email"),
#     (54, "O"),
#     (55, "S-Advertising&Marketing"),
#     (56, "S-Apps&Subscriptions"),
#     (57, "S-Automobile"),
#     (58, "S-BankCharges"),
#     (59, "S-ClothingHairMakeupCollabo"),
#     (60, "S-GSTHSTITCs"),
#     (61, "S-HumanIntertaction"),
#     (62, "S-Insurance"),
#     (63, "S-Materials"),
#     (64, "S-Meals&Entertainment"),
#     (65, "S-OfficeExpenses"),
#     (66, "S-RentorLeasePayments"),
#     (67, "S-Shipping"),
#     (68, "S-TDBank"),
#     (69, "S-TDVisa"),
#     (70, "S-Travel"),
#     (71, "S-Wages&Salaries"),
#     (72, "S-WebsiteDomains&Email"),
# ]


def PossibilityAcum(
    outputstring,
    Mapdict,
    path,
    setname,
):
    # print(outputstring)
    path = path + "//" + setname

    Mapdict = Mapdict.items()
    # print(Mapdict)
    # Change the directory
    path = os.path.abspath(path)

    # read original data

    rawlines = []

    def read_text_file(file_path, lines=rawlines):
        with open(file_path, encoding="cp1252", errors="ignore") as file_in:
            for line in file_in:
                if line != "":
                    lines.append(line)

    file_path = []

    for file in os.listdir(path):
        if file.endswith(".txt"):
            file_path.append(f"{path}//{file}")

    file_path = sorted(file_path)
    # read dataset path
    for i in file_path:
        read_text_file(i)

    rawlines = [x for x in rawlines if x != ""]
    # read output file
    # lines = outputstring.split()
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
            print(categories[i])
            mapcount = mapcount + 1
    categories = categories[0:mapcount]
    # print(categories)
    for i in range(len(categories)):
        categories[i] = categories[i][1]
        categories[i] = categories[i][2:]
    Mapdict = dict((x, y) for x, y in Mapdict)

    words = [i.split() for i in lines]
    possibility = []
    # Seperate words and possibilities
    for i in range(len(words)):
        if words[i] != []:
            possibility.append(words[i][6:-2])
            words[i] = words[i][0]
            # print(words[i])
    # print(len(possibility[1]))
    words[:] = [x for x in words if x]
    accum = [0 for i in range(len(Mapdict))]
    sentence = ""
    possibilitystore = []
    flag = 0
    outline = []
    print(len(rawlines))
    sentenceindex = 0
    rawsentence = rawlines[sentenceindex].split()
    # print(len(words))
    # print(len(possibility))
    index = 0
    while words:
        if rawsentence:
            concatwords = ""
            currentwords = rawsentence.pop(0)
            while concatwords != currentwords:
                pending = words.pop(0)
                if pending == []:
                    pending = ""
                concatwords = concatwords + pending
                pos = possibility.pop(0)
                for j in range(len(pos)):
                    pos[j] = float(pos[j])
                accum = [sum(x) for x in zip(accum, pos)]
            # print(pos[i])
            if flag == 0:
                possibilitystore.append(pos)
                flag = 1

            # print(words[i])
            sentence = sentence + " " + currentwords
        else:
            # if rawlines[sentenceindex] == 'SEND E-TFR FEE:
            #     print(accum)
            sentenceindex = sentenceindex + 1
            try:
                rawsentence = rawlines[sentenceindex].split()
            except:
                break
            # accum.reshape(-1, 1)
            # accum = normalize(np.exp(accum[1:-1]).reshape(1, -1), norm="l1", axis=1)
            # accum = accum[0]
            # for i in range(len(accum)):
            #     accum[i] = "{:.1%}".format(accum[i])
            sumup = 0
            possblemap = []
            for k in accum:
                sumup = sumup + k
            for k in range(len(accum)):
                possblemap.append(round(float(accum[k] / sumup), 10))
            # print(sumup)
            flag = 0
            total = 0
            # print(accum)
            simp_pos = possibilitystore.pop()
            # print(simp_pos)
            # for i in range(20):
            #     simp_pos[i] = simp_pos[i] * 20000
            # for k in simp_pos:
            #     total = total + k
            # # print(total)
            # for k in range(len(simp_pos)):
            #     simp_pos[k] = round(simp_pos[k] / total, 4)

            inv_map = {v: k for k, v in Mapdict.items()}
            possblehash = zip(inv_map, possblemap)
            possblehash = list(possblehash)
            possnumber = [0 for i in categories]

            for it in range(len(categories)):
                for j in range(len(possblehash)):
                    if categories[it] in possblehash[j][0]:
                        possnumber[it] = possnumber[it] + possblehash[j][1]
            categorieszip = list(zip(categories, possnumber))
            # print(categorieszip)
            # print(sentence + "  " + str(categorieszip))
            mydict = dict(categorieszip)
            mydict = {
                k: v
                for k, v in sorted(
                    mydict.items(), key=lambda item: item[1], reverse=True
                )
            }
            outline.append(sentence + "," + str(mydict))
            # print(mydict)
            accum = [0 for i in range(len(Mapdict))]
            sentence = ""
    with open("./customoutput/" + setname + ".csv", "wb") as file:
        for line in outline:
            file.write(line.encode())
            file.write("\n".encode())
    return outline
    # print(words)
    # print(len(possibility[0]))


def AcumAll(path, mapdict, outputstring, dataset_type):
    return PossibilityAcum(
        outputstring=outputstring,
        Mapdict=mapdict,
        path=path,
        setname=dataset_type,
    )

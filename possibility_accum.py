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


Mapdict = [
    (0, "B-Advertising"),
    (1, "B-BankCharges"),
    (2, "B-Catering"),
    (3, "B-Clothing&Hair&Makeup"),
    (4, "B-FreightCharge"),
    (5, "B-Insurance"),
    (6, "B-LOC"),
    (7, "B-MISC"),
    (8, "B-Meals&Entertainment"),
    (9, "B-ORG"),
    (10, "B-OfficeExpense"),
    (11, "B-PER"),
    (12, "B-Postage"),
    (13, "B-SmallEquipment"),
    (14, "B-SoftwareSubscription"),
    (15, "B-Telephone"),
    (16, "B-Travel&Transportation"),
    (17, "B-Unknown"),
    (18, "B-VehicleRepairs"),
    (19, "B-VehiclesGas"),
    (20, "B-VehiclesParking"),
    (21, "E-Advertising"),
    (22, "E-BankCharges"),
    (23, "E-Catering"),
    (24, "E-Clothing&Hair&Makeup"),
    (25, "E-FreightCharge"),
    (26, "E-Insurance"),
    (27, "E-LOC"),
    (28, "E-MISC"),
    (29, "E-Meals&Entertainment"),
    (30, "E-ORG"),
    (31, "E-OfficeExpense"),
    (32, "E-PER"),
    (33, "E-Postage"),
    (34, "E-SmallEquipment"),
    (35, "E-SoftwareSubscription"),
    (36, "E-Telephone"),
    (37, "E-Travel&Transportation"),
    (38, "E-Unknown"),
    (39, "E-VehicleRepairs"),
    (40, "E-VehiclesGas"),
    (41, "E-VehiclesParking"),
    (42, "I-Advertising"),
    (43, "I-BankCharges"),
    (44, "I-Catering"),
    (45, "I-Clothing&Hair&Makeup"),
    (46, "I-FreightCharge"),
    (47, "I-Insurance"),
    (48, "I-LOC"),
    (49, "I-MISC"),
    (50, "I-Meals&Entertainment"),
    (51, "I-ORG"),
    (52, "I-OfficeExpense"),
    (53, "I-PER"),
    (54, "I-Postage"),
    (55, "I-SmallEquipment"),
    (56, "I-SoftwareSubscription"),
    (57, "I-Telephone"),
    (58, "I-Travel&Transportation"),
    (59, "I-Unknown"),
    (60, "I-VehicleRepairs"),
    (61, "I-VehiclesGas"),
    (62, "I-VehiclesParking"),
    (63, "O"),
    (64, "S-Advertising"),
    (65, "S-BankCharges"),
    (66, "S-Catering"),
    (67, "S-Clothing&Hair&Makeup"),
    (68, "S-FreightCharge"),
    (69, "S-Insurance"),
    (70, "S-LOC"),
    (71, "S-MISC"),
    (72, "S-Meals&Entertainment"),
    (73, "S-ORG"),
    (74, "S-OfficeExpense"),
    (75, "S-PER"),
    (76, "S-Postage"),
    (77, "S-SmallEquipment"),
    (78, "S-SoftwareSubscription"),
    (79, "S-Telephone"),
    (80, "S-Travel&Transportation"),
    (81, "S-Unknown"),
    (82, "S-VehicleRepairs"),
    (83, "S-VehiclesGas"),
    (84, "S-VehiclesParking"),
]


def PossibilityAcum(
    outputfilepath,
    Mapdict=Mapdict,
    path="/home/lhy/Wowerz/NeuroNER-master/data/_new_dataset2022/en/valid",
):

    # Change the directory
    os.chdir(path)

    # read original data

    rawlines = []

    def read_text_file(file_path, lines=rawlines):
        with open(file_path) as file_in:

            for line in file_in:
                if line != "":
                    lines.append(line)

    file_path = []

    # iterate through all file
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path.append(f"{path}//{file}")

    file_path = sorted(file_path)
    # call read text file function
    for i in file_path:
        read_text_file(i)

    rawlines = [x for x in rawlines if x != ""]

    # process categories
    categories = list(Mapdict)[0:20]
    for i in range(len(categories)):
        categories[i] = categories[i][1]
        categories[i] = categories[i][2:]
    Mapdict = dict((x, y) for x, y in Mapdict)

    # read output file
    with open(outputfilepath + "/000_valid.txt") as file_in:
        lines = []
        for line in file_in:
            if line != "":
                lines.append(line)
    words = [i.split() for i in lines]
    possibility = []
    # Seperate words and possibilities
    for i in range(len(words)):
        if words[i] != []:
            possibility.append(words[i][6:-2])
            words[i] = words[i][0]
            # print(words[i])

    accum = [0 for i in range(85)]
    sentence = ""
    possibilitystore = []
    flag = 0
    outline = []

    sentenceindex = 0
    rawsentence = rawlines[sentenceindex].split()

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
            # print(pos[i])
            if flag == 0:
                possibilitystore.append(pos)
                flag = 1
            accum = [sum(x) for x in zip(accum, pos)]
            # print(words[i])
            sentence = sentence + " " + currentwords
        else:
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
                possblemap.append(round(float(accum[k] / sumup), 4))
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
            possblehash = zip(inv_map, simp_pos)
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
            accum = [0 for i in range(85)]
            sentence = ""
    with open("csvfile_5.csv", "wb") as file:
        for line in outline:
            file.write(line.encode())
            file.write("\n".encode())
    # print(words)
    # print(len(possibility[0]))


PossibilityAcum("/home/lhy/Wowerz/NeuroNER-master/output/en_2022-02-09_15-48-08-96993")

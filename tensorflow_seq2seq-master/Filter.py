import enchant
d = enchant.Dict("en_US")
keeplist=[]
with open("./data/normal.txt") as f:
    normallist=[]

    for index,line in enumerate(f.readlines()):
        wrongcount=0
        for word in line.split(" "):
            if d.check(word) is False and len(word) > 2:
            # print(len(word))
                wrongcount=1
                break;

        if wrongcount == 0 and len(line) > 14:
            normallist.append(line)
            keeplist.append(index)





## perfect filter 5W
with open("./data/prenormal.txt",'w')as f:
    count=0;
    for line in normallist:
        f.write(line)
        count+=1
    print(count)
with open("./data/simple.txt") as f:
    simplelist=[]
    for index, line in enumerate(f.readlines()):
        if index  in keeplist:
            simplelist.append(line)

with open("./data/presimple.txt" , "w") as f:
    for line in simplelist:
        f.write(line)


# 10W rough filter and not completed
# with open("./data/prenormal.txt",'w')as f:
#     valid=0
#     for line in normallist:
#         wrongcount=0
#         for word in line.split(" ") :
#             if d.check(word) is False :
#                 # print(len(word))
#                 wrongcount+=1
#                 break;
#
#         if wrongcount <= 6 and len(line) > 14:
#             valid+=1
#
#             f.write(line)
#
#     print(valid)




